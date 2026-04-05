"""Hybrid quantum-classical layer.

Bridges classical neural network layers with quantum circuit execution.
Handles encoding, circuit execution, measurement, and gradient routing.

Forward:
  1. Receive (batch_size, n_features) from classical backbone
  2. For each sample: encode features -> run variational circuit -> measure P(|1>)
  3. Return (batch_size, n_qubits) measurement probabilities

Backward:
  1. Receive dL/d(output) from classical head
  2. For quantum params: parameter-shift rule
  3. For input: finite-difference on encoding angles (optional)
  4. Return dL/d(input) for backbone

Uses exact expectation values (no shot noise) for training stability.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qsim.circuit import Circuit
from qsim.density_matrix import DensityMatrix
from qsim.gradient import PARAMETER_SHIFT
from qsim.measurement import expectation_value
from qsim.noise import NoiseModel
from qsim.observables import Observable
from qsim.parameters import Parameter
from qsim.statevector import Statevector
from quantum.ansatz import CNOTLadder, RingTopology
from quantum.encoding import AngleEncoder


class HybridQuantumClassicalLayer:
    """Quantum layer that sits between classical layers in a hybrid model.

    Encodes classical features into quantum states, processes through a
    variational circuit, and returns measurement probabilities.
    """

    def __init__(
        self,
        num_qubits: int,
        ansatz: CNOTLadder | RingTopology,
        encoder: Optional[AngleEncoder] = None,
        compute_input_grad: bool = False,
        init_strategy: str = "uniform",
        init_epsilon: float = 0.01,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.ansatz = ansatz
        self.encoder = encoder or AngleEncoder(num_qubits)
        self.compute_input_grad = compute_input_grad
        self.noise_model = noise_model

        self._encoding_circuit = self.encoder.circuit()
        self._ansatz_circuit = ansatz.circuit()

        self._observables = [Observable.z(i) for i in range(num_qubits)]

        self._trainable_params = ansatz.parameters
        self._param_values = ansatz.init_params(
            seed=42, strategy=init_strategy, epsilon=init_epsilon,
        )

        self._grad_params: dict[Parameter, float] = {p: 0.0 for p in self._trainable_params}

        self._param_array = np.array([self._param_values[p] for p in self._trainable_params])
        self._grad_array = np.zeros(len(self._trainable_params))

        self._input: Optional[np.ndarray] = None
        self._output: Optional[np.ndarray] = None
        self._was_unbatched = False

    @property
    def trainable_params(self) -> dict[Parameter, float]:
        return dict(self._param_values)

    @property
    def grad_params(self) -> dict[Parameter, float]:
        return dict(self._grad_params)

    def _build_full_circuit(self, feature_bindings: dict[Parameter, float]) -> Circuit:
        """Build encoding + ansatz circuit with all parameters bound."""
        full_bindings = {**feature_bindings, **self._param_values}
        full_circuit = Circuit(self.num_qubits, name="hybrid")
        full_circuit.compose(self._encoding_circuit)
        full_circuit.compose(self._ansatz_circuit)
        return full_circuit.bind_parameters(full_bindings)

    def _evaluate_sample(self, features: np.ndarray) -> np.ndarray:
        """Run one sample through the quantum circuit.

        Returns: array of shape (num_qubits,) with P(|1>) per qubit.
        P(|1>) = (1 - <Z>) / 2.

        When noise_model is set, uses DensityMatrix simulation for
        realistic noisy expectation values.
        """
        bindings = self.encoder.bind(features)
        bound_circuit = self._build_full_circuit(bindings)

        if self.noise_model is not None:
            dm = DensityMatrix.from_circuit(bound_circuit, self.noise_model)
            measurements = np.array([
                (1.0 - dm.expectation_value(obs)) / 2.0
                for obs in self._observables
            ])
        else:
            sv = Statevector.from_circuit(bound_circuit)
            measurements = np.array([
                (1.0 - expectation_value(sv, obs)) / 2.0
                for obs in self._observables
            ])
        return measurements

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: encode features and measure.

        x: (batch_size, n_features) or (n_features,)
        Returns: (batch_size, n_qubits) or (n_qubits,)
        """
        self._was_unbatched = (x.ndim == 1)
        if self._was_unbatched:
            x = x[np.newaxis]

        self._input = x
        batch_size = x.shape[0]
        output = np.zeros((batch_size, self.num_qubits))

        for i in range(batch_size):
            output[i] = self._evaluate_sample(x[i])

        self._output = output
        return output[0] if self._was_unbatched else output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients for quantum parameters and optionally input.

        Uses parameter-shift rule for quantum parameter gradients.
        Chain rule: dL/dθ = Σ_i dL/dy_i · dy_i/dθ
        """
        if self._was_unbatched:
            grad_output = grad_output[np.newaxis]

        batch_size = grad_output.shape[0]
        shift = PARAMETER_SHIFT

        self._grad_params = {p: 0.0 for p in self._trainable_params}

        grad_input = np.zeros_like(self._input) if self.compute_input_grad else np.zeros_like(self._input)

        for sample_idx in range(batch_size):
            features = self._input[sample_idx]
            feature_bindings = self.encoder.bind(features)
            base_bindings = {**feature_bindings, **self._param_values}

            for param in self._trainable_params:
                theta = self._param_values[param]

                shifted_plus = dict(base_bindings)
                shifted_plus[param] = theta + shift

                full_circuit_plus = Circuit(self.num_qubits, name="shift+")
                full_circuit_plus.compose(self._encoding_circuit)
                full_circuit_plus.compose(self._ansatz_circuit)
                bound_plus = full_circuit_plus.bind_parameters(shifted_plus)
                sv_plus = Statevector.from_circuit(bound_plus)

                shifted_minus = dict(base_bindings)
                shifted_minus[param] = theta - shift

                full_circuit_minus = Circuit(self.num_qubits, name="shift-")
                full_circuit_minus.compose(self._encoding_circuit)
                full_circuit_minus.compose(self._ansatz_circuit)
                bound_minus = full_circuit_minus.bind_parameters(shifted_minus)
                sv_minus = Statevector.from_circuit(bound_minus)

                for qubit_idx in range(self.num_qubits):
                    exp_plus = expectation_value(sv_plus, self._observables[qubit_idx])
                    exp_minus = expectation_value(sv_minus, self._observables[qubit_idx])
                    d_output_d_param = -(exp_plus - exp_minus) / (2 * np.sin(shift)) / 2.0
                    self._grad_params[param] += grad_output[sample_idx, qubit_idx] * d_output_d_param

            if self.compute_input_grad:
                eps = 1e-4
                for feat_idx in range(features.shape[0]):
                    features_plus = features.copy()
                    features_plus[feat_idx] += eps
                    output_plus = self._evaluate_sample(features_plus)

                    features_minus = features.copy()
                    features_minus[feat_idx] -= eps
                    output_minus = self._evaluate_sample(features_minus)

                    d_output_d_input = (output_plus - output_minus) / (2 * eps)
                    grad_input[sample_idx, feat_idx] = np.sum(
                        grad_output[sample_idx] * d_output_d_input
                    )

        return grad_input[0] if self._was_unbatched else grad_input

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return persistent (param_array, grad_array) pairs for optimizer.

        Returns the SAME array objects across calls — critical for optimizer
        compatibility (optimizer stores references to these arrays).
        """
        return [(self._param_array, self._grad_array)]

    def sync_from_array(self, _param_array: Optional[np.ndarray] = None) -> None:
        """Update internal parameter dict from the persistent flat array.

        Call after optimizer.step() to propagate array updates back to
        the parameter dict used by circuit binding.
        """
        for i, p in enumerate(self._trainable_params):
            self._param_values[p] = float(self._param_array[i])

    def sync_grads_to_array(self) -> None:
        """Copy internal gradient dict to the persistent flat array."""
        for i, p in enumerate(self._trainable_params):
            self._grad_array[i] = self._grad_params[p]
