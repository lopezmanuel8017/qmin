"""Quantum kernel for computing feature-space similarity.

Computes K(x_i, x_j) = |<0|U^dag(x_j) U(x_i)|0>|^2, the fidelity between
two quantum-encoded feature vectors. This kernel operates in 2^n-dimensional
Hilbert space, providing an inductive bias that classical kernels of polynomial
dimension cannot efficiently reproduce (Liu et al. 2021).

The kernel is trainable: the ansatz parameters affect the feature map U(x),
and can be optimized via the parameter-shift rule.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qsim.circuit import Circuit
from qsim.statevector import Statevector
from quantum.ansatz import CNOTLadder, RingTopology
from quantum.encoding import AngleEncoder


class QuantumKernel:
    """Trainable quantum kernel using fidelity in Hilbert space.

    K(x_i, x_j) = |<psi(x_j)|psi(x_i)>|^2
    where |psi(x)> = U_ansatz * U_encode(x) |0>
    """

    def __init__(
        self,
        num_qubits: int,
        ansatz: Optional[CNOTLadder | RingTopology] = None,
        encoder: Optional[AngleEncoder] = None,
        init_strategy: str = "uniform",
        init_epsilon: float = 0.01,
    ) -> None:
        self.num_qubits = num_qubits
        self.ansatz = ansatz or CNOTLadder(num_qubits, 1)
        self.encoder = encoder or AngleEncoder(num_qubits)

        self._encoding_circuit = self.encoder.circuit()
        self._ansatz_circuit = self.ansatz.circuit()

        self._trainable_params = self.ansatz.parameters
        self._param_values = self.ansatz.init_params(
            seed=42, strategy=init_strategy, epsilon=init_epsilon,
        )

        self._param_array = np.array([
            self._param_values[p] for p in self._trainable_params
        ])
        self._grad_array = np.zeros(len(self._trainable_params))

    def _encode(self, features: np.ndarray) -> Statevector:
        """Encode features into a quantum state: U_ansatz * U_encode(x) |0>."""
        bindings = self.encoder.bind(features)
        bindings.update(self._param_values)

        qc = Circuit(self.num_qubits, name="kernel_encode")
        qc.compose(self._encoding_circuit)
        qc.compose(self._ansatz_circuit)
        bound = qc.bind_parameters(bindings)
        return Statevector.from_circuit(bound)

    def compute_entry(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        """Compute K(x_i, x_j) = |<psi(x_j)|psi(x_i)>|^2."""
        sv_i = self._encode(x_i)
        sv_j = self._encode(x_j)
        overlap = sv_j.inner(sv_i)
        return float(abs(overlap) ** 2)

    def compute_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute the full kernel matrix K[i,j] = K(x_i, x_j).

        Exploits symmetry: only computes upper triangle. K[i,i] = 1.0.

        Args:
            features: (n_samples, n_features) array

        Returns:
            K: (n_samples, n_samples) symmetric positive semi-definite matrix
        """
        n = features.shape[0]
        K = np.zeros((n, n))

        states = [self._encode(features[i]) for i in range(n)]

        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                overlap = states[j].inner(states[i])
                val = float(abs(overlap) ** 2)
                K[i, j] = val
                K[j, i] = val

        return K

    def sync_from_array(self) -> None:
        """Update internal parameter dict from the persistent flat array."""
        for i, p in enumerate(self._trainable_params):
            self._param_values[p] = float(self._param_array[i])

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self._param_array, self._grad_array)]
