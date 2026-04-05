"""Quantum data encoding schemes.

Encoding classical data into quantum states is the first step in any
quantum machine learning pipeline.

AngleEncoder:
  Maps feature x_i to Ry(x_i) rotation on qubit i.
  Simple, NISQ-friendly, one feature per qubit.
  Reference: Schuld & Petruccione (2021), Ch. 5

TriValueEncoder:
  Maps RGB values (r, g, b) to Ry(r)·Rz(g)·Ry(b) on a single qubit.
  Compacts 3 channels into 1 qubit using sequential rotations.
  Reference: CE-QCNN (2025), Tri-Value Qubit Encoding
"""

from __future__ import annotations

import numpy as np

from qsim.circuit import Circuit
from qsim.parameters import Parameter


class AngleEncoder:
    """Encode features as Ry rotation angles on qubits.

    Given features [x_0, x_1, ..., x_{n-1}], applies Ry(x_i) to qubit i.
    Number of qubits = number of features.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self._params = [Parameter(f"enc_{i}") for i in range(num_qubits)]

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._params)

    def circuit(self) -> Circuit:
        """Return parameterized encoding circuit."""
        qc = Circuit(self.num_qubits, name="angle_encoding")
        for i, p in enumerate(self._params):
            qc.ry(p, i)
        return qc

    def bind(self, features: np.ndarray) -> dict[Parameter, float]:
        """Map a feature vector to parameter bindings.

        features: array of shape (num_qubits,)
        Returns: dict mapping Parameter -> float value
        """
        if features.shape != (self.num_qubits,):
            raise ValueError(
                f"Expected features shape ({self.num_qubits},), "
                f"got {features.shape}"
            )
        return {p: float(v) for p, v in zip(self._params, features)}


class TriValueEncoder:
    """Encode RGB triplets using Ry(r)·Rz(g)·Ry(b) per qubit.

    Each qubit encodes 3 values (e.g., RGB channels of a pixel patch).
    Number of qubits = number of triplets.

    Reference: CE-QCNN (2025), Tri-Value Qubit Encoding scheme.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self._params: list[tuple[Parameter, Parameter, Parameter]] = []
        for i in range(num_qubits):
            self._params.append((
                Parameter(f"tv_{i}_r"),
                Parameter(f"tv_{i}_g"),
                Parameter(f"tv_{i}_b"),
            ))

    @property
    def parameters(self) -> list[Parameter]:
        return [p for triplet in self._params for p in triplet]

    def circuit(self) -> Circuit:
        """Return parameterized encoding circuit."""
        qc = Circuit(self.num_qubits, name="trivalue_encoding")
        for i, (r, g, b) in enumerate(self._params):
            qc.ry(r, i).rz(g, i).ry(b, i)
        return qc

    def bind(self, rgb_values: np.ndarray) -> dict[Parameter, float]:
        """Map RGB triplets to parameter bindings.

        rgb_values: array of shape (num_qubits, 3) where columns are R, G, B.
        Returns: dict mapping Parameter -> float value
        """
        if rgb_values.shape != (self.num_qubits, 3):
            raise ValueError(
                f"Expected shape ({self.num_qubits}, 3), got {rgb_values.shape}"
            )
        bindings: dict[Parameter, float] = {}
        for i, (r_p, g_p, b_p) in enumerate(self._params):
            bindings[r_p] = float(rgb_values[i, 0])
            bindings[g_p] = float(rgb_values[i, 1])
            bindings[b_p] = float(rgb_values[i, 2])
        return bindings
