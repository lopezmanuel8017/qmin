"""Quantum gate definitions.

Each gate is a GateDefinition: an immutable value object holding the gate name,
qubit count, parameter count, a matrix factory, and the OpenQASM name.

Rotation gates follow R_k(θ) = exp(-iθσ_k/2) (Nielsen & Chuang, Ch. 4.2):
  Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
  Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
  Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class GateDefinition:
    """Immutable definition of a quantum gate."""

    name: str
    num_qubits: int
    num_params: int
    matrix_fn: Callable[..., np.ndarray]
    qasm_name: str
    label: Optional[str] = None

    def matrix(self, *params: float) -> np.ndarray:
        """Return the unitary matrix for given parameters."""
        if len(params) != self.num_params:
            raise ValueError(
                f"Gate {self.name} expects {self.num_params} params, got {len(params)}"
            )
        return self.matrix_fn(*params)

    def tensor(self, *params: float) -> np.ndarray:
        """Return matrix reshaped as (2,)*2k tensor for k-qubit gate."""
        m = self.matrix(*params)
        k = self.num_qubits
        return m.reshape((2,) * (2 * k))


H = GateDefinition(
    name="H", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    qasm_name="h",
)

X = GateDefinition(
    name="X", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array([[0, 1], [1, 0]], dtype=complex),
    qasm_name="x",
)

Y = GateDefinition(
    name="Y", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array([[0, -1j], [1j, 0]], dtype=complex),
    qasm_name="y",
)

Z = GateDefinition(
    name="Z", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array([[1, 0], [0, -1]], dtype=complex),
    qasm_name="z",
)

S = GateDefinition(
    name="S", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array([[1, 0], [0, 1j]], dtype=complex),
    qasm_name="s",
)

T = GateDefinition(
    name="T", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.array(
        [[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex
    ),
    qasm_name="t",
)

I_GATE = GateDefinition(
    name="I", num_qubits=1, num_params=0,
    matrix_fn=lambda: np.eye(2, dtype=complex),
    qasm_name="id",
)


def _rx_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


Rx = GateDefinition(
    name="Rx", num_qubits=1, num_params=1,
    matrix_fn=_rx_matrix, qasm_name="rx",
)


def _ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


Ry = GateDefinition(
    name="Ry", num_qubits=1, num_params=1,
    matrix_fn=_ry_matrix, qasm_name="ry",
)


def _rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


Rz = GateDefinition(
    name="Rz", num_qubits=1, num_params=1,
    matrix_fn=_rz_matrix, qasm_name="rz",
)


def _phase_matrix(phi: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


Phase = GateDefinition(
    name="Phase", num_qubits=1, num_params=1,
    matrix_fn=_phase_matrix, qasm_name="p",
)


CNOT = GateDefinition(
    name="CNOT", num_qubits=2, num_params=0,
    matrix_fn=lambda: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    ),
    qasm_name="cx",
)

CZ = GateDefinition(
    name="CZ", num_qubits=2, num_params=0,
    matrix_fn=lambda: np.diag([1, 1, 1, -1]).astype(complex),
    qasm_name="cz",
)

SWAP = GateDefinition(
    name="SWAP", num_qubits=2, num_params=0,
    matrix_fn=lambda: np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    ),
    qasm_name="swap",
)


def _controlled_rotation(rotation_fn: Callable[[float], np.ndarray]):
    """Build a controlled-rotation matrix factory from a single-qubit rotation."""
    def matrix_fn(theta: float) -> np.ndarray:
        r = rotation_fn(theta)
        m = np.eye(4, dtype=complex)
        m[2:, 2:] = r
        return m
    return matrix_fn


CRx = GateDefinition(
    name="CRx", num_qubits=2, num_params=1,
    matrix_fn=_controlled_rotation(_rx_matrix), qasm_name="crx",
)

CRy = GateDefinition(
    name="CRy", num_qubits=2, num_params=1,
    matrix_fn=_controlled_rotation(_ry_matrix), qasm_name="cry",
)

CRz = GateDefinition(
    name="CRz", num_qubits=2, num_params=1,
    matrix_fn=_controlled_rotation(_rz_matrix), qasm_name="crz",
)

GATE_REGISTRY: dict[str, GateDefinition] = {
    g.name: g
    for g in [
        H, X, Y, Z, S, T, I_GATE,
        Rx, Ry, Rz, Phase,
        CNOT, CZ, SWAP, CRx, CRy, CRz,
    ]
}
