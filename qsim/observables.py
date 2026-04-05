"""Pauli observables for expectation value computation.

An Observable is a Hermitian operator expressed as a sum of Pauli tensor products:
  O = Σ_i c_i · P_i
where each P_i is a tensor product of single-qubit Pauli operators {I, X, Y, Z}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

I_MATRIX = np.eye(2, dtype=complex)
X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_LABELS: dict[str, np.ndarray] = {
    "I": I_MATRIX, "X": X_MATRIX, "Y": Y_MATRIX, "Z": Z_MATRIX,
}


@dataclass(frozen=True)
class PauliTerm:
    """A tensor product of Pauli operators with a coefficient.

    Example: 0.5 * Z_0 Z_2 is PauliTerm(coeff=0.5, ops=((0, 'Z'), (2, 'Z')))
    Non-specified qubits are implicitly I (identity).

    ops is stored as a tuple of (qubit, label) pairs for immutability.
    """

    coeff: complex
    ops: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        for qubit, label in self.ops:
            if label not in PAULI_LABELS:
                raise ValueError(f"Invalid Pauli label: '{label}'")
            if qubit < 0:
                raise ValueError(f"Negative qubit index: {qubit}")

    @property
    def qubits(self) -> tuple[int, ...]:
        return tuple(sorted(q for q, _ in self.ops))

    @property
    def ops_dict(self) -> dict[int, str]:
        return dict(self.ops)

    def matrix_on_qubit(self, qubit: int) -> np.ndarray:
        """Return the 2x2 Pauli matrix acting on the given qubit."""
        d = self.ops_dict
        label = d.get(qubit, "I")
        return PAULI_LABELS[label]

    def __repr__(self) -> str:
        if not self.ops:
            return f"{self.coeff} * I"
        op_str = " ".join(f"{label}{qubit}" for qubit, label in sorted(self.ops))
        return f"{self.coeff} * {op_str}"


class Observable:
    """A Hermitian observable as a sum of PauliTerms.

    H = Σ_i c_i · P_i where P_i is a tensor product of Paulis.
    """

    def __init__(self, terms: Sequence[PauliTerm]) -> None:
        self._terms = list(terms)

    @property
    def terms(self) -> list[PauliTerm]:
        return list(self._terms)

    @property
    def num_qubits_required(self) -> int:
        """Minimum number of qubits needed."""
        if not self._terms:
            return 0
        all_qubits = [q for t in self._terms for q in t.qubits]
        if not all_qubits:
            return 0
        return max(all_qubits) + 1

    def __add__(self, other: Observable) -> Observable:
        return Observable(self._terms + other._terms)

    def __mul__(self, scalar: complex) -> Observable:
        return Observable(
            [PauliTerm(t.coeff * scalar, t.ops) for t in self._terms]
        )

    def __rmul__(self, scalar: complex) -> Observable:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return " + ".join(repr(t) for t in self._terms)

    @classmethod
    def z(cls, qubit: int) -> Observable:
        """Single-qubit Z observable."""
        return cls([PauliTerm(1.0, ((qubit, "Z"),))])

    @classmethod
    def zz(cls, qubit0: int, qubit1: int) -> Observable:
        """Two-qubit ZZ observable."""
        return cls([PauliTerm(1.0, ((qubit0, "Z"), (qubit1, "Z")))])

    @classmethod
    def x(cls, qubit: int) -> Observable:
        """Single-qubit X observable."""
        return cls([PauliTerm(1.0, ((qubit, "X"),))])

    @classmethod
    def identity(cls) -> Observable:
        """Identity observable (always measures 1)."""
        return cls([PauliTerm(1.0, ())])

    @classmethod
    def from_pauli_string(cls, pauli_str: str, coeff: complex = 1.0) -> Observable:
        """Parse 'ZZIY' -> Z0 Z1 Y3 (skip I).

        Characters map to qubits left-to-right: position 0 -> qubit 0.
        """
        ops: list[tuple[int, str]] = []
        for i, ch in enumerate(pauli_str):
            if ch not in PAULI_LABELS:
                raise ValueError(f"Invalid Pauli character: '{ch}'")
            if ch != "I":
                ops.append((i, ch))
        return cls([PauliTerm(coeff, tuple(ops))])
