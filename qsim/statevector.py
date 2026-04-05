"""Statevector quantum circuit simulator using tensor contraction.

The core idea: never build 2^n × 2^n gate matrices. Instead, reshape the
statevector to a rank-n tensor (2,2,...,2) and apply gates via np.einsum
on the target qubit axes.

Complexity per gate application:
  Single-qubit: O(2^n) — contracts (2,2) gate with (2,)*n tensor over 1 axis
  Two-qubit:    O(2^n) — contracts (2,2,2,2) gate over 2 axes

Compare to full matrix multiplication: O(4^n) for Kronecker product + matvec.
For n=12, tensor contraction is ~4000x faster.

Reference: Markov & Shi (2008), "Simulating Quantum Computation by Contracting
Tensor Networks."
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .circuit import Circuit, Instruction


class Statevector:
    """Statevector simulator using tensor contraction via np.einsum."""

    def __init__(
        self,
        num_qubits: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._n = num_qubits
        if initial_state is not None:
            if initial_state.shape != (2**num_qubits,):
                raise ValueError(
                    f"initial_state shape must be ({2**num_qubits},), "
                    f"got {initial_state.shape}"
                )
            self._data = initial_state.astype(complex).copy()
        else:
            self._data = np.zeros(2**num_qubits, dtype=complex)
            self._data[0] = 1.0

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def data(self) -> np.ndarray:
        """Flat (2^n,) complex array. Returns a copy."""
        return self._data.copy()

    @property
    def tensor(self) -> np.ndarray:
        """View as (2,2,...,2) tensor (n axes). Returns a copy."""
        return self._data.copy().reshape((2,) * self._n)

    def _apply_single_qubit_gate(
        self, gate_matrix: np.ndarray, target: int
    ) -> None:
        """Apply a 2x2 unitary to the target qubit via einsum. O(2^n).

        Reshapes statevector to (2,)*n tensor, contracts gate matrix over
        the target qubit axis, then flattens back.
        """
        n = self._n
        t = self._data.reshape((2,) * n)

        state_indices = list(range(n))
        gate_out = n
        out_indices = list(range(n))
        out_indices[target] = gate_out

        result = np.einsum(
            gate_matrix, [gate_out, target],
            t, state_indices,
            out_indices,
        )
        self._data = result.reshape(2**n)

    def _apply_two_qubit_gate(
        self, gate_matrix: np.ndarray, qubit0: int, qubit1: int
    ) -> None:
        """Apply a 4x4 unitary to (qubit0, qubit1) via einsum. O(2^n).

        The gate matrix is reshaped to (2,2,2,2) tensor where axes are
        (out0, out1, in0, in1). Contracts over in0=qubit0, in1=qubit1.
        """
        n = self._n
        t = self._data.reshape((2,) * n)
        gate_tensor = gate_matrix.reshape(2, 2, 2, 2)

        state_indices = list(range(n))
        out0 = n
        out1 = n + 1
        out_indices = list(range(n))
        out_indices[qubit0] = out0
        out_indices[qubit1] = out1

        result = np.einsum(
            gate_tensor, [out0, out1, qubit0, qubit1],
            t, state_indices,
            out_indices,
        )
        self._data = result.reshape(2**n)

    def apply_instruction(self, inst: Instruction) -> None:
        """Apply a single (fully bound) instruction to the statevector."""
        if inst.is_parameterized:
            raise ValueError(
                "Cannot simulate unbound parameters. "
                "Call circuit.bind_parameters() first."
            )
        matrix = inst.gate.matrix(*inst.params)

        if inst.gate.num_qubits == 1:
            self._apply_single_qubit_gate(matrix, inst.qubits[0])
        elif inst.gate.num_qubits == 2:
            self._apply_two_qubit_gate(matrix, inst.qubits[0], inst.qubits[1])
        else:
            raise NotImplementedError("Gates on >2 qubits not supported")  # pragma: no cover

    def evolve(self, circuit: Circuit) -> Statevector:
        """Apply an entire circuit. Returns self for chaining."""
        if circuit.is_parameterized():
            raise ValueError("Circuit has unbound parameters.")
        for inst in circuit.instructions:
            self.apply_instruction(inst)
        return self

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities for all computational basis states."""
        return np.abs(self._data) ** 2

    def probability(self, bitstring: str) -> float:
        """Probability of measuring a specific bitstring like '101'."""
        from .utils import computational_basis_index

        idx = computational_basis_index(bitstring)
        return float(np.abs(self._data[idx]) ** 2)

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> Statevector:
        """Create |00...0> and evolve through the circuit."""
        sv = cls(circuit.num_qubits)
        sv.evolve(circuit)
        return sv

    def copy(self) -> Statevector:
        """Return an independent copy of this statevector."""
        return Statevector(self._n, self._data)

    def inner(self, other: Statevector) -> complex:
        """Compute <self|other>."""
        return complex(np.vdot(self._data, other._data))

    def norm(self) -> float:
        """L2 norm of the statevector (should be 1.0 for valid states)."""
        return float(np.linalg.norm(self._data))

    def __repr__(self) -> str:
        return f"Statevector(num_qubits={self._n}, norm={self.norm():.6f})"
