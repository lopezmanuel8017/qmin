"""Density matrix quantum simulator with noise support.

Represents mixed quantum states as density matrices ρ (2^n × 2^n).
Supports:
  - Unitary evolution: ρ' = U ρ U†
  - Noise channels via Kraus operators: ρ' = Σ_k E_k ρ E_k†
  - Integration with NoiseModel for per-gate noise

The density matrix is stored as a (2^n, 2^n) complex array internally,
reshaped to (2,)*2n for efficient einsum-based gate application (mirroring
the Statevector tensor contraction approach).

Memory: O(4^n) vs O(2^n) for statevector. Practical limit ~10-12 qubits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .observables import Observable
from .circuit import Circuit, Instruction
from .noise import NoiseChannel, NoiseModel
from .statevector import Statevector


class DensityMatrix:
    """Density matrix simulator supporting noise channels."""

    def __init__(
        self,
        num_qubits: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._n = num_qubits
        dim = 2**num_qubits
        if initial_state is not None:
            if initial_state.shape != (dim, dim):
                raise ValueError(
                    f"initial_state shape must be ({dim}, {dim}), "
                    f"got {initial_state.shape}"
                )
            self._data = initial_state.astype(complex).copy()
        else:
            self._data = np.zeros((dim, dim), dtype=complex)
            self._data[0, 0] = 1.0

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def data(self) -> np.ndarray:
        """(2^n, 2^n) density matrix. Returns a copy."""
        return self._data.copy()

    def _apply_single_qubit_unitary(
        self, gate_matrix: np.ndarray, target: int
    ) -> None:
        """Apply ρ' = U_target ρ U_target† via two einsum contractions.

        Reshape ρ to (2,)*2n tensor. First n axes are 'ket' indices,
        last n are 'bra' indices.
        """
        n = self._n
        dim = 2**n
        rho = self._data.reshape((2,) * (2 * n))
        U = gate_matrix
        U_dag = gate_matrix.conj().T

        ket_indices = list(range(2 * n))
        u_out = 2 * n
        out_indices = list(ket_indices)
        out_indices[target] = u_out
        rho = np.einsum(
            U, [u_out, target],
            rho, ket_indices,
            out_indices,
        )

        bra_idx = n + target
        bra_indices = list(range(2 * n))
        bra_indices[target] = u_out
        u_dag_out = 2 * n + 1
        out_indices2 = list(bra_indices)
        out_indices2[bra_idx] = u_dag_out
        rho = np.einsum(
            U_dag, [bra_idx, u_dag_out],
            rho, bra_indices,
            out_indices2,
        )

        self._data = rho.reshape(dim, dim)

    def _apply_two_qubit_unitary(
        self, gate_matrix: np.ndarray, qubit0: int, qubit1: int
    ) -> None:
        """Apply ρ' = U_{q0,q1} ρ U†_{q0,q1} via einsum."""
        n = self._n
        dim = 2**n
        rho = self._data.reshape((2,) * (2 * n))
        U = gate_matrix.reshape(2, 2, 2, 2)
        U_dag = gate_matrix.conj().T.reshape(2, 2, 2, 2)

        ket_indices = list(range(2 * n))
        out0 = 2 * n
        out1 = 2 * n + 1
        out_indices = list(ket_indices)
        out_indices[qubit0] = out0
        out_indices[qubit1] = out1
        rho = np.einsum(
            U, [out0, out1, qubit0, qubit1],
            rho, ket_indices,
            out_indices,
        )

        bra0 = n + qubit0
        bra1 = n + qubit1
        cur_indices = list(range(2 * n))
        cur_indices[qubit0] = out0
        cur_indices[qubit1] = out1
        udag_out0 = 2 * n + 2
        udag_out1 = 2 * n + 3
        final_indices = list(cur_indices)
        final_indices[bra0] = udag_out0
        final_indices[bra1] = udag_out1
        rho = np.einsum(
            U_dag, [bra0, bra1, udag_out0, udag_out1],
            rho, cur_indices,
            final_indices,
        )

        self._data = rho.reshape(dim, dim)

    def apply_instruction(self, inst: Instruction) -> None:
        """Apply a unitary gate instruction."""
        if inst.is_parameterized:
            raise ValueError("Cannot simulate unbound parameters.")
        matrix = inst.gate.matrix(*inst.params)

        if inst.gate.num_qubits == 1:
            self._apply_single_qubit_unitary(matrix, inst.qubits[0])
        elif inst.gate.num_qubits == 2:
            self._apply_two_qubit_unitary(matrix, inst.qubits[0], inst.qubits[1])
        else:
            raise NotImplementedError("Gates on >2 qubits not supported")  # pragma: no cover

    def apply_noise(self, channel: NoiseChannel, target: int) -> None:
        """Apply a single-qubit noise channel: ρ' = Σ_k E_k ρ E_k†."""
        result = np.zeros_like(self._data)
        for E in channel.kraus_ops:
            saved = self._data.copy()
            self._apply_single_qubit_unitary(E, target)
            result += self._data
            self._data = saved
        self._data = result

    def evolve(
        self,
        circuit: Circuit,
        noise_model: Optional[NoiseModel] = None,
    ) -> DensityMatrix:
        """Apply circuit with optional noise after each gate."""
        if circuit.is_parameterized():
            raise ValueError("Circuit has unbound parameters.")
        for inst in circuit.instructions:
            self.apply_instruction(inst)
            if noise_model is not None:
                channel = noise_model.get_gate_noise(inst.gate.name)
                if channel is not None:
                    for q in inst.qubits:
                        self.apply_noise(channel, q)
        return self

    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        noise_model: Optional[NoiseModel] = None,
    ) -> DensityMatrix:
        """Create |0><0| and evolve through the circuit."""
        dm = cls(circuit.num_qubits)
        dm.evolve(circuit, noise_model)
        return dm

    @classmethod
    def from_statevector(cls, sv: Statevector) -> DensityMatrix:
        """Create |ψ><ψ| from a pure statevector."""
        data = sv.data
        rho = np.outer(data, data.conj())
        return cls(sv.num_qubits, rho)

    def expectation_value(self, observable: Observable) -> float:
        """Compute Tr(O · ρ) for a Pauli observable.

        Uses the identity: Tr(O·ρ) = Σ_term coeff * Tr(P_term · ρ)
        """
        from .observables import PAULI_LABELS

        result = 0.0
        for term in observable.terms:
            dim = 2**self._n
            pauli = np.eye(dim, dtype=complex)
            for qubit, label in term.ops:
                single = PAULI_LABELS[label]
                full = np.eye(1, dtype=complex)
                for q in range(self._n):
                    if q == qubit:
                        full = np.kron(full, single)
                    else:
                        full = np.kron(full, np.eye(2, dtype=complex))
                pauli = pauli @ full
            result += term.coeff * np.trace(pauli @ self._data)
        return float(np.real(result))

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities: diagonal of ρ."""
        return np.real(np.diag(self._data))

    def trace(self) -> float:
        """Trace of the density matrix (should be 1.0)."""
        return float(np.real(np.trace(self._data)))

    def purity(self) -> float:
        """Tr(ρ²). 1.0 for pure states, 1/d for maximally mixed."""
        return float(np.real(np.trace(self._data @ self._data)))

    def __repr__(self) -> str:
        return f"DensityMatrix(num_qubits={self._n}, trace={self.trace():.6f}, purity={self.purity():.6f})"
