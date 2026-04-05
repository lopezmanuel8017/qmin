"""Quantum circuit abstraction.

A Circuit is an ordered sequence of Instructions on n qubits. It supports:
- Fluent API for gate application: circuit.h(0).cx(0, 1).rz(theta, 1)
- Symbolic parameters via Parameter objects
- Parameter binding (returns a new circuit, original is preserved)
- Circuit composition and depth calculation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import gates as g
from .parameters import Parameter, ParameterValueType


@dataclass(frozen=True)
class Instruction:
    """A single gate application: gate + target qubits + parameter values."""

    gate: g.GateDefinition
    qubits: tuple[int, ...]
    params: tuple[ParameterValueType, ...]

    @property
    def is_parameterized(self) -> bool:
        return any(isinstance(p, Parameter) for p in self.params)


class Circuit:
    """A quantum circuit: ordered sequence of Instructions on n qubits."""

    def __init__(self, num_qubits: int, name: str = "circuit") -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self._num_qubits = num_qubits
        self._name = name
        self._instructions: list[Instruction] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def name(self) -> str:
        return self._name

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        return tuple(self._instructions)

    @property
    def parameters(self) -> set[Parameter]:
        """All unbound Parameters in this circuit."""
        params: set[Parameter] = set()
        for inst in self._instructions:
            for p in inst.params:
                if isinstance(p, Parameter):
                    params.add(p)
        return params

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    @property
    def ordered_parameters(self) -> list[Parameter]:
        """Parameters in order of first appearance (stable ordering for gradients)."""
        seen: set[Parameter] = set()
        ordered: list[Parameter] = []
        for inst in self._instructions:
            for p in inst.params:
                if isinstance(p, Parameter) and p not in seen:
                    seen.add(p)
                    ordered.append(p)
        return ordered

    @property
    def depth(self) -> int:
        """Circuit depth (longest path through the circuit DAG)."""
        if not self._instructions:
            return 0
        qubit_depth = [0] * self._num_qubits
        for inst in self._instructions:
            max_d = max(qubit_depth[q] for q in inst.qubits)
            for q in inst.qubits:
                qubit_depth[q] = max_d + 1
        return max(qubit_depth)

    def h(self, qubit: int) -> Circuit:
        self._append(g.H, (qubit,), ())
        return self

    def x(self, qubit: int) -> Circuit:
        self._append(g.X, (qubit,), ())
        return self

    def y(self, qubit: int) -> Circuit:
        self._append(g.Y, (qubit,), ())
        return self

    def z(self, qubit: int) -> Circuit:
        self._append(g.Z, (qubit,), ())
        return self

    def s(self, qubit: int) -> Circuit:
        self._append(g.S, (qubit,), ())
        return self

    def t(self, qubit: int) -> Circuit:
        self._append(g.T, (qubit,), ())
        return self

    def rx(self, theta: ParameterValueType, qubit: int) -> Circuit:
        self._append(g.Rx, (qubit,), (theta,))
        return self

    def ry(self, theta: ParameterValueType, qubit: int) -> Circuit:
        self._append(g.Ry, (qubit,), (theta,))
        return self

    def rz(self, theta: ParameterValueType, qubit: int) -> Circuit:
        self._append(g.Rz, (qubit,), (theta,))
        return self

    def phase(self, phi: ParameterValueType, qubit: int) -> Circuit:
        self._append(g.Phase, (qubit,), (phi,))
        return self

    def cx(self, control: int, target: int) -> Circuit:
        self._append(g.CNOT, (control, target), ())
        return self

    def cnot(self, control: int, target: int) -> Circuit:
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> Circuit:
        self._append(g.CZ, (control, target), ())
        return self

    def swap(self, qubit0: int, qubit1: int) -> Circuit:
        self._append(g.SWAP, (qubit0, qubit1), ())
        return self

    def crx(self, theta: ParameterValueType, control: int, target: int) -> Circuit:
        self._append(g.CRx, (control, target), (theta,))
        return self

    def cry(self, theta: ParameterValueType, control: int, target: int) -> Circuit:
        self._append(g.CRy, (control, target), (theta,))
        return self

    def crz(self, theta: ParameterValueType, control: int, target: int) -> Circuit:
        self._append(g.CRz, (control, target), (theta,))
        return self

    def apply(
        self,
        gate: g.GateDefinition,
        qubits: tuple[int, ...],
        params: tuple[ParameterValueType, ...] = (),
    ) -> Circuit:
        """Generic gate application for extensibility."""
        self._append(gate, qubits, params)
        return self

    def _append(
        self,
        gate: g.GateDefinition,
        qubits: tuple[int, ...],
        params: tuple[ParameterValueType, ...],
    ) -> None:
        """Validate and append an instruction."""
        if gate.num_qubits != len(qubits):
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_qubits} qubits, "
                f"got {len(qubits)}"
            )
        if gate.num_params != len(params):
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_params} params, "
                f"got {len(params)}"
            )
        for q in qubits:
            if q < 0 or q >= self._num_qubits:
                raise ValueError(
                    f"Qubit {q} out of range [0, {self._num_qubits})"
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubit indices: {qubits}")
        self._instructions.append(Instruction(gate, qubits, params))

    def compose(
        self,
        other: Circuit,
        qubit_map: Optional[dict[int, int]] = None,
    ) -> Circuit:
        """Append another circuit's instructions onto this circuit.

        qubit_map: {other_qubit: self_qubit}. If None, identity mapping.
        """
        if qubit_map is None:
            qubit_map = {i: i for i in range(other.num_qubits)}
        for inst in other.instructions:
            mapped_qubits = tuple(qubit_map[q] for q in inst.qubits)
            self._append(inst.gate, mapped_qubits, inst.params)
        return self

    def bind_parameters(self, param_dict: dict[Parameter, float]) -> Circuit:
        """Return a NEW circuit with specified Parameters replaced by floats.

        Partial binding is allowed: unbound Parameters remain symbolic.
        The original circuit is not modified.
        """
        new_circuit = Circuit(self._num_qubits, name=self._name)
        for inst in self._instructions:
            new_params = tuple(
                param_dict.get(p, p) if isinstance(p, Parameter) else p
                for p in inst.params
            )
            new_circuit._instructions.append(
                Instruction(inst.gate, inst.qubits, new_params)
            )
        return new_circuit

    def is_parameterized(self) -> bool:
        return len(self.parameters) > 0

    def __len__(self) -> int:
        return len(self._instructions)

    def __repr__(self) -> str:
        return (
            f"Circuit(num_qubits={self._num_qubits}, "
            f"instructions={len(self._instructions)}, "
            f"params={self.num_parameters})"
        )
