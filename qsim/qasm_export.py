"""OpenQASM 2.0 and 3.0 circuit serialization.

Exports a bound (non-parameterized) Circuit to QASM string format,
enabling execution on IBM Quantum hardware via Qiskit Runtime or
direct submission.

OpenQASM 2.0: https://arxiv.org/abs/1707.03429
OpenQASM 3.0: https://arxiv.org/abs/2104.14722
"""

from __future__ import annotations

from .circuit import Circuit, Instruction


def to_qasm2(
    circuit: Circuit,
    register_name: str = "q",
    classical_register: bool = True,
) -> str:
    """Export circuit to OpenQASM 2.0 string.

    Raises ValueError if circuit has unbound parameters.
    """
    if circuit.is_parameterized():
        raise ValueError(
            "Cannot export parameterized circuit to QASM. "
            "Bind parameters first."
        )

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg {register_name}[{circuit.num_qubits}];",
    ]

    if classical_register:
        lines.append(f"creg c[{circuit.num_qubits}];")

    for inst in circuit.instructions:
        lines.append(_instruction_to_qasm2(inst, register_name))

    return "\n".join(lines) + "\n"


def _instruction_to_qasm2(inst: Instruction, reg: str) -> str:
    """Convert a single instruction to a QASM 2.0 line."""
    qasm_name = inst.gate.qasm_name
    qubit_args = ",".join(f"{reg}[{q}]" for q in inst.qubits)

    if inst.gate.num_params == 0:
        return f"{qasm_name} {qubit_args};"
    else:
        param_str = ",".join(f"{float(p):.15g}" for p in inst.params)
        return f"{qasm_name}({param_str}) {qubit_args};"


def to_qasm3(
    circuit: Circuit,
    register_name: str = "q",
) -> str:
    """Export circuit to OpenQASM 3.0 string.

    Raises ValueError if circuit has unbound parameters.
    """
    if circuit.is_parameterized():
        raise ValueError(
            "Cannot export parameterized circuit to QASM. "
            "Bind parameters first."
        )

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{circuit.num_qubits}] {register_name};",
        f"bit[{circuit.num_qubits}] c;",
    ]

    for inst in circuit.instructions:
        lines.append(_instruction_to_qasm3(inst, register_name))

    return "\n".join(lines) + "\n"


def _instruction_to_qasm3(inst: Instruction, reg: str) -> str:
    """Convert a single instruction to a QASM 3.0 line."""
    qasm_name = inst.gate.qasm_name
    qubit_args = ", ".join(f"{reg}[{q}]" for q in inst.qubits)

    if inst.gate.num_params == 0:
        return f"{qasm_name} {qubit_args};"
    else:
        param_str = ", ".join(f"{float(p):.15g}" for p in inst.params)
        return f"{qasm_name}({param_str}) {qubit_args};"
