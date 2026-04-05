"""Qubit ordering utilities and validation helpers.

Convention: big-endian (qubit 0 = most significant bit).
This matches IBM/Qiskit convention: |q0 q1 ... qn-1> maps to
index q0 * 2^(n-1) + q1 * 2^(n-2) + ... + qn-1 * 2^0.
"""

QUBIT_ORDER = "big-endian"


def validate_qubit_indices(qubits: tuple[int, ...], n_qubits: int) -> None:
    """Raise ValueError if any qubit index is out of range or duplicated."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    for q in qubits:
        if q < 0 or q >= n_qubits:
            raise ValueError(
                f"Qubit index {q} out of range [0, {n_qubits})"
            )
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"Duplicate qubit indices: {qubits}")


def computational_basis_index(bitstring: str) -> int:
    """Convert a bitstring to its computational basis index (big-endian).

    '101' -> 5, '00' -> 0, '1' -> 1
    """
    if not bitstring:
        raise ValueError("Bitstring cannot be empty")
    if not all(c in "01" for c in bitstring):
        raise ValueError(f"Invalid bitstring: '{bitstring}' (must contain only '0' and '1')")
    return int(bitstring, 2)


def index_to_bitstring(index: int, n_qubits: int) -> str:
    """Convert a computational basis index to a bitstring (big-endian).

    (5, 3) -> '101', (0, 2) -> '00'
    """
    if index < 0 or index >= 2**n_qubits:
        raise ValueError(
            f"Index {index} out of range [0, {2**n_qubits}) for {n_qubits} qubits"
        )
    return format(index, f"0{n_qubits}b")
