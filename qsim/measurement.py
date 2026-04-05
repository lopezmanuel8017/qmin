"""Measurement engine: expectation values, sampling, and partial measurement.

Supports two strategies for expectation values:
1. Z-only fast path: O(2^n), no copies, no matrix ops. Used when all
   operators in a PauliTerm are Z or I.
2. General path: Apply Pauli to |ψ> to get |φ>, compute <ψ|φ>. Requires
   one statevector copy. Still O(2^n).
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .observables import PAULI_LABELS, Observable, PauliTerm
from .statevector import Statevector
from .utils import index_to_bitstring


def expectation_value(sv: Statevector, observable: Observable) -> float:
    """Compute <ψ|O|ψ> exactly (no sampling noise)."""
    total = 0.0
    for term in observable.terms:
        total += term.coeff * _expectation_pauli_term(sv, term)
    return float(np.real(total))


def _expectation_pauli_term(sv: Statevector, term: PauliTerm) -> complex:
    """Compute <ψ|P|ψ> for a single Pauli tensor product term.

    Dispatches to Z-only fast path when possible.
    """
    ops = term.ops_dict
    if all(label in ("Z", "I") for label in ops.values()):
        z_qubits = tuple(q for q, label in ops.items() if label == "Z")
        return complex(_expectation_z_only(sv, z_qubits))

    phi = sv.copy()
    for qubit, pauli_label in ops.items():
        matrix = PAULI_LABELS[pauli_label]
        phi._apply_single_qubit_gate(matrix, qubit)
    return sv.inner(phi)


def _expectation_z_only(sv: Statevector, z_qubits: tuple[int, ...]) -> float:
    """Fast path for Z-only observables.

    <ψ| Z_i Z_j ... |ψ> = Σ_k |a_k|² · (-1)^(parity of bits i,j,... in k)

    This is O(2^n) with no copies or matrix operations. Most common in VQE/QAOA.
    """
    if not z_qubits:
        return 1.0

    probs = sv.probabilities()
    n = sv.num_qubits

    indices = np.arange(2**n)
    parity = np.zeros(2**n, dtype=int)
    for q in z_qubits:
        bit_position = n - 1 - q
        parity ^= (indices >> bit_position) & 1

    signs = 1 - 2 * parity
    return float(np.sum(probs * signs))


def sample(
    sv: Statevector,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> Counter:
    """Shot-based measurement simulation.

    Returns Counter of bitstrings, e.g. Counter({'00': 512, '11': 512}).
    """
    rng = np.random.default_rng(seed)
    probs = sv.probabilities()
    indices = rng.choice(2**sv.num_qubits, size=shots, p=probs)

    results: Counter = Counter()
    for idx in indices:
        bs = index_to_bitstring(int(idx), sv.num_qubits)
        results[bs] += 1
    return results


def partial_measure(
    sv: Statevector,
    qubits: tuple[int, ...],
    seed: Optional[int] = None,
) -> tuple[str, Statevector]:
    """Measure a subset of qubits, collapsing the state.

    Returns (measured_bitstring, post_measurement_statevector).
    The post-measurement state is renormalized.
    """
    rng = np.random.default_rng(seed)
    n = sv.num_qubits
    probs = sv.probabilities()

    outcome_probs: dict[str, float] = {}
    for idx in range(2**n):
        full_bs = index_to_bitstring(idx, n)
        measured_bs = "".join(full_bs[q] for q in qubits)
        outcome_probs[measured_bs] = outcome_probs.get(measured_bs, 0.0) + probs[idx]

    outcomes = list(outcome_probs.keys())
    outcome_p = np.array([outcome_probs[o] for o in outcomes])
    chosen_idx = rng.choice(len(outcomes), p=outcome_p)
    measured_result = outcomes[chosen_idx]

    new_data = sv.data
    for idx in range(2**n):
        full_bs = index_to_bitstring(idx, n)
        measured_bs = "".join(full_bs[q] for q in qubits)
        if measured_bs != measured_result:
            new_data[idx] = 0.0

    norm = np.linalg.norm(new_data)
    new_data /= norm

    return measured_result, Statevector(n, new_data)
