"""Quantum noise channels and noise models.

Noise channels are represented as sets of Kraus operators {E_k} satisfying
the completeness relation: Σ_k E_k† E_k = I.

A NoiseModel maps gate names to noise channels, applied after each gate
during density matrix simulation.

Supported channels:
  - Depolarizing: with probability p, replace qubit with maximally mixed state
  - Amplitude damping: models T1 energy relaxation
  - Phase damping: models T2 dephasing (loss of coherence)
  - Readout error: bit-flip applied post-measurement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class NoiseChannel:
    """A quantum noise channel defined by its Kraus operators.

    Noise is applied as: ρ' = Σ_k E_k ρ E_k†
    Completeness: Σ_k E_k† E_k = I (trace-preserving).
    """

    kraus_ops: tuple[np.ndarray, ...]

    def validate(self) -> bool:
        """Check the completeness relation Σ_k E_k† E_k ≈ I."""
        d = self.kraus_ops[0].shape[0]
        total = sum(E.conj().T @ E for E in self.kraus_ops)
        return bool(np.allclose(total, np.eye(d)))

    @property
    def num_ops(self) -> int:
        return len(self.kraus_ops)


def depolarizing(p: float) -> NoiseChannel:
    """Single-qubit depolarizing channel.

    With probability p, the qubit state is replaced with I/2.
    ε_p(ρ) = (1-p)ρ + p·I/2

    Kraus operators: √(1 - 3p/4)·I, √(p/4)·X, √(p/4)·Y, √(p/4)·Z
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    return NoiseChannel(kraus_ops=(
        np.sqrt(1 - 3 * p / 4) * _I,
        np.sqrt(p / 4) * _X,
        np.sqrt(p / 4) * _Y,
        np.sqrt(p / 4) * _Z,
    ))


def amplitude_damping(gamma: float) -> NoiseChannel:
    """Amplitude damping channel (models T1 relaxation).

    |1> decays to |0> with probability gamma.
    Kraus operators:
      E0 = [[1, 0], [0, √(1-γ)]]
      E1 = [[0, √γ], [0, 0]]
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return NoiseChannel(kraus_ops=(E0, E1))


def phase_damping(lambda_: float) -> NoiseChannel:
    """Phase damping channel (models T2 dephasing).

    Off-diagonal elements of the density matrix decay by √(1-λ).
    Kraus operators:
      E0 = [[1, 0], [0, √(1-λ)]]
      E1 = [[0, 0], [0, √λ]]
    """
    if not 0 <= lambda_ <= 1:
        raise ValueError(f"lambda must be in [0, 1], got {lambda_}")
    E0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_)]], dtype=complex)
    E1 = np.array([[0, 0], [0, np.sqrt(lambda_)]], dtype=complex)
    return NoiseChannel(kraus_ops=(E0, E1))


@dataclass
class ReadoutError:
    """Readout error model: bit-flip on measurement with probability p.

    Applied after measurement, not during circuit evolution.
    """

    p: float

    def __post_init__(self) -> None:
        if not 0 <= self.p <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {self.p}")

    def apply(self, bitstring: str, rng: Optional[np.random.Generator] = None) -> str:
        """Apply readout errors to a measurement bitstring."""
        if rng is None:
            rng = np.random.default_rng()
        bits = list(bitstring)
        for i in range(len(bits)):
            if rng.random() < self.p:
                bits[i] = "1" if bits[i] == "0" else "0"
        return "".join(bits)


class NoiseModel:
    """Maps gate names to noise channels.

    After each gate is applied in density matrix simulation, the
    corresponding noise channel (if registered) is applied to the
    target qubit(s).
    """

    def __init__(self) -> None:
        self._gate_noise: dict[str, NoiseChannel] = {}
        self.readout_error: Optional[ReadoutError] = None

    def add_gate_noise(self, gate_name: str, channel: NoiseChannel) -> None:
        """Register a noise channel for a gate type.

        gate_name should match GateDefinition.name (e.g., "CNOT", "Rx").
        """
        self._gate_noise[gate_name] = channel

    def get_gate_noise(self, gate_name: str) -> Optional[NoiseChannel]:
        return self._gate_noise.get(gate_name)

    def set_readout_error(self, p: float) -> None:
        self.readout_error = ReadoutError(p)

    @property
    def gate_names(self) -> list[str]:
        return list(self._gate_noise.keys())
