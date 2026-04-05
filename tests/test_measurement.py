"""Tests for qsim/measurement.py — expectation values, sampling, partial measurement."""

import numpy as np
import pytest

from qsim.circuit import Circuit
from qsim.measurement import (
    _expectation_z_only,
    expectation_value,
    partial_measure,
    sample,
)
from qsim.observables import Observable
from qsim.statevector import Statevector


class TestExpectationValueZ:
    def test_z_on_zero(self):
        """<0|Z|0> = +1"""
        sv = Statevector(1)
        obs = Observable.z(0)
        assert np.isclose(expectation_value(sv, obs), 1.0)

    def test_z_on_one(self):
        """<1|Z|1> = -1"""
        qc = Circuit(1)
        qc.x(0)
        sv = Statevector.from_circuit(qc)
        obs = Observable.z(0)
        assert np.isclose(expectation_value(sv, obs), -1.0)

    def test_z_on_plus_state(self):
        """<+|Z|+> = 0 (equal superposition)"""
        qc = Circuit(1)
        qc.h(0)
        sv = Statevector.from_circuit(qc)
        obs = Observable.z(0)
        assert np.isclose(expectation_value(sv, obs), 0.0)

    def test_zz_on_bell_state(self):
        """<Φ+|Z⊗Z|Φ+> = +1 (perfect correlation)"""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        obs = Observable.zz(0, 1)
        assert np.isclose(expectation_value(sv, obs), 1.0)

    def test_z0_on_bell_state(self):
        """<Φ+|Z⊗I|Φ+> = 0 (each qubit individually random)"""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        obs = Observable.z(0)
        assert np.isclose(expectation_value(sv, obs), 0.0)

    def test_identity_observable(self):
        """<ψ|I|ψ> = 1 for any state."""
        sv = Statevector(2)
        obs = Observable.identity()
        assert np.isclose(expectation_value(sv, obs), 1.0)


class TestExpectationValueGeneral:
    def test_x_on_plus_state(self):
        """<+|X|+> = 1"""
        qc = Circuit(1)
        qc.h(0)
        sv = Statevector.from_circuit(qc)
        obs = Observable.x(0)
        assert np.isclose(expectation_value(sv, obs), 1.0)

    def test_x_on_zero(self):
        """<0|X|0> = 0"""
        sv = Statevector(1)
        obs = Observable.x(0)
        assert np.isclose(expectation_value(sv, obs), 0.0)

    def test_sum_observable(self):
        """<0|Z + Z|0> = 2 (same qubit, so Z appears twice with coeff 1 each)."""
        sv = Statevector(1)
        obs = Observable.z(0) + Observable.z(0)
        assert np.isclose(expectation_value(sv, obs), 2.0)

    def test_scaled_observable(self):
        sv = Statevector(1)
        obs = Observable.z(0) * 0.5
        assert np.isclose(expectation_value(sv, obs), 0.5)

    def test_pauli_string_observable(self):
        """Test from_pauli_string integration."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        obs = Observable.from_pauli_string("ZZ")
        assert np.isclose(expectation_value(sv, obs), 1.0)


class TestExpectationZOnlyFastPath:
    def test_matches_general_path(self):
        """Z-only fast path should give same result as general path."""
        qc = Circuit(3)
        qc.h(0).cx(0, 1).ry(1.0, 2)
        sv = Statevector.from_circuit(qc)

        # Fast path
        fast = _expectation_z_only(sv, (0, 2))

        # General path via Observable
        obs = Observable.zz(0, 2)
        general = expectation_value(sv, obs)

        assert np.isclose(fast, general)

    def test_empty_z_qubits_is_one(self):
        """No Z operators = identity = 1."""
        sv = Statevector(2)
        assert np.isclose(_expectation_z_only(sv, ()), 1.0)


class TestSample:
    def test_deterministic_state(self):
        """Sampling |00> should always give '00'."""
        sv = Statevector(2)
        counts = sample(sv, shots=100, seed=42)
        assert counts["00"] == 100

    def test_bell_state_distribution(self):
        """Bell state should give ~50% |00>, ~50% |11>."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        counts = sample(sv, shots=10000, seed=42)

        # Should only have '00' and '11'
        assert set(counts.keys()).issubset({"00", "11"})

        # Chi-squared test: expected 5000 each
        observed = np.array([counts.get("00", 0), counts.get("11", 0)])
        expected = np.array([5000, 5000])
        chi_sq = np.sum((observed - expected) ** 2 / expected)
        # p < 0.001 threshold for 1 dof is 10.83
        assert chi_sq < 10.83, f"Chi-squared {chi_sq} too high"

    def test_seed_reproducibility(self):
        sv = Statevector(1)
        qc = Circuit(1)
        qc.h(0)
        sv.evolve(qc)
        c1 = sample(sv, shots=100, seed=123)
        c2 = sample(sv, shots=100, seed=123)
        assert c1 == c2

    def test_custom_shots(self):
        sv = Statevector(1)
        counts = sample(sv, shots=50, seed=0)
        total = sum(counts.values())
        assert total == 50


class TestPartialMeasure:
    def test_bell_state_collapse(self):
        """Measuring qubit 0 of Bell state collapses both qubits."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)

        result, post_sv = partial_measure(sv, (0,), seed=42)
        assert result in ("0", "1")

        # Post-measurement state should be either |00> or |11>
        if result == "0":
            assert np.isclose(post_sv.probability("00"), 1.0)
        else:
            assert np.isclose(post_sv.probability("11"), 1.0)

    def test_product_state_partial_measure(self):
        """Measuring qubit 0 of |10> gives '1', leaves qubit 1 as |0>."""
        qc = Circuit(2)
        qc.x(0)
        sv = Statevector.from_circuit(qc)

        result, post_sv = partial_measure(sv, (0,), seed=0)
        assert result == "1"
        assert np.isclose(post_sv.norm(), 1.0)

    def test_post_measurement_normalized(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        _, post_sv = partial_measure(sv, (0,), seed=0)
        assert np.isclose(post_sv.norm(), 1.0)

    def test_measure_multiple_qubits(self):
        qc = Circuit(3)
        qc.x(0).x(2)  # |101>
        sv = Statevector.from_circuit(qc)
        result, post_sv = partial_measure(sv, (0, 2), seed=0)
        assert result == "11"
        assert np.isclose(post_sv.norm(), 1.0)
