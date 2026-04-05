"""Tests for qsim/noise.py — noise channels and noise models."""

import numpy as np
import pytest

from qsim.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping,
    depolarizing,
    phase_damping,
)


class TestNoiseChannel:
    def test_depolarizing_completeness(self):
        """Σ E_k† E_k = I for depolarizing channel."""
        for p in [0.0, 0.01, 0.1, 0.5, 1.0]:
            channel = depolarizing(p)
            assert channel.validate()

    def test_amplitude_damping_completeness(self):
        for gamma in [0.0, 0.1, 0.5, 1.0]:
            channel = amplitude_damping(gamma)
            assert channel.validate()

    def test_phase_damping_completeness(self):
        for lam in [0.0, 0.1, 0.5, 1.0]:
            channel = phase_damping(lam)
            assert channel.validate()

    def test_depolarizing_identity_at_zero(self):
        """p=0 depolarizing is identity (one Kraus op ≈ I)."""
        channel = depolarizing(0.0)
        np.testing.assert_allclose(channel.kraus_ops[0], np.eye(2), atol=1e-10)

    def test_depolarizing_num_ops(self):
        assert depolarizing(0.1).num_ops == 4

    def test_amplitude_damping_num_ops(self):
        assert amplitude_damping(0.1).num_ops == 2

    def test_phase_damping_num_ops(self):
        assert phase_damping(0.1).num_ops == 2

    def test_invalid_probability_depolarizing(self):
        with pytest.raises(ValueError):
            depolarizing(-0.1)
        with pytest.raises(ValueError):
            depolarizing(1.1)

    def test_invalid_probability_amplitude_damping(self):
        with pytest.raises(ValueError):
            amplitude_damping(-0.1)

    def test_invalid_probability_phase_damping(self):
        with pytest.raises(ValueError):
            phase_damping(1.5)


class TestReadoutError:
    def test_no_error(self):
        err = ReadoutError(0.0)
        rng = np.random.default_rng(0)
        result = err.apply("0101", rng)
        assert result == "0101"

    def test_full_error(self):
        err = ReadoutError(1.0)
        rng = np.random.default_rng(0)
        result = err.apply("0000", rng)
        assert result == "1111"

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            ReadoutError(-0.5)

    def test_default_rng(self):
        err = ReadoutError(0.5)
        result = err.apply("01")
        assert len(result) == 2


class TestNoiseModel:
    def test_add_and_get_gate_noise(self):
        model = NoiseModel()
        ch = depolarizing(0.01)
        model.add_gate_noise("CNOT", ch)
        assert model.get_gate_noise("CNOT") is ch
        assert model.get_gate_noise("H") is None

    def test_gate_names(self):
        model = NoiseModel()
        model.add_gate_noise("H", depolarizing(0.01))
        model.add_gate_noise("CNOT", depolarizing(0.05))
        assert set(model.gate_names) == {"H", "CNOT"}

    def test_readout_error(self):
        model = NoiseModel()
        model.set_readout_error(0.1)
        assert model.readout_error is not None
        assert model.readout_error.p == 0.1
