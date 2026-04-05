"""Tests for qsim/density_matrix.py — density matrix simulator with noise."""

import numpy as np
import pytest

from qsim.circuit import Circuit
from qsim.density_matrix import DensityMatrix
from qsim.measurement import expectation_value
from qsim.noise import NoiseModel, amplitude_damping, depolarizing, phase_damping
from qsim.observables import Observable
from qsim.statevector import Statevector


class TestDensityMatrixBasic:
    def test_initial_state(self):
        dm = DensityMatrix(2)
        assert dm.num_qubits == 2
        np.testing.assert_allclose(dm.trace(), 1.0)
        np.testing.assert_allclose(dm.purity(), 1.0)

    def test_invalid_qubits(self):
        with pytest.raises(ValueError):
            DensityMatrix(0)

    def test_custom_initial_state(self):
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        dm = DensityMatrix(1, rho)
        np.testing.assert_allclose(dm.trace(), 1.0)

    def test_invalid_initial_state_shape(self):
        with pytest.raises(ValueError, match="shape"):
            DensityMatrix(1, np.eye(4, dtype=complex))


class TestNoiselessMatchesStatevector:
    """Without noise, DensityMatrix should match Statevector results."""

    def test_h_gate(self):
        qc = Circuit(1).h(0)
        sv = Statevector.from_circuit(qc)
        dm = DensityMatrix.from_circuit(qc)

        sv_exp = expectation_value(sv, Observable.z(0))
        dm_exp = dm.expectation_value(Observable.z(0))
        np.testing.assert_allclose(dm_exp, sv_exp, atol=1e-10)

    def test_bell_state(self):
        qc = Circuit(2).h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        dm = DensityMatrix.from_circuit(qc)

        for obs in [Observable.z(0), Observable.z(1), Observable.zz(0, 1)]:
            sv_exp = expectation_value(sv, obs)
            dm_exp = dm.expectation_value(obs)
            np.testing.assert_allclose(dm_exp, sv_exp, atol=1e-10)

    def test_probabilities_match(self):
        qc = Circuit(2).h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        dm = DensityMatrix.from_circuit(qc)
        np.testing.assert_allclose(dm.probabilities(), sv.probabilities(), atol=1e-10)

    def test_from_statevector(self):
        qc = Circuit(2).h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        dm = DensityMatrix.from_statevector(sv)
        np.testing.assert_allclose(dm.probabilities(), sv.probabilities(), atol=1e-10)
        np.testing.assert_allclose(dm.purity(), 1.0, atol=1e-10)

    def test_parametric_circuit(self):
        from qsim.parameters import Parameter
        theta = Parameter("theta")
        qc = Circuit(1).ry(theta, 0)
        bound = qc.bind_parameters({theta: np.pi / 3})
        sv = Statevector.from_circuit(bound)
        dm = DensityMatrix.from_circuit(bound)
        np.testing.assert_allclose(
            dm.expectation_value(Observable.z(0)),
            expectation_value(sv, Observable.z(0)),
            atol=1e-10,
        )


class TestNoiseEffects:
    def test_depolarizing_reduces_purity(self):
        """Depolarizing noise on a pure state should reduce purity."""
        qc = Circuit(1).h(0)
        model = NoiseModel()
        model.add_gate_noise("H", depolarizing(0.5))
        dm = DensityMatrix.from_circuit(qc, model)
        assert dm.purity() < 1.0 - 0.01

    def test_depolarizing_full_maximally_mixed(self):
        """Full depolarizing (p=1) after any gate → maximally mixed."""
        qc = Circuit(1).x(0)
        model = NoiseModel()
        model.add_gate_noise("X", depolarizing(1.0))
        dm = DensityMatrix.from_circuit(qc, model)
        np.testing.assert_allclose(dm.data, np.eye(2) / 2, atol=1e-10)

    def test_amplitude_damping_relaxation(self):
        """Excited state |1> with amplitude damping decays toward |0>."""
        qc = Circuit(1).x(0)
        model = NoiseModel()
        model.add_gate_noise("X", amplitude_damping(0.9))
        dm = DensityMatrix.from_circuit(qc, model)
        probs = dm.probabilities()
        assert probs[0] > 0.8

    def test_phase_damping_coherence_loss(self):
        """Phase damping on a superposition should reduce off-diagonal elements."""
        qc = Circuit(1).h(0)
        model = NoiseModel()
        model.add_gate_noise("H", phase_damping(0.9))
        dm = DensityMatrix.from_circuit(qc, model)
        assert abs(dm.data[0, 1]) < 0.2

    def test_noisy_differs_from_noiseless(self):
        """Noisy expectation values should differ from ideal."""
        qc = Circuit(1).ry(np.pi / 3, 0)
        dm_ideal = DensityMatrix.from_circuit(qc)
        model = NoiseModel()
        model.add_gate_noise("Ry", depolarizing(0.5))
        dm_noisy = DensityMatrix.from_circuit(qc, model)

        ideal_exp = dm_ideal.expectation_value(Observable.z(0))
        noisy_exp = dm_noisy.expectation_value(Observable.z(0))
        assert not np.isclose(ideal_exp, noisy_exp, atol=0.01)

    def test_trace_preserved_under_noise(self):
        """Noise channels are trace-preserving."""
        qc = Circuit(2).h(0).cx(0, 1)
        model = NoiseModel()
        model.add_gate_noise("H", depolarizing(0.1))
        model.add_gate_noise("CNOT", depolarizing(0.05))
        dm = DensityMatrix.from_circuit(qc, model)
        np.testing.assert_allclose(dm.trace(), 1.0, atol=1e-10)

    def test_no_noise_for_unregistered_gates(self):
        """Gates without registered noise should be noiseless."""
        qc = Circuit(1).h(0).x(0)
        model = NoiseModel()
        model.add_gate_noise("X", depolarizing(1.0))
        dm = DensityMatrix.from_circuit(qc, model)
        assert dm.trace() > 0.99

    def test_unbound_parameters_raise(self):
        from qsim.parameters import Parameter
        qc = Circuit(1).ry(Parameter("t"), 0)
        with pytest.raises(ValueError, match="unbound"):
            DensityMatrix.from_circuit(qc)

    def test_unbound_apply_instruction_raises(self):
        from qsim.parameters import Parameter
        qc = Circuit(1).ry(Parameter("t"), 0)
        dm = DensityMatrix(1)
        with pytest.raises(ValueError, match="unbound"):
            dm.apply_instruction(qc.instructions[0])

    def test_repr(self):
        dm = DensityMatrix(1)
        r = repr(dm)
        assert "DensityMatrix" in r
        assert "trace" in r
        assert "purity" in r


class TestHybridLayerWithNoise:
    def test_noisy_forward(self):
        """HybridQuantumClassicalLayer with noise_model should run."""
        from pipeline.hybrid_layer import HybridQuantumClassicalLayer
        from quantum.ansatz import CNOTLadder

        model = NoiseModel()
        model.add_gate_noise("Ry", depolarizing(0.01))

        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(
            2, ansatz, noise_model=model,
        )
        x = np.array([[0.5, 1.0]])
        out = layer.forward(x)
        assert out.shape == (1, 2)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_noisy_differs_from_noiseless(self):
        """Noisy layer output should differ from noiseless."""
        from pipeline.hybrid_layer import HybridQuantumClassicalLayer
        from quantum.ansatz import CNOTLadder

        ansatz_clean = CNOTLadder(2, 1)
        layer_clean = HybridQuantumClassicalLayer(2, ansatz_clean)

        model = NoiseModel()
        model.add_gate_noise("Ry", depolarizing(0.3))
        model.add_gate_noise("Rz", depolarizing(0.3))
        ansatz_noisy = CNOTLadder(2, 1)
        layer_noisy = HybridQuantumClassicalLayer(
            2, ansatz_noisy, noise_model=model,
        )

        x = np.array([[0.5, 1.0]])
        out_clean = layer_clean.forward(x)
        out_noisy = layer_noisy.forward(x)
        assert not np.allclose(out_clean, out_noisy, atol=0.01)
