"""Tests for quantum/encoding.py — data encoding schemes."""

import numpy as np
import pytest

from quantum.encoding import AngleEncoder, TriValueEncoder
from qsim.measurement import expectation_value
from qsim.observables import Observable
from qsim.statevector import Statevector


class TestAngleEncoder:
    def test_circuit_structure(self):
        enc = AngleEncoder(3)
        qc = enc.circuit()
        assert qc.num_qubits == 3
        assert len(qc.instructions) == 3
        assert all(inst.gate.name == "Ry" for inst in qc.instructions)

    def test_parameters(self):
        enc = AngleEncoder(4)
        assert len(enc.parameters) == 4
        assert enc.parameters[0].name == "enc_0"

    def test_bind(self):
        enc = AngleEncoder(2)
        features = np.array([0.5, 1.0])
        bindings = enc.bind(features)
        assert len(bindings) == 2

    def test_bind_wrong_shape_raises(self):
        enc = AngleEncoder(2)
        with pytest.raises(ValueError, match="Expected features shape"):
            enc.bind(np.array([0.5]))

    def test_encode_and_simulate(self):
        """Encode features, simulate, verify measurement makes sense."""
        enc = AngleEncoder(2)
        features = np.array([0.0, np.pi])

        qc = enc.circuit()
        bindings = enc.bind(features)
        bound_qc = qc.bind_parameters(bindings)
        sv = Statevector.from_circuit(bound_qc)

        assert np.isclose(expectation_value(sv, Observable.z(0)), 1.0, atol=1e-10)
        assert np.isclose(expectation_value(sv, Observable.z(1)), -1.0, atol=1e-10)

    def test_intermediate_angles(self):
        """Ry(π/2)|0> puts qubit in equal superposition: <Z> = 0."""
        enc = AngleEncoder(1)
        features = np.array([np.pi / 2])
        qc = enc.circuit()
        bound = qc.bind_parameters(enc.bind(features))
        sv = Statevector.from_circuit(bound)
        assert np.isclose(expectation_value(sv, Observable.z(0)), 0.0, atol=1e-10)


class TestTriValueEncoder:
    def test_circuit_structure(self):
        enc = TriValueEncoder(2)
        qc = enc.circuit()
        assert qc.num_qubits == 2
        assert len(qc.instructions) == 6

    def test_gate_sequence(self):
        enc = TriValueEncoder(1)
        qc = enc.circuit()
        gate_names = [inst.gate.name for inst in qc.instructions]
        assert gate_names == ["Ry", "Rz", "Ry"]

    def test_parameters(self):
        enc = TriValueEncoder(2)
        assert len(enc.parameters) == 6

    def test_bind(self):
        enc = TriValueEncoder(2)
        rgb = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        bindings = enc.bind(rgb)
        assert len(bindings) == 6

    def test_bind_wrong_shape_raises(self):
        enc = TriValueEncoder(2)
        with pytest.raises(ValueError, match="Expected shape"):
            enc.bind(np.array([0.1, 0.2, 0.3]))

    def test_encode_zeros(self):
        """All zeros: Ry(0)·Rz(0)·Ry(0) = I, state should be |0>."""
        enc = TriValueEncoder(1)
        rgb = np.array([[0.0, 0.0, 0.0]])
        qc = enc.circuit()
        bound = qc.bind_parameters(enc.bind(rgb))
        sv = Statevector.from_circuit(bound)
        assert np.isclose(expectation_value(sv, Observable.z(0)), 1.0, atol=1e-10)

    def test_encode_nonzero(self):
        """Nonzero RGB values should change the state from |0>."""
        enc = TriValueEncoder(1)
        rgb = np.array([[1.0, 0.5, 0.3]])
        qc = enc.circuit()
        bound = qc.bind_parameters(enc.bind(rgb))
        sv = Statevector.from_circuit(bound)
        z_exp = expectation_value(sv, Observable.z(0))
        assert not np.isclose(z_exp, 1.0)
