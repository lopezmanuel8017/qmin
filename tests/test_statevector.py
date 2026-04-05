"""Tests for qsim/statevector.py — simulation engine correctness."""

import numpy as np
import pytest

from qsim.circuit import Circuit
from qsim.gates import H
from qsim.parameters import Parameter
from qsim.statevector import Statevector


class TestInitialization:
    def test_default_is_zero_state(self):
        sv = Statevector(2)
        assert np.allclose(sv.data, [1, 0, 0, 0])

    def test_single_qubit_zero(self):
        sv = Statevector(1)
        assert np.allclose(sv.data, [1, 0])

    def test_custom_initial_state(self):
        state = np.array([0, 1, 0, 0], dtype=complex)
        sv = Statevector(2, initial_state=state)
        assert np.allclose(sv.data, [0, 1, 0, 0])

    def test_wrong_initial_state_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Statevector(2, initial_state=np.array([1, 0]))

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            Statevector(0)

    def test_norm_is_one(self):
        sv = Statevector(3)
        assert np.isclose(sv.norm(), 1.0)


class TestSingleQubitGates:
    def test_x_on_zero(self):
        """X|0> = |1>"""
        qc = Circuit(1)
        qc.x(0)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 1])

    def test_h_on_zero(self):
        """H|0> = (|0> + |1>) / √2"""
        qc = Circuit(1)
        qc.h(0)
        sv = Statevector.from_circuit(qc)
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(sv.data, expected)

    def test_x_on_second_qubit(self):
        """I⊗X |00> = |01>"""
        qc = Circuit(2)
        qc.x(1)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 1, 0, 0])

    def test_x_on_first_qubit(self):
        """X⊗I |00> = |10>"""
        qc = Circuit(2)
        qc.x(0)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 0, 1, 0])

    def test_norm_preserved(self):
        """Gate application preserves statevector norm."""
        qc = Circuit(3)
        qc.h(0).rx(1.23, 1).ry(0.5, 2).rz(2.1, 0)
        sv = Statevector.from_circuit(qc)
        assert np.isclose(sv.norm(), 1.0)

    def test_gate_on_last_qubit_of_four(self):
        """X on qubit 3 of 4-qubit system: |0000> -> |0001>"""
        qc = Circuit(4)
        qc.x(3)
        sv = Statevector.from_circuit(qc)
        expected = np.zeros(16, dtype=complex)
        expected[1] = 1.0
        assert np.allclose(sv.data, expected)

    def test_matches_kronecker_product(self):
        """Compare einsum result to explicit Kronecker product for H⊗I."""
        qc = Circuit(2)
        qc.h(0)
        sv = Statevector.from_circuit(qc)

        h_mat = H.matrix()
        full = np.kron(h_mat, np.eye(2))
        expected = full @ np.array([1, 0, 0, 0], dtype=complex)
        assert np.allclose(sv.data, expected)


class TestTwoQubitGates:
    def test_bell_state(self):
        """H(0), CNOT(0,1) -> (|00> + |11>) / √2"""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(sv.data, expected)

    def test_cnot_no_flip_when_control_zero(self):
        """CNOT(0,1)|00> = |00>"""
        qc = Circuit(2)
        qc.cx(0, 1)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [1, 0, 0, 0])

    def test_cnot_flips_when_control_one(self):
        """CNOT(0,1)|10> = |11>"""
        qc = Circuit(2)
        qc.x(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 0, 0, 1])

    def test_non_adjacent_cnot(self):
        """CNOT(0,3) on 4-qubit system: |1000> -> |1001>"""
        qc = Circuit(4)
        qc.x(0).cx(0, 3)
        sv = Statevector.from_circuit(qc)
        expected = np.zeros(16, dtype=complex)
        expected[0b1001] = 1.0
        assert np.allclose(sv.data, expected)

    def test_non_adjacent_cnot_reverse(self):
        """CNOT(3,0) on 4 qubits: |0001> -> |1001>"""
        qc = Circuit(4)
        qc.x(3).cx(3, 0)
        sv = Statevector.from_circuit(qc)
        expected = np.zeros(16, dtype=complex)
        expected[0b1001] = 1.0
        assert np.allclose(sv.data, expected)

    def test_swap(self):
        """SWAP(0,1)|10> = |01>"""
        qc = Circuit(2)
        qc.x(0).swap(0, 1)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 1, 0, 0])

    def test_cz(self):
        """CZ|11> = -|11>"""
        qc = Circuit(2)
        qc.x(0).x(1).cz(0, 1)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [0, 0, 0, -1])

    def test_norm_preserved_two_qubit(self):
        qc = Circuit(3)
        qc.h(0).cx(0, 1).cx(1, 2)
        sv = Statevector.from_circuit(qc)
        assert np.isclose(sv.norm(), 1.0)

    def test_matches_kronecker_cnot(self):
        """Verify CNOT via einsum matches full Kronecker for 3-qubit system."""
        qc = Circuit(3)
        qc.h(0).cx(0, 2)
        sv = Statevector.from_circuit(qc)

        I2 = np.eye(2, dtype=complex)
        p0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p1 = np.array([[0, 0], [0, 1]], dtype=complex)
        x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        cnot_full = np.kron(np.kron(p0, I2), I2) + np.kron(np.kron(p1, I2), x_mat)
        h_full = np.kron(np.kron(H.matrix(), I2), I2)
        init = np.zeros(8, dtype=complex)
        init[0] = 1.0
        expected = cnot_full @ h_full @ init
        assert np.allclose(sv.data, expected)


class TestGHZState:
    def test_ghz3(self, ghz3_circuit):
        sv = Statevector.from_circuit(ghz3_circuit)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        assert np.allclose(sv.data, expected)


class TestStatevectorMethods:
    def test_probability_bell(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        sv = Statevector.from_circuit(qc)
        assert np.isclose(sv.probability("00"), 0.5)
        assert np.isclose(sv.probability("11"), 0.5)
        assert np.isclose(sv.probability("01"), 0.0)
        assert np.isclose(sv.probability("10"), 0.0)

    def test_probabilities_sum_to_one(self):
        qc = Circuit(3)
        qc.h(0).rx(1.0, 1).ry(0.5, 2).cx(0, 2)
        sv = Statevector.from_circuit(qc)
        assert np.isclose(np.sum(sv.probabilities()), 1.0)

    def test_copy_is_independent(self):
        sv = Statevector(2)
        sv2 = sv.copy()
        qc = Circuit(2)
        qc.x(0)
        sv.evolve(qc)
        assert not np.allclose(sv.data, sv2.data)
        assert np.allclose(sv2.data, [1, 0, 0, 0])

    def test_inner_product_orthogonal(self):
        sv0 = Statevector(1)
        sv1 = Statevector(1, np.array([0, 1], dtype=complex))
        assert np.isclose(sv0.inner(sv1), 0.0)

    def test_inner_product_same_state(self):
        sv = Statevector(2)
        assert np.isclose(sv.inner(sv), 1.0)

    def test_tensor_property(self):
        sv = Statevector(2)
        t = sv.tensor
        assert t.shape == (2, 2)
        assert np.isclose(t[0, 0], 1.0)

    def test_data_returns_copy(self):
        sv = Statevector(1)
        d = sv.data
        d[0] = 999
        assert np.isclose(sv.data[0], 1.0)

    def test_repr(self):
        sv = Statevector(2)
        r = repr(sv)
        assert "num_qubits=2" in r
        assert "norm=1.0" in r


class TestErrorHandling:
    def test_unbound_parameter_raises_evolve(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        with pytest.raises(ValueError, match="unbound parameters"):
            Statevector.from_circuit(qc)

    def test_unbound_parameter_raises_apply_instruction(self):
        """Direct apply_instruction with unbound param (not reachable via evolve)."""
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        sv = Statevector(1)
        with pytest.raises(ValueError, match="Cannot simulate unbound"):
            sv.apply_instruction(qc.instructions[0])

    def test_empty_circuit(self):
        """Empty circuit leaves state unchanged."""
        qc = Circuit(2)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [1, 0, 0, 0])

    def test_identity_like_circuit(self):
        """Rx(0) leaves state unchanged."""
        qc = Circuit(1)
        qc.rx(0.0, 0)
        sv = Statevector.from_circuit(qc)
        assert np.allclose(sv.data, [1, 0])
