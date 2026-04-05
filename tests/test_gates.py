"""Tests for qsim/gates.py — gate definitions and unitarity."""

import numpy as np
import pytest

from qsim.gates import (
    CNOT, CRx, CRy, CRz, CZ, GATE_REGISTRY, H, I_GATE, Phase, Rx, Ry, Rz,
    S, SWAP, T, X, Y, Z
)

FIXED_SINGLE = [H, X, Y, Z, S, T, I_GATE]
PARAMETERIZED_SINGLE = [Rx, Ry, Rz, Phase]
FIXED_TWO = [CNOT, CZ, SWAP]
PARAMETERIZED_TWO = [CRx, CRy, CRz]
ALL_GATES = FIXED_SINGLE + PARAMETERIZED_SINGLE + FIXED_TWO + PARAMETERIZED_TWO


def _is_unitary(m: np.ndarray, tol: float = 1e-12) -> bool:
    """Check U†U = I."""
    product = m.conj().T @ m
    return np.allclose(product, np.eye(m.shape[0]), atol=tol)


class TestFixedSingleQubitGates:
    @pytest.mark.parametrize("gate", FIXED_SINGLE)
    def test_is_unitary(self, gate):
        m = gate.matrix()
        assert _is_unitary(m), f"{gate.name} is not unitary"

    @pytest.mark.parametrize("gate", FIXED_SINGLE)
    def test_shape(self, gate):
        m = gate.matrix()
        assert m.shape == (2, 2)

    @pytest.mark.parametrize("gate", FIXED_SINGLE)
    def test_tensor_shape(self, gate):
        t = gate.tensor()
        assert t.shape == (2, 2)

    def test_hadamard_values(self):
        h = H.matrix()
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(h, expected)

    def test_pauli_x_values(self):
        x = X.matrix()
        assert np.allclose(x, np.array([[0, 1], [1, 0]]))

    def test_pauli_y_values(self):
        y = Y.matrix()
        assert np.allclose(y, np.array([[0, -1j], [1j, 0]]))

    def test_pauli_z_values(self):
        z = Z.matrix()
        assert np.allclose(z, np.array([[1, 0], [0, -1]]))

    def test_identity_values(self):
        i = I_GATE.matrix()
        assert np.allclose(i, np.eye(2))

    def test_s_is_sqrt_z(self):
        """S² = Z"""
        s = S.matrix()
        z = Z.matrix()
        assert np.allclose(s @ s, z)

    def test_t_is_sqrt_s(self):
        """T² = S"""
        t_mat = T.matrix()
        s_mat = S.matrix()
        assert np.allclose(t_mat @ t_mat, s_mat)


class TestParameterizedSingleQubitGates:
    @pytest.mark.parametrize("gate", PARAMETERIZED_SINGLE)
    @pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
    def test_is_unitary(self, gate, theta):
        m = gate.matrix(theta)
        assert _is_unitary(m), f"{gate.name}({theta}) is not unitary"

    @pytest.mark.parametrize("gate", [Rx, Ry, Rz])
    def test_zero_angle_is_identity(self, gate):
        """R_k(0) = I for all rotation gates."""
        m = gate.matrix(0.0)
        assert np.allclose(m, np.eye(2), atol=1e-12)

    def test_rx_pi_is_minus_i_x(self):
        """Rx(π) = -iX"""
        m = Rx.matrix(np.pi)
        expected = -1j * X.matrix()
        assert np.allclose(m, expected)

    def test_ry_pi_is_minus_i_y(self):
        """Ry(π) = -iY"""
        m = Ry.matrix(np.pi)
        expected = -1j * Y.matrix()
        assert np.allclose(m, expected)

    def test_rz_pi_is_minus_i_z(self):
        """Rz(π) = -iZ"""
        m = Rz.matrix(np.pi)
        expected = -1j * Z.matrix()
        assert np.allclose(m, expected)

    def test_phase_zero_is_identity(self):
        m = Phase.matrix(0.0)
        assert np.allclose(m, np.eye(2))

    def test_phase_pi_is_z(self):
        """P(π) = Z (up to global phase — but P(π) = diag(1,-1) = Z exactly)."""
        m = Phase.matrix(np.pi)
        z = Z.matrix()
        assert np.allclose(m, z)

    def test_wrong_param_count_raises(self):
        with pytest.raises(ValueError, match="expects 1 params"):
            Rx.matrix()

    @pytest.mark.parametrize("gate", PARAMETERIZED_SINGLE)
    def test_tensor_shape(self, gate):
        t = gate.tensor(1.0)
        assert t.shape == (2, 2)


class TestFixedTwoQubitGates:
    @pytest.mark.parametrize("gate", FIXED_TWO)
    def test_is_unitary(self, gate):
        m = gate.matrix()
        assert _is_unitary(m), f"{gate.name} is not unitary"

    @pytest.mark.parametrize("gate", FIXED_TWO)
    def test_shape(self, gate):
        m = gate.matrix()
        assert m.shape == (4, 4)

    @pytest.mark.parametrize("gate", FIXED_TWO)
    def test_tensor_shape(self, gate):
        t = gate.tensor()
        assert t.shape == (2, 2, 2, 2)

    def test_cnot_truth_table(self):
        """CNOT |c,t> -> |c, c XOR t>"""
        m = CNOT.matrix()
        assert np.allclose(m @ [1, 0, 0, 0], [1, 0, 0, 0])
        assert np.allclose(m @ [0, 1, 0, 0], [0, 1, 0, 0])
        assert np.allclose(m @ [0, 0, 1, 0], [0, 0, 0, 1])
        assert np.allclose(m @ [0, 0, 0, 1], [0, 0, 1, 0])

    def test_cz_is_symmetric(self):
        """CZ is symmetric: swapping control/target gives the same gate."""
        m = CZ.matrix()
        assert np.allclose(m, m.T)

    def test_swap_action(self):
        """SWAP |01> = |10>"""
        m = SWAP.matrix()
        assert np.allclose(m @ [0, 1, 0, 0], [0, 0, 1, 0])
        assert np.allclose(m @ [0, 0, 1, 0], [0, 1, 0, 0])

    def test_swap_is_self_inverse(self):
        """SWAP² = I"""
        m = SWAP.matrix()
        assert np.allclose(m @ m, np.eye(4))


class TestParameterizedTwoQubitGates:
    @pytest.mark.parametrize("gate", PARAMETERIZED_TWO)
    @pytest.mark.parametrize("theta", [0.0, np.pi / 3, np.pi])
    def test_is_unitary(self, gate, theta):
        m = gate.matrix(theta)
        assert _is_unitary(m), f"{gate.name}({theta}) is not unitary"

    @pytest.mark.parametrize("gate", PARAMETERIZED_TWO)
    def test_zero_angle_is_identity(self, gate):
        """CR_k(0) = I (no rotation when angle=0)."""
        m = gate.matrix(0.0)
        assert np.allclose(m, np.eye(4), atol=1e-12)

    def test_crx_control_zero_no_rotation(self):
        """CRx with control=|0> leaves target unchanged."""
        m = CRx.matrix(np.pi)
        assert np.allclose(m @ [1, 0, 0, 0], [1, 0, 0, 0])
        assert np.allclose(m @ [0, 1, 0, 0], [0, 1, 0, 0])

    @pytest.mark.parametrize("gate", PARAMETERIZED_TWO)
    def test_shape(self, gate):
        m = gate.matrix(1.0)
        assert m.shape == (4, 4)


class TestGateRegistry:
    def test_all_gates_registered(self):
        for gate in ALL_GATES:
            assert gate.name in GATE_REGISTRY
            assert GATE_REGISTRY[gate.name] is gate

    def test_registry_count(self):
        assert len(GATE_REGISTRY) == len(ALL_GATES)

    def test_lookup_by_name(self):
        assert GATE_REGISTRY["H"] is H
        assert GATE_REGISTRY["CNOT"] is CNOT
        assert GATE_REGISTRY["Rx"] is Rx


class TestGateDefinition:
    def test_wrong_param_count_raises(self):
        with pytest.raises(ValueError, match="expects 0 params"):
            H.matrix(1.0)

    def test_qasm_name_set(self):
        assert H.qasm_name == "h"
        assert CNOT.qasm_name == "cx"
        assert Rx.qasm_name == "rx"
