"""Tests for qsim/observables.py — Pauli operators and observables."""

import pytest

from qsim.observables import (
    I_MATRIX, X_MATRIX, Y_MATRIX, Z_MATRIX,
    Observable, PauliTerm,
)
import numpy as np


class TestPauliMatrices:
    def test_identity(self):
        assert np.allclose(I_MATRIX, np.eye(2))

    def test_x_squared_is_identity(self):
        assert np.allclose(X_MATRIX @ X_MATRIX, np.eye(2))

    def test_y_squared_is_identity(self):
        assert np.allclose(Y_MATRIX @ Y_MATRIX, np.eye(2))

    def test_z_squared_is_identity(self):
        assert np.allclose(Z_MATRIX @ Z_MATRIX, np.eye(2))

    def test_xy_is_iz(self):
        """XY = iZ"""
        assert np.allclose(X_MATRIX @ Y_MATRIX, 1j * Z_MATRIX)


class TestPauliTerm:
    def test_basic_construction(self):
        term = PauliTerm(0.5, ((0, "Z"), (2, "Z")))
        assert term.coeff == 0.5
        assert term.qubits == (0, 2)

    def test_ops_dict(self):
        term = PauliTerm(1.0, ((1, "X"), (3, "Y")))
        d = term.ops_dict
        assert d == {1: "X", 3: "Y"}

    def test_matrix_on_qubit(self):
        term = PauliTerm(1.0, ((0, "Z"),))
        assert np.allclose(term.matrix_on_qubit(0), Z_MATRIX)
        assert np.allclose(term.matrix_on_qubit(1), I_MATRIX)

    def test_identity_term(self):
        term = PauliTerm(2.0, ())
        assert term.qubits == ()
        assert repr(term) == "2.0 * I"

    def test_repr_single(self):
        term = PauliTerm(1.0, ((0, "Z"),))
        assert "Z0" in repr(term)

    def test_repr_multi(self):
        term = PauliTerm(0.5, ((0, "Z"), (2, "X")))
        r = repr(term)
        assert "0.5" in r

    def test_invalid_pauli_label_raises(self):
        with pytest.raises(ValueError, match="Invalid Pauli label"):
            PauliTerm(1.0, ((0, "W"),))

    def test_negative_qubit_raises(self):
        with pytest.raises(ValueError, match="Negative qubit index"):
            PauliTerm(1.0, ((-1, "Z"),))


class TestObservable:
    def test_z_factory(self):
        obs = Observable.z(0)
        assert len(obs.terms) == 1
        assert obs.terms[0].coeff == 1.0
        assert obs.num_qubits_required == 1

    def test_zz_factory(self):
        obs = Observable.zz(0, 1)
        assert obs.num_qubits_required == 2

    def test_x_factory(self):
        obs = Observable.x(0)
        assert len(obs.terms) == 1

    def test_identity_factory(self):
        obs = Observable.identity()
        assert obs.num_qubits_required == 0

    def test_addition(self):
        obs1 = Observable.z(0)
        obs2 = Observable.z(1)
        combined = obs1 + obs2
        assert len(combined.terms) == 2

    def test_scalar_multiplication(self):
        obs = Observable.z(0) * 0.5
        assert obs.terms[0].coeff == 0.5

    def test_rmul(self):
        obs = 0.5 * Observable.z(0)
        assert obs.terms[0].coeff == 0.5

    def test_from_pauli_string_zz(self):
        obs = Observable.from_pauli_string("ZZ")
        assert obs.num_qubits_required == 2
        term = obs.terms[0]
        d = term.ops_dict
        assert d == {0: "Z", 1: "Z"}

    def test_from_pauli_string_with_identity(self):
        obs = Observable.from_pauli_string("ZIZ")
        term = obs.terms[0]
        d = term.ops_dict
        assert d == {0: "Z", 2: "Z"}

    def test_from_pauli_string_with_coeff(self):
        obs = Observable.from_pauli_string("XY", coeff=0.5)
        assert obs.terms[0].coeff == 0.5

    def test_from_pauli_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            Observable.from_pauli_string("ZWX")

    def test_num_qubits_empty(self):
        obs = Observable([])
        assert obs.num_qubits_required == 0

    def test_repr(self):
        obs = Observable.z(0) + Observable.z(1)
        r = repr(obs)
        assert "Z0" in r

    def test_terms_returns_copy(self):
        obs = Observable.z(0)
        terms = obs.terms
        terms.append(PauliTerm(1.0, ()))
        assert len(obs.terms) == 1
