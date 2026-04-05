"""Shared test fixtures for the quantum simulator test suite."""

import pytest

from qsim.circuit import Circuit
from qsim.parameters import Parameter


@pytest.fixture
def bell_circuit():
    """H(0), CNOT(0,1) -> Bell state |Φ+> = (|00> + |11>) / √2"""
    qc = Circuit(2, name="bell")
    qc.h(0).cx(0, 1)
    return qc


@pytest.fixture
def ghz3_circuit():
    """3-qubit GHZ state: (|000> + |111>) / √2"""
    qc = Circuit(3, name="ghz3")
    qc.h(0).cx(0, 1).cx(0, 2)
    return qc


@pytest.fixture
def parameterized_circuit():
    """Rx(theta) on qubit 0, Ry(phi) on qubit 1 of a 2-qubit circuit."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = Circuit(2, name="param_test")
    qc.rx(theta, 0).ry(phi, 1)
    return qc, theta, phi
