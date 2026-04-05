"""Tests for qsim/qasm_export.py — OpenQASM serialization."""

import numpy as np
import pytest

from qsim.circuit import Circuit
from qsim.parameters import Parameter
from qsim.qasm_export import to_qasm2, to_qasm3


class TestQASM2:
    def test_bell_circuit(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        qasm = to_qasm2(qc)

        assert "OPENQASM 2.0;" in qasm
        assert 'include "qelib1.inc";' in qasm
        assert "qreg q[2];" in qasm
        assert "creg c[2];" in qasm
        assert "h q[0];" in qasm
        assert "cx q[0],q[1];" in qasm

    def test_parameterized_gate(self):
        qc = Circuit(1)
        qc.rx(1.5707963267948966, 0)
        qasm = to_qasm2(qc)
        assert "rx(1.5707963267949)" in qasm

    def test_no_classical_register(self):
        qc = Circuit(1)
        qc.h(0)
        qasm = to_qasm2(qc, classical_register=False)
        assert "creg" not in qasm

    def test_custom_register_name(self):
        qc = Circuit(1)
        qc.h(0)
        qasm = to_qasm2(qc, register_name="qr")
        assert "qreg qr[1];" in qasm
        assert "h qr[0];" in qasm

    def test_all_fixed_gates(self):
        qc = Circuit(2)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.cx(0, 1).cz(0, 1).swap(0, 1)
        qasm = to_qasm2(qc)

        for name in ["h", "x", "y", "z", "s", "t", "cx", "cz", "swap"]:
            assert name in qasm

    def test_all_parameterized_gates(self):
        qc = Circuit(2)
        qc.rx(1.0, 0).ry(2.0, 0).rz(3.0, 0).phase(0.5, 0)
        qc.crx(1.0, 0, 1).cry(2.0, 0, 1).crz(3.0, 0, 1)
        qasm = to_qasm2(qc)

        for name in ["rx", "ry", "rz", "p", "crx", "cry", "crz"]:
            assert name in qasm

    def test_unbound_parameters_raise(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        with pytest.raises(ValueError, match="Bind parameters"):
            to_qasm2(qc)

    def test_empty_circuit(self):
        qc = Circuit(1)
        qasm = to_qasm2(qc)
        assert "qreg q[1];" in qasm

    def test_ends_with_newline(self):
        qc = Circuit(1)
        qc.h(0)
        qasm = to_qasm2(qc)
        assert qasm.endswith("\n")


class TestQASM3:
    def test_bell_circuit(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        qasm = to_qasm3(qc)

        assert "OPENQASM 3.0;" in qasm
        assert 'include "stdgates.inc";' in qasm
        assert "qubit[2] q;" in qasm
        assert "bit[2] c;" in qasm
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm

    def test_parameterized_gate(self):
        qc = Circuit(1)
        qc.ry(np.pi, 0)
        qasm = to_qasm3(qc)
        assert "ry(" in qasm

    def test_unbound_parameters_raise(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        with pytest.raises(ValueError, match="Bind parameters"):
            to_qasm3(qc)

    def test_custom_register_name(self):
        qc = Circuit(1)
        qc.h(0)
        qasm = to_qasm3(qc, register_name="qr")
        assert "qubit[1] qr;" in qasm
        assert "h qr[0];" in qasm

    def test_ends_with_newline(self):
        qc = Circuit(1)
        qc.h(0)
        qasm = to_qasm3(qc)
        assert qasm.endswith("\n")
