"""Tests for qsim/circuit.py — circuit abstraction and parameter management."""

import pytest

from qsim.circuit import Circuit
from qsim.gates import CNOT, H, Rx
from qsim.parameters import Parameter, ParameterVector


class TestCircuitCreation:
    def test_basic_creation(self):
        qc = Circuit(3)
        assert qc.num_qubits == 3
        assert len(qc) == 0
        assert qc.depth == 0

    def test_zero_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            Circuit(0)

    def test_name(self):
        qc = Circuit(2, name="test")
        assert qc.name == "test"

    def test_repr(self):
        qc = Circuit(2)
        qc.h(0)
        r = repr(qc)
        assert "num_qubits=2" in r
        assert "instructions=1" in r


class TestFluentAPI:
    def test_chaining(self):
        qc = Circuit(2)
        result = qc.h(0).cx(0, 1).rz(1.0, 1)
        assert result is qc
        assert len(qc) == 3

    def test_all_single_qubit_gates(self):
        qc = Circuit(1)
        qc.h(0).x(0).y(0).z(0).s(0).t(0)
        qc.rx(1.0, 0).ry(1.0, 0).rz(1.0, 0).phase(1.0, 0)
        assert len(qc) == 10

    def test_all_two_qubit_gates(self):
        qc = Circuit(2)
        qc.cx(0, 1).cnot(1, 0).cz(0, 1).swap(0, 1)
        qc.crx(1.0, 0, 1).cry(1.0, 0, 1).crz(1.0, 0, 1)
        assert len(qc) == 7

    def test_apply_generic(self):
        qc = Circuit(2)
        qc.apply(H, (0,))
        qc.apply(CNOT, (0, 1))
        qc.apply(Rx, (0,), (1.5,))
        assert len(qc) == 3


class TestValidation:
    def test_qubit_out_of_range(self):
        qc = Circuit(2)
        with pytest.raises(ValueError, match="out of range"):
            qc.h(2)

    def test_negative_qubit(self):
        qc = Circuit(2)
        with pytest.raises(ValueError, match="out of range"):
            qc.h(-1)

    def test_duplicate_qubits(self):
        qc = Circuit(2)
        with pytest.raises(ValueError, match="Duplicate"):
            qc.cx(0, 0)

    def test_wrong_qubit_count(self):
        qc = Circuit(2)
        with pytest.raises(ValueError, match="requires 1 qubits"):
            qc.apply(H, (0, 1))

    def test_wrong_param_count(self):
        qc = Circuit(2)
        with pytest.raises(ValueError, match="requires 0 params"):
            qc.apply(H, (0,), (1.0,))


class TestInstructions:
    def test_instruction_properties(self):
        qc = Circuit(2)
        qc.rx(1.5, 0)
        inst = qc.instructions[0]
        assert inst.gate is Rx
        assert inst.qubits == (0,)
        assert inst.params == (1.5,)
        assert not inst.is_parameterized

    def test_parameterized_instruction(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        inst = qc.instructions[0]
        assert inst.is_parameterized
        assert inst.params == (theta,)

    def test_instructions_tuple_is_copy(self):
        qc = Circuit(1)
        qc.h(0)
        insts = qc.instructions
        assert isinstance(insts, tuple)
        assert len(insts) == 1


class TestParameters:
    def test_no_parameters(self):
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        assert qc.num_parameters == 0
        assert not qc.is_parameterized()

    def test_single_parameter(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        assert qc.num_parameters == 1
        assert qc.is_parameterized()
        assert theta in qc.parameters

    def test_reused_parameter(self):
        """Same Parameter used twice counts as one."""
        theta = Parameter("theta")
        qc = Circuit(2)
        qc.rx(theta, 0).ry(theta, 1)
        assert qc.num_parameters == 1

    def test_ordered_parameters_stable(self):
        """Parameters ordered by first appearance."""
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = Circuit(3)
        qc.rx(b, 0).ry(a, 1).rz(c, 2).rx(a, 0)
        ordered = qc.ordered_parameters
        assert ordered == [b, a, c]

    def test_parameter_vector(self):
        pv = ParameterVector("theta", 3)
        qc = Circuit(3)
        for i in range(3):
            qc.ry(pv[i], i)
        assert qc.num_parameters == 3


class TestBindParameters:
    def test_full_binding(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        bound = qc.bind_parameters({theta: 1.5})
        assert not bound.is_parameterized()
        assert bound.instructions[0].params == (1.5,)

    def test_original_unchanged(self):
        theta = Parameter("theta")
        qc = Circuit(1)
        qc.rx(theta, 0)
        _ = qc.bind_parameters({theta: 1.5})
        assert qc.is_parameterized()

    def test_partial_binding(self):
        a = Parameter("a")
        b = Parameter("b")
        qc = Circuit(2)
        qc.rx(a, 0).ry(b, 1)
        bound = qc.bind_parameters({a: 1.0})
        assert bound.is_parameterized()
        assert bound.num_parameters == 1
        assert bound.instructions[0].params == (1.0,)
        assert bound.instructions[1].params[0] is b

    def test_bind_preserves_fixed_params(self):
        theta = Parameter("theta")
        qc = Circuit(2)
        qc.rx(theta, 0).ry(2.0, 1)
        bound = qc.bind_parameters({theta: 1.0})
        assert bound.instructions[1].params == (2.0,)


class TestCompose:
    def test_simple_compose(self):
        qc1 = Circuit(2)
        qc1.h(0)
        qc2 = Circuit(2)
        qc2.cx(0, 1)
        qc1.compose(qc2)
        assert len(qc1) == 2

    def test_compose_with_qubit_map(self):
        qc1 = Circuit(4)
        qc1.h(0)
        sub = Circuit(2)
        sub.cx(0, 1)
        qc1.compose(sub, qubit_map={0: 2, 1: 3})
        inst = qc1.instructions[1]
        assert inst.qubits == (2, 3)

    def test_compose_returns_self(self):
        qc1 = Circuit(2)
        qc2 = Circuit(2)
        qc2.h(0)
        result = qc1.compose(qc2)
        assert result is qc1


class TestDepth:
    def test_empty_circuit(self):
        qc = Circuit(2)
        assert qc.depth == 0

    def test_single_gate(self):
        qc = Circuit(2)
        qc.h(0)
        assert qc.depth == 1

    def test_parallel_gates(self):
        """Two gates on different qubits: depth 1 each, total depth 1."""
        qc = Circuit(2)
        qc.h(0).h(1)
        assert qc.depth == 1

    def test_sequential_gates(self):
        """Two gates on same qubit: depth 2."""
        qc = Circuit(1)
        qc.h(0).x(0)
        assert qc.depth == 2

    def test_bell_circuit_depth(self):
        """H(0), CNOT(0,1): depth 2."""
        qc = Circuit(2)
        qc.h(0).cx(0, 1)
        assert qc.depth == 2
