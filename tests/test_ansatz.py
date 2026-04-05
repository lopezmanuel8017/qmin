"""Tests for quantum/ansatz.py — variational circuit ansatze."""

import numpy as np
import pytest

from quantum.ansatz import CNOTLadder, MultiTopology, RingTopology
from qsim.statevector import Statevector


class TestCNOTLadder:
    def test_creation(self):
        ansatz = CNOTLadder(4, 2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_layers == 2

    def test_parameter_count(self):
        """2 params per qubit per layer (Ry + Rz)."""
        ansatz = CNOTLadder(4, 3)
        assert ansatz.num_parameters == 4 * 3 * 2

    def test_circuit_structure(self):
        ansatz = CNOTLadder(3, 1)
        qc = ansatz.circuit()
        assert qc.num_qubits == 3
        assert len(qc) == 8

    def test_circuit_gate_types(self):
        ansatz = CNOTLadder(3, 1)
        qc = ansatz.circuit()
        gate_names = [inst.gate.name for inst in qc.instructions]
        assert gate_names.count("Ry") == 3
        assert gate_names.count("Rz") == 3
        assert gate_names.count("CNOT") == 2

    def test_cnot_targets(self):
        """CNOT chain: (0,1), (1,2), ..."""
        ansatz = CNOTLadder(4, 1)
        qc = ansatz.circuit()
        cnots = [inst for inst in qc.instructions if inst.gate.name == "CNOT"]
        assert cnots[0].qubits == (0, 1)
        assert cnots[1].qubits == (1, 2)
        assert cnots[2].qubits == (2, 3)

    def test_circuit_depth(self):
        ansatz = CNOTLadder(3, 1)
        qc = ansatz.circuit()
        assert qc.depth >= 3

    def test_init_params(self):
        ansatz = CNOTLadder(2, 1)
        params = ansatz.init_params(seed=42)
        assert len(params) == ansatz.num_parameters
        assert all(0 <= v < 2 * np.pi for v in params.values())

    def test_simulate(self):
        """Circuit with bound params should produce a valid state."""
        ansatz = CNOTLadder(3, 2)
        qc = ansatz.circuit()
        params = ansatz.init_params(seed=0)
        bound = qc.bind_parameters(params)
        sv = Statevector.from_circuit(bound)
        assert np.isclose(sv.norm(), 1.0)

    def test_single_qubit(self):
        """With 1 qubit, no CNOTs are added."""
        ansatz = CNOTLadder(1, 2)
        qc = ansatz.circuit()
        cnots = [inst for inst in qc.instructions if inst.gate.name == "CNOT"]
        assert len(cnots) == 0
        assert ansatz.num_parameters == 4

    def test_parameters_property(self):
        ansatz = CNOTLadder(2, 1)
        params = ansatz.parameters
        assert len(params) == 4
        assert all(p.name.startswith("ladder_") for p in params)

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            CNOTLadder(0, 1)

    def test_invalid_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            CNOTLadder(2, 0)


class TestRingTopology:
    def test_creation(self):
        ansatz = RingTopology(4, 2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_layers == 2

    def test_parameter_count(self):
        ansatz = RingTopology(4, 3)
        assert ansatz.num_parameters == 4 * 3 * 2

    def test_includes_wrap_around_cnot(self):
        """Ring should include CNOT(n-1, 0) per layer."""
        ansatz = RingTopology(4, 1)
        qc = ansatz.circuit()
        cnots = [inst for inst in qc.instructions if inst.gate.name == "CNOT"]
        assert len(cnots) == 4
        assert cnots[-1].qubits == (3, 0)

    def test_more_cnots_than_ladder(self):
        """Ring has one extra CNOT per layer compared to ladder."""
        ladder = CNOTLadder(4, 2)
        ring = RingTopology(4, 2)
        ladder_cnots = sum(1 for inst in ladder.circuit().instructions if inst.gate.name == "CNOT")
        ring_cnots = sum(1 for inst in ring.circuit().instructions if inst.gate.name == "CNOT")
        assert ring_cnots == ladder_cnots + 2

    def test_init_params(self):
        ansatz = RingTopology(3, 1)
        params = ansatz.init_params(seed=42)
        assert len(params) == ansatz.num_parameters

    def test_simulate(self):
        ansatz = RingTopology(4, 2)
        qc = ansatz.circuit()
        params = ansatz.init_params(seed=1)
        bound = qc.bind_parameters(params)
        sv = Statevector.from_circuit(bound)
        assert np.isclose(sv.norm(), 1.0)

    def test_parameters_property(self):
        ansatz = RingTopology(3, 1)
        params = ansatz.parameters
        assert len(params) == 6
        assert all(p.name.startswith("ring_") for p in params)

    def test_min_qubits_raises(self):
        with pytest.raises(ValueError, match="requires >= 2 qubits"):
            RingTopology(1, 1)

    def test_invalid_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            RingTopology(3, 0)


class TestMultiTopology:
    def test_creation(self):
        topologies = [CNOTLadder(4, 2), RingTopology(4, 2)]
        mt = MultiTopology(topologies)
        assert mt.num_qubits == 4
        assert mt.output_dim == 8

    def test_parameters_combined(self):
        t1 = CNOTLadder(3, 1)
        t2 = RingTopology(3, 1)
        mt = MultiTopology([t1, t2])
        assert mt.num_parameters == t1.num_parameters + t2.num_parameters

    def test_circuits(self):
        t1 = CNOTLadder(3, 1)
        t2 = RingTopology(3, 1)
        mt = MultiTopology([t1, t2])
        circuits = mt.circuits()
        assert len(circuits) == 2
        assert circuits[0].name == "cnot_ladder"
        assert circuits[1].name == "ring_topology"

    def test_init_params(self):
        t1 = CNOTLadder(3, 1)
        t2 = RingTopology(3, 1)
        mt = MultiTopology([t1, t2])
        params = mt.init_params(seed=0)
        assert len(params) == mt.num_parameters

    def test_parameters_property(self):
        t1 = CNOTLadder(3, 1)
        t2 = RingTopology(3, 1)
        mt = MultiTopology([t1, t2])
        params = mt.parameters
        assert len(params) == t1.num_parameters + t2.num_parameters

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiTopology([])

    def test_mismatched_qubits_raises(self):
        t1 = CNOTLadder(3, 1)
        t2 = CNOTLadder(4, 1)
        with pytest.raises(ValueError, match="same num_qubits"):
            MultiTopology([t1, t2])


class TestInitStrategies:
    """Test configurable parameter initialization strategies."""

    def test_identity_block_values_near_zero(self):
        ansatz = CNOTLadder(4, 2)
        params = ansatz.init_params(seed=0, strategy="identity_block", epsilon=0.01)
        values = np.array(list(params.values()))
        assert np.all(np.abs(values) < 0.1)

    def test_identity_block_ring(self):
        ansatz = RingTopology(3, 2)
        params = ansatz.init_params(seed=0, strategy="identity_block")
        values = np.array(list(params.values()))
        assert np.all(np.abs(values) < 0.1)

    def test_small_random_values_smaller_than_uniform(self):
        ansatz = CNOTLadder(4, 2)
        uniform = ansatz.init_params(seed=0, strategy="uniform")
        small = ansatz.init_params(seed=0, strategy="small_random")
        uniform_range = max(uniform.values()) - min(uniform.values())
        small_range = max(small.values()) - min(small.values())
        assert small_range < uniform_range

    def test_invalid_strategy_raises(self):
        ansatz = CNOTLadder(2, 1)
        with pytest.raises(ValueError, match="Unknown init strategy"):
            ansatz.init_params(strategy="bogus")

    def test_invalid_strategy_ring_raises(self):
        ansatz = RingTopology(2, 1)
        with pytest.raises(ValueError, match="Unknown init strategy"):
            ansatz.init_params(strategy="bogus")

    def test_backward_compatible_default(self):
        """Default 'uniform' produces same results as before."""
        ansatz = CNOTLadder(3, 1)
        params = ansatz.init_params(seed=42)
        assert len(params) == ansatz.num_parameters
        assert all(0 <= v < 2 * np.pi for v in params.values())

    def test_multi_topology_forwards_strategy(self):
        t1 = CNOTLadder(3, 1)
        t2 = RingTopology(3, 1)
        mt = MultiTopology([t1, t2])
        params = mt.init_params(seed=0, strategy="identity_block", epsilon=0.005)
        values = np.array(list(params.values()))
        assert np.all(np.abs(values) < 0.05)

    def test_small_random_ring(self):
        ansatz = RingTopology(3, 2)
        params = ansatz.init_params(seed=0, strategy="small_random")
        values = np.array(list(params.values()))
        assert np.max(np.abs(values)) < np.pi
