"""Variational ansatze (parameterized quantum circuits) for quantum ML.

Each ansatz builds a parameterized Circuit with trainable rotation gates
and entangling layers. Multiple topologies are provided for the multi-topology
QCNN approach.

CNOTLadder:
  Ry+Rz per qubit, then CNOT chain (0→1, 1→2, ..., n-2→n-1). Repeat L layers.
  Linear entanglement — good for nearest-neighbor hardware connectivity.

RingTopology:
  Same rotations but CNOT ring (includes n-1→0 wrap-around).
  All-to-all effective connectivity over depth.

MultiTopology:
  Runs multiple topology circuits and concatenates measurement outputs.
  Provides diverse feature extraction perspectives.
"""

from __future__ import annotations

import numpy as np

from qsim.circuit import Circuit
from qsim.parameters import Parameter


class CNOTLadder:
    """Variational ansatz with linear CNOT entanglement.

    Structure per layer:
      - Ry(θ_i) on each qubit i
      - Rz(φ_i) on each qubit i
      - CNOT(0,1), CNOT(1,2), ..., CNOT(n-2,n-1)
    """

    def __init__(self, num_qubits: int, num_layers: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._params: list[Parameter] = []

        for layer in range(num_layers):
            for qubit in range(num_qubits):
                self._params.append(Parameter(f"ladder_L{layer}_q{qubit}_ry"))
                self._params.append(Parameter(f"ladder_L{layer}_q{qubit}_rz"))

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._params)

    @property
    def num_parameters(self) -> int:
        return len(self._params)

    def circuit(self) -> Circuit:
        """Build the parameterized ansatz circuit."""
        qc = Circuit(self.num_qubits, name="cnot_ladder")
        idx = 0
        for _layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qc.ry(self._params[idx], qubit)
                qc.rz(self._params[idx + 1], qubit)
                idx += 2
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
        return qc

    def init_params(
        self,
        seed: int = 0,
        strategy: str = "uniform",
        epsilon: float = 0.01,
    ) -> dict[Parameter, float]:
        """Parameter initialization with configurable strategy.

        Strategies:
          - "uniform": Uniform(0, 2π). Standard but causes barren plateaus at scale.
          - "identity_block": Normal(0, epsilon). Circuit starts near identity,
            avoiding exponential gradient vanishing (Grant et al. 2019).
          - "small_random": Normal(0, scale) where scale shrinks with circuit size.
        """
        rng = np.random.default_rng(seed)
        n = len(self._params)
        if strategy == "uniform":
            values = rng.uniform(0, 2 * np.pi, n)
        elif strategy == "identity_block":
            values = rng.normal(0, epsilon, n)
        elif strategy == "small_random":
            scale = np.pi / (2 * np.sqrt(self.num_layers * self.num_qubits))
            values = rng.normal(0, scale, n)
        else:
            raise ValueError(f"Unknown init strategy: {strategy!r}")
        return {p: float(v) for p, v in zip(self._params, values)}


class RingTopology:
    """Variational ansatz with ring CNOT entanglement.

    Same as CNOTLadder but with an additional CNOT(n-1, 0) per layer,
    closing the ring for all-to-all effective connectivity over depth.
    """

    def __init__(self, num_qubits: int, num_layers: int) -> None:
        if num_qubits < 2:
            raise ValueError(f"Ring topology requires >= 2 qubits, got {num_qubits}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._params: list[Parameter] = []

        for layer in range(num_layers):
            for qubit in range(num_qubits):
                self._params.append(Parameter(f"ring_L{layer}_q{qubit}_ry"))
                self._params.append(Parameter(f"ring_L{layer}_q{qubit}_rz"))

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._params)

    @property
    def num_parameters(self) -> int:
        return len(self._params)

    def circuit(self) -> Circuit:
        """Build the parameterized ansatz circuit."""
        qc = Circuit(self.num_qubits, name="ring_topology")
        idx = 0
        for _layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qc.ry(self._params[idx], qubit)
                qc.rz(self._params[idx + 1], qubit)
                idx += 2
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            qc.cx(self.num_qubits - 1, 0)
        return qc

    def init_params(
        self,
        seed: int = 0,
        strategy: str = "uniform",
        epsilon: float = 0.01,
    ) -> dict[Parameter, float]:
        """Parameter initialization with configurable strategy.

        See CNOTLadder.init_params for strategy descriptions.
        """
        rng = np.random.default_rng(seed)
        n = len(self._params)
        if strategy == "uniform":
            values = rng.uniform(0, 2 * np.pi, n)
        elif strategy == "identity_block":
            values = rng.normal(0, epsilon, n)
        elif strategy == "small_random":
            scale = np.pi / (2 * np.sqrt(self.num_layers * self.num_qubits))
            values = rng.normal(0, scale, n)
        else:
            raise ValueError(f"Unknown init strategy: {strategy!r}")
        return {p: float(v) for p, v in zip(self._params, values)}


class MultiTopology:
    """Run multiple ansatz topologies and concatenate measurement outputs.

    Each topology produces num_qubits measurement values. The total output
    dimension is num_topologies * num_qubits.
    """

    def __init__(self, topologies: list[CNOTLadder | RingTopology]) -> None:
        if not topologies:
            raise ValueError("At least one topology required")
        self.topologies = topologies
        self._num_qubits = topologies[0].num_qubits
        for t in topologies:
            if t.num_qubits != self._num_qubits:
                raise ValueError("All topologies must have the same num_qubits")

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def output_dim(self) -> int:
        return len(self.topologies) * self._num_qubits

    @property
    def parameters(self) -> list[Parameter]:
        return [p for t in self.topologies for p in t.parameters]

    @property
    def num_parameters(self) -> int:
        return sum(t.num_parameters for t in self.topologies)

    def circuits(self) -> list[Circuit]:
        """Return list of circuits, one per topology."""
        return [t.circuit() for t in self.topologies]

    def init_params(
        self,
        seed: int = 0,
        strategy: str = "uniform",
        epsilon: float = 0.01,
    ) -> dict[Parameter, float]:
        result: dict[Parameter, float] = {}
        for i, t in enumerate(self.topologies):
            result.update(t.init_params(seed=seed + i, strategy=strategy, epsilon=epsilon))
        return result
