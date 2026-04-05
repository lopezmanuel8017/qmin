"""quantum — Quantum ML components: encoding, ansatze, attention, kernels, diagnostics."""

from .ansatz import CNOTLadder, MultiTopology, RingTopology
from .attention import QuantumSelfAttention
from .diagnostics import estimate_gradient_variance
from .encoding import AngleEncoder, TriValueEncoder
from .kernel import QuantumKernel

__all__ = [
    "AngleEncoder", "TriValueEncoder",
    "CNOTLadder", "RingTopology", "MultiTopology",
    "QuantumSelfAttention", "QuantumKernel",
    "estimate_gradient_variance",
]
