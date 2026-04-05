"""qsim — Pure NumPy quantum circuit simulator.

Provides statevector and density matrix simulation with tensor contraction,
parameterized circuits, noise channels, gradient computation via
parameter-shift rule, and OpenQASM export.
"""

from .circuit import Circuit, Instruction
from .density_matrix import DensityMatrix
from .gradient import (
    compute_cost_and_gradient,
    numerical_gradient,
    parameter_shift_gradient,
)
from .measurement import expectation_value, partial_measure, sample
from .noise import (
    NoiseChannel,
    NoiseModel,
    ReadoutError,
    amplitude_damping,
    depolarizing,
    phase_damping,
)
from .observables import Observable, PauliTerm
from .parameters import Parameter, ParameterVector
from .qasm_export import to_qasm2, to_qasm3
from .statevector import Statevector

__version__ = "0.2.0"

__all__ = [
    "Circuit",
    "DensityMatrix",
    "Instruction",
    "NoiseChannel",
    "NoiseModel",
    "Observable",
    "Parameter",
    "ParameterVector",
    "PauliTerm",
    "ReadoutError",
    "Statevector",
    "amplitude_damping",
    "compute_cost_and_gradient",
    "depolarizing",
    "expectation_value",
    "numerical_gradient",
    "parameter_shift_gradient",
    "partial_measure",
    "phase_damping",
    "sample",
    "to_qasm2",
    "to_qasm3",
]
