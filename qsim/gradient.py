"""Gradient computation via the parameter-shift rule.

Parameter-shift rule (Schuld et al., 2019, Eq. 5):
  df/dθ = [f(θ + π/2) - f(θ - π/2)] / 2

Valid for gates whose generator G has eigenvalues ±1/2, which includes all
standard rotation gates Rx, Ry, Rz (generator = σ_k/2).

General formula for shift s:
  df/dθ = [f(θ + s) - f(θ - s)] / (2·sin(s))

For s = π/2: sin(π/2) = 1, so df/dθ = [f(θ + π/2) - f(θ - π/2)] / 2.

Cost: 2 × num_parameters circuit evaluations for the full gradient.
Each evaluation is O(depth × 2^n) for n qubits.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit
from .measurement import expectation_value
from .observables import Observable
from .parameters import Parameter
from .statevector import Statevector

PARAMETER_SHIFT = np.pi / 2


def parameter_shift_gradient(
    circuit: Circuit,
    observable: Observable,
    param_values: dict[Parameter, float],
    shift: float = PARAMETER_SHIFT,
) -> dict[Parameter, float]:
    """Compute d<O>/d(θ_i) for each parameter via the parameter-shift rule.

    Returns {parameter: gradient_value}.
    """
    gradients: dict[Parameter, float] = {}
    sin_shift = np.sin(shift)

    for param in circuit.ordered_parameters:
        theta = param_values[param]

        shifted_plus = dict(param_values)
        shifted_plus[param] = theta + shift
        bound_plus = circuit.bind_parameters(shifted_plus)
        sv_plus = Statevector.from_circuit(bound_plus)
        exp_plus = expectation_value(sv_plus, observable)

        shifted_minus = dict(param_values)
        shifted_minus[param] = theta - shift
        bound_minus = circuit.bind_parameters(shifted_minus)
        sv_minus = Statevector.from_circuit(bound_minus)
        exp_minus = expectation_value(sv_minus, observable)

        gradients[param] = (exp_plus - exp_minus) / (2 * sin_shift)

    return gradients


def numerical_gradient(
    circuit: Circuit,
    observable: Observable,
    param_values: dict[Parameter, float],
    epsilon: float = 1e-7,
) -> dict[Parameter, float]:
    """Finite-difference gradient (for validation/debugging only).

    Uses central difference: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / (2ε).
    """
    gradients: dict[Parameter, float] = {}

    for param in circuit.ordered_parameters:
        theta = param_values[param]

        shifted_plus = dict(param_values)
        shifted_plus[param] = theta + epsilon
        bound_plus = circuit.bind_parameters(shifted_plus)
        exp_plus = expectation_value(Statevector.from_circuit(bound_plus), observable)

        shifted_minus = dict(param_values)
        shifted_minus[param] = theta - epsilon
        bound_minus = circuit.bind_parameters(shifted_minus)
        exp_minus = expectation_value(Statevector.from_circuit(bound_minus), observable)

        gradients[param] = (exp_plus - exp_minus) / (2 * epsilon)

    return gradients


def compute_cost_and_gradient(
    circuit: Circuit,
    observable: Observable,
    param_values: dict[Parameter, float],
) -> tuple[float, dict[Parameter, float]]:
    """Compute both the expectation value and its gradient.

    Total cost: 1 + 2·num_parameters circuit evaluations.
    """
    bound = circuit.bind_parameters(param_values)
    sv = Statevector.from_circuit(bound)
    cost = expectation_value(sv, observable)
    grad = parameter_shift_gradient(circuit, observable, param_values)
    return cost, grad
