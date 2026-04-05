"""Diagnostics for variational quantum circuits.

Provides tools to detect barren plateaus by estimating gradient variance
across random parameter initializations. Exponentially vanishing gradient
variance is the signature of a barren plateau (McClean et al. 2018).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from qsim.gradient import parameter_shift_gradient
from qsim.observables import Observable
from quantum.ansatz import CNOTLadder, RingTopology


def estimate_gradient_variance(
    ansatz: Union[CNOTLadder, RingTopology],
    observable: Optional[Observable] = None,
    strategy: str = "uniform",
    epsilon: float = 0.01,
    n_samples: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate gradient variance across random initializations.

    Samples n_samples random parameter sets using the given strategy,
    computes parameter-shift gradients for each, and returns variance
    statistics. Low variance indicates a barren plateau.

    Args:
        ansatz: Variational ansatz to analyze.
        observable: Observable to measure. Defaults to Z on qubit 0.
        strategy: Initialization strategy ("uniform", "identity_block", "small_random").
        epsilon: Scale for identity_block initialization.
        n_samples: Number of random initializations to sample.
        seed: Base random seed.

    Returns:
        Dict with keys:
          - "mean_grad_variance": average variance across all parameters.
          - "per_param_variance": dict mapping parameter name to its variance.
          - "num_samples": number of samples used.
    """
    if observable is None:
        observable = Observable.z(0)

    circuit = ansatz.circuit()
    param_names = [p.name for p in ansatz.parameters]
    n_params = ansatz.num_parameters

    all_grads = np.zeros((n_samples, n_params))

    for i in range(n_samples):
        param_values = ansatz.init_params(seed=seed + i, strategy=strategy, epsilon=epsilon)
        grads = parameter_shift_gradient(circuit, observable, param_values)
        for j, p in enumerate(ansatz.parameters):
            all_grads[i, j] = grads[p]

    variances = np.var(all_grads, axis=0)

    per_param_variance = {
        name: float(var) for name, var in zip(param_names, variances)
    }

    return {
        "mean_grad_variance": float(np.mean(variances)),
        "per_param_variance": per_param_variance,
        "num_samples": n_samples,
    }
