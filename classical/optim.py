"""Optimizers with manual update rules.

AdamW (Loshchilov & Hutter, 2019):
  Decoupled weight decay — applies weight decay directly to weight update,
  not folded into gradient as L2 regularization.

  Update rule:
    m_t = β₁·m_{t-1} + (1-β₁)·g_t
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - lr·(m̂_t/(√v̂_t + ε) + λ·θ_{t-1})

  Key difference from Adam: the weight decay term λ·θ is not part of the
  gradient, so it doesn't interact with the adaptive learning rate. This
  matters when parameter scales vary (quantum vs classical params).

SGD with Momentum:
  v_t = μ·v_{t-1} + g_t
  θ_t = θ_{t-1} - lr·v_t
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class AdamW:
    """AdamW optimizer (Loshchilov & Hutter, 2019)."""

    def __init__(
        self,
        params: Sequence[tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        """Perform one optimization step."""
        self.t += 1
        for i, (param, grad) in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                + self.weight_decay * param)

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for _, grad in self.params:
            grad.fill(0.0)


class SGD:
    """Stochastic Gradient Descent with momentum."""

    def __init__(
        self,
        params: Sequence[tuple[np.ndarray, np.ndarray]],
        lr: float = 0.01,
        momentum: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p, _ in self.params]

    def step(self) -> None:
        for i, (param, grad) in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            param -= self.lr * self.velocities[i]

    def zero_grad(self) -> None:
        for _, grad in self.params:
            grad.fill(0.0)
