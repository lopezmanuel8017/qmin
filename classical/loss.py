"""Loss functions with manual gradient computation.

All losses implement:
  - forward(predictions, targets) -> scalar loss
  - backward() -> gradient w.r.t. predictions

References:
  - Cross-entropy with log-softmax: Goodfellow et al. (2016), Ch. 6.2.2
    Uses log-sum-exp trick for numerical stability: log softmax(x)_j = x_j - log(Σ exp(x_i))
  - SmoothL1: Girshick (2015), "Fast R-CNN"
  - KLDivergence: Hinton et al. (2015), "Distilling the Knowledge"
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss with log-softmax fusion for numerical stability.

    Input: logits (N, C) or (C,), targets as integer class indices (N,) or scalar.
    Output: scalar loss = -1/N Σ log(softmax(logits)[target])

    Numerically stable via log-sum-exp trick:
      log softmax(x)_j = x_j - max(x) - log(Σ exp(x_i - max(x)))
    """

    def __init__(self) -> None:
        self._probs: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._was_unbatched = False

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self._was_unbatched = (logits.ndim == 1)
        if self._was_unbatched:
            logits = logits[np.newaxis]
            targets = np.atleast_1d(targets)

        self._targets = targets.astype(int)

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self._probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

        N = logits.shape[0]
        log_probs = np.log(self._probs[np.arange(N), self._targets] + 1e-15)
        return float(-log_probs.mean())

    def backward(self) -> np.ndarray:
        N = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(N), self._targets] -= 1.0
        grad /= N
        return grad[0] if self._was_unbatched else grad


class SmoothL1Loss:
    """Smooth L1 (Huber) loss for bounding box regression.

    L(x) = 0.5 * x² if |x| < 1, else |x| - 0.5

    dL/dx = x if |x| < 1, else sign(x)

    Reference: Girshick (2015), "Fast R-CNN"
    """

    def __init__(self) -> None:
        self._diff: Optional[np.ndarray] = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self._diff = predictions - targets
        abs_diff = np.abs(self._diff)
        loss = np.where(abs_diff < 1.0, 0.5 * self._diff**2, abs_diff - 0.5)
        return float(loss.mean())

    def backward(self) -> np.ndarray:
        abs_diff = np.abs(self._diff)
        grad = np.where(abs_diff < 1.0, self._diff, np.sign(self._diff))
        return grad / self._diff.size


class KLDivergenceLoss:
    """KL divergence for knowledge distillation.

    KL(p || q) = Σ p_i * log(p_i / q_i)

    Used with temperature-scaled softmax for distillation:
      L_KD = T² · KL(softmax(teacher/T) || softmax(student/T))

    Reference: Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature
        self._student_probs: Optional[np.ndarray] = None
        self._teacher_probs: Optional[np.ndarray] = None

    def _softmax_with_temp(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)

    def forward(self, student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
        self._student_probs = self._softmax_with_temp(student_logits)
        self._teacher_probs = self._softmax_with_temp(teacher_logits)

        kl = self._teacher_probs * np.log(
            (self._teacher_probs + 1e-15) / (self._student_probs + 1e-15)
        )
        return float(self.temperature**2 * kl.sum(axis=-1).mean())

    def backward(self) -> np.ndarray:
        N = self._student_probs.shape[0] if self._student_probs.ndim > 1 else 1
        grad = self.temperature * (self._student_probs - self._teacher_probs) / N
        return grad
