"""Knowledge distillation for quantum RPN training.

Trains a quantum RPN surrogate to mimic a classical teacher RPN.
Uses a mixed loss combining task loss (ground truth) and distillation loss
(soft targets from teacher).

L = α · L_task(student, ground_truth) + (1-α) · L_KD(student, teacher/T)

Reference: Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"
Quantum-Eyes architecture: classical teacher RPN -> quantum student RPN.
"""

from __future__ import annotations

import numpy as np

from classical.detection_head import RPNHead
from classical.loss import CrossEntropyLoss, KLDivergenceLoss, SmoothL1Loss
from classical.optim import AdamW


class DistillationTrainer:
    """Train a student model to mimic a teacher using knowledge distillation.

    The teacher provides soft targets via temperature-scaled softmax.
    The student learns from both hard labels and teacher's soft predictions.
    """

    def __init__(
        self,
        teacher: RPNHead,
        student: RPNHead,
        alpha: float = 0.5,
        temperature: float = 3.0,
        lr: float = 1e-3,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.alpha = alpha

        self.task_loss_fn = CrossEntropyLoss()
        self.kd_loss_fn = KLDivergenceLoss(temperature=temperature)
        self.reg_loss_fn = SmoothL1Loss()

        self.optimizer = AdamW(student.parameters(), lr=lr)

    def train_step(
        self,
        feature_map: np.ndarray,
        cls_targets: np.ndarray,
        reg_targets: np.ndarray,
    ) -> dict[str, float]:
        """One distillation training step.

        Args:
            feature_map: (N, C, H, W) input features
            cls_targets: (N, num_anchors*2, H, W) ground truth class labels
            reg_targets: (N, num_anchors*4, H, W) ground truth bbox deltas

        Returns:
            dict with 'total_loss', 'task_loss', 'kd_loss'
        """
        teacher_cls, _ = self.teacher.forward(feature_map)
        student_cls, student_reg = self.student.forward(feature_map)

        s_cls_flat = _reshape_cls(student_cls)
        t_cls_flat = _reshape_cls(teacher_cls)
        targets_flat = _reshape_cls_targets(cls_targets)

        task_loss = self.task_loss_fn.forward(s_cls_flat, targets_flat)

        kd_loss = self.kd_loss_fn.forward(s_cls_flat, t_cls_flat)

        s_reg_flat = _reshape_reg(student_reg)
        targets_reg_flat = _reshape_reg(reg_targets)
        reg_loss = self.reg_loss_fn.forward(s_reg_flat, targets_reg_flat)

        total_loss = self.alpha * (task_loss + reg_loss) + (1 - self.alpha) * kd_loss

        grad_task = self.task_loss_fn.backward()
        grad_kd = self.kd_loss_fn.backward()
        grad_cls_flat = self.alpha * grad_task + (1 - self.alpha) * grad_kd

        grad_reg_flat = self.alpha * self.reg_loss_fn.backward()

        grad_cls = _unreshape_cls(grad_cls_flat, student_cls.shape)
        grad_reg = _unreshape_reg(grad_reg_flat, student_reg.shape)

        self.student.backward(grad_cls, grad_reg)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "kd_loss": kd_loss,
            "reg_loss": reg_loss,
        }


def _reshape_cls(cls_scores: np.ndarray) -> np.ndarray:
    """(N, num_anchors*2, H, W) -> (N*H*W*num_anchors, 2)"""
    if cls_scores.ndim == 3:
        cls_scores = cls_scores[np.newaxis]
    N, C, H, W = cls_scores.shape
    num_anchors = C // 2
    return cls_scores.reshape(N, num_anchors, 2, H, W).transpose(0, 3, 4, 1, 2).reshape(-1, 2)


def _reshape_cls_targets(targets: np.ndarray) -> np.ndarray:
    """(N, num_anchors*2, H, W) -> (N*H*W*num_anchors,) as class indices."""
    if targets.ndim == 3:
        targets = targets[np.newaxis]
    N, C, H, W = targets.shape
    num_anchors = C // 2
    reshaped = targets.reshape(N, num_anchors, 2, H, W).transpose(0, 3, 4, 1, 2).reshape(-1, 2)
    return np.argmax(reshaped, axis=1)


def _reshape_reg(reg: np.ndarray) -> np.ndarray:
    """(N, num_anchors*4, H, W) -> (N*H*W*num_anchors, 4)"""
    if reg.ndim == 3:
        reg = reg[np.newaxis]
    N, C, H, W = reg.shape
    num_anchors = C // 4
    return reg.reshape(N, num_anchors, 4, H, W).transpose(0, 3, 4, 1, 2).reshape(-1, 4)


def _unreshape_cls(grad_flat: np.ndarray, target_shape: tuple) -> np.ndarray:
    """(N*H*W*num_anchors, 2) -> (N, num_anchors*2, H, W)"""
    if len(target_shape) == 3:
        N = 1
        C, H, W = target_shape
    else:
        N, C, H, W = target_shape
    num_anchors = C // 2
    grad = grad_flat.reshape(N, H, W, num_anchors, 2).transpose(0, 3, 4, 1, 2).reshape(N, C, H, W)
    return grad[0] if len(target_shape) == 3 else grad


def _unreshape_reg(grad_flat: np.ndarray, target_shape: tuple) -> np.ndarray:
    """(N*H*W*num_anchors, 4) -> (N, num_anchors*4, H, W)"""
    if len(target_shape) == 3:
        N = 1
        C, H, W = target_shape
    else:
        N, C, H, W = target_shape
    num_anchors = C // 4
    grad = grad_flat.reshape(N, H, W, num_anchors, 4).transpose(0, 3, 4, 1, 2).reshape(N, C, H, W)
    return grad[0] if len(target_shape) == 3 else grad
