"""Tests for pipeline/distillation.py — knowledge distillation training."""

import numpy as np

from classical.detection_head import RPNHead
from pipeline.distillation import (
    DistillationTrainer,
    _reshape_cls, _reshape_cls_targets, _reshape_reg,
    _unreshape_cls, _unreshape_reg,
)


class TestReshapeHelpers:
    def test_reshape_cls_roundtrip(self):
        """Reshape + unreshape should recover original shape."""
        x = np.random.default_rng(0).normal(size=(2, 6, 4, 4))
        flat = _reshape_cls(x)
        assert flat.shape == (2 * 4 * 4 * 3, 2)
        recovered = _unreshape_cls(flat, x.shape)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_reshape_cls_unbatched(self):
        x = np.random.default_rng(1).normal(size=(6, 4, 4))
        flat = _reshape_cls(x)
        assert flat.shape == (4 * 4 * 3, 2)

    def test_reshape_reg_roundtrip(self):
        x = np.random.default_rng(2).normal(size=(2, 12, 4, 4))
        flat = _reshape_reg(x)
        assert flat.shape == (2 * 4 * 4 * 3, 4)
        recovered = _unreshape_reg(flat, x.shape)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_reshape_reg_unbatched(self):
        x = np.random.default_rng(3).normal(size=(12, 3, 3))
        flat = _reshape_reg(x)
        assert flat.shape == (3 * 3 * 3, 4)

    def test_reshape_cls_targets(self):
        targets = np.zeros((1, 4, 2, 2))
        targets[:, 0, :, :] = 1
        targets[:, 3, :, :] = 1
        flat = _reshape_cls_targets(targets)
        assert flat.shape == (1 * 2 * 2 * 2,)
        assert set(flat.tolist()).issubset({0, 1})

    def test_reshape_cls_targets_unbatched(self):
        targets = np.zeros((4, 2, 2))
        targets[0, :, :] = 1
        flat = _reshape_cls_targets(targets)
        assert flat.shape == (2 * 2 * 2,)

    def test_unreshape_cls_unbatched(self):
        flat = np.random.default_rng(4).normal(size=(4 * 4 * 3, 2))
        result = _unreshape_cls(flat, (6, 4, 4))
        assert result.shape == (6, 4, 4)

    def test_unreshape_reg_unbatched(self):
        flat = np.random.default_rng(5).normal(size=(4 * 4 * 3, 4))
        result = _unreshape_reg(flat, (12, 4, 4))
        assert result.shape == (12, 4, 4)


class TestDistillationTrainer:
    def test_train_step(self):
        teacher = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)
        student = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)

        trainer = DistillationTrainer(
            teacher, student, alpha=0.5, temperature=3.0, lr=0.01
        )

        rng = np.random.default_rng(6)
        feature_map = rng.normal(size=(1, 32, 4, 4))

        cls_targets = np.zeros((1, 4, 4, 4))
        cls_targets[:, 0, :, :] = 1
        reg_targets = np.zeros((1, 8, 4, 4))

        losses = trainer.train_step(feature_map, cls_targets, reg_targets)
        assert "total_loss" in losses
        assert "task_loss" in losses
        assert "kd_loss" in losses
        assert "reg_loss" in losses
        assert np.isfinite(losses["total_loss"])

    def test_loss_changes_over_steps(self):
        teacher = RPNHead(in_channels=16, mid_channels=32, num_anchors=1)
        student = RPNHead(in_channels=16, mid_channels=32, num_anchors=1)

        trainer = DistillationTrainer(teacher, student, alpha=0.5, lr=0.01)

        rng = np.random.default_rng(7)
        feature_map = rng.normal(size=(1, 16, 3, 3))
        cls_targets = np.zeros((1, 2, 3, 3))
        cls_targets[:, 0, :, :] = 1
        reg_targets = np.zeros((1, 4, 3, 3))

        losses = []
        for _ in range(5):
            result = trainer.train_step(feature_map, cls_targets, reg_targets)
            losses.append(result["total_loss"])

        assert not all(np.isclose(loss, losses[0]) for loss in losses)
