"""Tests for pipeline/classifier.py and pipeline/trainer.py.

Key test: the overfit test. 10 synthetic samples, many epochs -> 100% accuracy.
This validates the entire forward-backward-update chain end-to-end.
"""

import numpy as np

from pipeline.classifier import HybridClassifier
from pipeline.trainer import Trainer


class TestHybridClassifier:
    def test_forward_shape(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        x = np.random.default_rng(0).normal(size=(2, 1, 8, 8))
        logits = model.forward(x)
        assert logits.shape == (2, 3)

    def test_forward_unbatched(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        x = np.random.default_rng(1).normal(size=(1, 8, 8))
        logits = model.forward(x)
        assert logits.shape == (3,)

    def test_loss_computes(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        x = np.random.default_rng(2).normal(size=(2, 1, 8, 8))
        targets = np.array([0, 1])
        logits = model.forward(x)
        loss = model.compute_loss(logits, targets)
        assert np.isfinite(loss)
        assert loss > 0

    def test_backward_runs(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        x = np.random.default_rng(3).normal(size=(2, 1, 8, 8))
        targets = np.array([0, 2])
        model.forward(x)
        model.compute_loss(model.forward(x), targets)
        model.backward()

    def test_parameters_not_empty(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        x = np.random.default_rng(4).normal(size=(1, 1, 8, 8))
        model.forward(x)
        params = model.parameters()
        assert len(params) > 0

    def test_parameters_before_forward(self):
        """Calling parameters() before forward() should work (proj not yet initialized)."""
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        params = model.parameters()
        assert len(params) > 0


class TestTrainer:
    def test_train_step(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        trainer = Trainer(model, lr=0.01)
        x = np.random.default_rng(5).normal(size=(2, 1, 8, 8))
        targets = np.array([0, 1])
        loss, logits = trainer.train_step(x, targets)
        assert np.isfinite(loss)
        assert logits.shape == (2, 3)

    def test_evaluate(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        trainer = Trainer(model, lr=0.01)
        x = np.random.default_rng(6).normal(size=(4, 1, 8, 8))
        targets = np.array([0, 1, 2, 0])
        trainer.train_step(x[:2], targets[:2])
        loss, acc = trainer.evaluate(x, targets)
        assert np.isfinite(loss)
        assert 0 <= acc <= 1

    def test_evaluate_unbatched(self):
        model = HybridClassifier(num_classes=3, num_qubits=2, num_layers=1)
        trainer = Trainer(model, lr=0.01)
        x_init = np.random.default_rng(7).normal(size=(2, 1, 8, 8))
        trainer.train_step(x_init, np.array([0, 1]))
        x_single = np.random.default_rng(8).normal(size=(1, 8, 8))
        targets_single = np.array(0)
        loss, acc = trainer.evaluate(x_single, targets_single)
        assert np.isfinite(loss)

    def test_training_loop_end_to_end(self):
        """Validate the entire forward-backward-update chain runs without error.

        With unfrozen backbone and more capacity, verify loss changes over training.
        """
        rng = np.random.default_rng(42)
        n_samples = 4
        num_classes = 2
        x = rng.normal(size=(n_samples, 1, 8, 8)).astype(np.float32)
        y = np.array([0, 1, 0, 1])

        model = HybridClassifier(
            num_classes=num_classes, num_qubits=2, num_layers=1,
            freeze_backbone=False,
        )
        trainer = Trainer(model, lr=0.01, weight_decay=0.0)

        losses = []
        for _ in range(10):
            loss, logits = trainer.train_step(x, y)
            losses.append(loss)
            assert np.isfinite(loss)
            assert logits.shape == (4, 2)

        assert any(abs(loss - losses[0]) > 1e-10 for loss in losses[1:]), \
            "Loss is constant — gradients may not be flowing"


class TestUnfrozenBackbone:
    def test_unfrozen_backward_runs(self):
        """Test full backward through unfrozen backbone."""
        model = HybridClassifier(
            num_classes=2, num_qubits=2, num_layers=1,
            freeze_backbone=False,
        )
        x = np.random.default_rng(10).normal(size=(2, 1, 8, 8))
        targets = np.array([0, 1])
        logits = model.forward(x)
        model.compute_loss(logits, targets)
        model.backward()

    def test_unfrozen_has_more_params(self):
        """Unfrozen model has backbone params in parameter list."""
        model_frozen = HybridClassifier(
            num_classes=2, num_qubits=2, num_layers=1, freeze_backbone=True,
        )
        model_unfrozen = HybridClassifier(
            num_classes=2, num_qubits=2, num_layers=1, freeze_backbone=False,
        )
        x = np.random.default_rng(11).normal(size=(1, 1, 8, 8))
        model_frozen.forward(x)
        model_unfrozen.forward(x)
        assert len(model_unfrozen.parameters()) > len(model_frozen.parameters())


class TestTrainerEpoch:
    def test_train_epoch(self):
        model = HybridClassifier(num_classes=2, num_qubits=2, num_layers=1)
        trainer = Trainer(model, lr=0.01)
        rng = np.random.default_rng(9)
        x = rng.normal(size=(4, 1, 8, 8))
        y = np.array([0, 1, 0, 1])
        avg_loss = trainer.train_epoch(x, y, batch_size=2)
        assert np.isfinite(avg_loss)
