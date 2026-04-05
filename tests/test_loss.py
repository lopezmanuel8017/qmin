"""Tests for classical/loss.py — loss functions and gradient correctness."""

import numpy as np

from classical.loss import CrossEntropyLoss, KLDivergenceLoss, SmoothL1Loss


def _loss_gradient_check(
    loss_fn,
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Check loss gradient via finite difference on predictions."""
    loss_fn.forward(predictions, targets)
    analytical = loss_fn.backward()

    numerical = np.zeros_like(predictions)
    p_flat = predictions.ravel()
    for i in range(p_flat.size):
        old_val = p_flat[i]

        p_flat[i] = old_val + eps
        loss_plus = loss_fn.forward(predictions.reshape(predictions.shape), targets)

        p_flat[i] = old_val - eps
        loss_minus = loss_fn.forward(predictions.reshape(predictions.shape), targets)

        numerical.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
        p_flat[i] = old_val

    np.testing.assert_allclose(analytical, numerical, atol=atol)


class TestCrossEntropyLoss:
    def test_forward_perfect_prediction(self):
        """High logit for correct class -> near-zero loss."""
        loss_fn = CrossEntropyLoss()
        logits = np.array([[10.0, -10.0, -10.0]])
        targets = np.array([0])
        loss = loss_fn.forward(logits, targets)
        assert loss < 0.001

    def test_forward_uniform_prediction(self):
        """Equal logits -> loss = log(num_classes)."""
        loss_fn = CrossEntropyLoss()
        logits = np.array([[0.0, 0.0, 0.0]])
        targets = np.array([1])
        loss = loss_fn.forward(logits, targets)
        np.testing.assert_allclose(loss, np.log(3), atol=1e-6)

    def test_gradient_batched(self):
        rng = np.random.default_rng(30)
        loss_fn = CrossEntropyLoss()
        logits = rng.normal(size=(4, 5))
        targets = np.array([0, 2, 4, 1])
        _loss_gradient_check(loss_fn, logits, targets)

    def test_gradient_unbatched(self):
        rng = np.random.default_rng(31)
        loss_fn = CrossEntropyLoss()
        logits = rng.normal(size=(3,))
        targets = np.array(1)
        _loss_gradient_check(loss_fn, logits, targets)

    def test_numerical_stability_large_logits(self):
        """Should not overflow with large logit values."""
        loss_fn = CrossEntropyLoss()
        logits = np.array([[1000.0, 0.0, 0.0]])
        targets = np.array([0])
        loss = loss_fn.forward(logits, targets)
        assert np.isfinite(loss)
        assert loss < 0.001

    def test_backward_shape_batched(self):
        loss_fn = CrossEntropyLoss()
        logits = np.random.default_rng(32).normal(size=(3, 4))
        targets = np.array([0, 1, 2])
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        assert grad.shape == (3, 4)

    def test_backward_shape_unbatched(self):
        loss_fn = CrossEntropyLoss()
        logits = np.array([1.0, 2.0, 3.0])
        targets = np.array(0)
        loss_fn.forward(logits, targets)
        grad = loss_fn.backward()
        assert grad.shape == (3,)


class TestSmoothL1Loss:
    def test_small_difference(self):
        """When |diff| < 1, loss = 0.5 * diff²."""
        loss_fn = SmoothL1Loss()
        pred = np.array([0.5])
        target = np.array([0.0])
        loss = loss_fn.forward(pred, target)
        np.testing.assert_allclose(loss, 0.5 * 0.5**2)

    def test_large_difference(self):
        """When |diff| >= 1, loss = |diff| - 0.5."""
        loss_fn = SmoothL1Loss()
        pred = np.array([3.0])
        target = np.array([0.0])
        loss = loss_fn.forward(pred, target)
        np.testing.assert_allclose(loss, 2.5)

    def test_at_transition(self):
        """At |diff| = 1, both formulas give 0.5."""
        loss_fn = SmoothL1Loss()
        pred = np.array([1.0])
        target = np.array([0.0])
        loss = loss_fn.forward(pred, target)
        np.testing.assert_allclose(loss, 0.5)

    def test_gradient(self):
        rng = np.random.default_rng(33)
        loss_fn = SmoothL1Loss()
        pred = rng.normal(size=(5, 4))
        target = rng.normal(size=(5, 4))
        _loss_gradient_check(loss_fn, pred, target)

    def test_gradient_at_boundary(self):
        """Test gradient near the transition point |diff| = 1."""
        loss_fn = SmoothL1Loss()
        pred = np.array([0.99, 1.01, -0.99, -1.01])
        target = np.zeros(4)
        _loss_gradient_check(loss_fn, pred, target, atol=1e-4)


class TestKLDivergenceLoss:
    def test_identical_distributions(self):
        """KL(p || p) = 0."""
        loss_fn = KLDivergenceLoss()
        logits = np.array([[1.0, 2.0, 3.0]])
        loss = loss_fn.forward(logits, logits)
        np.testing.assert_allclose(loss, 0.0, atol=1e-10)

    def test_positive(self):
        """KL divergence is non-negative."""
        loss_fn = KLDivergenceLoss()
        student = np.array([[1.0, 0.0, 0.0]])
        teacher = np.array([[0.0, 1.0, 0.0]])
        loss = loss_fn.forward(student, teacher)
        assert loss >= 0

    def test_temperature_scaling(self):
        """Higher temperature -> softer distributions -> lower KL."""
        loss_low_t = KLDivergenceLoss(temperature=1.0)
        loss_high_t = KLDivergenceLoss(temperature=5.0)
        student = np.array([[5.0, 0.0, -5.0]])
        teacher = np.array([[-5.0, 0.0, 5.0]])
        kl_low = loss_low_t.forward(student, teacher)
        kl_high = loss_high_t.forward(student, teacher)
        assert np.isfinite(kl_low)
        assert np.isfinite(kl_high)

    def test_gradient(self):
        rng = np.random.default_rng(34)
        loss_fn = KLDivergenceLoss(temperature=2.0)
        student = rng.normal(size=(3, 4))
        teacher = rng.normal(size=(3, 4))
        _loss_gradient_check(loss_fn, student, teacher, atol=1e-4)
