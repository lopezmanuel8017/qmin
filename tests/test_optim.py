"""Tests for classical/optim.py — optimizer correctness."""

import numpy as np

from classical.optim import AdamW, SGD


class TestAdamW:
    def _make_params(self):
        """Create simple params with known gradients."""
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        return [(param, grad)]

    def test_step_updates_params(self):
        params = self._make_params()
        opt = AdamW(params, lr=0.01, weight_decay=0.0)
        old_param = params[0][0].copy()
        opt.step()
        assert not np.allclose(params[0][0], old_param)

    def test_step_count(self):
        params = self._make_params()
        opt = AdamW(params)
        assert opt.t == 0
        opt.step()
        assert opt.t == 1
        opt.step()
        assert opt.t == 2

    def test_zero_grad(self):
        params = self._make_params()
        opt = AdamW(params)
        opt.zero_grad()
        np.testing.assert_allclose(params[0][1], 0.0)

    def test_decoupled_weight_decay(self):
        """AdamW's weight decay is decoupled — verify it differs from Adam (L2).

        With L2 reg, the effective gradient would be g + λ·θ, then Adam processes it.
        With AdamW, the gradient g goes through Adam, then λ·θ is subtracted separately.
        These give different results when adaptive learning rates vary.
        """
        param_aw = np.array([5.0])
        grad_aw = np.array([0.1])
        opt_aw = AdamW([(param_aw, grad_aw)], lr=0.1, weight_decay=0.1,
                       betas=(0.9, 0.999))
        opt_aw.step()
        val_aw = param_aw[0]

        param_l2 = np.array([5.0])
        grad_l2 = np.array([0.1 + 0.1 * 5.0])
        opt_l2 = AdamW([(param_l2, grad_l2)], lr=0.1, weight_decay=0.0,
                       betas=(0.9, 0.999))
        opt_l2.step()
        val_l2 = param_l2[0]

        assert not np.isclose(val_aw, val_l2), (
            f"AdamW ({val_aw}) should differ from Adam+L2 ({val_l2})"
        )

    def test_bias_correction(self):
        """Early steps should have bias correction effect."""
        param = np.array([0.0])
        grad = np.array([1.0])
        opt = AdamW([(param, grad)], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)

        opt.step()
        assert abs(param[0]) > 0.05

    def test_convergence_on_quadratic(self):
        """AdamW should minimize f(x) = x² from x=5."""
        param = np.array([5.0])
        grad = np.zeros(1)
        opt = AdamW([(param, grad)], lr=0.1, weight_decay=0.0)

        for _ in range(200):
            grad[:] = 2 * param
            opt.step()

        assert abs(param[0]) < 0.1, f"Failed to converge: x={param[0]}"


class TestSGD:
    def test_step_updates_params(self):
        param = np.array([1.0, 2.0])
        grad = np.array([0.5, 0.5])
        opt = SGD([(param, grad)], lr=0.1)
        old = param.copy()
        opt.step()
        np.testing.assert_allclose(param, old - 0.1 * grad)

    def test_momentum_accumulation(self):
        """With momentum, velocity should accumulate across steps."""
        param = np.array([1.0])
        grad = np.array([1.0])
        opt = SGD([(param, grad)], lr=0.1, momentum=0.9)

        opt.step()
        np.testing.assert_allclose(param, [0.9])

        opt.step()
        np.testing.assert_allclose(param, [0.71])

    def test_zero_grad(self):
        param = np.array([1.0])
        grad = np.array([0.5])
        opt = SGD([(param, grad)])
        opt.zero_grad()
        np.testing.assert_allclose(grad, 0.0)

    def test_convergence_on_quadratic(self):
        param = np.array([5.0])
        grad = np.zeros(1)
        opt = SGD([(param, grad)], lr=0.01)

        for _ in range(500):
            grad[:] = 2 * param
            opt.step()

        assert abs(param[0]) < 0.1, f"Failed to converge: x={param[0]}"
