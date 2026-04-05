"""Tests for classical/layers.py — forward correctness and gradient checking.

Every backward() is validated against numerical finite-difference.
Pattern: perturb each param by ε=1e-5, compute loss change, compare
to analytical gradient. Agreement to 1e-5 relative error.
"""

import numpy as np

from classical.layers import (
    BatchNorm1d, Conv2d, Flatten, Layer, Linear, MaxPool2d, ReLU,
)


def _numerical_gradient_check(
    layer: Layer,
    x: np.ndarray,
    upstream_grad: np.ndarray,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Check input gradient via finite difference.

    For each element x[i]:
      numerical_grad[i] ≈ (f(x+ε) - f(x-ε)) / (2ε)
    where f(x) = sum(upstream_grad * forward(x)).
    """
    layer.forward(x)
    analytical = layer.backward(upstream_grad)

    numerical = np.zeros_like(x)
    x_flat = x.ravel()
    for i in range(x_flat.size):
        old_val = x_flat[i]

        x_flat[i] = old_val + eps
        out_plus = layer.forward(x.reshape(x.shape))
        loss_plus = np.sum(upstream_grad * out_plus)

        x_flat[i] = old_val - eps
        out_minus = layer.forward(x.reshape(x.shape))
        loss_minus = np.sum(upstream_grad * out_minus)

        numerical.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
        x_flat[i] = old_val

    layer.forward(x)

    np.testing.assert_allclose(
        analytical, numerical, atol=atol, rtol=rtol,
        err_msg="Input gradient mismatch"
    )


def _param_gradient_check(
    layer: Layer,
    x: np.ndarray,
    upstream_grad: np.ndarray,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Check parameter gradients via finite difference."""
    layer.forward(x)
    layer.backward(upstream_grad)

    for param, grad in layer.parameters():
        numerical = np.zeros_like(param)
        p_flat = param.ravel()
        for i in range(p_flat.size):
            old_val = p_flat[i]

            p_flat[i] = old_val + eps
            out_plus = layer.forward(x)
            loss_plus = np.sum(upstream_grad * out_plus)

            p_flat[i] = old_val - eps
            out_minus = layer.forward(x)
            loss_minus = np.sum(upstream_grad * out_minus)

            numerical.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
            p_flat[i] = old_val

        np.testing.assert_allclose(
            grad, numerical, atol=atol, rtol=rtol,
            err_msg=f"Parameter gradient mismatch for shape {param.shape}"
        )


class TestLinear:
    def test_forward_shape_batched(self):
        layer = Linear(4, 3)
        x = np.random.default_rng(0).normal(size=(5, 4))
        out = layer.forward(x)
        assert out.shape == (5, 3)

    def test_forward_shape_unbatched(self):
        layer = Linear(4, 3)
        x = np.random.default_rng(0).normal(size=(4,))
        out = layer.forward(x)
        assert out.shape == (3,)

    def test_forward_value(self):
        layer = Linear(2, 2, bias=False)
        layer.weight = np.eye(2)
        x = np.array([3.0, 4.0])
        out = layer.forward(x)
        np.testing.assert_allclose(out, [3.0, 4.0])

    def test_input_gradient_batched(self):
        rng = np.random.default_rng(1)
        layer = Linear(4, 3)
        x = rng.normal(size=(5, 4))
        grad_out = rng.normal(size=(5, 3))
        _numerical_gradient_check(layer, x, grad_out)

    def test_input_gradient_unbatched(self):
        rng = np.random.default_rng(2)
        layer = Linear(3, 2)
        x = rng.normal(size=(3,))
        grad_out = rng.normal(size=(2,))
        _numerical_gradient_check(layer, x, grad_out)

    def test_param_gradient_batched(self):
        rng = np.random.default_rng(3)
        layer = Linear(4, 3)
        x = rng.normal(size=(5, 4))
        grad_out = rng.normal(size=(5, 3))
        _param_gradient_check(layer, x, grad_out)

    def test_param_gradient_unbatched(self):
        rng = np.random.default_rng(4)
        layer = Linear(3, 2)
        x = rng.normal(size=(3,))
        grad_out = rng.normal(size=(2,))
        _param_gradient_check(layer, x, grad_out)

    def test_no_bias(self):
        layer = Linear(3, 2, bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1
        x = np.random.default_rng(5).normal(size=(2, 3))
        out = layer.forward(x)
        assert out.shape == (2, 2)

    def test_no_bias_gradient(self):
        rng = np.random.default_rng(6)
        layer = Linear(3, 2, bias=False)
        x = rng.normal(size=(4, 3))
        grad_out = rng.normal(size=(4, 2))
        _numerical_gradient_check(layer, x, grad_out)
        _param_gradient_check(layer, x, grad_out)


class TestReLU:
    def test_forward(self):
        layer = ReLU()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        out = layer.forward(x)
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 2.0])

    def test_backward(self):
        layer = ReLU()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        layer.forward(x)
        grad = layer.backward(np.ones(4))
        np.testing.assert_allclose(grad, [0.0, 0.0, 1.0, 1.0])

    def test_gradient_check(self):
        rng = np.random.default_rng(7)
        layer = ReLU()
        x = rng.normal(size=(3, 4))
        grad_out = rng.normal(size=(3, 4))
        _numerical_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_no_parameters(self):
        assert ReLU().parameters() == []


class TestFlatten:
    def test_forward_4d(self):
        layer = Flatten()
        x = np.zeros((2, 3, 4, 5))
        out = layer.forward(x)
        assert out.shape == (2, 60)

    def test_forward_1d(self):
        layer = Flatten()
        x = np.array([1.0, 2.0])
        out = layer.forward(x)
        np.testing.assert_allclose(out, x)

    def test_backward_restores_shape(self):
        layer = Flatten()
        x = np.random.default_rng(8).normal(size=(2, 3, 4))
        layer.forward(x)
        grad_out = np.random.default_rng(9).normal(size=(2, 12))
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (2, 3, 4)

    def test_no_parameters(self):
        assert Flatten().parameters() == []


class TestMaxPool2d:
    def test_forward_shape(self):
        layer = MaxPool2d(2)
        x = np.random.default_rng(10).normal(size=(2, 3, 4, 4))
        out = layer.forward(x)
        assert out.shape == (2, 3, 2, 2)

    def test_forward_value(self):
        layer = MaxPool2d(2)
        x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
        out = layer.forward(x)
        expected = np.array([[[[5, 7], [13, 15]]]])
        np.testing.assert_allclose(out, expected)

    def test_forward_unbatched(self):
        layer = MaxPool2d(2)
        x = np.arange(16, dtype=float).reshape(1, 4, 4)
        out = layer.forward(x)
        assert out.shape == (1, 2, 2)

    def test_backward_routes_to_max(self):
        layer = MaxPool2d(2)
        x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
        layer.forward(x)
        grad_out = np.ones((1, 1, 2, 2))
        grad_in = layer.backward(grad_out)
        assert grad_in[0, 0, 1, 1] == 1.0
        assert grad_in[0, 0, 0, 0] == 0.0

    def test_backward_unbatched(self):
        layer = MaxPool2d(2)
        x = np.arange(16, dtype=float).reshape(1, 4, 4)
        layer.forward(x)
        grad_out = np.ones((1, 2, 2))
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (1, 4, 4)

    def test_no_parameters(self):
        assert MaxPool2d(2).parameters() == []


class TestConv2d:
    def test_forward_shape(self):
        layer = Conv2d(3, 16, kernel_size=3, padding=1)
        x = np.random.default_rng(11).normal(size=(2, 3, 8, 8))
        out = layer.forward(x)
        assert out.shape == (2, 16, 8, 8)

    def test_forward_shape_no_padding(self):
        layer = Conv2d(1, 1, kernel_size=3)
        x = np.random.default_rng(12).normal(size=(1, 1, 5, 5))
        out = layer.forward(x)
        assert out.shape == (1, 1, 3, 3)

    def test_forward_unbatched(self):
        layer = Conv2d(1, 2, kernel_size=3)
        x = np.random.default_rng(13).normal(size=(1, 5, 5))
        out = layer.forward(x)
        assert out.shape == (2, 3, 3)

    def test_input_gradient(self):
        rng = np.random.default_rng(14)
        layer = Conv2d(1, 2, kernel_size=3, padding=0)
        x = rng.normal(size=(2, 1, 5, 5))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _numerical_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_param_gradient(self):
        rng = np.random.default_rng(15)
        layer = Conv2d(1, 2, kernel_size=3, padding=0)
        x = rng.normal(size=(2, 1, 5, 5))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _param_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_input_gradient_with_padding(self):
        rng = np.random.default_rng(16)
        layer = Conv2d(1, 1, kernel_size=3, padding=1)
        x = rng.normal(size=(1, 1, 4, 4))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _numerical_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_no_bias(self):
        layer = Conv2d(1, 1, kernel_size=3, bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1
        rng = np.random.default_rng(17)
        x = rng.normal(size=(1, 1, 5, 5))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _numerical_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_stride(self):
        layer = Conv2d(1, 1, kernel_size=3, stride=2)
        x = np.random.default_rng(18).normal(size=(1, 1, 7, 7))
        out = layer.forward(x)
        assert out.shape == (1, 1, 3, 3)

    def test_backward_unbatched(self):
        rng = np.random.default_rng(19)
        layer = Conv2d(1, 2, kernel_size=3, padding=0)
        x = rng.normal(size=(1, 5, 5))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (1, 5, 5)


class TestBatchNorm1d:
    def test_forward_shape(self):
        layer = BatchNorm1d(4)
        x = np.random.default_rng(19).normal(size=(8, 4))
        out = layer.forward(x)
        assert out.shape == (8, 4)

    def test_output_normalized(self):
        """After batchnorm (gamma=1, beta=0), output should be ~zero mean, ~unit var."""
        layer = BatchNorm1d(4)
        x = np.random.default_rng(20).normal(loc=5, scale=3, size=(100, 4))
        out = layer.forward(x)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=0), 1.0, atol=0.02)

    def test_input_gradient(self):
        rng = np.random.default_rng(21)
        layer = BatchNorm1d(3)
        x = rng.normal(size=(8, 3))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _numerical_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_param_gradient(self):
        rng = np.random.default_rng(22)
        layer = BatchNorm1d(3)
        x = rng.normal(size=(8, 3))
        out = layer.forward(x)
        grad_out = rng.normal(size=out.shape)
        _param_gradient_check(layer, x, grad_out, atol=1e-4)

    def test_running_stats_updated(self):
        layer = BatchNorm1d(2, momentum=0.1)
        x = np.random.default_rng(23).normal(loc=3, scale=2, size=(10, 2))
        layer.forward(x)
        assert not np.allclose(layer.running_mean, 0.0)

    def test_eval_mode(self):
        layer = BatchNorm1d(2)
        x_train = np.random.default_rng(24).normal(loc=5, size=(50, 2))
        layer.forward(x_train)

        layer.training = False
        x_test = np.random.default_rng(25).normal(loc=5, size=(3, 2))
        out = layer.forward(x_test)
        assert out.shape == (3, 2)

    def test_1d_input(self):
        """BatchNorm on a single (unbatched) 1D input."""
        layer = BatchNorm1d(3)
        x = np.array([1.0, 2.0, 3.0])
        out = layer.forward(x)
        assert out.shape == (3,)
        grad = layer.backward(np.ones(3))
        assert grad.shape == (3,)


class TestLayerBase:
    def test_base_parameters_empty(self):
        assert Layer().parameters() == []
