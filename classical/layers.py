"""Classical neural network layers with manual forward/backward.

Every layer implements:
  - forward(x) -> y  (caches input for backward)
  - backward(grad_output) -> grad_input  (also stores param gradients)
  - parameters() -> list of (param, grad) pairs

Backprop formulas are derived from first principles and validated against
numerical finite-difference (see tests/test_layers.py).

References:
  - Conv2d im2col: Chellapilla et al. (2006), "High Performance Convolutional
    Neural Networks for Document Processing"
  - BatchNorm: Ioffe & Szegedy (2015), "Batch Normalization"
  - Backprop: Goodfellow, Bengio & Courville (2016), "Deep Learning", Ch. 6
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Layer:
    """Base class for all layers."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (param, grad) tuples. Empty for parameter-free layers."""
        return []


class Linear(Layer):
    """Fully connected layer: y = xW^T + b.

    Forward: y_j = Σ_i x_i W_{ji} + b_j
    Backward:
      dL/dx_i = Σ_j dL/dy_j · W_{ji}          (grad_input)
      dL/dW_{ji} = Σ_batch dL/dy_j · x_i       (weight gradient)
      dL/db_j = Σ_batch dL/dy_j                 (bias gradient)

    Reference: Goodfellow et al. (2016), Ch. 6.5.2
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.default_rng(42).normal(0, scale, (out_features, in_features))
        self.grad_weight = np.zeros_like(self.weight)
        self._use_bias = bias
        if bias:
            self.bias = np.zeros(out_features)
            self.grad_bias = np.zeros(out_features)
        else:
            self.bias = None
            self.grad_bias = None
        self._input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        out = x @ self.weight.T
        if self._use_bias:
            out = out + self.bias
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self._input
        if x.ndim == 1:
            self.grad_weight = np.outer(grad_output, x)
        else:
            self.grad_weight = grad_output.T @ x

        if self._use_bias:
            if grad_output.ndim == 1:
                self.grad_bias = grad_output.copy()
            else:
                self.grad_bias = grad_output.sum(axis=0)

        grad_input = grad_output @ self.weight
        return grad_input

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = [(self.weight, self.grad_weight)]
        if self._use_bias:
            params.append((self.bias, self.grad_bias))
        return params


class ReLU(Layer):
    """ReLU activation: y = max(0, x).

    Backward: dL/dx = dL/dy · 1_{x > 0}
    At x=0, gradient is 0 (subgradient convention).
    """

    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._mask


class Flatten(Layer):
    """Flatten all dimensions except batch: (N, C, H, W) -> (N, C*H*W).

    Handles both batched (ndim > 1) and unbatched inputs.
    """

    def __init__(self) -> None:
        self._input_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        if x.ndim <= 1:
            return x
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self._input_shape)


class MaxPool2d(Layer):
    """2D max pooling with stride = kernel_size (non-overlapping).

    Forward: For each pooling window, take the max value.
    Backward: Route gradient to the position of the max in each window.

    Input shape: (N, C, H, W) or (C, H, W)
    """

    def __init__(self, kernel_size: int) -> None:
        self.kernel_size = kernel_size
        self._input: Optional[np.ndarray] = None
        self._max_indices: Optional[np.ndarray] = None
        self._was_unbatched = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._was_unbatched = (x.ndim == 3)
        if self._was_unbatched:
            x = x[np.newaxis]

        self._input = x
        k = self.kernel_size
        N, C, H, W = x.shape
        H_out = H // k
        W_out = W // k

        x_reshaped = x[:, :, :H_out * k, :W_out * k].reshape(N, C, H_out, k, W_out, k)
        out = x_reshaped.max(axis=(3, 5))

        self._max_indices = x_reshaped.reshape(N, C, H_out, k, W_out, k)
        return out[0] if self._was_unbatched else out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._was_unbatched:
            grad_output = grad_output[np.newaxis]

        x = self._input
        k = self.kernel_size
        N, C, H, W = x.shape
        H_out = H // k
        W_out = W // k

        grad_input = np.zeros_like(x)
        x_reshaped = x[:, :, :H_out * k, :W_out * k].reshape(N, C, H_out, k, W_out, k)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        window = x_reshaped[n, c, i, :, j, :]
                        max_idx = np.unravel_index(window.argmax(), window.shape)
                        grad_input[n, c, i * k + max_idx[0], j * k + max_idx[1]] = \
                            grad_output[n, c, i, j]

        return grad_input[0] if self._was_unbatched else grad_input


class Conv2d(Layer):
    """2D convolution using im2col for efficient implementation.

    im2col transforms the convolution into a matrix multiplication:
      1. Unfold input patches into columns (im2col)
      2. Multiply: output = weight_matrix @ col_matrix + bias
      3. Reshape to output spatial dimensions

    Forward: y[n,f,i,j] = Σ_{c,kh,kw} x[n,c,i+kh,j+kw] · W[f,c,kh,kw] + b[f]
    Backward:
      dL/dW[f,c,kh,kw] = Σ_{n,i,j} dL/dy[n,f,i,j] · x[n,c,i+kh,j+kw]
      dL/dx: col2im of (W^T @ dL/dy_reshaped)

    Reference: Chellapilla et al. (2006)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        rng = np.random.default_rng(42)
        self.weight = rng.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size))
        self.grad_weight = np.zeros_like(self.weight)

        self._use_bias = bias
        if bias:
            self.bias = np.zeros(out_channels)
            self.grad_bias = np.zeros(out_channels)
        else:
            self.bias = None
            self.grad_bias = None

        self._input: Optional[np.ndarray] = None
        self._col: Optional[np.ndarray] = None
        self._was_unbatched = False

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        """Transform input to column matrix for matrix-multiply convolution.

        Input: (N, C, H, W)
        Output: (N, C*kH*kW, H_out*W_out)
        """
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        col = np.zeros((N, C, k, k, H_out, W_out), dtype=x.dtype)
        for i in range(k):
            i_max = i + s * H_out
            for j in range(k):
                j_max = j + s * W_out
                col[:, :, i, j, :, :] = x[:, :, i:i_max:s, j:j_max:s]

        return col.reshape(N, C * k * k, H_out * W_out)

    def _col2im(self, col: np.ndarray, input_shape: tuple) -> np.ndarray:
        """Inverse of im2col: accumulate column gradients back to input shape.

        col: (N, C*kH*kW, H_out*W_out)
        Returns: (N, C, H, W)
        """
        N, C, H, W = input_shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        col_reshaped = col.reshape(N, C, k, k, H_out, W_out)
        x = np.zeros(input_shape, dtype=col.dtype)

        for i in range(k):
            i_max = i + s * H_out
            for j in range(k):
                j_max = j + s * W_out
                x[:, :, i:i_max:s, j:j_max:s] += col_reshaped[:, :, i, j, :, :]

        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._was_unbatched = (x.ndim == 3)
        if self._was_unbatched:
            x = x[np.newaxis]

        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )

        self._input = x
        N, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        col = self._im2col(x)
        self._col = col

        W_col = self.weight.reshape(self.out_channels, -1)

        out = np.zeros((N, self.out_channels, H_out * W_out), dtype=x.dtype)
        for n in range(N):
            out[n] = W_col @ col[n]

        out = out.reshape(N, self.out_channels, H_out, W_out)
        if self._use_bias:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out[0] if self._was_unbatched else out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._was_unbatched:
            grad_output = grad_output[np.newaxis]

        N, F, H_out, W_out = grad_output.shape
        grad_out_reshaped = grad_output.reshape(N, F, -1)

        W_col = self.weight.reshape(self.out_channels, -1)

        self.grad_weight = np.zeros_like(self.weight)
        grad_col = np.zeros_like(self._col)

        for n in range(N):
            self.grad_weight += (grad_out_reshaped[n] @ self._col[n].T).reshape(
                self.weight.shape
            )
            grad_col[n] = W_col.T @ grad_out_reshaped[n]

        if self._use_bias:
            self.grad_bias = grad_output.sum(axis=(0, 2, 3))

        grad_input = self._col2im(grad_col, self._input.shape)

        if self.padding > 0:
            p = self.padding
            grad_input = grad_input[:, :, p:-p, p:-p]

        return grad_input[0] if self._was_unbatched else grad_input

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = [(self.weight, self.grad_weight)]
        if self._use_bias:
            params.append((self.bias, self.grad_bias))
        return params


class BatchNorm1d(Layer):
    """1D Batch Normalization (Ioffe & Szegedy, 2015).

    Forward (training):
      μ = mean(x, axis=0)
      σ² = var(x, axis=0)
      x_hat = (x - μ) / √(σ² + ε)
      y = γ · x_hat + β

    Running stats updated with momentum for inference.

    Backward:
      dL/dγ = Σ dL/dy · x_hat
      dL/dβ = Σ dL/dy
      dL/dx: see Ioffe & Szegedy (2015), Section 3
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.grad_gamma = np.zeros(num_features)
        self.grad_beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True

        self._x_hat: Optional[np.ndarray] = None
        self._std_inv: Optional[np.ndarray] = None
        self._input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x

        if self.training:
            if x.ndim == 1:
                mean = x.mean()
                var = x.var()
            else:
                mean = x.mean(axis=0)
                var = x.var(axis=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            std_inv = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x - mean) * std_inv
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (x - self.running_mean) * std_inv

        self._x_hat = x_hat
        self._std_inv = std_inv
        return self.gamma * x_hat + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x_hat = self._x_hat
        std_inv = self._std_inv

        if grad_output.ndim == 1:
            N = 1
        else:
            N = grad_output.shape[0]

        self.grad_gamma = (grad_output * x_hat).sum(axis=0) if grad_output.ndim > 1 \
            else grad_output * x_hat
        self.grad_beta = grad_output.sum(axis=0) if grad_output.ndim > 1 \
            else grad_output.copy()

        dx_hat = grad_output * self.gamma
        grad_input = (1.0 / N) * std_inv * (
            N * dx_hat - dx_hat.sum(axis=0) - x_hat * (dx_hat * x_hat).sum(axis=0)
        )
        return grad_input

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]
