"""Quantum kernel-based proposal re-ranker for object detection.

Uses a trained quantum kernel to compute pairwise similarity between
proposal features, then re-ranks proposals via kernel-weighted score
aggregation. This gives the quantum component a well-motivated role:
computing similarity in 2^n-dimensional Hilbert space.

Architecture:
  1. Receive proposal features (n_proposals, feature_dim) and base scores (n_proposals,)
  2. Project features to num_qubits dimensions
  3. Compute quantum kernel matrix K (n_proposals x n_proposals)
  4. Re-rank: new_scores = softmax(K) @ base_scores
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from classical.layers import Linear
from qsim.gradient import PARAMETER_SHIFT
from quantum.kernel import QuantumKernel


class QuantumKernelReranker:
    """Re-ranks proposals using quantum kernel similarity."""

    def __init__(
        self,
        feature_dim: int,
        num_qubits: int = 4,
        init_strategy: str = "uniform",
        init_epsilon: float = 0.01,
    ) -> None:
        self.feature_dim = feature_dim
        self.num_qubits = num_qubits

        self.proj = Linear(feature_dim, num_qubits)

        self.kernel = QuantumKernel(
            num_qubits,
            init_strategy=init_strategy,
            init_epsilon=init_epsilon,
        )

        self._proj_features: Optional[np.ndarray] = None
        self._kernel_matrix: Optional[np.ndarray] = None
        self._attn_weights: Optional[np.ndarray] = None
        self._base_scores: Optional[np.ndarray] = None

    def forward(
        self,
        proposal_features: np.ndarray,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        """Re-rank proposals using quantum kernel similarity.

        Args:
            proposal_features: (n_proposals, feature_dim)
            base_scores: (n_proposals,) original confidence scores

        Returns:
            reranked_scores: (n_proposals,)
        """
        self._base_scores = base_scores

        self._proj_features = self.proj.forward(proposal_features)

        self._kernel_matrix = self.kernel.compute_matrix(self._proj_features)

        shifted = self._kernel_matrix - self._kernel_matrix.max(axis=1, keepdims=True)
        exp_k = np.exp(shifted)
        self._attn_weights = exp_k / exp_k.sum(axis=1, keepdims=True)

        reranked = self._attn_weights @ base_scores
        return reranked

    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass — gradient for quantum kernel parameters.

        Uses parameter-shift rule on the kernel entries.
        """
        n = len(grad_output)
        shift = PARAMETER_SHIFT

        grad_attn = np.outer(grad_output, self._base_scores)

        grad_kernel = np.zeros_like(self._kernel_matrix)
        for i in range(n):
            p = self._attn_weights[i]
            dp = grad_attn[i]
            grad_kernel[i] = p * (dp - np.sum(dp * p))

        self.kernel._grad_array[:] = 0.0
        params = self.kernel._trainable_params
        param_values = self.kernel._param_values

        for p_idx, param in enumerate(params):
            theta = param_values[param]

            shifted_plus = dict(param_values)
            shifted_plus[param] = theta + shift

            shifted_minus = dict(param_values)
            shifted_minus[param] = theta - shift

            for i in range(n):
                for j in range(i, n):
                    if abs(grad_kernel[i, j] + grad_kernel[j, i]) < 1e-12:
                        continue

                    old_vals = dict(self.kernel._param_values)

                    self.kernel._param_values = shifted_plus
                    k_plus = self.kernel.compute_entry(
                        self._proj_features[i], self._proj_features[j],
                    )
                    self.kernel._param_values = shifted_minus
                    k_minus = self.kernel.compute_entry(
                        self._proj_features[i], self._proj_features[j],
                    )
                    self.kernel._param_values = old_vals

                    dk = (k_plus - k_minus) / (2 * np.sin(shift))

                    self.kernel._grad_array[p_idx] += grad_kernel[i, j] * dk
                    if i != j:
                        self.kernel._grad_array[p_idx] += grad_kernel[j, i] * dk

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = list(self.proj.parameters())
        params.extend(self.kernel.parameters())
        return params

    def sync_from_array(self) -> None:
        self.kernel.sync_from_array()
