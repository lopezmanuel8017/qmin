"""Tests for pipeline/quantum_reranker.py — kernel-based proposal re-ranking."""

import numpy as np

from pipeline.quantum_reranker import QuantumKernelReranker


class TestQuantumKernelReranker:
    def test_forward_shape(self):
        reranker = QuantumKernelReranker(feature_dim=8, num_qubits=2)
        features = np.random.default_rng(0).normal(size=(5, 8))
        scores = np.random.default_rng(1).uniform(0.3, 0.9, size=(5,))
        result = reranker.forward(features, scores)
        assert result.shape == (5,)

    def test_reranked_scores_differ(self):
        """Re-ranked scores should generally differ from originals."""
        reranker = QuantumKernelReranker(feature_dim=8, num_qubits=2)
        features = np.random.default_rng(0).normal(size=(4, 8))
        scores = np.array([0.9, 0.5, 0.3, 0.1])
        result = reranker.forward(features, scores)
        assert not np.allclose(result, scores)

    def test_reranked_scores_bounded(self):
        """Output should be bounded by [min(scores), max(scores)]."""
        reranker = QuantumKernelReranker(feature_dim=4, num_qubits=2)
        features = np.random.default_rng(2).normal(size=(3, 4))
        scores = np.array([0.2, 0.5, 0.8])
        result = reranker.forward(features, scores)
        assert np.all(result >= scores.min() - 1e-10)
        assert np.all(result <= scores.max() + 1e-10)

    def test_backward_runs(self):
        reranker = QuantumKernelReranker(feature_dim=4, num_qubits=2)
        features = np.random.default_rng(3).normal(size=(3, 4))
        scores = np.array([0.5, 0.3, 0.7])
        reranker.forward(features, scores)
        reranker.backward(np.ones(3))

    def test_parameters(self):
        reranker = QuantumKernelReranker(feature_dim=4, num_qubits=2)
        params = reranker.parameters()
        assert len(params) >= 2

    def test_sync_from_array(self):
        reranker = QuantumKernelReranker(feature_dim=4, num_qubits=2)
        reranker.sync_from_array()

    def test_backward_negligible_gradient_skip(self):
        """Exercise the gradient skip path when grad_kernel entries are tiny."""
        reranker = QuantumKernelReranker(feature_dim=4, num_qubits=2)
        features = np.random.default_rng(10).normal(size=(3, 4))
        scores = np.array([0.5, 0.5, 0.5])
        reranker.forward(features, scores)
        reranker.backward(np.full(3, 1e-15))
