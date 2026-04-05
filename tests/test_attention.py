"""Tests for quantum/attention.py — quantum self-attention."""

import numpy as np
import pytest

from quantum.attention import QuantumSelfAttention


class TestQuantumSelfAttention:
    def test_forward_shape(self):
        attn = QuantumSelfAttention(feature_dim=8, sequence_length=4)
        x = np.random.default_rng(0).normal(size=(4, 8))
        out = attn.forward(x)
        assert out.shape == (4, 8)

    def test_attention_weights_sum_to_one(self):
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=3)
        x = np.random.default_rng(1).normal(size=(3, 4))
        attn.forward(x)
        weights = attn._attn_weights
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_attention_weights_non_negative(self):
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=3)
        x = np.random.default_rng(2).normal(size=(3, 4))
        attn.forward(x)
        assert np.all(attn._attn_weights >= 0)

    def test_backward_shape(self):
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=3)
        x = np.random.default_rng(3).normal(size=(3, 4))
        attn.forward(x)
        grad = attn.backward(np.ones((3, 4)))
        assert grad.shape == (3, 4)

    def test_parameters(self):
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=4)
        params = attn.parameters()
        assert len(params) == 4

    def test_quantum_score_is_bounded(self):
        """Quantum attention scores (Z expectation) should be in [-1, 1]."""
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=4)
        q = np.random.default_rng(4).normal(size=(4,))
        k = np.random.default_rng(5).normal(size=(4,))
        score = attn._quantum_attention_score(q, k)
        assert -1.0 <= score <= 1.0

    def test_num_qubits(self):
        attn = QuantumSelfAttention(feature_dim=8, sequence_length=8)
        assert attn.num_qubits == 3

    def test_num_qubits_min(self):
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=2)
        assert attn.num_qubits == 2


class TestAttentionGradients:
    """Validate that backward() produces correct gradients via finite-difference."""

    def _make_attn_and_input(self, seed=42):
        """Small attention module for gradient tests."""
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=2)
        x = np.random.default_rng(seed).normal(size=(2, 4))
        return attn, x

    def test_grad_W_q_nonzero(self):
        attn, x = self._make_attn_and_input()
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_q) > 1e-8)

    def test_grad_W_k_nonzero(self):
        attn, x = self._make_attn_and_input()
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_k) > 1e-8)

    def test_quantum_param_gradients_nonzero(self):
        attn, x = self._make_attn_and_input()
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn._grad_array) > 1e-8)

    def _scalar_loss(self, attn, x):
        """Sum of all forward outputs — a simple scalar loss."""
        return np.sum(attn.forward(x))

    def test_backward_numerical_gradient_W_q(self):
        """Finite-difference validation of grad_W_q."""
        attn, x = self._make_attn_and_input(seed=10)
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        analytical = attn.grad_W_q.copy()

        eps = 1e-5
        for idx in [(0, 0), (1, 2), (3, 1)]:
            orig = attn.W_q[idx]
            attn.W_q[idx] = orig + eps
            loss_plus = self._scalar_loss(attn, x)
            attn.W_q[idx] = orig - eps
            loss_minus = self._scalar_loss(attn, x)
            attn.W_q[idx] = orig

            numerical = (loss_plus - loss_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical[idx], numerical, atol=1e-3,
                err_msg=f"W_q[{idx}]: analytical={analytical[idx]:.6f}, numerical={numerical:.6f}",
            )

    def test_backward_numerical_gradient_W_k(self):
        """Finite-difference validation of grad_W_k."""
        attn, x = self._make_attn_and_input(seed=11)
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        analytical = attn.grad_W_k.copy()

        eps = 1e-5
        for idx in [(0, 0), (2, 3)]:
            orig = attn.W_k[idx]
            attn.W_k[idx] = orig + eps
            loss_plus = self._scalar_loss(attn, x)
            attn.W_k[idx] = orig - eps
            loss_minus = self._scalar_loss(attn, x)
            attn.W_k[idx] = orig

            numerical = (loss_plus - loss_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical[idx], numerical, atol=1e-3,
                err_msg=f"W_k[{idx}]: analytical={analytical[idx]:.6f}, numerical={numerical:.6f}",
            )

    def test_backward_numerical_gradient_quantum_params(self):
        """Finite-difference validation of quantum parameter gradients."""
        attn, x = self._make_attn_and_input(seed=12)
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        analytical = attn._grad_array.copy()

        eps = 1e-5
        for idx in [0, 1, 2]:
            orig = attn._param_array[idx]
            attn._param_array[idx] = orig + eps
            attn.sync_from_array()
            loss_plus = self._scalar_loss(attn, x)

            attn._param_array[idx] = orig - eps
            attn.sync_from_array()
            loss_minus = self._scalar_loss(attn, x)

            attn._param_array[idx] = orig
            attn.sync_from_array()

            numerical = (loss_plus - loss_minus) / (2 * eps)
            np.testing.assert_allclose(
                analytical[idx], numerical, atol=1e-3,
                err_msg=f"param[{idx}]: analytical={analytical[idx]:.6f}, numerical={numerical:.6f}",
            )

    def test_sync_from_array(self):
        """Verify sync_from_array propagates optimizer updates to param_values."""
        attn, _ = self._make_attn_and_input()
        original_vals = [attn._param_values[p] for p in attn._quantum_params]

        attn._param_array[:] = 0.0
        attn.sync_from_array()

        for p in attn._quantum_params:
            assert attn._param_values[p] == 0.0

        for i, p in enumerate(attn._quantum_params):
            attn._param_array[i] = original_vals[i]
        attn.sync_from_array()

    def test_backward_input_gradient_nonzero(self):
        """grad_input should be non-zero (all three paths contribute)."""
        attn, x = self._make_attn_and_input()
        attn.forward(x)
        grad_input = attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(grad_input) > 1e-8)

    def test_interaction_vector_truncate(self):
        """Truncate mode: interaction is first num_qubits elements."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="truncate",
        )
        q = np.array([1.0, 2.0, 3.0, 4.0])
        k = np.array([0.5, 0.5, 0.5, 0.5])
        interaction = attn._interaction_vector(q, k)
        np.testing.assert_allclose(interaction, [0.5, 1.0])

    def test_interaction_vector_reuploading(self):
        """Reuploading mode: interaction covers all feature dims (padded)."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="reuploading",
        )
        q = np.array([1.0, 2.0, 3.0, 4.0])
        k = np.array([0.5, 0.5, 0.5, 0.5])
        interaction = attn._interaction_vector(q, k)
        np.testing.assert_allclose(interaction, [0.5, 1.0, 1.5, 2.0])

    def test_score_circuit_has_all_parameters(self):
        """The cached score circuit should contain encoding + variational params."""
        attn = QuantumSelfAttention(feature_dim=4, sequence_length=2)
        circuit_params = set(attn._score_circuit.parameters)
        expected = set(attn._encoding_params) | set(attn._quantum_params)
        assert circuit_params == expected


class TestEncodingModes:
    """Test different encoding modes for quantum attention."""

    def test_truncate_backward_compatible(self):
        """Truncate mode should behave like the original single-layer encoding."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="truncate",
        )
        assert attn.num_reuploading_layers == 1
        assert attn._num_encoding_dims == attn.num_qubits

    def test_reuploading_parameter_count(self):
        """Reuploading should have more quantum params than truncate."""
        attn_t = QuantumSelfAttention(
            feature_dim=8, sequence_length=4, encoding_mode="truncate",
        )
        attn_r = QuantumSelfAttention(
            feature_dim=8, sequence_length=4, encoding_mode="reuploading",
        )
        assert len(attn_r._quantum_params) > len(attn_t._quantum_params)
        assert attn_r.num_reuploading_layers == 4

    def test_reuploading_max_layers_cap(self):
        attn = QuantumSelfAttention(
            feature_dim=64, sequence_length=4, encoding_mode="reuploading",
            max_reuploading_layers=3,
        )
        assert attn.num_reuploading_layers == 3

    def test_reuploading_score_bounded(self):
        attn = QuantumSelfAttention(
            feature_dim=8, sequence_length=4, encoding_mode="reuploading",
        )
        q = np.random.default_rng(0).normal(size=(8,))
        k = np.random.default_rng(1).normal(size=(8,))
        score = attn._quantum_attention_score(q, k)
        assert -1.0 <= score <= 1.0

    def test_reuploading_forward_shape(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="reuploading",
        )
        x = np.random.default_rng(0).normal(size=(2, 4))
        out = attn.forward(x)
        assert out.shape == (2, 4)

    def test_projection_mode_parameters(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="projection",
        )
        params = attn.parameters()
        assert len(params) == 5
        assert attn.W_compress.shape == (attn.num_qubits, 4)

    def test_projection_forward_shape(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="projection",
        )
        x = np.random.default_rng(0).normal(size=(2, 4))
        out = attn.forward(x)
        assert out.shape == (2, 4)

    def test_projection_score_bounded(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="projection",
        )
        q = np.random.default_rng(0).normal(size=(4,))
        k = np.random.default_rng(1).normal(size=(4,))
        score = attn._quantum_attention_score(q, k)
        assert -1.0 <= score <= 1.0

    def test_invalid_encoding_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding_mode"):
            QuantumSelfAttention(feature_dim=4, sequence_length=2, encoding_mode="bad")

    def test_reuploading_backward_W_q_nonzero(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="reuploading",
        )
        x = np.random.default_rng(42).normal(size=(2, 4))
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_q) > 1e-8)

    def test_projection_backward_W_compress_nonzero(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="projection",
        )
        x = np.random.default_rng(42).normal(size=(2, 4))
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_compress) > 1e-8)

    def test_identity_block_init_strategy(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, init_strategy="identity_block",
        )
        values = np.array([attn._param_values[p] for p in attn._quantum_params])
        assert np.all(np.abs(values) < 0.1)

    def test_small_random_init_strategy(self):
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, init_strategy="small_random",
        )
        values = np.array([attn._param_values[p] for p in attn._quantum_params])
        assert np.max(np.abs(values)) < np.pi

    def test_invalid_init_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown init strategy"):
            QuantumSelfAttention(
                feature_dim=4, sequence_length=2, init_strategy="bogus",
            )

    def test_truncate_padding_short_features(self):
        """Truncate mode with features shorter than num_qubits zero-pads."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="truncate",
        )
        q = np.array([1.0])
        k = np.array([0.5])
        interaction = attn._interaction_vector(q, k)
        assert len(interaction) == 2
        np.testing.assert_allclose(interaction, [0.5, 0.0])

    def test_reuploading_backward_gradient_flow(self):
        """Reuploading mode backward should produce non-zero Q/K grads."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2,
            encoding_mode="reuploading", max_reuploading_layers=2,
        )
        x = np.random.default_rng(99).normal(size=(2, 4))
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_q) > 1e-8)
        assert np.any(np.abs(attn._grad_array) > 1e-8)

    def test_reuploading_padding_short_features(self):
        """When feature_dim < num_reuploading_layers * num_qubits, zero-pad."""
        attn = QuantumSelfAttention(
            feature_dim=3, sequence_length=2, encoding_mode="reuploading",
        )
        q = np.array([1.0, 2.0, 3.0])
        k = np.array([0.5, 0.5, 0.5])
        interaction = attn._interaction_vector(q, k)
        assert len(interaction) == attn._num_encoding_dims
        assert interaction[-1] == 0.0

    def test_truncate_backward_full(self):
        """Truncate mode backward exercises the truncate branch in backward."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="truncate",
        )
        x = np.random.default_rng(50).normal(size=(2, 4))
        attn.forward(x)
        attn.backward(np.ones((2, 4)))
        assert np.any(np.abs(attn.grad_W_q) > 1e-8)

    def test_gradient_skip_path(self):
        """Exercise the gradient skip (grad_scores near zero)."""
        attn = QuantumSelfAttention(
            feature_dim=4, sequence_length=2, encoding_mode="truncate",
        )
        x = np.random.default_rng(60).normal(size=(2, 4))
        attn.forward(x)
        grad = np.zeros((2, 4))
        grad[0, 0] = 1e-15
        grad_input = attn.backward(grad)
        assert grad_input.shape == (2, 4)
