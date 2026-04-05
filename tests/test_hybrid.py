"""Tests for pipeline/hybrid_layer.py — quantum-classical bridge."""

import numpy as np

from pipeline.hybrid_layer import HybridQuantumClassicalLayer
from quantum.ansatz import CNOTLadder


class TestHybridLayerForward:
    def test_output_shape_batched(self):
        ansatz = CNOTLadder(4, 1)
        layer = HybridQuantumClassicalLayer(4, ansatz)
        x = np.random.default_rng(0).uniform(0, np.pi, (3, 4))
        out = layer.forward(x)
        assert out.shape == (3, 4)

    def test_output_shape_unbatched(self):
        ansatz = CNOTLadder(4, 1)
        layer = HybridQuantumClassicalLayer(4, ansatz)
        x = np.random.default_rng(1).uniform(0, np.pi, (4,))
        out = layer.forward(x)
        assert out.shape == (4,)

    def test_output_range(self):
        """P(|1>) should be in [0, 1]."""
        ansatz = CNOTLadder(3, 1)
        layer = HybridQuantumClassicalLayer(3, ansatz)
        x = np.random.default_rng(2).uniform(0, 2 * np.pi, (2, 3))
        out = layer.forward(x)
        assert np.all(out >= -1e-10)
        assert np.all(out <= 1 + 1e-10)

    def test_zero_input_near_zero_output(self):
        """With zero features (Ry(0) = I), output depends only on ansatz params."""
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        x = np.zeros((2,))
        out = layer.forward(x)
        assert out.shape == (2,)


class TestHybridLayerBackward:
    def test_gradient_shapes(self):
        ansatz = CNOTLadder(3, 1)
        layer = HybridQuantumClassicalLayer(3, ansatz)
        x = np.random.default_rng(3).uniform(0, np.pi, (2, 3))
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (2, 3)

    def test_gradient_shapes_unbatched(self):
        ansatz = CNOTLadder(3, 1)
        layer = HybridQuantumClassicalLayer(3, ansatz)
        x = np.random.default_rng(4).uniform(0, np.pi, (3,))
        layer.forward(x)
        grad_out = np.ones(3)
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == (3,)

    def test_param_gradients_nonzero(self):
        """Parameter-shift should produce non-zero gradients."""
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        x = np.array([1.0, 0.5])
        layer.forward(x)
        grad_out = np.array([1.0, 1.0])
        layer.backward(grad_out)

        grads = layer.grad_params
        assert any(abs(g) > 1e-10 for g in grads.values()), \
            "Expected non-zero gradients"

    def test_input_gradient_with_compute(self):
        """When compute_input_grad=True, input gradient should be non-zero."""
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz, compute_input_grad=True)
        x = np.array([1.0, 0.5])
        layer.forward(x)
        grad_in = layer.backward(np.array([1.0, 1.0]))
        assert np.any(np.abs(grad_in) > 1e-8)


class TestHybridLayerParameters:
    def test_parameters_method(self):
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        params = layer.parameters()
        assert len(params) == 1
        param_arr, grad_arr = params[0]
        assert param_arr.shape == grad_arr.shape
        assert len(param_arr) == ansatz.num_parameters

    def test_trainable_params(self):
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        tp = layer.trainable_params
        assert len(tp) == ansatz.num_parameters

    def test_sync_from_array(self):
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        param_arr, _ = layer.parameters()[0]
        param_arr[:] = 0.0
        layer.sync_from_array()
        tp = layer.trainable_params
        assert all(v == 0.0 for v in tp.values())

    def test_sync_grads_to_array(self):
        ansatz = CNOTLadder(2, 1)
        layer = HybridQuantumClassicalLayer(2, ansatz)
        x = np.array([1.0, 0.5])
        layer.forward(x)
        layer.backward(np.array([1.0, 1.0]))
        layer.sync_grads_to_array()
        _, grad_arr = layer.parameters()[0]
        assert np.any(np.abs(grad_arr) > 1e-10)
