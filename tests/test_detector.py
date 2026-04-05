"""Tests for pipeline/detector.py — hybrid quantum detector pipeline."""

import numpy as np

from pipeline.detector import HybridDetector


class TestHybridDetector:
    def test_forward_shapes(self):
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=3)
        x = np.random.default_rng(0).normal(size=(1, 3, 16, 16))
        cls_scores, bbox_deltas, quantum_scores = det.forward(x)
        assert cls_scores.shape[0] == 1
        assert bbox_deltas.shape[0] == 1
        assert quantum_scores.shape[-1] == 2

    def test_forward_unbatched(self):
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(1).normal(size=(1, 16, 16))
        cls_scores, _, _ = det.forward(x)
        assert cls_scores.ndim == 3

    def test_detect_with_anchors(self):
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(2).normal(size=(1, 16, 16))
        anchors = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [10, 10, 20, 20],
        ], dtype=float)
        boxes, scores = det.detect(x, anchors, score_threshold=0.0)
        assert boxes.ndim == 2
        assert scores.ndim == 1
        if len(boxes) > 0:
            assert boxes.shape[1] == 4

    def test_detect_high_threshold(self):
        """Very high threshold should filter all detections."""
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(3).normal(size=(1, 16, 16))
        anchors = np.array([[0, 0, 10, 10]], dtype=float)
        boxes, scores = det.detect(x, anchors, score_threshold=0.999)
        assert boxes.ndim == 2
        assert scores.ndim == 1

    def test_detect_empty_after_filter(self):
        """All scores below threshold -> empty result."""
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(4).normal(size=(1, 16, 16))
        anchors = np.array([[0, 0, 10, 10]], dtype=float)
        boxes, scores = det.detect(x, anchors, score_threshold=1.0)
        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)

    def test_forward_batched(self):
        """Batched forward path (not unbatched)."""
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(5).normal(size=(2, 1, 16, 16))
        cls_scores, bbox_deltas, _ = det.forward(x)
        assert cls_scores.ndim == 4
        assert bbox_deltas.ndim == 4

    def test_detect_batched_cls_scores(self):
        """detect() with batched input that produces 4D cls_scores."""
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        x = np.random.default_rng(6).normal(size=(1, 1, 16, 16))
        anchors = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=float)
        boxes, _ = det.detect(x, anchors, score_threshold=0.0, iou_threshold=1.0)
        assert boxes.ndim == 2

    def test_parameters(self):
        det = HybridDetector(num_classes=2, num_anchors=2, num_qubits=2,
                             input_channels=1)
        params = det.parameters()
        assert len(params) > 0


class TestKernelRerankerDetector:
    """Test HybridDetector with quantum kernel re-ranking enabled."""

    def test_forward_shapes_kernel_mode(self):
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        x = np.random.default_rng(0).normal(size=(1, 16, 16))
        cls_scores, _, _ = det.forward(x)
        assert cls_scores.ndim == 3

    def test_detect_kernel_mode(self):
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        x = np.random.default_rng(1).normal(size=(1, 16, 16))
        anchors = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
        ], dtype=float)
        boxes, scores = det.detect(x, anchors, score_threshold=0.0)
        assert boxes.ndim == 2
        assert scores.ndim == 1

    def test_parameters_kernel_mode(self):
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        params = det.parameters()
        assert len(params) > 0

    def test_detect_empty_kernel_mode(self):
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        x = np.random.default_rng(2).normal(size=(1, 16, 16))
        anchors = np.array([[0, 0, 10, 10]], dtype=float)
        boxes, _ = det.detect(x, anchors, score_threshold=1.0)
        assert boxes.shape == (0, 4)

    def test_detect_kernel_batched_features(self):
        """Kernel reranker with batched input (4D features)."""
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        x = np.random.default_rng(7).normal(size=(1, 1, 16, 16))
        anchors = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [10, 10, 20, 20],
        ], dtype=float)
        boxes, _ = det.detect(x, anchors, score_threshold=0.0)
        assert boxes.ndim == 2

    def test_detect_kernel_unbatched_features(self):
        """Kernel reranker with unbatched input (3D features/backbone)."""
        det = HybridDetector(
            num_classes=2, num_anchors=2, num_qubits=2,
            input_channels=1, use_kernel_reranker=True,
        )
        x = np.random.default_rng(8).normal(size=(1, 16, 16))
        anchors = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
        ], dtype=float)
        boxes, _ = det.detect(x, anchors, score_threshold=0.0)
        assert boxes.ndim == 2
