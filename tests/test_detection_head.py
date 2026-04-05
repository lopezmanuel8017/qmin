"""Tests for classical/detection_head.py — RPN, BBox, IoU, NMS, deltas."""

import numpy as np

from classical.detection_head import (
    BBoxRegressor, RPNHead, apply_deltas, compute_iou, nms,
)


class TestRPNHead:
    def test_forward_shapes(self):
        rpn = RPNHead(in_channels=64, mid_channels=128, num_anchors=3)
        x = np.random.default_rng(0).normal(size=(2, 64, 8, 8))
        cls_scores, bbox_deltas = rpn.forward(x)
        assert cls_scores.shape == (2, 6, 8, 8)
        assert bbox_deltas.shape == (2, 12, 8, 8)

    def test_forward_unbatched(self):
        rpn = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)
        x = np.random.default_rng(1).normal(size=(32, 6, 6))
        cls_scores, bbox_deltas = rpn.forward(x)
        assert cls_scores.shape == (4, 6, 6)
        assert bbox_deltas.shape == (8, 6, 6)

    def test_backward_runs(self):
        rpn = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)
        x = np.random.default_rng(2).normal(size=(1, 32, 4, 4))
        cls_scores, bbox_deltas = rpn.forward(x)
        grad_cls = np.ones_like(cls_scores)
        grad_reg = np.ones_like(bbox_deltas)
        grad_input = rpn.backward(grad_cls, grad_reg)
        assert grad_input.shape == (1, 32, 4, 4)

    def test_backward_unbatched(self):
        rpn = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)
        x = np.random.default_rng(3).normal(size=(32, 4, 4))
        cls_scores, bbox_deltas = rpn.forward(x)
        grad_input = rpn.backward(np.ones_like(cls_scores), np.ones_like(bbox_deltas))
        assert grad_input.shape == (32, 4, 4)

    def test_parameters(self):
        rpn = RPNHead(in_channels=32, mid_channels=64, num_anchors=2)
        params = rpn.parameters()
        assert len(params) > 0


class TestBBoxRegressor:
    def test_forward_shape(self):
        reg = BBoxRegressor(feature_dim=16, hidden_dim=8)
        x = np.random.default_rng(4).normal(size=(5, 16))
        out = reg.forward(x)
        assert out.shape == (5, 4)

    def test_backward_shape(self):
        reg = BBoxRegressor(feature_dim=8, hidden_dim=4)
        x = np.random.default_rng(5).normal(size=(3, 8))
        reg.forward(x)
        grad = reg.backward(np.ones((3, 4)))
        assert grad.shape == (3, 8)

    def test_parameters(self):
        reg = BBoxRegressor(feature_dim=8)
        params = reg.parameters()
        assert len(params) == 4


class TestComputeIoU:
    def test_identical_boxes(self):
        box = np.array([0, 0, 10, 10])
        assert np.isclose(compute_iou(box, box), 1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 5, 5])
        b = np.array([10, 10, 20, 20])
        assert np.isclose(compute_iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        assert np.isclose(compute_iou(a, b), 25 / 175)

    def test_containment(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([2, 2, 8, 8])
        assert np.isclose(compute_iou(a, b), 36 / 100)

    def test_zero_area_box(self):
        a = np.array([5, 5, 5, 5])
        b = np.array([0, 0, 10, 10])
        assert np.isclose(compute_iou(a, b), 0.0)

    def test_both_zero_area(self):
        """Two zero-area boxes: union=0, should return 0.0."""
        a = np.array([5, 5, 5, 5])
        b = np.array([5, 5, 5, 5])
        assert np.isclose(compute_iou(a, b), 0.0)


class TestNMS:
    def test_empty(self):
        result = nms(np.zeros((0, 4)), np.zeros(0))
        assert len(result) == 0

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        keep = nms(boxes, scores)
        np.testing.assert_array_equal(keep, [0])

    def test_no_overlap(self):
        boxes = np.array([[0, 0, 5, 5], [10, 10, 20, 20]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_high_overlap_suppression(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
        ])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_three_boxes_partial_suppression(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2
        assert 0 in keep
        assert 2 in keep


class TestApplyDeltas:
    def test_zero_deltas(self):
        """Zero deltas should return the anchor itself."""
        anchors = np.array([[10, 10, 30, 30]], dtype=float)
        deltas = np.array([[0, 0, 0, 0]], dtype=float)
        pred = apply_deltas(anchors, deltas)
        np.testing.assert_allclose(pred, anchors, atol=1e-10)

    def test_shift_center(self):
        """Positive dx should shift box right."""
        anchors = np.array([[0, 0, 10, 10]], dtype=float)
        deltas = np.array([[0.5, 0, 0, 0]], dtype=float)
        pred = apply_deltas(anchors, deltas)
        expected_cx = 5.0 + 0.5 * 10
        assert np.isclose(pred[0, 0] + pred[0, 2], 2 * expected_cx)

    def test_scale_width(self):
        """Positive dw should increase width."""
        anchors = np.array([[0, 0, 10, 10]], dtype=float)
        deltas = np.array([[0, 0, np.log(2), 0]], dtype=float)
        pred = apply_deltas(anchors, deltas)
        pred_w = pred[0, 2] - pred[0, 0]
        assert np.isclose(pred_w, 20.0)

    def test_batch(self):
        anchors = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=float)
        deltas = np.zeros((2, 4))
        pred = apply_deltas(anchors, deltas)
        np.testing.assert_allclose(pred, anchors, atol=1e-10)
