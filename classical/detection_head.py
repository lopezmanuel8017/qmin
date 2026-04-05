"""Detection head components: RPN, bounding box regression, and NMS.

Region Proposal Network (RPN):
  Generates objectness scores and bounding box deltas for anchor boxes.
  Input: feature map (N, C, H, W)
  Output: objectness (N, num_anchors*2, H, W), bbox deltas (N, num_anchors*4, H, W)

BBoxRegressor:
  Refines bounding box predictions via fully connected layers.
  Input: (N, feature_dim)
  Output: (N, 4) — [dx, dy, dw, dh] deltas

NMS (Non-Maximum Suppression):
  Greedy algorithm to remove overlapping detections.
  Reference: Girshick (2015), "Fast R-CNN"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from classical.layers import Conv2d, Linear, ReLU


class RPNHead:
    """Region Proposal Network head.

    Uses a 3x3 conv for shared features, then two 1x1 conv branches:
    - Classification: objectness score (object vs background)
    - Regression: bounding box deltas (dx, dy, dw, dh)
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_anchors: int = 3,
    ) -> None:
        self.num_anchors = num_anchors

        self.conv = Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = ReLU()

        self.cls_conv = Conv2d(mid_channels, num_anchors * 2, kernel_size=1)

        self.reg_conv = Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

        self._shared_features: Optional[np.ndarray] = None
        self._was_unbatched = False

    def forward(
        self, feature_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            feature_map: (N, C, H, W) or (C, H, W)

        Returns:
            cls_scores: (N, num_anchors*2, H, W) objectness logits
            bbox_deltas: (N, num_anchors*4, H, W) regression deltas
        """
        self._was_unbatched = (feature_map.ndim == 3)
        if self._was_unbatched:
            feature_map = feature_map[np.newaxis]

        shared = self.relu.forward(self.conv.forward(feature_map))
        self._shared_features = shared

        cls_scores = self.cls_conv.forward(shared)
        bbox_deltas = self.reg_conv.forward(shared)

        if self._was_unbatched:
            return cls_scores[0], bbox_deltas[0]
        return cls_scores, bbox_deltas

    def backward(
        self, grad_cls: np.ndarray, grad_reg: np.ndarray
    ) -> np.ndarray:
        """Backward pass through both branches."""
        if self._was_unbatched:
            grad_cls = grad_cls[np.newaxis]
            grad_reg = grad_reg[np.newaxis]

        grad_shared_cls = self.cls_conv.backward(grad_cls)
        grad_shared_reg = self.reg_conv.backward(grad_reg)

        grad_shared = grad_shared_cls + grad_shared_reg

        grad_shared = self.relu.backward(grad_shared)
        grad_input = self.conv.backward(grad_shared)

        return grad_input[0] if self._was_unbatched else grad_input

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        params.extend(self.conv.parameters())
        params.extend(self.cls_conv.parameters())
        params.extend(self.reg_conv.parameters())
        return params


class BBoxRegressor:
    """Bounding box regression head.

    Two FC layers: feature_dim -> hidden -> 4 (dx, dy, dw, dh).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        self.fc1 = Linear(feature_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, 4)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, feature_dim) -> (N, 4)"""
        return self.fc2.forward(self.relu.forward(self.fc1.forward(x)))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = self.fc2.backward(grad_output)
        grad = self.relu.backward(grad)
        return self.fc1.backward(grad)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.fc1.parameters() + self.fc2.parameters()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes.

    Boxes are [x1, y1, x2, y2] format (top-left, bottom-right corners).
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Greedy Non-Maximum Suppression.

    Args:
        boxes: (N, 4) in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        iou_threshold: suppress boxes with IoU > threshold

    Returns:
        keep: array of indices to keep

    Reference: Girshick (2015), "Fast R-CNN"
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=int)


def apply_deltas(
    anchors: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    """Apply bounding box deltas to anchors.

    Anchors and output are in [x1, y1, x2, y2] format.
    Deltas are [dx, dy, dw, dh] in the parameterized form:
      dx = (x_pred - x_anchor) / w_anchor
      dy = (y_pred - y_anchor) / h_anchor
      dw = log(w_pred / w_anchor)
      dh = log(h_pred / h_anchor)

    Reference: Girshick et al. (2014), "Rich feature hierarchies"
    """
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + 0.5 * w
    cy = anchors[:, 1] + 0.5 * h

    pred_cx = deltas[:, 0] * w + cx
    pred_cy = deltas[:, 1] * h + cy
    pred_w = np.exp(deltas[:, 2]) * w
    pred_h = np.exp(deltas[:, 3]) * h

    pred_boxes = np.zeros_like(deltas)
    pred_boxes[:, 0] = pred_cx - 0.5 * pred_w
    pred_boxes[:, 1] = pred_cy - 0.5 * pred_h
    pred_boxes[:, 2] = pred_cx + 0.5 * pred_w
    pred_boxes[:, 3] = pred_cy + 0.5 * pred_h

    return pred_boxes
