"""Hybrid quantum-classical object detector pipeline.

Architecture:
  Input image -> Classical backbone -> Feature map
               -> [Quantum-enhanced RPN or Classical RPN] -> Proposals
               -> BBox Regressor -> Refined detections
               -> NMS -> Final detections

Supports:
  - Classical-only detection (baseline)
  - Quantum-enhanced detection via pass-through scoring (legacy)
  - Quantum kernel re-ranking of proposals (use_kernel_reranker=True)
  - Knowledge distillation from classical to quantum RPN
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from classical.detection_head import BBoxRegressor, RPNHead, apply_deltas, nms
from classical.layers import Conv2d, Linear, MaxPool2d, ReLU
from pipeline.hybrid_layer import HybridQuantumClassicalLayer
from pipeline.quantum_reranker import QuantumKernelReranker
from quantum.ansatz import CNOTLadder
from quantum.encoding import AngleEncoder


class HybridDetector:
    """Hybrid quantum-classical object detector.

    Uses a classical backbone for feature extraction and a quantum-enhanced
    pathway for region proposal scoring.
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_anchors: int = 3,
        num_qubits: int = 4,
        num_quantum_layers: int = 1,
        input_channels: int = 3,
        use_kernel_reranker: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.use_kernel_reranker = use_kernel_reranker

        self.conv1 = Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()

        self.rpn = RPNHead(in_channels=64, mid_channels=128, num_anchors=num_anchors)

        self.num_qubits = num_qubits
        if use_kernel_reranker:
            self.quantum_reranker = QuantumKernelReranker(
                feature_dim=64,
                num_qubits=num_qubits,
            )
            self.quantum_proj = None
            self.quantum_scorer = None
        else:
            ansatz = CNOTLadder(num_qubits, num_quantum_layers)
            encoder = AngleEncoder(num_qubits)
            self.quantum_scorer = HybridQuantumClassicalLayer(
                num_qubits, ansatz, encoder, compute_input_grad=False,
            )
            self.quantum_proj = Linear(num_anchors * 2, num_qubits)
            self.quantum_reranker = None

        self.bbox_regressor = BBoxRegressor(feature_dim=num_anchors * 4, hidden_dim=64)

        self._backbone_layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2,
        ]

    def forward(
        self,
        image: np.ndarray,
        _anchors: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            image: (N, C, H, W) or (C, H, W)
            anchors: (num_total_anchors, 4) anchor boxes in [x1,y1,x2,y2]

        Returns:
            cls_scores: objectness scores per anchor location
            bbox_deltas: bbox regression deltas
            quantum_scores: quantum-enhanced scores (per spatial location)
        """
        was_unbatched = (image.ndim == 3)
        if was_unbatched:
            image = image[np.newaxis]

        features = image
        for layer in self._backbone_layers:
            features = layer.forward(features)

        self._cached_features = features

        cls_scores, bbox_deltas = self.rpn.forward(features)

        if self.use_kernel_reranker:
            quantum_scores = np.zeros(features.shape[0])
        else:
            _ = cls_scores.shape[0]
            H, W = cls_scores.shape[2], cls_scores.shape[3]
            center_h, center_w = H // 2, W // 2
            spatial_features = cls_scores[:, :, center_h, center_w]
            q_input = self.quantum_proj.forward(spatial_features)
            quantum_scores = self.quantum_scorer.forward(q_input)

        if was_unbatched:
            return cls_scores[0], bbox_deltas[0], quantum_scores
        return cls_scores, bbox_deltas, quantum_scores

    def detect(
        self,
        image: np.ndarray,
        anchors: np.ndarray,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full detection pipeline with NMS.

        Returns:
            boxes: (K, 4) detected bounding boxes
            scores: (K,) confidence scores
        """
        cls_scores, bbox_deltas, _ = self.forward(image)

        if cls_scores.ndim == 4:
            cls_scores = cls_scores[0]
        C, H, W = cls_scores.shape
        num_anchors = C // 2

        cls_reshaped = cls_scores.reshape(num_anchors, 2, H, W)
        cls_exp = np.exp(cls_reshaped - cls_reshaped.max(axis=1, keepdims=True))
        cls_probs = cls_exp / cls_exp.sum(axis=1, keepdims=True)
        objectness = cls_probs[:, 1, :, :]

        if bbox_deltas.ndim == 4:
            bbox_deltas = bbox_deltas[0]
        reg_reshaped = bbox_deltas.reshape(num_anchors, 4, H, W)

        all_scores = objectness.reshape(-1)
        all_deltas = reg_reshaped.transpose(0, 2, 3, 1).reshape(-1, 4)

        total = len(all_scores)
        reps = (total + len(anchors) - 1) // len(anchors)
        anchors_tiled = np.tile(anchors, (reps, 1))[:total]

        pred_boxes = apply_deltas(anchors_tiled, all_deltas)

        mask = all_scores > score_threshold
        filtered_boxes = pred_boxes[mask]
        filtered_scores = all_scores[mask]

        if len(filtered_boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        if self.use_kernel_reranker and len(filtered_boxes) > 1:
            feat = self._cached_features[0]
            proposal_feat = feat.mean(axis=(1, 2))
            proposal_features = np.tile(proposal_feat, (len(filtered_boxes), 1))
            box_centers = (filtered_boxes[:, :2] + filtered_boxes[:, 2:]) / 2
            bc_max = box_centers.max()
            box_centers = box_centers / (bc_max + 1e-8)
            diversity = np.zeros((len(filtered_boxes), proposal_features.shape[1]))
            fill_cols = min(box_centers.shape[1], diversity.shape[1])
            diversity[:, :fill_cols] = box_centers[:, :fill_cols]
            proposal_features = proposal_features + 0.1 * diversity

            filtered_scores = self.quantum_reranker.forward(
                proposal_features, filtered_scores,
            )

        keep = nms(filtered_boxes, filtered_scores, iou_threshold)
        return filtered_boxes[keep], filtered_scores[keep]

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        for layer in self._backbone_layers:
            params.extend(layer.parameters())
        params.extend(self.rpn.parameters())
        if self.use_kernel_reranker:
            params.extend(self.quantum_reranker.parameters())
        else:
            params.extend(self.quantum_proj.parameters())
            params.extend(self.quantum_scorer.parameters())
        params.extend(self.bbox_regressor.parameters())
        return params
