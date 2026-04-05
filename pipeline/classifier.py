"""Hybrid quantum-classical classifier pipeline.

Architecture:
  Input (28x28) -> Conv2d(1,16,3) -> ReLU -> MaxPool(2)
               -> Conv2d(16,32,3) -> ReLU -> MaxPool(2)
               -> Flatten -> Linear(dim, num_qubits)
               -> [Quantum Layer]
               -> Linear(num_qubits, num_classes)

Supports:
  - Full end-to-end training (backbone + quantum + head)
  - Frozen backbone (train quantum + head only)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from classical.layers import Conv2d, Flatten, Linear, MaxPool2d, ReLU
from classical.loss import CrossEntropyLoss
from pipeline.hybrid_layer import HybridQuantumClassicalLayer
from quantum.ansatz import CNOTLadder
from quantum.encoding import AngleEncoder


class HybridClassifier:
    """Hybrid quantum-classical image classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        num_qubits: int = 8,
        num_layers: int = 2,
        input_channels: int = 1,
        freeze_backbone: bool = True,
        init_strategy: str = "uniform",
        init_epsilon: float = 0.01,
    ) -> None:
        self.num_classes = num_classes
        self.num_qubits = num_qubits
        self.freeze_backbone = freeze_backbone

        self.conv1 = Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2)
        self.flatten = Flatten()

        self._proj: Optional[Linear] = None
        self._proj_initialized = False

        ansatz = CNOTLadder(num_qubits, num_layers)
        encoder = AngleEncoder(num_qubits)
        self.quantum_layer = HybridQuantumClassicalLayer(
            num_qubits, ansatz, encoder,
            compute_input_grad=not freeze_backbone,
            init_strategy=init_strategy,
            init_epsilon=init_epsilon,
        )

        self.head = Linear(num_qubits, num_classes)
        self.loss_fn = CrossEntropyLoss()

        self._backbone_layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
        ]

    def _init_proj(self, flat_dim: int) -> None:
        self._proj = Linear(flat_dim, self.num_qubits)
        self._proj_initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the full pipeline.

        x: (batch_size, C, H, W) or (C, H, W)
        Returns: logits (batch_size, num_classes) or (num_classes,)
        """
        was_unbatched = (x.ndim == 3)
        if was_unbatched:
            x = x[np.newaxis]

        out = x
        for layer in self._backbone_layers:
            out = layer.forward(out)

        if not self._proj_initialized:
            self._init_proj(out.shape[-1])

        out = self._proj.forward(out)

        out = self.quantum_layer.forward(out)

        out = self.head.forward(out)

        return out[0] if was_unbatched else out

    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.loss_fn.forward(logits, targets)

    def backward(self) -> None:
        """Backward pass through the full pipeline."""
        grad = self.loss_fn.backward()
        grad = self.head.backward(grad)
        grad = self.quantum_layer.backward(grad)
        self.quantum_layer.sync_grads_to_array()
        grad = self._proj.backward(grad)

        if not self.freeze_backbone:
            for layer in reversed(self._backbone_layers):
                grad = layer.backward(grad)

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """All trainable parameters."""
        params = []
        if not self.freeze_backbone:
            for layer in self._backbone_layers:
                params.extend(layer.parameters())
        if self._proj is not None:
            params.extend(self._proj.parameters())
        params.extend(self.quantum_layer.parameters())
        params.extend(self.head.parameters())
        return params
