"""Training loop orchestration for hybrid quantum-classical models.

Handles the full forward-backward-update cycle, epoch management,
and basic logging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from classical.optim import AdamW
from pipeline.classifier import HybridClassifier


class Trainer:
    """Training orchestrator for HybridClassifier."""

    def __init__(
        self,
        model: HybridClassifier,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer: Optional[AdamW] = None
        self.lr = lr
        self.weight_decay = weight_decay
        self._initialized = False

    def _init_optimizer(self) -> None:
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self._initialized = True

    def train_step(
        self, x: np.ndarray, targets: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """One training step: forward, loss, backward, update.

        Returns: (loss, logits)
        """
        if not self._initialized:
            _ = self.model.forward(x)
            self._init_optimizer()

        logits = self.model.forward(x)
        loss = self.model.compute_loss(logits, targets)
        self.model.backward()

        self.optimizer.step()

        self.model.quantum_layer.sync_from_array()

        self.optimizer.zero_grad()
        return loss, logits

    def train_epoch(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 1,
    ) -> float:
        """Train for one epoch over the dataset.

        Returns: average loss over the epoch
        """
        n_samples = x_train.shape[0]
        indices = np.random.default_rng().permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            loss, _ = self.train_step(x_batch, y_batch)
            total_loss += loss
            n_batches += 1

        return total_loss / n_batches

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float, float]:
        """Evaluate model on test data.

        Returns: (loss, accuracy)
        """
        logits = self.model.forward(x_test)
        loss = self.model.compute_loss(logits, y_test)

        if logits.ndim == 1:
            predictions = np.array([np.argmax(logits)])
        else:
            predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == y_test))

        return loss, accuracy
