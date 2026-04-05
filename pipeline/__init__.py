"""pipeline — Hybrid quantum-classical model integration."""

from .classifier import HybridClassifier
from .detector import HybridDetector
from .distillation import DistillationTrainer
from .hybrid_layer import HybridQuantumClassicalLayer
from .quantum_reranker import QuantumKernelReranker
from .trainer import Trainer

__all__ = [
    "HybridClassifier", "HybridDetector", "DistillationTrainer",
    "HybridQuantumClassicalLayer", "QuantumKernelReranker", "Trainer",
]
