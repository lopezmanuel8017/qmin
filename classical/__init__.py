"""classical — Manual neural network layers, losses, and optimizers."""

from .detection_head import BBoxRegressor, RPNHead, apply_deltas, compute_iou, nms
from .layers import BatchNorm1d, Conv2d, Flatten, Linear, MaxPool2d, ReLU
from .loss import CrossEntropyLoss, KLDivergenceLoss, SmoothL1Loss
from .optim import AdamW, SGD

__all__ = [
    "BatchNorm1d", "BBoxRegressor", "Conv2d", "Flatten", "Linear", "MaxPool2d",
    "ReLU", "RPNHead",
    "CrossEntropyLoss", "KLDivergenceLoss", "SmoothL1Loss",
    "AdamW", "SGD",
    "apply_deltas", "compute_iou", "nms",
]
