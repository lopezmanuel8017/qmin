"""MNIST data loader — parses raw IDX binary format.

No external dependencies (no torchvision, sklearn, etc).
Downloads from the original Yann LeCun source if needed.

IDX file format:
  - 4 bytes: magic number
  - 4 bytes: number of items
  - 4 bytes: number of rows (images only)
  - 4 bytes: number of cols (images only)
  - rest: unsigned bytes (pixel values or labels)
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

FILENAMES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't exist."""
    if filepath.exists():
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)


def _parse_idx_images(filepath: Path) -> np.ndarray:
    """Parse IDX3 image file -> (N, 1, 28, 28) float32 array, normalized to [0, 1]."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0
    return images


def _parse_idx_labels(filepath: Path) -> np.ndarray:
    """Parse IDX1 label file -> (N,) int array."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)


def load_mnist(
    data_dir: str = "data/raw/mnist",
    subset: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.

    Returns: (x_train, y_train, x_test, y_test)
      x_train: (60000, 1, 28, 28) float32 normalized [0, 1]
      y_train: (60000,) int
      x_test: (10000, 1, 28, 28) float32 normalized [0, 1]
      y_test: (10000,) int

    If subset is specified, returns only the first `subset` samples.
    """
    data_path = Path(data_dir)

    for key, filename in FILENAMES.items():
        filepath = data_path / filename
        if not filepath.exists():
            _download_file(MNIST_URLS[key], filepath)  # pragma: no cover

    x_train = _parse_idx_images(data_path / FILENAMES["train_images"])
    y_train = _parse_idx_labels(data_path / FILENAMES["train_labels"])
    x_test = _parse_idx_images(data_path / FILENAMES["test_images"])
    y_test = _parse_idx_labels(data_path / FILENAMES["test_labels"])

    if subset is not None:
        x_train = x_train[:subset]
        y_train = y_train[:subset]
        x_test = x_test[:min(subset, len(x_test))]
        y_test = y_test[:min(subset, len(y_test))]

    return x_train, y_train, x_test, y_test
