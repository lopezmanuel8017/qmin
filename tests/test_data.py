"""Tests for data/mnist.py — IDX format parsing.

Tests create synthetic IDX files to validate parsing logic without
requiring the actual MNIST download.
"""

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from data.mnist import (
    FILENAMES, _download_file, _parse_idx_images, _parse_idx_labels, load_mnist,
)


def _create_fake_idx3(path: Path, images: np.ndarray) -> None:
    """Create a fake IDX3 image file."""
    n, _, h, w = images.shape
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, h, w))
        f.write(images.astype(np.uint8).tobytes())


def _create_fake_idx1(path: Path, labels: np.ndarray) -> None:
    """Create a fake IDX1 label file."""
    n = len(labels)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))
        f.write(labels.astype(np.uint8).tobytes())


class TestIDXParsing:
    def test_parse_images(self, tmp_path):
        """Parse synthetic IDX3 image file."""
        n, c, h, w = 5, 1, 4, 4
        raw = np.random.default_rng(0).integers(0, 256, (n, c, h, w), dtype=np.uint8)
        filepath = tmp_path / "images.gz"
        _create_fake_idx3(filepath, raw)

        images = _parse_idx_images(filepath)
        assert images.shape == (5, 1, 4, 4)
        assert images.dtype == np.float32
        assert images.max() <= 1.0
        assert images.min() >= 0.0
        np.testing.assert_allclose(images, raw / 255.0, atol=1e-6)

    def test_parse_labels(self, tmp_path):
        """Parse synthetic IDX1 label file."""
        labels = np.array([0, 3, 7, 1, 9], dtype=np.uint8)
        filepath = tmp_path / "labels.gz"
        _create_fake_idx1(filepath, labels)

        parsed = _parse_idx_labels(filepath)
        assert parsed.shape == (5,)
        assert parsed.dtype == int
        np.testing.assert_array_equal(parsed, labels)

    def test_28x28_images(self, tmp_path):
        """Standard MNIST dimensions."""
        raw = np.random.default_rng(1).integers(0, 256, (3, 1, 28, 28), dtype=np.uint8)
        filepath = tmp_path / "mnist28.gz"
        _create_fake_idx3(filepath, raw)

        images = _parse_idx_images(filepath)
        assert images.shape == (3, 1, 28, 28)

    def test_wrong_magic_images(self, tmp_path):
        filepath = tmp_path / "bad.gz"
        with gzip.open(filepath, "wb") as f:
            f.write(struct.pack(">IIII", 9999, 1, 2, 2))
            f.write(b"\x00" * 4)
        with pytest.raises(AssertionError, match="Invalid magic"):
            _parse_idx_images(filepath)

    def test_wrong_magic_labels(self, tmp_path):
        filepath = tmp_path / "bad.gz"
        with gzip.open(filepath, "wb") as f:
            f.write(struct.pack(">II", 9999, 1))
            f.write(b"\x00")
        with pytest.raises(AssertionError, match="Invalid magic"):
            _parse_idx_labels(filepath)


class TestDownloadFile:
    def test_skip_if_exists(self, tmp_path):
        """_download_file should skip if file already exists."""
        filepath = tmp_path / "existing.gz"
        filepath.write_text("existing")
        _download_file("https://example.com/fake", filepath)
        assert filepath.read_text() == "existing"

    def test_creates_parent_dir_and_attempts_download(self, tmp_path, monkeypatch):
        """When file doesn't exist, creates parent dir and calls urlretrieve."""
        filepath = tmp_path / "subdir" / "new_file.gz"
        downloaded = []

        def fake_urlretrieve(url, dest):
            downloaded.append((url, str(dest)))
            Path(dest).write_text("fake")

        monkeypatch.setattr("data.mnist.urllib.request.urlretrieve", fake_urlretrieve)
        _download_file("https://example.com/data.gz", filepath)
        assert len(downloaded) == 1
        assert filepath.parent.exists()


class TestLoadMnist:
    def test_load_from_synthetic_files(self, tmp_path):
        """Load MNIST from synthetic IDX files."""
        n_train, n_test = 10, 3
        rng = np.random.default_rng(99)

        train_imgs = rng.integers(0, 256, (n_train, 1, 28, 28), dtype=np.uint8)
        train_lbls = rng.integers(0, 10, n_train, dtype=np.uint8)
        test_imgs = rng.integers(0, 256, (n_test, 1, 28, 28), dtype=np.uint8)
        test_lbls = rng.integers(0, 10, n_test, dtype=np.uint8)

        _create_fake_idx3(tmp_path / FILENAMES["train_images"], train_imgs)
        _create_fake_idx1(tmp_path / FILENAMES["train_labels"], train_lbls)
        _create_fake_idx3(tmp_path / FILENAMES["test_images"], test_imgs)
        _create_fake_idx1(tmp_path / FILENAMES["test_labels"], test_lbls)

        x_train, y_train, x_test, y_test = load_mnist(str(tmp_path))
        assert x_train.shape == (10, 1, 28, 28)
        assert y_train.shape == (10,)
        assert x_test.shape == (3, 1, 28, 28)
        assert y_test.shape == (3,)

    def test_load_with_subset(self, tmp_path):
        """Load with subset parameter."""
        n_train, n_test = 10, 5
        rng = np.random.default_rng(100)

        train_imgs = rng.integers(0, 256, (n_train, 1, 28, 28), dtype=np.uint8)
        train_lbls = rng.integers(0, 10, n_train, dtype=np.uint8)
        test_imgs = rng.integers(0, 256, (n_test, 1, 28, 28), dtype=np.uint8)
        test_lbls = rng.integers(0, 10, n_test, dtype=np.uint8)

        _create_fake_idx3(tmp_path / FILENAMES["train_images"], train_imgs)
        _create_fake_idx1(tmp_path / FILENAMES["train_labels"], train_lbls)
        _create_fake_idx3(tmp_path / FILENAMES["test_images"], test_imgs)
        _create_fake_idx1(tmp_path / FILENAMES["test_labels"], test_lbls)

        x_train, y_train, x_test, y_test = load_mnist(str(tmp_path), subset=3)
        assert x_train.shape[0] == 3
        assert y_train.shape[0] == 3
        assert x_test.shape[0] == 3
        assert y_test.shape[0] == 3
