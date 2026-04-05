"""Tests for data/kitti.py — KITTI dataset loader."""

import numpy as np
import pytest

from data.kitti import (
    KITTI_CLASSES, _load_image, _parse_ppm, load_kitti_dataset,
    load_kitti_sample, parse_kitti_label,
)


class TestParseKittiLabel:
    def test_parse_car(self, tmp_path):
        label_file = tmp_path / "000001.txt"
        label_file.write_text(
            "Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 "
            "-0.65 1.71 46.70 -1.59\n"
        )
        anns = parse_kitti_label(label_file)
        assert len(anns) == 1
        assert anns[0]["class"] == "Car"
        assert anns[0]["class_id"] == 0
        np.testing.assert_allclose(anns[0]["bbox"], [587.01, 173.33, 614.12, 200.12])

    def test_parse_multiple_objects(self, tmp_path):
        label_file = tmp_path / "000002.txt"
        label_file.write_text(
            "Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 "
            "-0.65 1.71 46.70 -1.59\n"
            "Pedestrian 0.00 0 0.21 712.40 143.00 810.73 307.92 1.89 0.48 1.20 "
            "1.84 1.47 8.41 0.01\n"
        )
        anns = parse_kitti_label(label_file)
        assert len(anns) == 2
        assert anns[0]["class"] == "Car"
        assert anns[1]["class"] == "Pedestrian"

    def test_skip_dontcare(self, tmp_path):
        label_file = tmp_path / "000003.txt"
        label_file.write_text(
            "DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 "
            "-1000 -1000 -10\n"
        )
        anns = parse_kitti_label(label_file)
        assert len(anns) == 0

    def test_empty_file(self, tmp_path):
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")
        anns = parse_kitti_label(label_file)
        assert len(anns) == 0

    def test_short_line_skipped(self, tmp_path):
        label_file = tmp_path / "short.txt"
        label_file.write_text("Car 0.00 0\n")
        anns = parse_kitti_label(label_file)
        assert len(anns) == 0

    def test_unknown_class_skipped(self, tmp_path):
        label_file = tmp_path / "unknown.txt"
        label_file.write_text(
            "UnknownClass 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 "
            "-0.65 1.71 46.70 -1.59\n"
        )
        anns = parse_kitti_label(label_file)
        assert len(anns) == 0

    def test_truncated_and_occluded(self, tmp_path):
        label_file = tmp_path / "000004.txt"
        label_file.write_text(
            "Car 0.50 2 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 "
            "-0.65 1.71 46.70 -1.59\n"
        )
        anns = parse_kitti_label(label_file)
        assert anns[0]["truncated"] == 0.50
        assert anns[0]["occluded"] == 2


class TestLoadKittiSample:
    def test_load_with_synthetic_image(self, tmp_path):
        label_file = tmp_path / "000001.txt"
        label_file.write_text(
            "Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 "
            "-0.65 1.71 46.70 -1.59\n"
        )
        image_path = tmp_path / "000001.png"
        image, anns = load_kitti_sample(image_path, label_file, target_size=(64, 64))
        assert image.shape == (3, 64, 64)
        assert image.dtype == np.float32
        assert len(anns) == 1

    def test_load_missing_label(self, tmp_path):
        image_path = tmp_path / "000001.png"
        label_path = tmp_path / "missing.txt"
        image, anns = load_kitti_sample(image_path, label_path, target_size=(32, 32))
        assert image.shape == (3, 32, 32)
        assert len(anns) == 0


class TestLoadImage:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_image(tmp_path / "nonexistent.png", (64, 64))

    def test_unknown_format_returns_synthetic(self, tmp_path):
        img_file = tmp_path / "test.bin"
        img_file.write_bytes(b"\x00\x01\x02\x03")
        image = _load_image(img_file, (32, 32))
        assert image.shape == (3, 32, 32)

    def test_ppm_parsing(self, tmp_path):
        """Create a minimal PPM file and parse it."""
        w, h = 4, 3
        pixels = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
        ppm_data = f"P6\n{w} {h}\n255\n".encode() + pixels.tobytes()
        ppm_file = tmp_path / "test.ppm"
        ppm_file.write_bytes(ppm_data)

        image = _load_image(ppm_file, (2, 2))
        assert image.shape == (3, 2, 2)
        assert image.dtype == np.float32
        assert image.max() <= 1.0


class TestParsePPM:
    def test_basic(self):
        w, h = 3, 2
        pixels = np.arange(18, dtype=np.uint8).reshape(h, w, 3)
        raw = f"P6\n{w} {h}\n255\n".encode() + pixels.tobytes()
        image = _parse_ppm(raw, (2, 2))
        assert image.shape == (3, 2, 2)


class TestLoadKittiDataset:
    def test_empty_dir(self, tmp_path):
        images, anns = load_kitti_dataset(str(tmp_path))
        assert images == []
        assert anns == []

    def test_with_synthetic_data(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        for i in range(3):
            (img_dir / f"{i:06d}.bin").write_bytes(b"\x00" * 10)
            (lbl_dir / f"{i:06d}.txt").write_text(
                "Car 0.00 0 0 100 100 200 200 1 1 1 0 0 10 0\n"
            )

        images, anns = load_kitti_dataset(str(tmp_path), target_size=(32, 32))
        assert len(images) == 3
        assert len(anns) == 3
        assert images[0].shape == (3, 32, 32)
        assert len(anns[0]) == 1

    def test_max_samples(self, tmp_path):
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        for i in range(5):
            (img_dir / f"{i:06d}.bin").write_bytes(b"\x00" * 10)
            (lbl_dir / f"{i:06d}.txt").write_text("")

        images, _ = load_kitti_dataset(str(tmp_path), max_samples=2)
        assert len(images) == 2


class TestKittiClasses:
    def test_car_class(self):
        assert KITTI_CLASSES["Car"] == 0

    def test_dontcare(self):
        assert KITTI_CLASSES["DontCare"] == -1
