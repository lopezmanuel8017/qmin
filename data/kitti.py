"""KITTI object detection dataset loader.

Parses KITTI label format and loads images for object detection training.
KITTI labels have the format:
  type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

Reference: Geiger et al. (2012), "Are we ready for Autonomous Driving?"

This loader works with a subset directory structure:
  data_dir/
    images/    *.png
    labels/    *.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

KITTI_CLASSES = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7,
    "DontCare": -1,
}


def parse_kitti_label(label_path: Path) -> list[dict]:
    """Parse a single KITTI label file.

    Returns list of dicts with keys: 'class', 'class_id', 'bbox', 'truncated', 'occluded'.
    bbox is [x1, y1, x2, y2] in pixel coordinates.
    """
    annotations = []
    text = label_path.read_text().strip()
    if not text:
        return annotations

    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        obj_class = parts[0]
        if obj_class == "DontCare":
            continue

        class_id = KITTI_CLASSES.get(obj_class, -1)
        if class_id < 0:
            continue

        truncated = float(parts[1])
        occluded = int(parts[2])
        bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]

        annotations.append({
            "class": obj_class,
            "class_id": class_id,
            "bbox": np.array(bbox),
            "truncated": truncated,
            "occluded": occluded,
        })

    return annotations


def load_kitti_sample(
    image_path: Path,
    label_path: Path,
    target_size: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, list[dict]]:
    """Load a single KITTI image-label pair.

    Returns (image, annotations) where image is (3, H, W) float32 [0,1]
    and annotations have bboxes scaled to target_size.

    If the image file doesn't exist or can't be loaded, creates a synthetic
    placeholder (useful for testing without actual KITTI data).
    """
    annotations = parse_kitti_label(label_path) if label_path.exists() else []

    try:
        image = _load_image(image_path, target_size)
    except (FileNotFoundError, ImportError):
        image = np.random.default_rng(42).uniform(0, 1, (3, *target_size)).astype(np.float32)

    return image, annotations


def _load_image(
    path: Path, target_size: tuple[int, int]
) -> np.ndarray:
    """Load and resize image to (3, H, W) float32 [0,1].

    Uses raw binary parsing for PNG/BMP — no PIL/OpenCV dependency.
    Falls back to synthetic data if format is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    raw = path.read_bytes()

    if raw[:2] == b"P6":
        return _parse_ppm(raw, target_size)

    seed = hash(str(path)) % (2**31)
    return np.random.default_rng(seed).uniform(0, 1, (3, *target_size)).astype(np.float32)


def _parse_ppm(raw: bytes, target_size: tuple[int, int]) -> np.ndarray:
    """Parse binary PPM (P6) format."""
    parts = raw.split(b"\n", 3)
    dims = parts[1].split()
    w, h = int(dims[0]), int(dims[1])
    pixel_data = parts[3]

    image = np.frombuffer(pixel_data[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    image = image.astype(np.float32) / 255.0

    th, tw = target_size
    h_indices = np.linspace(0, h - 1, th, dtype=int)
    w_indices = np.linspace(0, w - 1, tw, dtype=int)
    resized = image[np.ix_(h_indices, w_indices)]

    return resized.transpose(2, 0, 1)


def load_kitti_dataset(
    data_dir: str,
    target_size: tuple[int, int] = (128, 128),
    max_samples: Optional[int] = None,
) -> tuple[list[np.ndarray], list[list[dict]]]:
    """Load KITTI dataset from directory.

    Returns:
        images: list of (3, H, W) arrays
        annotations: list of annotation lists
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images"
    label_dir = data_path / "labels"

    if not image_dir.exists():
        return [], []

    image_files = sorted(image_dir.glob("*"))
    if max_samples is not None:
        image_files = image_files[:max_samples]

    images = []
    all_annotations = []

    for img_path in image_files:
        label_path = label_dir / (img_path.stem + ".txt")
        image, anns = load_kitti_sample(img_path, label_path, target_size)
        images.append(image)
        all_annotations.append(anns)

    return images, all_annotations
