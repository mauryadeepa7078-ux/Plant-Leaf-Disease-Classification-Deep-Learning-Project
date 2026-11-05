"""Unit tests for the basic image-quality safeguards."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from utils.image_quality import image_quality_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_blank_grey_image_is_not_usable():
    quality = image_quality_metrics(Image.new("RGB", (320, 320), (128, 128, 128)))
    assert quality["usable"] is False
    assert quality["flags"]["not_leaf_like"] is True
    assert any("leaf-like colour" in warning for warning in quality["warnings"])


def test_real_leaf_image_passes_basic_quality():
    image_path = next((PROJECT_ROOT / "dataset" / "processed_224").rglob("*.jpg"))
    quality = image_quality_metrics(Image.open(image_path).convert("RGB"))
    assert quality["leaf_colour_ratio"] > 0.08
    assert quality["usable"] is True
