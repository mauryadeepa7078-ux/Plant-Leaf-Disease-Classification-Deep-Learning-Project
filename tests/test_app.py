"""Flask route and end-to-end prediction tests."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from PIL import Image

from utils.image_quality import image_quality_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "leafguard_mobilenetv2.keras"
PREDICTIONS = PROJECT_ROOT / "results" / "test_predictions.csv"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run train.py before prediction tests"
)


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"LeafGuard" in response.data
    assert b"Tomato" in response.data
    assert b"Apple" in response.data
    assert b"Grape" in response.data
    assert b"Corn" in response.data


def test_static_pages_load(client):
    assert client.get("/detect").status_code == 200
    assert client.get("/performance").status_code == 200
    assert client.get("/about").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/missing-page").status_code == 404


def test_invalid_extension_is_rejected(client):
    response = client.post(
        "/predict",
        data={"plant": "Tomato", "image": (io.BytesIO(b"not an image"), "note.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Unsupported file type" in response.data


def test_invalid_image_bytes_are_rejected(client):
    response = client.post(
        "/predict",
        data={"plant": "Tomato", "image": (io.BytesIO(b"GIF89a fake"), "leaf.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"could not be read as an image" in response.data


def _correct_samples_by_plant(limit: int = 1) -> dict[str, str]:
    if not PREDICTIONS.exists():
        return {}
    samples: dict[str, list[str]] = {}
    with PREDICTIONS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["correct"].lower() == "true":
                samples.setdefault(row["plant"], []).append(row["processed_path"])
    # Pick a correctly classified image that also passes the quality gate, so
    # the end-to-end test represents a normal, usable upload rather than a
    # deliberately blurry test image.
    chosen = {}
    for plant, paths in samples.items():
        for relative_path in paths:
            image = Image.open(PROJECT_ROOT / relative_path).convert("RGB")
            if image_quality_metrics(image)["usable"]:
                chosen[plant] = relative_path
                break
    return chosen


@requires_model
@pytest.mark.skipif(not PREDICTIONS.exists(), reason="Run evaluate.py first")
def test_all_four_plants_predict_through_website(client):
    samples = _correct_samples_by_plant()
    assert set(samples) == {"Tomato", "Apple", "Grape", "Corn"}, samples
    for plant, relative_path in samples.items():
        image = Image.open(PROJECT_ROOT / relative_path).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        response = client.post(
            "/predict",
            data={"plant": plant, "image": (buffer, f"{plant.lower()}.jpg")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert response.status_code == 302, (plant, response.status_code)
        result = client.get(response.headers["Location"])
        assert result.status_code == 200
        assert plant.encode() in result.data
        assert b"confidence" in result.data.lower()
        assert b"Needs a closer look" not in result.data


@requires_model
def test_low_quality_flat_image_is_uncertain(client):
    image = Image.new("RGB", (300, 300), (128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    response = client.post(
        "/predict",
        data={"plant": "", "image": (buffer, "blank.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert response.status_code == 302
    result = client.get(response.headers["Location"])
    assert result.status_code == 200
    assert b"Needs a closer look" in result.data or b"Uncertain" in result.data
