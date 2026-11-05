"""Model artefact tests (skipped until train.py/evaluate.py have been run)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.labels import CLASS_NAMES, IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "leafguard_mobilenetv2.keras"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run train.py to create the trained model"
)


def test_model_has_16_class_output():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    assert model.input_shape == (None, *IMAGE_SIZE, 3)
    assert model.output_shape == (None, len(CLASS_NAMES))


def test_model_scores_sum_to_one():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    batch = np.full((2, *IMAGE_SIZE, 3), 127.0, dtype=np.float32)
    scores = np.asarray(model(batch, training=False))
    assert scores.shape == (2, 16)
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(2), rtol=1e-5)


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="Run evaluate.py for metrics")
def test_metrics_match_test_split():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["class_count"] == 16
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["per_class"]) == 16
    assert sum(row["support"] for row in metrics["per_class"]) == metrics["test_images"]
    assert metrics["test_images"] > 0
