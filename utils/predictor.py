"""Reusable inference pipeline for the trained LeafGuard AI model.

Preprocessing is intentionally kept in one place: the trained Keras model
contains the MobileNetV2 rescaling layer, and this module always feeds a
224x224 RGB image with raw 0-255 values.  The same function is used by the
Flask website, the CLI and the test-suite.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_quality import image_quality_metrics, prepare_model_input
from .labels import (
    CLASS_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    LOW_MARGIN_THRESHOLD,
)
from .paths import MODEL_PATH, PREPROCESSING_PATH


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model files are unavailable."""


class LeafGuardPredictor:
    """Thread-safe, lazily loaded model wrapper."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        onnx_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.onnx_path = (
            Path(onnx_path)
            if onnx_path is not None
            else self.model_path.with_suffix(".onnx")
        )
        self._model = None
        self._session = None
        self._input_name: str | None = None
        self._backend: str | None = None
        self._lock = threading.Lock()
        self.loaded_at: float | None = None
        self.class_names: list[str] = list(CLASS_NAMES)

    @property
    def is_ready(self) -> bool:
        return self.onnx_path.exists() or self.model_path.exists()

    def load(self) -> None:
        if self._backend is not None:
            return
        if not self.is_ready:
            raise ModelNotReadyError(
                f"Model not found at {self.model_path} or {self.onnx_path}. "
                "Run `python train.py` first."
            )

        with self._lock:
            if self._backend is not None:
                return

            # ONNX Runtime is much lighter than TensorFlow and is the preferred
            # backend for deployment (for example on a 512 MB Render instance).
            if self.onnx_path.exists():
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(self.onnx_path), providers=["CPUExecutionProvider"]
                )
                outputs = session.get_outputs()
                if not outputs or outputs[0].shape[-1] not in (
                    len(CLASS_NAMES),
                    "16",
                ):
                    raise ModelNotReadyError(
                        f"ONNX model output {outputs[0].shape} does not match 16 classes."
                    )
                self._session = session
                self._input_name = session.get_inputs()[0].name
                self._backend = "onnx"
                self.loaded_at = time.time()
                return

            # Development fallback when only the Keras file is available.
            import tensorflow as tf

            model = tf.keras.models.load_model(self.model_path)
            output_shape = model.output_shape
            if not output_shape or output_shape[-1] != len(CLASS_NAMES):
                raise ModelNotReadyError(
                    f"Model outputs {output_shape}, expected {len(CLASS_NAMES)} classes."
                )
            self._model = model
            self._backend = "keras"
            self.loaded_at = time.time()

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_path": str(self.onnx_path if self._backend == "onnx" else self.model_path),
            "backend": self._backend or "not_loaded",
            "class_count": len(self.class_names),
            "input_size": [224, 224, 3],
            "preprocessing": "RGB, 224x224, raw 0-255; rescaling is inside the model",
        }
        if PREPROCESSING_PATH.exists():
            try:
                payload.update(json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        return payload

    def predict(
        self,
        image: Image.Image,
        selected_plant: str | None = None,
    ) -> dict[str, Any]:
        """Run a real forward pass and return a transparent prediction object."""
        self.load()

        quality = image_quality_metrics(image)
        batch = prepare_model_input(image)
        started = time.perf_counter()
        if self._backend == "onnx":
            assert self._session is not None and self._input_name is not None
            output = self._session.run(
                None, {self._input_name: batch.astype(np.float32)}
            )[0]
            probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
        else:
            assert self._model is not None
            probabilities = np.asarray(
                self._model(batch, training=False), dtype=np.float32
            ).reshape(-1)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if probabilities.size != len(CLASS_NAMES):
            raise ModelNotReadyError(
                f"Model returned {probabilities.size} scores, expected {len(CLASS_NAMES)}."
            )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = float(probabilities.sum())
        probabilities = probabilities / total if total > 0 else probabilities

        order = np.argsort(probabilities)[::-1]
        top_indices = [int(i) for i in order[:3]]
        top_probabilities = [float(probabilities[i]) for i in top_indices]
        best_index = top_indices[0]
        confidence = top_probabilities[0]
        margin = top_probabilities[0] - top_probabilities[1]

        from .labels import CLASSES

        best = CLASSES[best_index]
        reasons: list[str] = []
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Confidence {confidence * 100:.1f}% is below the "
                f"{LOW_CONFIDENCE_THRESHOLD * 100:.0f}% reliability threshold."
            )
        if margin < LOW_MARGIN_THRESHOLD:
            reasons.append(
                "The top two classes are close together, so the model is not "
                "clearly separating them."
            )
        if not quality["usable"]:
            reasons.append("The uploaded image did not pass the basic quality check.")

        selected = (selected_plant or "").strip().title() or None
        plant_mismatch = bool(selected and selected != best["plant"])

        top_predictions = []
        for index, probability in zip(top_indices, top_probabilities):
            item = CLASSES[index]
            top_predictions.append(
                {
                    "index": index,
                    "name": item["name"],
                    "plant": item["plant"],
                    "disease": item["disease"],
                    "probability": round(probability * 100.0, 2),
                    "is_healthy": item["healthy"],
                }
            )

        return {
            "predicted_index": best_index,
            "class_name": best["name"],
            "plant": best["plant"],
            "disease": best["disease"],
            "is_healthy": best["healthy"],
            "confidence": round(confidence * 100.0, 2),
            "confidence_raw": confidence,
            "margin": round(margin * 100.0, 2),
            "is_uncertain": bool(reasons),
            "uncertainty_reasons": reasons,
            "selected_plant": selected,
            "plant_mismatch": plant_mismatch,
            "top_predictions": top_predictions,
            "quality": quality,
            "latency_ms": round(latency_ms, 1),
            "thresholds": {
                "confidence": LOW_CONFIDENCE_THRESHOLD * 100.0,
                "margin": LOW_MARGIN_THRESHOLD * 100.0,
            },
            "info": best,
            "model": self.metadata(),
        }


_default_predictor: LeafGuardPredictor | None = None


def get_predictor() -> LeafGuardPredictor:
    """Process-wide singleton used by Flask."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = LeafGuardPredictor()
    return _default_predictor
