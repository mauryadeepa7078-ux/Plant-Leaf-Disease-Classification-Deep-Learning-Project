"""Export the trained Keras model to ONNX for lightweight Render inference.

TensorFlow is excellent for training but its runtime is large for a 512 MB
Render instance.  ONNX Runtime keeps the same trained weights while using much
less memory and starting faster.  The exported graph begins after the
training-only augmentation layers, so inference is deterministic.

Run:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.labels import CLASS_NAMES
from utils.paths import MODEL_PATH, MODELS_DIR, PROJECT_ROOT

ONNX_PATH = MODELS_DIR / "leafguard_mobilenetv2.onnx"
METADATA_PATH = MODELS_DIR / "onnx_export_metadata.json"


def build_clean_inference_model():
    """Remove training augmentation but keep rescaling, backbone and head."""
    model = tf.keras.models.load_model(MODEL_PATH)
    base = model.get_layer("mobilenetv2_1.00_224")
    head = model.get_layer("leafguard_classifier_head")

    inputs = layers.Input(shape=(224, 224, 3), name="leaf_rgb_0_255")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    outputs = head(x)
    clean = tf.keras.Model(inputs, outputs, name="leafguard_inference")
    return model, clean


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the Keras model first: {MODEL_PATH}")
    model, clean = build_clean_inference_model()

    # Compare the clean graph against the original saved model on real images.
    test_csv = PROJECT_ROOT / "dataset" / "splits" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("Run dataset_prep.py before exporting ONNX.")
    import pandas as pd

    frame = pd.read_csv(test_csv).head(8)
    images = []
    for relative in frame["processed_path"]:
        raw = tf.io.read_file(str(PROJECT_ROOT / relative))
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.LANCZOS3)
        images.append(np.asarray(image, dtype=np.float32))
    batch = np.stack(images)
    reference = np.asarray(model(batch, training=False))
    clean_output = np.asarray(clean(batch, training=False))
    max_diff = float(np.max(np.abs(reference - clean_output)))
    print(f"Clean-graph max difference vs saved model: {max_diff:.8f}")
    if max_diff > 1e-4:
        raise RuntimeError("Clean inference graph differs from the saved model.")

    import tf2onnx

    input_signature = (
        tf.TensorSpec((None, 224, 224, 3), tf.float32, name="leaf_rgb_0_255"),
    )
    print("Converting to ONNX ...")
    tf2onnx.convert.from_keras(
        clean,
        input_signature=input_signature,
        opset=17,
        output_path=str(ONNX_PATH),
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: batch.astype(np.float32)})[0]
    onnx_diff = float(np.max(np.abs(onnx_output - reference)))
    print(f"ONNX max difference vs saved model: {onnx_diff:.8f}")
    if onnx_diff > 1e-4:
        raise RuntimeError("ONNX output is too different from the Keras model.")

    metadata = {
        "source_model": MODEL_PATH.name,
        "onnx_model": ONNX_PATH.name,
        "opset": 17,
        "input_name": input_name,
        "input_shape": [None, 224, 224, 3],
        "input_value_range": [0.0, 255.0],
        "output_shape": [None, len(CLASS_NAMES)],
        "class_order": CLASS_NAMES,
        "max_abs_diff_vs_keras": onnx_diff,
        "filename": ONNX_PATH.name,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
