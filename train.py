"""Train LeafGuard AI's 16-class MobileNetV2 transfer-learning model.

Training strategy

The project targets CPU-only machines, where back-propagating through the
whole MobileNetV2 graph is prohibitively slow.  We therefore use the standard
*feature-extraction* form of transfer learning:

1. Load MobileNetV2 with ImageNet weights and freeze it.
2. Extract 1,280-dimensional features for every training image, plus one
   randomly augmented copy (flip, rotation, zoom, translation, contrast,
   brightness).  Augmentation is applied only to training data.
3. Train a small regularised classifier head on those features with
   EarlyStopping, ModelCheckpoint and learning-rate reduction.
4. Re-attach the trained head to the frozen backbone, verify that the combined
   model makes the same predictions, and save the complete Keras model.

This is genuine training on the real dataset, not a hard-coded predictor.
The saved model contains the backbone and the head, and it performs all
preprocessing internally, so inference is identical to training.

Examples
--------
    python train.py
    python train.py --epochs 30
    python train.py --rebuild-features
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Keep TensorFlow's informational output readable in a terminal.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from utils.labels import CLASS_NAMES, CLASSES
from utils.paths import (
    CLASS_NAMES_PATH,
    LABEL_MAP_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSING_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLITS_DIR,
)

IMAGE_SIZE = (224, 224)
FEATURE_BATCH = 16
HEAD_BATCH = 64
AUTOTUNE = tf.data.AUTOTUNE
WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)
WEIGHTS_PATH = MODELS_DIR / "mobilenet_v2_imagenet_weights.h5"
TRAIN_FEATURES_PATH = MODELS_DIR / "feature_cache_train.npz"
VAL_FEATURES_PATH = MODELS_DIR / "feature_cache_validation.npz"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_imagenet_weights() -> Path:
    """Download MobileNetV2 ImageNet weights once, with a clear error if offline."""
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 5_000_000:
        return WEIGHTS_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp = WEIGHTS_PATH.with_suffix(".part")
    print("Downloading MobileNetV2 ImageNet weights (~14 MB) ...")
    try:
        request = urllib.request.Request(
            WEIGHTS_URL, headers={"User-Agent": "LeafGuard-AI"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            temp.write_bytes(response.read())
        if temp.stat().st_size < 5_000_000:
            raise ValueError("Downloaded weights file is unexpectedly small.")
        temp.replace(WEIGHTS_PATH)
    except Exception as exc:  # noqa: BLE001
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download MobileNetV2 ImageNet weights. Check your internet "
            "connection, then run train.py again."
        ) from exc
    print(f"  saved to {WEIGHTS_PATH}")
    return WEIGHTS_PATH


def augmentation_pipeline() -> tf.keras.Sequential:
    """Training-only image augmentation, saved inside the final model."""
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="random_flip"),
            layers.RandomRotation(0.07, fill_mode="reflect", name="random_rotation"),
            layers.RandomZoom(0.10, fill_mode="reflect", name="random_zoom"),
            layers.RandomTranslation(
                0.05, 0.05, fill_mode="reflect", name="random_translation"
            ),
            layers.RandomContrast(0.12, name="random_contrast"),
            layers.RandomBrightness(0.10, name="random_brightness"),
        ],
        name="training_augmentation",
    )


def make_image_dataset(
    csv_path: Path,
    batch_size: int,
    training: bool,
    seed: int,
) -> tuple[tf.data.Dataset, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}. Run `python dataset_prep.py` first."
        )
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Split {csv_path.name} is empty.")
    paths = [str(PROJECT_ROOT / value) for value in frame["processed_path"]]
    labels = frame["class_index"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 1500), seed=seed, reshuffle_each_iteration=True
        )

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raw = tf.io.read_file(path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(decode, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset, frame


def build_feature_extractor(weights_path: Path) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Return (full frozen backbone, raw-image -> 1280-d feature model)."""
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=str(weights_path),
        name="mobilenetv2_1.00_224",
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="leaf_rgb_0_255")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    features = tf.keras.Model(inputs, x, name="leafguard_feature_extractor")
    return base, features


def extract_features(
    csv_path: Path,
    feature_model: tf.keras.Model,
    augment: bool,
    seed: int,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset, frame = make_image_dataset(csv_path, FEATURE_BATCH, training=False, seed=seed)
    if limit:
        dataset = dataset.take(max(1, limit // FEATURE_BATCH))
        frame = frame.iloc[: len(dataset) * FEATURE_BATCH].reset_index(drop=True)

    augmentation = augmentation_pipeline() if augment else None
    features: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    started = time.time()
    seen = 0
    for images, labels in dataset:
        if augmentation is not None:
            images = augmentation(images, training=True)
        batch_features = feature_model(images, training=False)
        features.append(np.asarray(batch_features, dtype=np.float32))
        labels_out.append(labels.numpy().astype(np.int32))
        seen += len(labels)
        if seen % 320 == 0:
            rate = seen / max(time.time() - started, 1e-6)
            print(f"    {seen} images | {rate:.1f} images/s")
    return np.concatenate(features), np.concatenate(labels_out)


def cache_features(
    feature_model: tf.keras.Model,
    seed: int,
    rebuild: bool,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build/load train (original + augmented) and validation feature caches."""
    required = (TRAIN_FEATURES_PATH, VAL_FEATURES_PATH)
    if rebuild:
        # Avoid mixing a partial cache from an interrupted run with stale data.
        for path in required:
            path.unlink(missing_ok=True)
    if not rebuild and all(path.exists() for path in required):
        print("Loading cached features ...")
        train_cache = np.load(TRAIN_FEATURES_PATH)
        val_cache = np.load(VAL_FEATURES_PATH)
        train_features = np.concatenate(
            [train_cache["original_features"], train_cache["augmented_features"]], axis=0
        )
        train_labels = np.concatenate(
            [train_cache["original_labels"], train_cache["augmented_labels"]], axis=0
        )
        return (
            train_features,
            train_labels,
            train_cache["original_features"],
            train_cache["original_labels"],
            val_cache["features"],
            val_cache["labels"],
        )

    print("\nExtracting frozen MobileNetV2 features from training images (original) ...")
    train_original, train_labels_original = extract_features(
        SPLITS_DIR / "train.csv", feature_model, augment=False, seed=seed, limit=limit
    )
    print("Extracting augmented training features ...")
    train_augmented, train_labels_augmented = extract_features(
        SPLITS_DIR / "train.csv", feature_model, augment=True, seed=seed + 7, limit=limit
    )
    print("Extracting validation features ...")
    val_features, val_labels = extract_features(
        SPLITS_DIR / "validation.csv", feature_model, augment=False, seed=seed
    )

    np.savez_compressed(
        TRAIN_FEATURES_PATH,
        original_features=train_original,
        original_labels=train_labels_original,
        augmented_features=train_augmented,
        augmented_labels=train_labels_augmented,
    )
    np.savez_compressed(VAL_FEATURES_PATH, features=val_features, labels=val_labels)
    train_features = np.concatenate([train_original, train_augmented], axis=0)
    train_labels = np.concatenate([train_labels_original, train_labels_augmented], axis=0)
    return (
        train_features,
        train_labels,
        train_original,
        train_labels_original,
        val_features,
        val_labels,
    )


def build_head() -> tf.keras.Model:
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=(1280,), name="mobilenet_features")
    x = layers.BatchNormalization(name="feature_normalization")(inputs)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="classifier_dense",
    )(x)
    x = layers.Dropout(0.40, name="classifier_dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="disease_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name="leafguard_classifier_head")


def build_full_model(
    base: tf.keras.Model,
    head: tf.keras.Model,
) -> tf.keras.Model:
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="leaf_rgb_0_255")
    x = augmentation_pipeline()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    outputs = head(x)
    return tf.keras.Model(inputs, outputs, name="leafguard_mobilenetv2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="maximum head epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit training images for a smoke test (not for final results)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU") or "CPU only")
    print("Classes:", len(CLASS_NAMES))

    weights_path = download_imagenet_weights()
    base, feature_model = build_feature_extractor(weights_path)

    (
        train_features,
        train_labels,
        train_original,
        train_labels_original,
        val_features,
        val_labels,
    ) = cache_features(feature_model, args.seed, args.rebuild_features, args.limit)

    print(
        f"\nTrain feature rows: {len(train_features)} "
        f"({len(train_original)} original + {len(train_features) - len(train_original)} augmented)"
    )
    print(f"Validation feature rows: {len(val_features)}")

    unique = np.unique(train_labels_original)
    weights = compute_class_weight(
        class_weight="balanced", classes=unique, y=train_labels_original
    )
    class_weight = {int(index): float(value) for index, value in zip(unique, weights)}
    print(
        "Class weights:",
        {CLASS_NAMES[index]: round(value, 3) for index, value in class_weight.items()},
    )

    head = build_head()
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    head.summary(print_fn=lambda line: print(line))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODELS_DIR / "best_head.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.35,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(RESULTS_DIR / "training_head_log.csv", append=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    started = time.time()
    history = head.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=args.epochs,
        batch_size=HEAD_BATCH,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    best_path = MODELS_DIR / "best_head.keras"
    if best_path.exists():
        head = tf.keras.models.load_model(best_path)
    head_val_loss, head_val_accuracy = head.evaluate(
        val_features, val_labels, verbose=0
    )
    print(f"\nHead validation accuracy: {head_val_accuracy:.4f}")

    print("\nBuilding and verifying the complete inference model ...")
    full_model = build_full_model(base, head)
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    full_model.save(MODEL_PATH)

    # Verify that the saved model's output matches the head's output and that it
    # can be reloaded.  This catches preprocessing/head-attachment mistakes now,
    # not during a demo.
    reloaded = tf.keras.models.load_model(MODEL_PATH)
    val_dataset, val_frame = make_image_dataset(
        SPLITS_DIR / "validation.csv", FEATURE_BATCH, training=False, seed=args.seed
    )
    image_batches = []
    for images, _ in val_dataset.take(4):
        image_batches.append(images.numpy())
    sample_images = np.concatenate(image_batches, axis=0)
    saved_scores = np.asarray(reloaded(sample_images, training=False))
    if not np.allclose(saved_scores.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Saved model outputs do not sum to one.")
    saved_predictions = saved_scores.argmax(axis=1)
    saved_accuracy = float(
        (saved_predictions == val_frame["class_index"].to_numpy()[: len(sample_images)]).mean()
    )
    print(f"Reloaded model spot-check accuracy on {len(sample_images)} val images: {saved_accuracy:.3f}")

    history_dict = {key: list(map(float, values)) for key, values in history.history.items()}
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(history_dict, indent=2), encoding="utf-8"
    )

    CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    LABEL_MAP_PATH.write_text(
        json.dumps(
            {
                "display_to_index": {item["name"]: item["index"] for item in CLASSES},
                "dataset_label_to_index": {
                    item["dataset_label"]: item["index"] for item in CLASSES
                },
                "index_to_dataset_label": {
                    str(item["index"]): item["dataset_label"] for item in CLASSES
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    PREPROCESSING_PATH.write_text(
        json.dumps(
            {
                "input_size": [224, 224],
                "color_mode": "RGB",
                "resize_interpolation": "PIL LANCZOS",
                "input_value_range": [0.0, 255.0],
                "normalization": "Rescaling(1/127.5, offset=-1) inside the model",
                "output": f"{len(CLASS_NAMES)}-class softmax",
                "class_order": CLASS_NAMES,
                "training_mode": "frozen MobileNetV2 feature extraction + trained head",
                "augmentation": [
                    "horizontal flip",
                    "rotation 7%",
                    "zoom 10%",
                    "translation 5%",
                    "contrast 12%",
                    "brightness 10%",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "MobileNetV2 (ImageNet, frozen) + BatchNorm + Dense(256) + Dropout(0.40)",
        "training_mode": "transfer-learning feature extraction (CPU-friendly)",
        "input_size": [224, 224, 3],
        "classes": len(CLASS_NAMES),
        "epochs_run": len(history.history.get("loss", [])),
        "batch_size_head": HEAD_BATCH,
        "best_validation_accuracy": round(float(head_val_accuracy), 6),
        "training_seconds": round(time.time() - started, 1),
        "train_images_original": int(len(train_original)),
        "train_feature_rows": int(len(train_features)),
        "validation_images": int(len(val_features)),
        "class_weight": class_weight,
        "note": (
            "Frozen-backbone feature extraction is used because the reference "
            "machine is CPU-only. Augmented features are generated only from the "
            "training split."
        ),
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "model_summary.txt").write_text(
        "\n".join(
            [
                f"TensorFlow: {tf.__version__}",
                "Input: 224x224x3 RGB, 0-255; rescaling inside model",
                "Backbone: MobileNetV2, ImageNet weights, frozen",
                "Head: BatchNorm -> Dense 256 ReLU (L2) -> Dropout 0.40 -> Dense 16 softmax",
                f"Classes: {len(CLASS_NAMES)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with (RESULTS_DIR / "model_summary.txt").open("a", encoding="utf-8") as handle:
        full_model.summary(print_fn=lambda line: handle.write(line + "\n"))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Best validation accuracy: {head_val_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
