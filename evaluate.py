"""Evaluate LeafGuard AI on the held-out test split and save real artefacts.

Outputs (results/)
------------------
model_metrics.json              machine-readable metrics used by the website
classification_report.txt/.csv  precision, recall and F1 for all 16 classes
confusion_matrix.csv/png        raw and normalised confusion matrices
accuracy_curve.png              training vs validation accuracy
loss_curve.png                  training vs validation loss
class_f1_scores.png             class-wise F1 bar chart
sample_predictions.png          one correctly classified test leaf per plant
test_predictions.csv            every test image with its prediction
class_wise_performance.csv      sortable per-class table

Run:
    python evaluate.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

from utils.labels import CLASS_NAMES, CLASSES
from utils.paths import (
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLITS_DIR,
    STATIC_DIR,
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PALETTE = "#2f6b3c"


def load_test_dataset() -> tuple[tf.data.Dataset, pd.DataFrame]:
    csv_path = SPLITS_DIR / "test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python dataset_prep.py` before evaluation."
        )
    frame = pd.read_csv(csv_path)
    paths = [str(PROJECT_ROOT / value) for value in frame["processed_path"]]
    labels = frame["class_index"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.LANCZOS3)
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, frame


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    labels = [name.replace(" ", "\n", 1) for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Number of test images"},
        ax=ax,
    )
    ax.set_title("LeafGuard AI - Confusion Matrix (held-out test set)", pad=16)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves() -> None:
    if not HISTORY_PATH.exists():
        print("Training history not found; skipping learning curves.")
        return
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    accuracy = history.get("accuracy", [])
    val_accuracy = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    if not accuracy or not loss:
        print("Training history is missing accuracy/loss values.")
        return

    epochs = np.arange(1, len(accuracy) + 1)
    boundary = int(history.get("stage1_actual_epochs", [0])[0])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(epochs, accuracy, marker="o", label="Training accuracy", color="#245c33")
    ax.plot(epochs, val_accuracy, marker="o", label="Validation accuracy", color="#d97706")
    if 0 < boundary < len(epochs):
        ax.axvline(boundary + 0.5, color="#64748b", linestyle="--", linewidth=1)
        ax.text(
            boundary + 0.7,
            min(accuracy + val_accuracy),
            "fine-tuning starts",
            fontsize=8,
            color="#475569",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Training and validation accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(epochs, loss, marker="o", label="Training loss", color="#245c33")
    ax.plot(epochs, val_loss, marker="o", label="Validation loss", color="#d97706")
    if 0 < boundary < len(epochs):
        ax.axvline(boundary + 0.5, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)


def plot_class_performance(per_class: pd.DataFrame) -> None:
    ordered = per_class.sort_values("f1_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colours = ["#c2410c" if value < 0.6 else "#2f6b3c" for value in ordered["f1_score"]]
    ax.barh(ordered["class_name"], ordered["f1_score"], color=colours)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1-score")
    ax.set_title("Class-wise F1-score on the held-out test set")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(ordered["f1_score"]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_f1_scores.png", dpi=180)
    plt.close(fig)


def save_sample_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Save a 2x2 figure with one correct test prediction per plant."""
    samples = []
    for plant in ("Tomato", "Apple", "Grape", "Corn"):
        candidate_indices = frame.index[
            (frame["plant"] == plant) & (y_true == y_pred)
        ].tolist()
        # Prefer a diseased example so the figure demonstrates an actual
        # classification decision rather than four healthy leaves.
        diseased = [
            index for index in candidate_indices if not CLASSES[int(y_true[index])]["healthy"]
        ]
        if diseased:
            candidate_indices = diseased
        if not candidate_indices:
            candidate_indices = frame.index[frame["plant"] == plant].tolist()
        if not candidate_indices:
            continue
        best = max(candidate_indices, key=lambda idx: confidence[idx])
        samples.append((plant, best))

    if not samples:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()
    for axis, (plant, index) in zip(axes, samples):
        row = frame.loc[index]
        image_path = PROJECT_ROOT / row["processed_path"]
        image = Image.open(image_path).convert("RGB")
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{plant}\nPredicted: {CLASS_NAMES[int(y_pred[index])]} "
            f"({confidence[index] * 100:.1f}%)",
            fontsize=9,
            color="#1f3d2b" if y_pred[index] == y_true[index] else "#b91c1c",
        )

        # Copy a web-friendly card image for the home page.
        image.resize((600, 600), Image.Resampling.LANCZOS).save(
            STATIC_DIR / "images" / f"{plant.lower()}.jpg",
            quality=90,
            optimize=True,
        )
        image.resize((480, 480), Image.Resampling.LANCZOS).save(
            STATIC_DIR / "samples" / f"{plant.lower()}_sample.jpg",
            quality=90,
            optimize=True,
        )

    for axis in axes[len(samples):]:
        axis.axis("off")
    fig.suptitle(
        "Correctly classified test leaves from all four supported plants",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample predictions for: {', '.join(item[0] for item in samples)}")


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "images").mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "samples").mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds, frame = load_test_dataset()
    print(f"Test images: {len(frame)}")

    probabilities = np.asarray(model.predict(test_ds, verbose=1), dtype=np.float32)
    y_true = frame["class_index"].to_numpy(dtype=np.int32)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    labels = list(range(len(CLASS_NAMES)))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    top3 = float(
        top_k_accuracy_score(y_true, probabilities, k=3, labels=labels)
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        RESULTS_DIR / "confusion_matrix.csv"
    )
    normalised = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(normalised, index=CLASS_NAMES, columns=CLASS_NAMES).round(4).to_csv(
        RESULTS_DIR / "confusion_matrix_normalized.csv"
    )
    plot_confusion_matrix(matrix, RESULTS_DIR / "confusion_matrix.png")

    per_class_rows = []
    for index, name in enumerate(CLASS_NAMES):
        item = CLASSES[index]
        true_mask = y_true == index
        pred_mask = y_pred == index
        support = int(true_mask.sum())
        correct = int((true_mask & pred_mask).sum())
        per_class_rows.append(
            {
                "class_index": index,
                "class_name": name,
                "plant": item["plant"],
                "disease": item["disease"],
                "support": support,
                "correct": correct,
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1_score": float(report_dict[name]["f1-score"]),
                "mean_confidence": float(confidence[pred_mask].mean()) if pred_mask.any() else 0.0,
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(RESULTS_DIR / "class_wise_performance.csv", index=False)
    plot_class_performance(per_class)

    predictions_frame = frame.copy()
    predictions_frame["true_class"] = [CLASS_NAMES[i] for i in y_true]
    predictions_frame["predicted_class"] = [CLASS_NAMES[i] for i in y_pred]
    predictions_frame["confidence"] = confidence
    predictions_frame["correct"] = y_true == y_pred
    predictions_frame[
        [
            "original_filename",
            "plant",
            "true_class",
            "predicted_class",
            "confidence",
            "correct",
            "processed_path",
        ]
    ].to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    plot_training_curves()
    save_sample_predictions(frame, y_true, y_pred, confidence)

    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "test_set": "15% held-out stratified group split (never used for training)",
        "test_images": int(len(frame)),
        "class_count": len(CLASS_NAMES),
        "accuracy": round(accuracy, 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "precision_weighted": round(float(precision_weighted), 6),
        "recall_weighted": round(float(recall_weighted), 6),
        "f1_weighted": round(float(f1_weighted), 6),
        "top3_accuracy": round(top3, 6),
        "mean_confidence": round(float(confidence.mean()), 6),
        "class_names": CLASS_NAMES,
        "per_class": per_class_rows,
        "confusion_matrix": matrix.tolist(),
        "notes": (
            "All values are computed from the trained model's predictions on the "
            "held-out test split. Confidence is not diagnostic certainty."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # A small file the website can use for the results-page summary.
    summary = {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "test_images": metrics["test_images"],
        "top3_accuracy": metrics["top3_accuracy"],
        "evaluated_at_utc": metrics["evaluated_at_utc"],
    }
    (RESULTS_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"Test accuracy : {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision_macro * 100:.2f}%")
    print(f"Macro recall   : {recall_macro * 100:.2f}%")
    print(f"Macro F1       : {f1_macro * 100:.2f}%")
    print(f"Top-3 accuracy : {top3 * 100:.2f}%")
    print("=" * 60)
    print(report_text)
    print(f"Metrics saved to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
