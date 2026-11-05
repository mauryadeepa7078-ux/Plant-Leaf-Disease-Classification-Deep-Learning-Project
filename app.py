"""LeafGuard AI Flask website.

Run locally:
    python app.py
Then open http://127.0.0.1:5000

The website performs a real forward pass through models/leafguard_mobilenetv2.keras.
It never returns a hard-coded disease or confidence value.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image

from utils.image_quality import (
    ImageValidationError,
    MAX_UPLOAD_BYTES,
    validate_upload,
)
from utils.labels import (
    CLASSES,
    CLASS_NAMES,
    PLANTS,
    PLANT_CLASSES,
    PLANT_META,
)
from utils.paths import (
    DATASET_SUMMARY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    SAVED_RESULTS_DIR,
    UPLOADS_DIR,
    ensure_directories,
)
from utils.predictor import ModelNotReadyError, get_predictor

RESULT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_RESULT_FILES = {
    "accuracy_curve.png",
    "loss_curve.png",
    "confusion_matrix.png",
    "confusion_matrix.csv",
    "class_f1_scores.png",
    "sample_predictions.png",
    "classification_report.txt",
    "classification_report.csv",
    "class_wise_performance.csv",
    "test_predictions.csv",
}

ensure_directories()

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 64 * 1024,
    SECRET_KEY=os.environ.get(
        "SECRET_KEY", "leafguard-ai-local-development-key-change-in-production"
    ),
    JSON_SORT_KEYS=False,
)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataset_summary() -> dict:
    return load_json(DATASET_SUMMARY_PATH) or {}


def model_metrics() -> dict:
    return load_json(METRICS_PATH) or {}


def save_upload_as_jpeg(image: Image.Image) -> str:
    """Save a web-display copy and return its filename under static/uploads."""
    filename = f"{uuid.uuid4().hex}.jpg"
    copy_image = image.copy()
    copy_image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    copy_image.convert("RGB").save(
        UPLOADS_DIR / filename, format="JPEG", quality=90, optimize=True
    )
    return filename


def plant_from_request() -> str | None:
    value = (request.form.get("plant") or request.args.get("plant") or "").strip().title()
    return value if value in PLANTS else None


@app.context_processor
def inject_globals() -> dict:
    return {
        "PLANTS": PLANTS,
        "PLANT_META": PLANT_META,
        "PLANT_CLASSES": PLANT_CLASSES,
        "CLASSES": CLASSES,
        "CLASS_NAMES": CLASS_NAMES,
        "model_ready": MODEL_PATH.exists(),
        "current_year": datetime.now().year,
    }


@app.template_filter("percentage")
def percentage(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.route("/")
def home():
    summary = dataset_summary()
    metrics = model_metrics()
    return render_template(
        "index.html",
        summary=summary,
        metrics=metrics,
        page_title="LeafGuard AI — Multi-plant leaf disease classification",
        active_page="home",
    )


@app.route("/detect")
def detect():
    selected = plant_from_request()
    return render_template(
        "detect.html",
        selected_plant=selected,
        page_title="Detect a leaf condition — LeafGuard AI",
        active_page="detect",
    )


@app.route("/predict", methods=["POST"])
def predict():
    selected_plant = plant_from_request()
    upload = request.files.get("image")

    if upload is None or not upload.filename:
        flash("Please choose a leaf image before analysing.", "error")
        return redirect(url_for("detect", plant=selected_plant or ""))

    try:
        raw_bytes = upload.read()
        image, metadata = validate_upload(
            raw_bytes, upload.filename, upload.mimetype
        )
    except ImageValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("detect", plant=selected_plant or ""))

    try:
        predictor = get_predictor()
        prediction = predictor.predict(image, selected_plant=selected_plant)
    except ModelNotReadyError:
        flash(
            "The trained model is not available yet. Run `python train.py` and "
            "restart the website.",
            "error",
        )
        return redirect(url_for("detect", plant=selected_plant or ""))
    except Exception:
        app.logger.exception("Prediction failed")
        flash(
            "The image could not be analysed due to an internal model error. "
            "Please try again with a normal JPG or PNG leaf photo.",
            "error",
        )
        return redirect(url_for("detect", plant=selected_plant or ""))

    image_filename = save_upload_as_jpeg(image)
    result_id = uuid.uuid4().hex
    payload = {
        "result_id": result_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_filename": image_filename,
        "upload_metadata": metadata,
        "selected_plant": selected_plant,
        "prediction": prediction,
    }
    # The prediction is always the model's global 16-class output. The selected
    # plant is stored only so the result page can flag a mismatch.
    (SAVED_RESULTS_DIR / f"{result_id}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return redirect(url_for("result", result_id=result_id))


@app.route("/result/<result_id>")
def result(result_id: str):
    if not RESULT_ID_PATTERN.fullmatch(result_id):
        abort(404)
    payload = load_json(SAVED_RESULTS_DIR / f"{result_id}.json")
    if payload is None:
        abort(404)
    prediction = payload["prediction"]
    return render_template(
        "results.html",
        result=payload,
        prediction=prediction,
        page_title=f"{prediction['plant']} — {prediction['disease']} | LeafGuard AI",
        active_page="detect",
    )


@app.route("/performance")
def performance():
    metrics = model_metrics()
    plots = {
        Path(name).stem: (RESULTS_DIR / name).exists()
        for name in (
            "accuracy_curve.png",
            "loss_curve.png",
            "confusion_matrix.png",
            "class_f1_scores.png",
            "sample_predictions.png",
        )
    }
    report_rows = metrics.get("per_class", []) if metrics else []
    return render_template(
        "performance.html",
        metrics=metrics,
        plots=plots,
        report_rows=report_rows,
        page_title="Model performance — LeafGuard AI",
        active_page="performance",
    )


@app.route("/about")
def about():
    summary = dataset_summary()
    return render_template(
        "about.html",
        summary=summary,
        page_title="About the project — LeafGuard AI",
        active_page="about",
    )


@app.route("/results/<path:filename>")
def results_file(filename: str):
    if filename not in ALLOWED_RESULT_FILES:
        abort(404)
    return send_from_directory(RESULTS_DIR, filename, max_age=300)


@app.route("/health")
def health():
    onnx_path = MODEL_PATH.with_suffix(".onnx")
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists() or onnx_path.exists(),
        "inference_backend": "onnx" if onnx_path.exists() else "keras",
        "model_path": str(MODEL_PATH),
        "onnx_model_path": str(onnx_path),
        "classes": len(CLASS_NAMES),
        "plants": PLANTS,
    }


@app.errorhandler(413)
def file_too_large(_error):
    flash("That image is larger than the 8 MB upload limit.", "error")
    return redirect(url_for("detect"))


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "404.html",
        page_title="Page not found — LeafGuard AI",
        active_page="",
    ), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template(
        "500.html",
        page_title="Something went wrong — LeafGuard AI",
        active_page="",
    ), 500


if __name__ == "__main__":
    # Default to the usual Flask port, but allow PORT=5050 when another local
    # application already owns port 5000.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
