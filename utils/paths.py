"""Project-wide paths and a tiny local-vendor bootstrap.

The normal installation uses ``pip install -r requirements.txt``.  The vendor
folder is only used by this development machine where pandas was repaired
locally; it has no effect when the folder does not exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed_224"
SPLITS_DIR = DATASET_DIR / "splits"
QUARANTINE_DIR = DATASET_DIR / "quarantine"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"
INSTANCE_DIR = PROJECT_ROOT / "instance"
SAVED_RESULTS_DIR = INSTANCE_DIR / "results"

MODEL_PATH = MODELS_DIR / "leafguard_mobilenetv2.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
DATASET_SUMMARY_PATH = DATASET_DIR / "dataset_summary.json"

ALL_DIRS = [
    DATASET_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    STATIC_DIR,
    UPLOADS_DIR,
    INSTANCE_DIR,
    SAVED_RESULTS_DIR,
]


def ensure_directories() -> None:
    """Create runtime directories if they are missing."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
