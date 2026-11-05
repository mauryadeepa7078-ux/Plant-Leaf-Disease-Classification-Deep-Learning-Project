"""Dataset split integrity tests (skipped until dataset_prep.py has been run)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "dataset"
SUMMARY = DATASET / "dataset_summary.json"
ALL_IMAGES = DATASET / "all_images.csv"

pytestmark = pytest.mark.skipif(
    not (SUMMARY.exists() and ALL_IMAGES.exists()),
    reason="Run dataset_prep.py to create dataset artefacts",
)


def load_rows():
    with ALL_IMAGES.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_summary_has_all_classes_and_plants():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["plant_types"] == 4
    assert summary["classes"] == 16
    assert summary["classes_per_plant"] == {"Tomato": 4, "Apple": 4, "Grape": 4, "Corn": 4}
    assert len(summary["images_per_class"]) == 16


def test_split_sizes_and_percentages():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = load_rows()
    assert summary["total_images"] == len(rows)
    counts = Counter(row["split"] for row in rows)
    assert counts["train"] + counts["validation"] + counts["test"] == len(rows)
    for split in ("train", "validation", "test"):
        percentage = counts[split] / len(rows)
        if split == "train":
            assert 0.65 <= percentage <= 0.75, (split, percentage)
        else:
            assert 0.12 <= percentage <= 0.18, (split, percentage)


def test_every_class_appears_in_every_split():
    rows = load_rows()
    table = defaultdict(set)
    for row in rows:
        table[row["class_name"]].add(row["split"])
    assert len(table) == 16
    for class_name, splits in table.items():
        assert splits == {"train", "validation", "test"}, class_name


def test_no_exact_duplicate_crosses_splits():
    rows = load_rows()
    splits_by_hash = defaultdict(set)
    for row in rows:
        splits_by_hash[row["pixel_hash"]].add(row["split"])
    leaks = {key: value for key, value in splits_by_hash.items() if len(value) > 1}
    assert not leaks


def test_no_group_crosses_splits():
    rows = load_rows()
    splits_by_group = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    leaks = {key: value for key, value in splits_by_group.items() if len(value) > 1}
    assert not leaks


def test_processed_images_are_rgb_224():
    rows = load_rows()
    # Sampling keeps the test fast while still catching resize/conversion errors.
    for row in rows[:: max(1, len(rows) // 40)]:
        image = Image.open(PROJECT_ROOT / row["processed_path"])
        assert image.size == (224, 224)
        assert image.mode == "RGB"
