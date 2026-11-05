"""Prepare the PlantVillage leaf dataset for LeafGuard AI.

Pipeline
--------
1. Verify that all 16 promised PlantVillage folders exist (never invent labels).
2. Read every image, reject corrupt files and convert working copies to RGB.
3. Remove exact duplicates using a hash of decoded pixels.
4. Group near-duplicates (perceptual dHash) so related photos stay in one split.
5. Resize every kept image to 224x224 and save a clean JPEG copy.
6. Split 70/15/15 with StratifiedGroupKFold, then write CSV manifests and a
   detailed dataset summary.

Run:
    python dataset_prep.py
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from utils.image_quality import open_image
from utils.labels import CLASSES, PLANTS, PLANT_CLASSES
from utils.paths import (
    DATASET_DIR,
    DATASET_SUMMARY_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    QUARANTINE_DIR,
    RAW_DIR,
    SPLITS_DIR,
)

RANDOM_SEED = 42
N_SPLITS = 20
TRAIN_FOLDS = set(range(0, 14))       # 14/20 = 70%
VAL_FOLDS = set(range(14, 17))        # 3/20 = 15%
TEST_FOLDS = set(range(17, 20))       # 3/20 = 15%
NEAR_DUPLICATE_DISTANCE = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DatasetError(RuntimeError):
    """Raised when the downloaded dataset does not match the promised classes."""


def discover_classes() -> dict[str, list[Path]]:
    """Return source folder -> sorted list of image paths, or fail clearly."""
    if not RAW_DIR.exists():
        raise DatasetError(
            f"Raw dataset folder not found: {RAW_DIR}\n"
            "Run `python scripts/download_dataset.py` first."
        )

    found: dict[str, list[Path]] = {}
    missing: list[str] = []
    for item in CLASSES:
        folder = RAW_DIR / item["dataset_label"]
        if not folder.is_dir():
            missing.append(item["dataset_label"])
            continue
        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            missing.append(item["dataset_label"])
        found[item["dataset_label"]] = paths

    if missing:
        raise DatasetError(
            "Missing or empty PlantVillage class folders:\n  - "
            + "\n  - ".join(missing)
            + "\nDownload the official dataset instead of substituting a class."
        )

    total = sum(len(paths) for paths in found.values())
    print(f"Verified {len(found)} class folders with {total} raw images.")
    return found


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """64-bit perceptual difference hash (robust to resizing/compression)."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def exact_hash(image: Image.Image) -> str:
    """Hash decoded pixels so files with different metadata still collide."""
    digest = hashlib.sha256()
    digest.update(b"RGB")
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def collect_images(found: dict[str, list[Path]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Validate, deduplicate and hash raw images. Returns records and reports."""
    records: list[dict] = []
    corrupt: list[dict] = []
    exact_duplicates: list[dict] = []
    seen_hashes: dict[str, dict] = {}
    total = sum(len(paths) for paths in found.values())
    processed_count = 0

    for item in CLASSES:
        label = item["dataset_label"]
        class_records: list[dict] = []
        for path in found[label]:
            processed_count += 1
            try:
                image = open_image(path)
                image.load()
            except Exception as exc:  # noqa: BLE001 - report every corrupt file
                corrupt.append(
                    {
                        "class_name": item["name"],
                        "dataset_label": label,
                        "filename": path.name,
                        "error": str(exc)[:300],
                    }
                )
                continue

            pixel_hash = exact_hash(image)
            if pixel_hash in seen_hashes:
                first = seen_hashes[pixel_hash]
                exact_duplicates.append(
                    {
                        "class_name": item["name"],
                        "dataset_label": label,
                        "filename": path.name,
                        "duplicate_of": first["filename"],
                        "duplicate_of_class": first["class_name"],
                        "pixel_hash": pixel_hash,
                        "cross_class": first["class_index"] != item["index"],
                    }
                )
                continue

            record = {
                "image_id": f"{item['index']:02d}_{len(class_records):05d}",
                "class_index": item["index"],
                "class_name": item["name"],
                "dataset_label": label,
                "plant": item["plant"],
                "disease": item["disease"],
                "healthy": item["healthy"],
                "source_path": path,
                "filename": path.name,
                "width": image.width,
                "height": image.height,
                "pixel_hash": pixel_hash,
                "dhash": difference_hash(image),
                "group_id": "",
            }
            seen_hashes[pixel_hash] = record
            class_records.append(record)
            records.append(record)

            if processed_count % 500 == 0 or processed_count == total:
                print(
                    f"  inspected {processed_count}/{total} "
                    f"({len(records)} unique, {len(corrupt)} corrupt, "
                    f"{len(exact_duplicates)} duplicate)"
                )

        print(f"  {item['name']:<32s} {len(class_records):4d} unique images")

    return records, corrupt, exact_duplicates


def assign_near_duplicate_groups(records: list[dict]) -> list[dict]:
    """Union-find within each class using perceptual hash distance."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["class_index"]].append(record)

    group_summaries: list[dict] = []
    next_group = 0
    for class_index in sorted(by_class):
        items = by_class[class_index]
        union = UnionFind(len(items))
        compare_count = 0
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                compare_count += 1
                if hamming(items[left]["dhash"], items[right]["dhash"]) <= NEAR_DUPLICATE_DISTANCE:
                    union.union(left, right)
        roots: dict[int, list[int]] = defaultdict(list)
        for index in range(len(items)):
            roots[union.find(index)].append(index)
        for members in roots.values():
            group_id = f"g{next_group:05d}"
            next_group += 1
            for index in members:
                items[index]["group_id"] = group_id
            if len(members) > 1:
                group_summaries.append(
                    {
                        "group_id": group_id,
                        "class_name": items[0]["class_name"],
                        "images": len(members),
                        "filenames": "; ".join(items[index]["filename"] for index in members),
                    }
                )
        print(
            f"  near-duplicate scan: {items[0]['class_name']:<32s} "
            f"{compare_count:6d} pairs, {len(roots):4d} groups"
        )
    return group_summaries


def write_processed_images(records: list[dict]) -> None:
    """Create 224x224 RGB JPEG copies used by training and inference."""
    if PROCESSED_DIR.exists():
        # Safety: only delete a directory inside this project.
        resolved = PROCESSED_DIR.resolve()
        if PROJECT_ROOT.resolve() not in resolved.parents:
            raise DatasetError(f"Refusing to delete unexpected path: {resolved}")
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        destination_dir = PROCESSED_DIR / record["dataset_label"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{Path(record['filename']).stem}.jpg"
        if destination.exists():
            destination = destination_dir / f"{Path(record['filename']).stem}_{record['image_id']}.jpg"
        image = open_image(record["source_path"]).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=92, optimize=True)
        record["processed_path"] = destination
        if index % 500 == 0 or index == len(records):
            print(f"  wrote {index}/{len(records)} processed images")


def stratified_group_split(records: list[dict]) -> None:
    """Assign 70/15/15 folds while keeping near-duplicate groups together."""
    labels = np.array([record["class_index"] for record in records])
    groups = np.array([record["group_id"] for record in records])
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    folds = np.full(len(records), -1, dtype=int)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups)
    ):
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise DatasetError("Split generation failed to assign every image to a fold.")

    for record, fold in zip(records, folds):
        if int(fold) in TRAIN_FOLDS:
            record["split"] = "train"
        elif int(fold) in VAL_FOLDS:
            record["split"] = "validation"
        elif int(fold) in TEST_FOLDS:
            record["split"] = "test"
        else:  # pragma: no cover - defensive
            raise DatasetError(f"Unexpected fold {fold}")
        record["fold"] = int(fold)


def check_leakage(records: list[dict]) -> dict:
    """Verify that duplicate groups and exact hashes never cross splits."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["group_id"]].add(record["split"])
        hash_splits[record["pixel_hash"]].add(record["split"])

    bad_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    bad_hashes = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    return {
        "group_leaks": len(bad_groups),
        "exact_hash_leaks": len(bad_hashes),
        "bad_groups": list(bad_groups.items())[:10],
        "bad_hashes": list(bad_hashes.items())[:10],
        "passed": not bad_groups and not bad_hashes,
    }


def build_summary(
    records: list[dict],
    raw_total: int,
    corrupt: list[dict],
    duplicates: list[dict],
    near_groups: list[dict],
    leakage: dict,
) -> dict:
    frame = pd.DataFrame(records)
    per_class_split = (
        frame.groupby(["class_name", "dataset_label", "plant", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for split in ("train", "validation", "test"):
        if split not in per_class_split:
            per_class_split[split] = 0
    per_class_split["total"] = per_class_split[["train", "validation", "test"]].sum(axis=1)

    images_per_class = {
        row["class_name"]: int(row["total"])
        for _, row in per_class_split.iterrows()
    }
    split_counts = frame["split"].value_counts().to_dict()
    total = len(records)
    imbalance_ratio = max(images_per_class.values()) / min(images_per_class.values())

    # Write compact CSV summaries.
    distribution = per_class_split[
        ["plant", "class_name", "dataset_label", "train", "validation", "test", "total"]
    ].sort_values(["plant", "class_name"])
    distribution.to_csv(DATASET_DIR / "class_distribution.csv", index=False)

    plant_counts = (
        frame.groupby(["plant", "split"]).size().unstack(fill_value=0)
    )
    for split in ("train", "validation", "test"):
        if split not in plant_counts:
            plant_counts[split] = 0
    plant_counts["total"] = plant_counts[["train", "validation", "test"]].sum(axis=1)
    plant_counts.to_csv(DATASET_DIR / "plant_distribution.csv")

    if duplicates:
        pd.DataFrame(duplicates).to_csv(
            DATASET_DIR / "duplicate_report.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "class_name",
                "dataset_label",
                "filename",
                "duplicate_of",
                "duplicate_of_class",
                "pixel_hash",
                "cross_class",
            ]
        ).to_csv(DATASET_DIR / "duplicate_report.csv", index=False)

    summary = {
        "dataset_name": "PlantVillage 16-class subset",
        "source": "https://github.com/spMohanty/PlantVillage-Dataset",
        "license": "CC BY-SA 3.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "image_size": [224, 224],
        "raw_images_found": raw_total,
        "corrupt_images_removed": len(corrupt),
        "exact_duplicates_removed": len(duplicates),
        "near_duplicate_groups": len(near_groups),
        "near_duplicate_images_grouped": int(sum(item["images"] for item in near_groups)),
        "total_images": total,
        "plant_types": len(PLANTS),
        "classes": len(CLASSES),
        "classes_per_plant": {plant: len(PLANT_CLASSES[plant]) for plant in PLANTS},
        "images_per_class": images_per_class,
        "split_counts": {key: int(split_counts.get(key, 0)) for key in ("train", "validation", "test")},
        "split_percentages": {
            key: round(100.0 * split_counts.get(key, 0) / total, 2)
            for key in ("train", "validation", "test")
        },
        "plant_distribution": {
            plant: {key: int(plant_counts.loc[plant, key]) for key in ("train", "validation", "test", "total")}
            for plant in plant_counts.index
        },
        "class_imbalance_ratio": round(imbalance_ratio, 3),
        "class_imbalance_note": (
            "Balanced (max/min <= 1.25)."
            if imbalance_ratio <= 1.25
            else "Imbalance detected; class weights are used during training."
        ),
        "duplicate_check": leakage,
        "corrupt_files": corrupt,
        "split_method": (
            "StratifiedGroupKFold(n_splits=20); folds 0-13 train, 14-16 validation, "
            "17-19 test; near-duplicate groups kept together."
        ),
    }
    return summary


def main() -> int:
    print("=" * 72)
    print("LeafGuard AI - dataset preparation")
    print("=" * 72)
    found = discover_classes()
    raw_total = sum(len(paths) for paths in found.values())
    records, corrupt, duplicates = collect_images(found)

    if len(records) < 16 * 40:
        raise DatasetError(
            f"Only {len(records)} usable images; this is too few for a 16-class split."
        )

    print("\nGrouping near-duplicate images ...")
    near_groups = assign_near_duplicate_groups(records)

    print("\nWriting 224x224 RGB images ...")
    write_processed_images(records)

    print("\nCreating stratified group split ...")
    random.seed(RANDOM_SEED)
    stratified_group_split(records)

    leakage = check_leakage(records)
    if not leakage["passed"]:
        raise DatasetError(
            "Leakage check failed; duplicate groups or exact hashes cross splits. "
            f"{leakage['bad_groups'][:3]}"
        )

    frame = pd.DataFrame(
        [
            {
                "image_id": record["image_id"],
                "class_index": record["class_index"],
                "class_name": record["class_name"],
                "dataset_label": record["dataset_label"],
                "plant": record["plant"],
                "disease": record["disease"],
                "split": record["split"],
                "fold": record["fold"],
                "group_id": record["group_id"],
                "dhash": record["dhash"],
                "pixel_hash": record["pixel_hash"],
                "source_path": record["source_path"].relative_to(PROJECT_ROOT).as_posix(),
                "processed_path": record["processed_path"].relative_to(PROJECT_ROOT).as_posix(),
                "original_filename": record["filename"],
            }
            for record in records
        ]
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATASET_DIR / "all_images.csv", index=False)
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split].sort_values("image_id")
        split_frame.to_csv(SPLITS_DIR / f"{split}.csv", index=False)
        print(f"  {split:<10s}: {len(split_frame)} images")

    summary = build_summary(
        records, raw_total, corrupt, duplicates, near_groups, leakage
    )
    DATASET_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nDataset summary")
    print(f"  total usable images : {summary['total_images']}")
    print(f"  plants / classes    : {summary['plant_types']} / {summary['classes']}")
    print(f"  train/val/test      : {summary['split_counts']}")
    print(f"  corrupt removed     : {summary['corrupt_images_removed']}")
    print(f"  exact duplicates    : {summary['exact_duplicates_removed']}")
    print(f"  leakage check       : {'PASS' if leakage['passed'] else 'FAIL'}")
    print(f"  summary file        : {DATASET_SUMMARY_PATH}")

    if corrupt:
        quarantine = QUARANTINE_DIR / "corrupt"
        quarantine.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(corrupt).to_csv(quarantine / "corrupt_files.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
