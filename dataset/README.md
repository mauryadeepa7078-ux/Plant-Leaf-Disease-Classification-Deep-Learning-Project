# Dataset directory

This directory is **not committed to Git** because it contains several
thousand photographs. Recreate it with the documented scripts.

## Layout after running the pipeline

```text
dataset/
├── raw/                       original PlantVillage JPEGs, one folder per class
├── processed_224/             224 × 224 RGB JPEG copies used by the model
├── splits/
│   ├── train.csv              70% of images
│   ├── validation.csv         15%
│   └── test.csv               15%
├── all_images.csv             full manifest with labels, split, group and hashes
├── class_distribution.csv     per-class train/validation/test counts
├── plant_distribution.csv     per-plant counts
├── dataset_summary.json       machine-readable summary used by the website
├── duplicate_report.csv       exact duplicates removed (empty when none)
├── download_manifest.csv      source URL and SHA-256 for each downloaded file
└── SOURCE.md                  dataset source, licence and class-folder verification
```

## Commands

```powershell
# Download the deterministic 250-images-per-class subset
python scripts/download_dataset.py

# Clean, resize, deduplicate and create the 70/15/15 splits
python dataset_prep.py
```

## Split rules

- 70% training, 15% validation, 15% testing.
- Stratified by class with `StratifiedGroupKFold(n_splits=20)`.
- Folds 0–13 = training, 14–16 = validation, 17–19 = testing.
- Perceptual near-duplicate groups are kept entirely within one split.
- Exact duplicates are removed before splitting.
- No test image is used for fitting, early stopping or checkpoint selection.

The prepared run in this workspace has **4,000 images**:
2,801 training, 600 validation and 599 test. Each class contributes 250 images.

See [SOURCE.md](SOURCE.md) for attribution and licence details.
