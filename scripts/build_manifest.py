"""Build dataset/download_manifest.csv from the downloaded raw images.

Records the exact source URL and SHA-256 hash of every downloaded file so the
subset can be audited and reproduced. Run after scripts/download_dataset.py.

    python scripts/build_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.labels import CLASSES  # noqa: E402

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
OUTPUT = PROJECT_ROOT / "dataset" / "download_manifest.csv"
REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    rows = []
    for item in CLASSES:
        folder = RAW_DIR / item["dataset_label"]
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"{item['name']:<34s} {len(files):4d}")
        for path in files:
            source_path = path.relative_to(PROJECT_ROOT).as_posix()
            github_path = f"raw/color/{item['dataset_label']}/{path.name}"
            url = (
                f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
                + urllib.parse.quote(github_path, safe="/")
            )
            rows.append(
                {
                    "plant": item["plant"],
                    "display_label": item["name"],
                    "source_class": item["dataset_label"],
                    "filename": path.name,
                    "relative_path": source_path,
                    "source_url": url,
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
