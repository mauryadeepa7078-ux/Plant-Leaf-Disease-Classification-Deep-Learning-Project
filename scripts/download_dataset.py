"""Download a reproducible 16-class PlantVillage subset from the official GitHub mirror.

Why this script?
----------------
LeafGuard AI does not commit thousands of leaf photographs to the repository.
This script downloads a balanced, deterministic subset directly from the public
PlantVillage repository maintained by the dataset authors (spMohanty).

Source : https://github.com/spMohanty/PlantVillage-Dataset
Mirror : https://huggingface.co/datasets/mohanty/PlantVillage
License: CC BY-SA 3.0 (see dataset/SOURCE.md)
Paper  : Mohanty, Hughes & Salathe (2016), Frontiers in Plant Science 7:1419.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --per-class 250

The default is 250 images per class (4,000 images before cleaning). This keeps
training practical on a CPU while leaving a meaningful held-out test set. The
dataset preparation step removes corrupt images and exact/near duplicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "download_manifest.csv"
SOURCE_DOC_PATH = PROJECT_ROOT / "dataset" / "SOURCE.md"
LISTING_CACHE_DIR = PROJECT_ROOT / "dataset" / ".file_listings"

REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
GITHUB_API = "https://api.github.com"

# The 16 exact directory names from PlantVillage.  Do not rename or invent
# folders: these are the labels that actually exist in the public dataset.
CLASSES = [
    # plant, display label, PlantVillage source directory
    ("Tomato", "Tomato Healthy", "Tomato___healthy"),
    ("Tomato", "Tomato Early Blight", "Tomato___Early_blight"),
    ("Tomato", "Tomato Late Blight", "Tomato___Late_blight"),
    ("Tomato", "Tomato Bacterial Spot", "Tomato___Bacterial_spot"),
    ("Apple", "Apple Healthy", "Apple___healthy"),
    ("Apple", "Apple Scab", "Apple___Apple_scab"),
    ("Apple", "Apple Black Rot", "Apple___Black_rot"),
    ("Apple", "Apple Cedar Apple Rust", "Apple___Cedar_apple_rust"),
    ("Grape", "Grape Healthy", "Grape___healthy"),
    ("Grape", "Grape Black Rot", "Grape___Black_rot"),
    ("Grape", "Grape Esca", "Grape___Esca_(Black_Measles)"),
    ("Grape", "Grape Leaf Blight", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"),
    ("Corn", "Corn Healthy", "Corn_(maize)___healthy"),
    ("Corn", "Corn Common Rust", "Corn_(maize)___Common_rust_"),
    ("Corn", "Corn Northern Leaf Blight", "Corn_(maize)___Northern_Leaf_Blight"),
    (
        "Corn",
        "Corn Cercospora Leaf Spot",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def github_request(url: str) -> object:
    """GET a GitHub API URL with a polite User-Agent and simple retries."""
    headers = {
        "User-Agent": "LeafGuard-AI-academic-project",
        "Accept": "application/vnd.github+json",
    }
    token = None
    try:
        import os

        token = os.environ.get("GITHUB_TOKEN")
    except Exception:
        pass
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry all network failures
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GitHub API request failed: {url}\n{last_error}")


def list_class_images_html(source_dir: str) -> list[dict]:
    """List files from the public GitHub tree page (no API token required).

    GitHub embeds the folder's tree as JSON in the page.  Large folders expose
    the first 1,000 entries, which is still more than the 250 images sampled
    per class.  This route avoids the unauthenticated API rate limit while
    continuing to use the dataset authors' official repository.
    """
    page_url = (
        f"https://github.com/{REPO}/tree/{BRANCH}/raw/color/"
        + urllib.parse.quote(source_dir, safe="")
    )
    request = urllib.request.Request(
        page_url, headers={"User-Agent": "LeafGuard-AI-academic-project"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        html = response.read().decode("utf-8", "replace")

    marker = 'react-app.embeddedData">'
    start = html.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find embedded tree data for {source_dir}")
    start += len(marker)
    end = html.find("</script>", start)
    if end < 0:
        raise RuntimeError(f"Malformed GitHub page for {source_dir}")
    try:
        payload = json.loads(html[start:end])
        items = payload["payload"]["codeViewTreeRoute"]["tree"]["items"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse GitHub listing for {source_dir}") from exc

    files = []
    for item in items:
        name = item.get("name", "")
        if not name or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        path = item.get("path") or f"raw/color/{source_dir}/{name}"
        files.append(
            {
                "name": name,
                "path": path,
                "download_url": (
                    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
                    + urllib.parse.quote(path, safe="/")
                ),
            }
        )
    if not files:
        raise RuntimeError(f"No images found in GitHub folder for {source_dir}")
    return files


def list_class_images(source_dir: str) -> list[dict]:
    """Return a cached deterministic listing for one PlantVillage class."""
    LISTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LISTING_CACHE_DIR / (
        hashlib.sha1(source_dir.encode("utf-8")).hexdigest() + ".json"
    )
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) >= 200:
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    files = list_class_images_html(source_dir)
    cache_path.write_text(json.dumps(files), encoding="utf-8")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(item: dict, destination: Path) -> tuple[bool, str]:
    """Download one image atomically, returning (downloaded, error)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        return False, ""
    url = item.get("download_url")
    if not url:
        return False, "missing download_url"
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error = ""
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "LeafGuard-AI-academic-project"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if len(data) < 1024 or not data.startswith(b"\xff\xd8"):
                raise ValueError(f"not a valid JPEG ({len(data)} bytes)")
            tmp.write_bytes(data)
            tmp.replace(destination)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            tmp.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    return False, last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=250,
        help="number of images to download per class (default: 250)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_class < 200:
        parser.error("Use at least 200 images per class for a meaningful split.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict, Path, str, str]] = []
    selected_summary = {}

    print(f"Reading class listings from GitHub: {REPO}")
    for index, (plant, label, source_dir) in enumerate(CLASSES):
        files = list_class_images(source_dir)
        if len(files) < args.per_class:
            print(
                f"ERROR: {source_dir} only exposes {len(files)} files; "
                f"need {args.per_class}. Stop instead of silently replacing the class."
            )
            return 2
        files = sorted(files, key=lambda item: item["name"])
        rng = random.Random(args.seed + index)
        chosen = rng.sample(files, args.per_class)
        selected_summary[source_dir] = {
            "plant": plant,
            "display_label": label,
            "available_in_repo": len(files),
            "selected": len(chosen),
        }
        print(f"  {plant:6s} {label:32s} {len(files):4d} available -> {len(chosen)} selected")
        for item in chosen:
            jobs.append((item, RAW_DIR / source_dir / item["name"], plant, label))

    print(f"\nDownloading {len(jobs)} images to {RAW_DIR} ...")
    failures: list[tuple[str, str]] = []
    downloaded = skipped = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(download_one, item, path): (item, path, plant, label)
            for item, path, plant, label in jobs
        }
        for count, future in enumerate(as_completed(futures), start=1):
            item, path, plant, label = futures[future]
            try:
                was_downloaded, error = future.result()
            except Exception as exc:  # noqa: BLE001
                was_downloaded, error = False, str(exc)
            if error:
                failures.append((str(path), error))
            elif was_downloaded:
                downloaded += 1
            else:
                skipped += 1
            if count % 100 == 0 or count == len(jobs):
                elapsed = time.time() - started
                print(
                    f"  {count}/{len(jobs)} | new {downloaded} | existing {skipped} "
                    f"| failed {len(failures)} | {elapsed:.0f}s"
                )

    if failures:
        print("\nSome downloads failed:")
        for path, error in failures[:20]:
            print(f"  {path}: {error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    # Write a manifest with hashes for reproducibility and duplicate auditing.
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, path, plant, label in jobs:
        rows.append(
            {
                "plant": plant,
                "display_label": label,
                "source_class": path.parent.name,
                "filename": path.name,
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_url": item.get("download_url", ""),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written to {MANIFEST_PATH}")

    SOURCE_DOC_PATH.write_text(
        """# Dataset source and license

## PlantVillage

- **Dataset:** PlantVillage (color, unsegmented images)
- **Authors:** David P. Hughes and Marcel Salathe; dataset assembled by
  Sharada P. Mohanty, David P. Hughes and Marcel Salathe (EPFL / Penn State).
- **Official repository:** https://github.com/spMohanty/PlantVillage-Dataset
- **Hugging Face mirror:** https://huggingface.co/datasets/mohanty/PlantVillage
- **Paper:** Mohanty, S. P., Hughes, D. P., & Salathe, M. (2016).
  *Using Deep Learning for Image-Based Plant Disease Detection.*
  Frontiers in Plant Science, 7, 1419.
  https://doi.org/10.3389/fpls.2016.01419
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported
  (CC BY-SA 3.0), as declared by the dataset mirror:
  https://huggingface.co/datasets/mohanty/PlantVillage

This project uses a deterministic subset of **250 images per class** (4,000
images before cleaning) from the 16 PlantVillage class folders required by the
brief. The exact files, their SHA-256 hashes and source URLs are recorded in
`dataset/download_manifest.csv`.

> **Attribution / share-alike:** PlantVillage images remain under CC BY-SA 3.0.
> If you redistribute the images or derived dataset, retain the attribution
> above and share adaptations under the same license. The source code of
> LeafGuard AI is a separate work; check the repository LICENSE for its terms.

PlantVillage images were collected under controlled conditions with leaves
placed against plain backgrounds. This is useful for a teaching project but
does not represent the diversity of real farm photographs.

## Class-folder note

The PlantVillage folder `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
combines two visually related corn foliar diseases (Cercospora leaf spot and
gray leaf spot). LeafGuard AI reports this class as **Corn Cercospora Leaf
Spot**, matching the brief, while the About page explains that the public
dataset groups it with gray leaf spot. No class was invented or substituted.
""",
        encoding="utf-8",
    )
    print(f"Source/license notes written to {SOURCE_DOC_PATH}")

    summary_path = PROJECT_ROOT / "dataset" / "download_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": f"https://github.com/{REPO}",
                "license": "CC BY-SA 3.0",
                "seed": args.seed,
                "per_class": args.per_class,
                "classes": selected_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Download summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
