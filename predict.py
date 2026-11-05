"""Command-line prediction for one leaf image.

Examples
--------
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --plant Tomato
    python predict.py path/to/leaf.jpg --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.image_quality import ImageValidationError, open_image
from utils.predictor import ModelNotReadyError, get_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a leaf image with LeafGuard AI")
    parser.add_argument("image", type=Path, help="path to a JPG/PNG/WEBP leaf image")
    parser.add_argument(
        "--plant",
        choices=["Tomato", "Apple", "Grape", "Corn"],
        help="optional plant selected by the user; used as a consistency check",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        print(f"ERROR: file not found: {args.image}", file=sys.stderr)
        return 2
    try:
        image = open_image(args.image)
    except ImageValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = get_predictor().predict(image, selected_plant=args.plant)
    except ModelNotReadyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    status = "UNCERTAIN" if result["is_uncertain"] else "PREDICTION"
    print(f"\n{status}")
    print(f"  Plant      : {result['plant']}")
    print(f"  Condition  : {result['disease']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    if result["plant_mismatch"]:
        print(
            f"  Note       : user selected {result['selected_plant']}, "
            f"but the model predicts {result['plant']}."
        )
    print("\n  Top 3:")
    for item in result["top_predictions"]:
        print(f"    {item['name']:<32s} {item['probability']:6.2f}%")
    print("\n  Image quality:")
    quality = result["quality"]
    print(
        f"    {quality['width']}x{quality['height']} | brightness "
        f"{quality['brightness']:.1f} | focus {quality['focus_score']:.4f} | "
        f"score {quality['quality_score']:.0f}/100"
    )
    for warning in quality["warnings"]:
        print(f"    ! {warning}")
    if result["is_uncertain"]:
        print("\n  This result is marked uncertain. Do not treat it as a diagnosis.")
        for reason in result["uncertainty_reasons"]:
            print(f"    - {reason}")
    print(
        "\n  Reminder: confidence is not a guarantee of diagnostic accuracy. "
        "Consult an agricultural expert for treatment decisions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
