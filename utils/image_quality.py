"""Image loading, validation and simple quality checks.

These checks are deliberately conservative.  They catch unusable uploads
(tiny files, extreme darkness, almost no detail) without pretending to be a
full out-of-distribution detector.  The model's confidence is checked
separately in ``utils.predictor``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .labels import IMAGE_SIZE

# Guard against decompression-bomb images while still allowing modern phone
# photographs (which are commonly 12-48 megapixels).
Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_SIDE = 100

# Calibrated for 0-1 grayscale Laplacian variance; sharp leaf photos usually
# score far above this.  It is intentionally low so that it only catches
# severely out-of-focus uploads.
FOCUS_WARN_THRESHOLD = 0.0009
CONTRAST_WARN_THRESHOLD = 12.0
DARK_WARN_THRESHOLD = 35.0
BRIGHT_WARN_THRESHOLD = 230.0
LEAF_COLOUR_WARN_THRESHOLD = 0.08


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be treated as a leaf photograph."""


def _validate_upload_metadata(filename: str, content_type: str | None, size: int) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            "Unsupported file type. Please upload a JPG, PNG or WEBP image."
        )
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        # Some browsers send application/octet-stream; the extension and PIL
        # decoding below still provide a meaningful check, so only reject a
        # clearly wrong image MIME type.
        if content_type.lower().startswith("image/"):
            raise ImageValidationError("Unsupported image format. Use JPG, PNG or WEBP.")
    if size <= 0:
        raise ImageValidationError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise ImageValidationError("The image is larger than the 8 MB upload limit.")


def open_image(source: str | Path | bytes | io.BytesIO) -> Image.Image:
    """Open an image safely, honour EXIF rotation and convert it to RGB."""
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        with Image.open(source) as opened:
            opened.verify()
        if isinstance(source, io.BytesIO):
            source.seek(0)
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "The file could not be read as an image. It may be corrupt or not a real photo."
        ) from exc


def validate_upload(
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Validate uploaded bytes and return a decoded RGB image plus metadata."""
    _validate_upload_metadata(filename, content_type, len(file_bytes))
    image = open_image(file_bytes)
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise ImageValidationError(
            f"Image is too small ({width}x{height}). Please use an image of at least "
            f"{MIN_SIDE}x{MIN_SIDE} pixels."
        )
    if max(width, height) > 12000:
        raise ImageValidationError("Image dimensions are unexpectedly large.")
    metadata = {
        "filename": Path(filename).name,
        "width": width,
        "height": height,
        "mode": "RGB",
        "bytes": len(file_bytes),
    }
    return image, metadata


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian, a lightweight focus measure."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    centre = gray[1:-1, 1:-1]
    lap = (
        -4.0 * centre
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def image_quality_metrics(image: Image.Image) -> dict[str, Any]:
    """Return interpretable quality measurements and warning flags."""
    width, height = image.size
    working = image.copy()
    working.thumbnail((512, 512), Image.Resampling.LANCZOS)
    array = np.asarray(working.convert("RGB"), dtype=np.float32)
    gray = np.asarray(working.convert("L"), dtype=np.float32) / 255.0

    brightness = float(gray.mean() * 255.0)
    contrast = float(gray.std() * 255.0)
    focus = _laplacian_variance(gray)

    # Hasler-Susstrunk colourfulness: low values often mean a grey/blank image.
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # Fraction of pixels in green/yellow/brown hue ranges with usable
    # saturation. PlantVillage leaves score well above 0.15; a grey blank or
    # a screenshot scores near zero. This is only a coarse sanity check, not a
    # botanical segmentation.
    hsv = np.asarray(working.convert("HSV"), dtype=np.float32)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf_mask = (
        (hue >= 15)
        & (hue <= 135)
        & (saturation >= 45)
        & (value >= 20)
        & (value <= 250)
    )
    leaf_colour_ratio = float(leaf_mask.mean())

    warnings: list[str] = []
    if min(width, height) < 220:
        warnings.append("Low resolution: the leaf may not fill enough of the frame.")
    if brightness < DARK_WARN_THRESHOLD:
        warnings.append("The image is very dark; retake it in softer, even light.")
    if brightness > BRIGHT_WARN_THRESHOLD:
        warnings.append("The image is very bright; avoid direct glare or overexposure.")
    if contrast < CONTRAST_WARN_THRESHOLD:
        warnings.append("The image has little tonal contrast; the leaf may be hard to see.")
    if focus < FOCUS_WARN_THRESHOLD:
        warnings.append("The image looks blurry or out of focus.")
    if colorfulness < 3.0:
        warnings.append("Very little colour was detected; check that the photo shows a leaf.")
    if leaf_colour_ratio < LEAF_COLOUR_WARN_THRESHOLD:
        warnings.append(
            "Very little leaf-like colour was detected. Make sure the image shows "
            "a leaf rather than a screenshot, document or unrelated object."
        )

    critical_flags = {
        "too_blurry": focus < FOCUS_WARN_THRESHOLD,
        "too_dark": brightness < DARK_WARN_THRESHOLD,
        "too_bright": brightness > BRIGHT_WARN_THRESHOLD,
        "too_flat": contrast < CONTRAST_WARN_THRESHOLD,
        "too_small": min(width, height) < MIN_SIDE,
        "not_leaf_like": leaf_colour_ratio < LEAF_COLOUR_WARN_THRESHOLD,
    }
    usable = not any(critical_flags.values())

    # A transparent 0-100 presentation score.  It is a quality aid, not a
    # diagnostic probability.
    score = 100.0
    score -= 20 if critical_flags["too_blurry"] else 0
    score -= 15 if critical_flags["too_dark"] or critical_flags["too_bright"] else 0
    score -= 10 if critical_flags["too_flat"] else 0
    score -= 18 if critical_flags["not_leaf_like"] else 0
    score -= 10 if min(width, height) < 220 else 0
    score = max(0.0, min(100.0, score))

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "focus_score": round(focus, 5),
        "colorfulness": round(colorfulness, 2),
        "leaf_colour_ratio": round(leaf_colour_ratio, 4),
        "quality_score": round(score, 1),
        "usable": usable,
        "warnings": warnings,
        "flags": critical_flags,
    }


def prepare_model_input(image: Image.Image) -> np.ndarray:
    """Resize exactly like training and return a raw 0-255 RGB float tensor."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)
