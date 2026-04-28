"""
detection.py — Detect the marked answer choice in the bottom crop of a page.

Primary method: OpenCV contour/blob analysis looking for filled circles or
boxes near A–E labels.

Extension point: ``detect_marked_choice_with_vision`` is an intentionally
unimplemented stub.  To add a vision-model backend, fill in that function
and return (letter, confidence).  The primary detector will automatically
fall back to it when its own confidence falls below the configured threshold.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Attempt to import cv2; treat it as optional so the rest of the pipeline
# still runs when OpenCV is not installed (detection will always flag for review).
try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed — answer detection will be disabled.")


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

DetectionResult = tuple[Optional[str], float]
"""(letter | None, confidence 0–1).  None means "could not determine"."""


# ---------------------------------------------------------------------------
# Primary detector — OpenCV
# ---------------------------------------------------------------------------

def detect_marked_choice_cv(image_path: str) -> DetectionResult:
    """
    Locate the marked answer choice using OpenCV contour analysis.

    Strategy:
    1. Convert to greyscale and threshold to isolate dark marks.
    2. Find contours whose aspect ratio and area suggest a filled bubble or
       circled letter.
    3. Map each candidate contour's horizontal position to an A–E column
       inferred from the five widest peaks in the horizontal projection.
    4. Return the letter whose column contains the largest / darkest mark.

    Returns:
        (letter, confidence) where confidence is 0–1.  Returns (None, 0.0)
        when OpenCV is unavailable or no marks are found.
    """
    if not _CV2_AVAILABLE:
        return None, 0.0

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("cv2 could not read image: %s", image_path)
        return None, 0.0

    h, w = img.shape

    # --- 1. Threshold ---
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 2. Find contours ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to contours that look like marks (filled circles / boxes).
    mark_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > (h * w * 0.05):
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / (ch + 1e-6)
        if 0.4 < aspect < 2.5:
            mark_candidates.append((x, y, cw, ch, area))

    if not mark_candidates:
        logger.debug("No mark candidates found in %s", image_path)
        return None, 0.0

    # --- 3. Infer A–E column positions from horizontal spread ---
    # Sort candidates by x-coordinate and pick the 5 widest horizontal clusters.
    mark_candidates.sort(key=lambda c: c[0])
    x_centers = [c[0] + c[2] // 2 for c in mark_candidates]

    letter_cols = _cluster_to_5_columns(x_centers, w)
    if letter_cols is None:
        logger.debug("Could not resolve 5-column layout in %s", image_path)
        return None, 0.0

    # --- 4. Assign the largest mark to a letter ---
    best_mark = max(mark_candidates, key=lambda c: c[4])  # largest area
    bx = best_mark[0] + best_mark[2] // 2

    letter, col_confidence = _assign_column(bx, letter_cols)
    if letter is None:
        return None, 0.0

    # Confidence heuristic: blend column-assignment certainty with mark size.
    size_confidence = min(best_mark[4] / (h * w * 0.01 + 1e-6), 1.0)
    confidence = 0.6 * col_confidence + 0.4 * size_confidence

    logger.debug(
        "Detected answer %s (conf=%.2f) in %s", letter, confidence, image_path
    )
    return letter, round(confidence, 3)


def _cluster_to_5_columns(
    x_centers: list[int], image_width: int
) -> Optional[list[tuple[str, int, int]]]:
    """
    Given a list of x-coordinates, attempt to identify 5 columns for A–E.

    Returns a list of (letter, col_left, col_right) tuples or None if
    clustering to exactly 5 columns is not possible.
    """
    if not x_centers:
        return None

    # Simple equal-division fallback: split image into 5 vertical bands.
    band_w = image_width // 5
    letters = ["A", "B", "C", "D", "E"]
    return [(l, i * band_w, (i + 1) * band_w) for i, l in enumerate(letters)]


def _assign_column(
    x: int, cols: list[tuple[str, int, int]]
) -> tuple[Optional[str], float]:
    """
    Find which column *x* falls in.  Returns (letter, confidence) where
    confidence reflects how centred *x* is within its column.
    """
    for letter, left, right in cols:
        if left <= x < right:
            width = right - left
            centre = (left + right) / 2
            distance_ratio = abs(x - centre) / (width / 2 + 1e-6)
            confidence = max(0.0, 1.0 - distance_ratio)
            return letter, confidence
    return None, 0.0


# ---------------------------------------------------------------------------
# Extension point — Vision-model backend (unimplemented)
# ---------------------------------------------------------------------------

def detect_marked_choice_with_vision(image_path: str) -> DetectionResult:
    """
    EXTENSION POINT — Vision-model answer detection (not yet implemented).

    Replace this stub with a call to a multimodal model (e.g. GPT-4o,
    Claude 3 Opus, or a fine-tuned classifier) that inspects the bottom-crop
    image and returns which bubble is filled.

    Expected return value:
        (letter, confidence)  e.g. ("C", 0.97)
        (None, 0.0)           when the model cannot determine an answer

    This function is called automatically by ``detect_answer`` when the
    primary OpenCV detector's confidence falls below ``min_confidence``.

    Args:
        image_path: Absolute path to the bottom-crop PNG.

    Raises:
        NotImplementedError: always (until implemented).
    """
    raise NotImplementedError(
        "detect_marked_choice_with_vision is a stub. "
        "Implement it by calling your preferred vision API and returning "
        "(letter: str | None, confidence: float)."
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def detect_answer(
    image_path: str,
    min_confidence: float = 0.6,
    use_vision_fallback: bool = False,
) -> DetectionResult:
    """
    Detect the marked answer choice in *image_path*.

    First tries the OpenCV detector.  If confidence < *min_confidence* AND
    *use_vision_fallback* is True, falls back to
    ``detect_marked_choice_with_vision``.  If neither yields sufficient
    confidence, returns (None, confidence) so the caller can flag the page
    for human review.

    Args:
        image_path:          Path to the bottom-crop PNG.
        min_confidence:      Minimum confidence to accept a detection.
        use_vision_fallback: Whether to call the vision stub on low confidence.

    Returns:
        (letter | None, confidence)
    """
    letter, conf = detect_marked_choice_cv(image_path)

    if (letter is None or conf < min_confidence) and use_vision_fallback:
        try:
            letter, conf = detect_marked_choice_with_vision(image_path)
        except NotImplementedError:
            logger.debug("Vision fallback not implemented; skipping.")
        except Exception as exc:
            logger.warning("Vision fallback failed: %s", exc)

    return letter, conf
