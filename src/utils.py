"""
utils.py — Shared helpers: PDF rendering, image cropping, review CSV/JSON I/O.
"""

import csv
import json
import logging
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Attempt PyMuPDF import.
try:
    import fitz  # PyMuPDF

    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.error("PyMuPDF (fitz) not installed — PDF rendering unavailable.")

try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# PDF → image
# ---------------------------------------------------------------------------

def render_page_to_image(pdf_path: str, page_index: int, dpi: int = 300) -> Image.Image:
    """
    Render page *page_index* (0-based) of *pdf_path* to a PIL Image.

    Args:
        pdf_path:   Path to the PDF file.
        page_index: 0-based page number.
        dpi:        Rendering resolution (300 recommended for OCR quality).

    Returns:
        PIL.Image in RGB mode.

    Raises:
        ImportError: if PyMuPDF is not installed.
        IndexError:  if *page_index* is out of range.
    """
    if not _FITZ_AVAILABLE:
        raise ImportError("PyMuPDF is required for PDF rendering.")

    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        raise IndexError(f"Page {page_index} out of range (document has {len(doc)} pages).")

    page = doc.load_page(page_index)
    zoom = dpi / 72.0  # PDF native resolution is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def page_count(pdf_path: str) -> int:
    """Return the number of pages in *pdf_path*."""
    if not _FITZ_AVAILABLE:
        raise ImportError("PyMuPDF is required.")
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


# ---------------------------------------------------------------------------
# Image cropping
# ---------------------------------------------------------------------------

def crop_top(image: Image.Image, fraction: float = 0.5) -> Image.Image:
    """Return the top *fraction* of *image*."""
    w, h = image.size
    return image.crop((0, 0, w, int(h * fraction)))


def crop_bottom(image: Image.Image, start_fraction: float = 0.5) -> Image.Image:
    """Return the portion of *image* below *start_fraction* of its height."""
    w, h = image.size
    return image.crop((0, int(h * start_fraction), w, h))


def save_temp_image(image: Image.Image, suffix: str = ".png") -> str:
    """
    Save *image* to a temporary file and return its path.

    The caller is responsible for deleting the file when done.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    image.save(path, format="PNG")
    return path


def _refine_figure_crop(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Shrink a loose figure box around the dominant non-text content inside it.

    This is mainly a safeguard for model outputs that return a very large box,
    including the full page. If refinement cannot confidently isolate a
    smaller region, the original box is returned.
    """
    left, top, right, bottom = box
    region = image.crop(box)
    rw, rh = region.size
    if rw < 40 or rh < 40 or not _CV2_AVAILABLE:
        return box

    region_np = np.array(region.convert("L"))
    _, thresh = cv2.threshold(region_np, 235, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(200, int(rw * rh * 0.0015))
    candidates: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:
            continue
        # Ignore long thin contours that are usually text rules/underlines.
        aspect = w / max(h, 1)
        if aspect > 18 or aspect < 0.05:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return box

    x1 = min(x for x, y, w, h in candidates)
    y1 = min(y for x, y, w, h in candidates)
    x2 = max(x + w for x, y, w, h in candidates)
    y2 = max(y + h for x, y, w, h in candidates)

    refined_area = max(1, (x2 - x1) * (y2 - y1))
    original_area = max(1, rw * rh)

    # If refinement does not materially shrink the crop, keep the original.
    if refined_area > original_area * 0.92:
        return box

    pad_x = max(8, int((x2 - x1) * 0.04))
    pad_y = max(8, int((y2 - y1) * 0.04))
    new_left = left + max(0, x1 - pad_x)
    new_top = top + max(0, y1 - pad_y)
    new_right = left + min(rw, x2 + pad_x)
    new_bottom = top + min(rh, y2 + pad_y)
    return new_left, new_top, new_right, new_bottom


def materialise_figures(
    figures: list[dict],
    page_images: list[Image.Image],
    figures_dir: str,
    stem: str,
) -> list[dict]:
    """
    Crop figure boxes from rendered page images and persist them.

    Returns a new list of figure dicts including ``latex_path``.
    """
    out: list[dict] = []
    base_dir = Path(figures_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for idx, fig in enumerate(figures, start=1):
        page_no = int(fig.get("page", 1))
        if page_no < 1 or page_no > len(page_images):
            logger.warning("Skipping figure with invalid page reference: %s", fig)
            continue

        image = page_images[page_no - 1]
        w, h = image.size
        x = max(0.0, min(1.0, float(fig.get("x", 0.0))))
        y = max(0.0, min(1.0, float(fig.get("y", 0.0))))
        width = max(0.0, min(1.0 - x, float(fig.get("width", 0.0))))
        height = max(0.0, min(1.0 - y, float(fig.get("height", 0.0))))

        if width <= 0 or height <= 0:
            logger.warning("Skipping zero-sized figure crop: %s", fig)
            continue

        left = int(w * x)
        top = int(h * y)
        right = int(w * (x + width))
        bottom = int(h * (y + height))
        left, top, right, bottom = _refine_figure_crop(image, (left, top, right, bottom))
        cropped = image.crop((left, top, right, bottom))

        filename = f"{stem}_fig_{idx}.png"
        path = base_dir / filename
        cropped.save(path, format="PNG")

        saved = dict(fig)
        saved["latex_path"] = f"figures/{filename}"
        out.append(saved)

    return out


def build_zip_bundle(tex_content: str, figures_dir: str, tex_name: str = "output.tex") -> bytes:
    """Return a zip archive containing the TeX file and any extracted figures."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(tex_name, tex_content)
        base_dir = Path(figures_dir)
        if base_dir.exists():
            for path in sorted(base_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(base_dir.parent)))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Review CSV
# ---------------------------------------------------------------------------

REVIEW_CSV_FIELDS = [
    "filename",
    "page",
    "detected_answer",
    "confidence",
    "notes",
]


def write_review_row(
    csv_path: str,
    filename: str,
    page: int,
    detected_answer: Optional[str],
    confidence: float,
    notes: str = "",
) -> None:
    """Append one row to the review CSV, creating it with a header if needed."""
    path = Path(csv_path)
    needs_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "filename": filename,
                "page": page,
                "detected_answer": detected_answer or "",
                "confidence": f"{confidence:.3f}",
                "notes": notes,
            }
        )


# ---------------------------------------------------------------------------
# Corrections JSON
# ---------------------------------------------------------------------------

def load_corrections(corrections_path: str) -> dict:
    """
    Load corrections.json and return a dict keyed by ``"filename:page"``.

    Returns an empty dict if the file does not exist.
    """
    path = Path(corrections_path)
    if not path.exists():
        return {}
    with open(corrections_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_correction(corrections_path: str, filename: str, page: int, letter: str) -> None:
    """Upsert one correction into corrections.json."""
    corrections = load_corrections(corrections_path)
    key = f"{filename}:{page}"
    corrections[key] = letter.upper()
    Path(corrections_path).parent.mkdir(parents=True, exist_ok=True)
    with open(corrections_path, "w", encoding="utf-8") as fh:
        json.dump(corrections, fh, indent=2)


def get_correction(corrections: dict, filename: str, page: int) -> Optional[str]:
    """Look up a human correction for (filename, page), or return None."""
    return corrections.get(f"{filename}:{page}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load JSON config from *config_path*.  Returns {} if file is absent."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
