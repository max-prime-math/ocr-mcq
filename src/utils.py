"""
utils.py — Shared helpers: PDF rendering, image cropping, review CSV/JSON I/O.
"""

import csv
import json
import logging
import os
import tempfile
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
