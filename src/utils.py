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
    region_area = max(1, rw * rh)
    noise_area = max(18, int(region_area * 0.00003))
    text_height = max(18, int(rh * 0.07))
    text_width = max(140, int(rw * 0.28))
    graphic_area = max(240, int(region_area * 0.0005))
    long_span = max(40, int(max(rw, rh) * 0.12))
    gap = max(18, int(max(rw, rh) * 0.035))
    label_gap = max(12, int(max(rw, rh) * 0.025))

    components: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < noise_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue

        fill = area / max(1, w * h)
        is_long_rule = (w >= long_span and h <= max(14, int(rh * 0.025))) or (
            h >= long_span and w <= max(14, int(rw * 0.025))
        )
        is_text_like = (
            h <= text_height
            and w <= text_width
            and area <= max(1500, int(region_area * 0.008))
            and fill <= 0.7
            and not is_long_rule
        )
        is_graphic = is_long_rule or area >= graphic_area or w >= long_span or h >= long_span
        components.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "fill": fill,
                "text_like": is_text_like,
                "graphic": is_graphic,
            }
        )

    graphic_components = [comp for comp in components if comp["graphic"] and not comp["text_like"]]
    if not graphic_components:
        return box

    graphic_components.sort(key=lambda comp: comp["area"], reverse=True)
    seed = graphic_components[0]
    cluster = [seed]
    x1 = seed["x"]
    y1 = seed["y"]
    x2 = seed["x"] + seed["w"]
    y2 = seed["y"] + seed["h"]

    changed = True
    while changed:
        changed = False
        for comp in graphic_components:
            if comp in cluster:
                continue
            cx1 = comp["x"]
            cy1 = comp["y"]
            cx2 = comp["x"] + comp["w"]
            cy2 = comp["y"] + comp["h"]
            if cx2 < x1 - gap or cx1 > x2 + gap or cy2 < y1 - gap or cy1 > y2 + gap:
                continue
            cluster.append(comp)
            x1 = min(x1, cx1)
            y1 = min(y1, cy1)
            x2 = max(x2, cx2)
            y2 = max(y2, cy2)
            changed = True

    label_candidates = [
        comp
        for comp in components
        if comp["text_like"]
        and comp["x"] + comp["w"] >= x1 - label_gap
        and comp["x"] <= x2 + label_gap
        and comp["y"] + comp["h"] >= y1 - label_gap
        and comp["y"] <= y2 + label_gap
    ]
    if label_candidates:
        x1 = min([x1] + [comp["x"] for comp in label_candidates])
        y1 = min([y1] + [comp["y"] for comp in label_candidates])
        x2 = max([x2] + [comp["x"] + comp["w"] for comp in label_candidates])
        y2 = max([y2] + [comp["y"] + comp["h"] for comp in label_candidates])

    refined_area = max(1, (x2 - x1) * (y2 - y1))
    original_area = region_area

    # If refinement does not materially shrink the crop, keep the original.
    if refined_area > original_area * 0.92:
        return box

    pad_x = max(8, int((x2 - x1) * 0.04))
    pad_y = max(8, int((y2 - y1) * 0.04))
    new_left = left + max(0, x1 - pad_x)
    new_top = top + max(0, y1 - pad_y)
    new_right = left + min(rw, x2 + pad_x)
    new_bottom = top + min(rh, y2 + pad_y)

    # Second pass: trim off dense caption text bands that sit below the
    # figure with a clear whitespace gap between them.
    candidate = image.crop((new_left, new_top, new_right, new_bottom))
    cw, ch = candidate.size
    if ch >= 120:
        candidate_np = np.array(candidate.convert("L"))
        ink = candidate_np < 235
        row_density = ink.mean(axis=1)

        bands: list[tuple[int, int, float, float]] = []
        in_band = False
        start = 0
        for i, value in enumerate(row_density):
            if value > 0.01 and not in_band:
                in_band = True
                start = i
            elif value <= 0.01 and in_band:
                end = i - 1
                if end - start >= 3:
                    segment = row_density[start : end + 1]
                    bands.append((start, end, float(segment.mean()), float(segment.max())))
                in_band = False
        if in_band:
            end = ch - 1
            if end - start >= 3:
                segment = row_density[start : end + 1]
                bands.append((start, end, float(segment.mean()), float(segment.max())))

        if len(bands) >= 2:
            last_start, last_end, last_mean, last_max = bands[-1]
            prev_end = bands[-2][1]
            gap_rows = last_start - prev_end - 1
            if (
                last_start >= int(ch * 0.72)
                and gap_rows >= max(14, int(ch * 0.03))
                and last_mean >= 0.05
                and last_max >= 0.15
            ):
                trimmed_bottom = new_top + max(0, last_start - 6)
                if trimmed_bottom - new_top >= int(ch * 0.6):
                    new_bottom = trimmed_bottom

    return new_left, new_top, new_right, new_bottom


def _reject_figure_crop(cropped: Image.Image, page_size: tuple[int, int]) -> bool:
    """
    Return True when a proposed figure crop is obviously low-value.

    Heuristics:
    - nearly blank crops
    - tiny crops
    - very large crops that are mostly text / whitespace
    """
    cw, ch = cropped.size
    pw, ph = page_size
    if cw < 40 or ch < 40:
        return True

    gray = np.array(cropped.convert("L")) if _CV2_AVAILABLE else None
    if gray is None:
        return False

    ink = gray < 245
    ink_ratio = float(ink.mean())
    if ink_ratio < 0.003:
        return True

    crop_area = cw * ch
    page_area = max(1, pw * ph)
    if crop_area > page_area * 0.2 and ink_ratio < 0.05:
        return True

    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small = 0
    large = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max(300, crop_area * 0.01):
            large += 1
        elif area > 8:
            small += 1

    if large == 0 and small > 25:
        return True
    return False


def _rect_area(rect) -> float:
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def _rect_intersection(a, b) -> float:
    left = max(float(a.x0), float(b.x0))
    top = max(float(a.y0), float(b.y0))
    right = min(float(a.x1), float(b.x1))
    bottom = min(float(a.y1), float(b.y1))
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _union_rects(rects: list):
    x0 = min(float(rect.x0) for rect in rects)
    y0 = min(float(rect.y0) for rect in rects)
    x1 = max(float(rect.x1) for rect in rects)
    y1 = max(float(rect.y1) for rect in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _normalise_pdf_rect(raw_rect, page_rect):
    rect = fitz.Rect(raw_rect)
    rect = fitz.Rect(
        min(rect.x0, rect.x1),
        min(rect.y0, rect.y1),
        max(rect.x0, rect.x1),
        max(rect.y0, rect.y1),
    )
    clipped = rect & page_rect
    if clipped.is_empty or clipped.width <= 1 or clipped.height <= 1:
        return None
    return clipped


def _pixel_box_to_pdf_rect(box: tuple[int, int, int, int], image_size: tuple[int, int], page_rect):
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return None
    left, top, right, bottom = box
    sx = page_rect.width / iw
    sy = page_rect.height / ih
    rect = fitz.Rect(left * sx, top * sy, right * sx, bottom * sy)
    return _normalise_pdf_rect(rect, page_rect)


def _render_pdf_clip(page, clip_rect, image_size: tuple[int, int]) -> Image.Image | None:
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return None
    sx = iw / page.rect.width
    sy = ih / page.rect.height
    pix = page.get_pixmap(matrix=fitz.Matrix(sx, sy), clip=clip_rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _extract_pdf_figure_crop(
    pdf_path: str,
    page_index: int,
    image_size: tuple[int, int],
    original_box: tuple[int, int, int, int],
    refined_box: tuple[int, int, int, int],
) -> Image.Image | None:
    """
    Render a figure directly from PDF page objects when possible.

    The raster-refined box is used only as a hint to choose overlapping PDF
    image objects. If no suitable object is found, return ``None`` and let the
    caller fall back to raster cropping.
    """
    if not _FITZ_AVAILABLE:
        return None

    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            return None
        page = doc.load_page(page_index)
        page_rect = page.rect
        preferred_rect = _pixel_box_to_pdf_rect(refined_box, image_size, page_rect)
        source_rect = _pixel_box_to_pdf_rect(original_box, image_size, page_rect)
        if preferred_rect is None or source_rect is None:
            return None

        blocks = page.get_text("dict").get("blocks", [])
        image_rects = []
        min_area = max(64.0, _rect_area(page_rect) * 0.002)
        for block in blocks:
            if block.get("type") != 1:
                continue
            rect = _normalise_pdf_rect(block.get("bbox"), page_rect)
            if rect is None or _rect_area(rect) < min_area:
                continue
            if _rect_intersection(rect, source_rect) <= 0:
                continue
            image_rects.append(rect)

        if not image_rects:
            return None

        def score(rect) -> tuple[float, float, float]:
            inter_pref = _rect_intersection(rect, preferred_rect)
            inter_src = _rect_intersection(rect, source_rect)
            area = _rect_area(rect)
            contains_center = 0.0
            center = fitz.Point((preferred_rect.x0 + preferred_rect.x1) / 2, (preferred_rect.y0 + preferred_rect.y1) / 2)
            if rect.contains(center):
                contains_center = 1.0
            return (
                inter_pref / max(1.0, _rect_area(preferred_rect)),
                contains_center,
                inter_src / max(1.0, area),
            )

        image_rects.sort(key=score, reverse=True)
        return _render_pdf_clip(page, image_rects[0], image_size)
    finally:
        doc.close()


def materialise_figures(
    figures: list[dict],
    page_images: list[Image.Image],
    figures_dir: str,
    stem: str,
    pdf_path: str | None = None,
    page_numbers: list[int] | None = None,
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
        original_box = (left, top, right, bottom)
        refined_box = _refine_figure_crop(image, original_box)
        raster_cropped = image.crop(refined_box)
        cropped = None
        if pdf_path and page_numbers and 0 <= page_no - 1 < len(page_numbers):
            extracted = _extract_pdf_figure_crop(
                pdf_path,
                page_numbers[page_no - 1],
                image.size,
                original_box,
                refined_box,
            )
            if extracted is not None and not _reject_figure_crop(extracted, image.size):
                cropped = extracted
        if cropped is None:
            cropped = raster_cropped
        if _reject_figure_crop(cropped, image.size):
            logger.warning("Rejecting low-value figure crop: %s", fig)
            continue

        filename = f"{stem}_fig_{idx}.png"
        path = base_dir / filename
        cropped.save(path, format="PNG")

        saved = dict(fig)
        saved["latex_path"] = f"figures/{filename}"
        out.append(saved)

    return out


def build_zip_bundle(
    tex_content: str,
    figures_dir: str | None,
    tex_name: str = "output.tex",
    extra_files: list[tuple[str, str]] | None = None,
    extra_text_files: list[tuple[str, str]] | None = None,
) -> bytes:
    """Return a zip archive containing the TeX file and any extracted figures."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(tex_name, tex_content)
        if figures_dir:
            base_dir = Path(figures_dir)
        else:
            base_dir = None
        if base_dir is not None and base_dir.exists():
            for path in sorted(base_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(base_dir.parent)))
        for src, arcname in extra_files or []:
            src_path = Path(src)
            if src_path.exists() and src_path.is_file():
                zf.write(src_path, arcname=arcname)
        for arcname, content in extra_text_files or []:
            zf.writestr(arcname, content)
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
