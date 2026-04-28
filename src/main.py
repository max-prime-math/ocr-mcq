"""
main.py — Entry point for the OCR-MCQ PDF → LaTeX converter.

Typical usage:
    python src/main.py --input input_pdfs --output output_tex --combine
    python src/main.py --input input_pdfs --review
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable when running as `python src/main.py`.
sys.path.insert(0, str(Path(__file__).parent))

from cache import MathpixCache
from detection import detect_answer
from latex_writer import append_to_combined, finalise_combined, write_tex_file
from ocr import get_ocr_text
from parsing import parse_question
from utils import (
    crop_bottom,
    crop_top,
    get_correction,
    load_config,
    load_corrections,
    page_count,
    render_page_to_image,
    save_correction,
    save_temp_image,
    write_review_row,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCR-MCQ: convert multiple-choice PDFs to exam-class LaTeX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="input_pdfs", help="Directory containing input PDFs.")
    p.add_argument("--output", default="output_tex", help="Directory for generated .tex files.")
    p.add_argument("--combine", action="store_true", help="Merge all questions into one output.tex.")
    p.add_argument("--cache", default="cache/mathpix", help="Mathpix cache directory.")
    p.add_argument("--review", action="store_true", help="Interactive review mode.")
    p.add_argument("--force-ocr", action="store_true", help="Re-call Mathpix even if cached.")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    p.add_argument("--top-crop", type=float, default=None, help="Top crop fraction (0–1).")
    p.add_argument("--bottom-crop-start", type=float, default=None, help="Bottom crop start fraction (0–1).")
    p.add_argument("--min-confidence", type=float, default=None, help="Minimum detection confidence (0–1).")
    p.add_argument("--config", default="config.json", help="Path to JSON config file.")
    p.add_argument("--review-csv", default="review/review.csv", help="Path to review CSV.")
    p.add_argument("--corrections", default="review/corrections.json", help="Path to corrections JSON.")
    return p


# ---------------------------------------------------------------------------
# Review mode
# ---------------------------------------------------------------------------

def run_review_mode(args, cfg: dict) -> None:
    """
    Interactive loop: display each flagged page's bottom crop, prompt for
    the correct answer, and save to corrections.json.
    """
    import csv

    csv_path = args.review_csv
    if not Path(csv_path).exists():
        print(f"No review CSV found at {csv_path}. Run without --review first.")
        return

    corrections = load_corrections(args.corrections)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    flagged = [r for r in rows if not r["detected_answer"] or float(r["confidence"]) < float(
        cfg.get("min_confidence", 0.6)
    )]

    if not flagged:
        print("No pages flagged for review.")
        return

    print(f"{len(flagged)} page(s) flagged for review.\n")

    for row in flagged:
        fname = row["filename"]
        page = int(row["page"])
        key = f"{fname}:{page}"

        if key in corrections:
            print(f"  [{fname} p{page}] already corrected → {corrections[key]}, skipping.")
            continue

        # Try to display the bottom crop image.
        pdf_path = str(Path(args.input) / fname)
        if Path(pdf_path).exists():
            try:
                img = render_page_to_image(pdf_path, page)
                bottom_crop_start = float(cfg.get("bottom_crop_start", 0.5))
                bottom = crop_bottom(img, bottom_crop_start)
                tmp = save_temp_image(bottom)
                _show_image(tmp)
                Path(tmp).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not display image for %s p%d: %s", fname, page, exc)
        else:
            print(f"  (PDF not found at {pdf_path}; cannot display image)")

        print(f"\n  File: {fname}  Page: {page}")
        print(f"  Detected: {row['detected_answer'] or 'none'}  Confidence: {row['confidence']}")
        print(f"  Notes: {row['notes']}")

        while True:
            answer = input("  Enter correct answer [A/B/C/D/E] or 's' to skip: ").strip().upper()
            if answer in ("A", "B", "C", "D", "E"):
                save_correction(args.corrections, fname, page, answer)
                print(f"  Saved: {fname}:{page} → {answer}\n")
                break
            elif answer == "S":
                print("  Skipped.\n")
                break
            else:
                print("  Invalid input. Enter A, B, C, D, E, or S.")

    print("Review complete.")


def _show_image(path: str) -> None:
    """Open *path* with the system image viewer (best-effort)."""
    import subprocess, platform

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        print(f"  (Could not open image automatically; view it at: {path})")


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(
    pdf_path: str,
    page_index: int,
    cfg: dict,
    cache: MathpixCache,
    force_ocr: bool,
) -> dict:
    """
    Process a single PDF page and return a result dict with keys:
        parsed, answer, confidence, flagged, error
    """
    top_fraction = cfg.get("top_crop", 0.5)
    bottom_start = cfg.get("bottom_crop_start", 0.5)
    min_conf = cfg.get("min_confidence", 0.6)

    # Render page.
    img = render_page_to_image(pdf_path, page_index, dpi=cfg.get("dpi", 300))

    # OCR on top crop.
    top_img = crop_top(img, top_fraction)
    top_tmp = save_temp_image(top_img)
    try:
        ocr_result = get_ocr_text(top_tmp, cache=cache, force=force_ocr)
    finally:
        Path(top_tmp).unlink(missing_ok=True)

    # Prefer latex_styled for math fidelity; fall back to plain text.
    raw_text = ocr_result.get("latex_styled") or ocr_result.get("text", "")
    parsed = parse_question(raw_text)

    # Answer detection on bottom crop.
    bottom_img = crop_bottom(img, bottom_start)
    bottom_tmp = save_temp_image(bottom_img)
    try:
        letter, confidence = detect_answer(
            bottom_tmp,
            min_confidence=min_conf,
            use_vision_fallback=cfg.get("use_vision_fallback", False),
        )
    finally:
        Path(bottom_tmp).unlink(missing_ok=True)

    flagged = letter is None or confidence < min_conf

    return {
        "parsed": parsed,
        "answer": letter,
        "confidence": confidence,
        "flagged": flagged,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Merge config file with CLI overrides (CLI wins).
    cfg = load_config(args.config)
    if args.top_crop is not None:
        cfg["top_crop"] = args.top_crop
    if args.bottom_crop_start is not None:
        cfg["bottom_crop_start"] = args.bottom_crop_start
    if args.min_confidence is not None:
        cfg["min_confidence"] = args.min_confidence

    # --review mode is a separate interactive workflow.
    if args.review:
        run_review_mode(args, cfg)
        return

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in %s", input_dir)
        sys.exit(1)

    cache = MathpixCache(args.cache)
    corrections = load_corrections(args.corrections)
    combined_path = str(output_dir / "output.tex")

    # Summary counters.
    total_pages = 0
    successful = 0
    flagged_count = 0
    error_count = 0

    for pdf_path in pdf_files:
        fname = pdf_path.name
        logger.info("Processing %s", fname)

        try:
            n_pages = page_count(str(pdf_path))
        except Exception as exc:
            logger.error("Could not open %s: %s", fname, exc)
            error_count += 1
            continue

        page_results: list[tuple] = []

        for page_idx in range(n_pages):
            total_pages += 1
            logger.debug("  Page %d / %d", page_idx + 1, n_pages)

            try:
                result = process_page(
                    str(pdf_path), page_idx, cfg, cache, args.force_ocr
                )
            except Exception as exc:
                logger.error("  Error on page %d of %s: %s", page_idx, fname, exc)
                error_count += 1
                write_review_row(
                    args.review_csv, fname, page_idx, None, 0.0, notes=str(exc)
                )
                continue

            # Apply human correction if available.
            correction = get_correction(corrections, fname, page_idx)
            if correction:
                result["answer"] = correction
                result["flagged"] = False
                logger.debug("  Applied correction: %s", correction)

            if result["flagged"]:
                flagged_count += 1
                write_review_row(
                    args.review_csv,
                    fname,
                    page_idx,
                    result["answer"],
                    result["confidence"],
                    notes="low confidence" if result["answer"] else "not detected",
                )
            else:
                successful += 1

            page_results.append((result["parsed"], result["answer"]))

        # Write output.
        if args.combine:
            append_to_combined(page_results, combined_path, fname)
        else:
            out_name = pdf_path.stem + ".tex"
            write_tex_file(page_results, str(output_dir / out_name))

    if args.combine and pdf_files:
        finalise_combined(combined_path)

    # Summary.
    print("\n" + "=" * 50)
    print(f"  Total pages processed : {total_pages}")
    print(f"  Successful            : {successful}")
    print(f"  Flagged for review    : {flagged_count}")
    print(f"  Errors                : {error_count}")
    print("=" * 50)
    if flagged_count:
        print(f"\n  Run with --review to resolve {flagged_count} flagged page(s).")
        print(f"  Review CSV: {args.review_csv}")


if __name__ == "__main__":
    main()
