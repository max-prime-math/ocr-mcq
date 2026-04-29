"""
main.py — Entry point for the OCR-MCQ PDF → LaTeX converter.

Uses Claude Vision to extract question stems, answer choices, and the
marked correct answer from each PDF page in a single API call.

Typical usage:
    python src/main.py --input input_pdfs --output output_tex --combine
    python src/main.py --input input_pdfs --review
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from cache import MathpixCache as VisionCache
from latex_writer import append_to_combined, finalise_combined, write_tex_file
from ocr import extract_page, should_extract_figures, should_retry_with_next_page
from parsing import ParsedQuestion
from utils import (
    crop_bottom,
    get_correction,
    load_config,
    load_corrections,
    materialise_figures,
    page_count,
    render_page_to_image,
    save_correction,
    save_temp_image,
    write_review_row,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"


def _normalise_figure_mode(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value or "auto").strip().lower()
    if text in {"on", "off", "auto"}:
        return text
    return "auto"


def _result_notes(data: dict, pages_used: int, figures: list[dict], page_index: int, next_page_index: int | None) -> list[str]:
    notes: list[str] = []
    question = (data.get("question") or "").strip().lower()
    answer = data.get("correct_answer")
    solution = (data.get("solution") or "").strip()
    if "solution on the next page" in question:
        notes.append("question says solution on next page")
    if pages_used > 1:
        notes.append("used multiple pages")
    if answer is None:
        notes.append("correct answer not detected")
    if figures:
        notes.append(f"kept {len(figures)} figure crop(s)")
    if next_page_index is not None and answer is None and not solution:
        notes.append("page may need next-page pairing")
    return notes


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
    p.add_argument("--cache", default="cache/vision", help="Vision cache directory.")
    p.add_argument("--review", action="store_true", help="Interactive review mode.")
    p.add_argument("--force-ocr", action="store_true", help="Re-call Claude even if a cached result exists.")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    p.add_argument("--bottom-crop-start", type=float, default=None, help="Where the answer area starts for review display (0–1).")
    p.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence to auto-accept an answer (0–1).")
    p.add_argument("--model", default=None, help=f"Claude model ID (default: {DEFAULT_MODEL}).")
    p.add_argument(
        "--figure-mode",
        choices=["off", "auto", "on"],
        default=None,
        help="Control figure extraction: off, auto-detect per question, or always on.",
    )
    p.add_argument("--config", default="config.json", help="Path to JSON config file.")
    p.add_argument("--review-csv", default="review/review.csv", help="Path to review CSV.")
    p.add_argument("--corrections", default="review/corrections.json", help="Path to corrections JSON.")
    return p


# ---------------------------------------------------------------------------
# Review mode
# ---------------------------------------------------------------------------

def run_review_mode(args, cfg: dict) -> None:
    """Show each flagged page, prompt for the correct letter, save corrections."""
    import csv

    csv_path = args.review_csv
    if not Path(csv_path).exists():
        print(f"No review CSV found at {csv_path}. Run without --review first.")
        return

    corrections = load_corrections(args.corrections)
    min_conf = float(cfg.get("min_confidence", 0.6))
    bottom_start = float(cfg.get("bottom_crop_start", 0.5))

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    flagged = [r for r in rows if not r["detected_answer"] or float(r["confidence"]) < min_conf]

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

        pdf_path = str(Path(args.input) / fname)
        if Path(pdf_path).exists():
            try:
                img = render_page_to_image(pdf_path, page)
                bottom = crop_bottom(img, bottom_start)
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
    import platform, subprocess
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
    next_page_index: int | None,
    cfg: dict,
    cache: VisionCache,
    client: anthropic.Anthropic,
    force_ocr: bool,
    figures_dir: str,
    source_stem: str,
) -> dict:
    """
    Render one PDF page, optionally plus its continuation page, and extract
    question + answer via Claude Vision.

    Returns a dict with keys: parsed, answer, confidence, flagged, error.
    """
    model = cfg.get("model", DEFAULT_MODEL)

    primary_dpi = int(cfg.get("dpi", 220))
    fallback_dpi = int(cfg.get("fallback_dpi", max(primary_dpi, 240)))
    figure_mode = _normalise_figure_mode(cfg.get("figure_mode", cfg.get("include_figures", "auto")))

    page_images = [render_page_to_image(pdf_path, page_index, dpi=primary_dpi)]

    tmp = save_temp_image(page_images[0])
    try:
        data = extract_page(
            tmp,
            client=client,
            cache=cache,
            force=force_ocr,
            model=model,
            include_figures=False,
        )
    finally:
        Path(tmp).unlink(missing_ok=True)

    wants_figures = figure_mode == "on" or (figure_mode == "auto" and should_extract_figures(data))

    used_next_page = next_page_index is not None and should_retry_with_next_page(data)
    if used_next_page:
        page_images = [render_page_to_image(pdf_path, page_index, dpi=fallback_dpi)]
        page_images.append(render_page_to_image(pdf_path, next_page_index, dpi=fallback_dpi))
        tmp = save_temp_image(page_images[0])
        tmp2 = save_temp_image(page_images[1])
        try:
            data = extract_page(
                tmp,
                client=client,
                cache=cache,
                force=force_ocr,
                model=model,
                second_image_path=tmp2,
                include_figures=wants_figures,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)
            Path(tmp2).unlink(missing_ok=True)
    elif wants_figures:
        page_images = [render_page_to_image(pdf_path, page_index, dpi=fallback_dpi)]
        tmp = save_temp_image(page_images[0])
        try:
            data = extract_page(
                tmp,
                client=client,
                cache=cache,
                force=force_ocr,
                model=model,
                include_figures=True,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

    pages_used = int(data.get("pages_used") or 1)
    if used_next_page and (data.get("correct_answer") is not None or data.get("solution")):
        pages_used = max(pages_used, 2)
    figures = materialise_figures(
        data.get("figures", []),
        page_images[:pages_used],
        figures_dir,
        f"{source_stem}_p{page_index + 1}",
        pdf_path=pdf_path,
        page_numbers=[page_index + offset for offset in range(pages_used)],
    )
    question_figures = [fig for fig in figures if fig.get("section", "question") == "question"]
    include_solution_figures = bool(cfg.get("include_solution_figures", False))
    solution_figures = [fig for fig in figures if fig.get("section") == "solution"] if include_solution_figures else []
    question_tables = [tbl for tbl in data.get("tables", []) if tbl.get("section", "question") == "question"]
    solution_tables = [tbl for tbl in data.get("tables", []) if tbl.get("section") == "solution"]

    parsed = ParsedQuestion(
        question=data["question"],
        choices=data.get("choices", {}),
        solution=data.get("solution"),
        figures=question_figures,
        solution_figures=solution_figures,
        tables=question_tables,
        solution_tables=solution_tables,
    )
    answer = data.get("correct_answer")  # letter or None
    confidence = 1.0 if answer else 0.0
    flagged = answer is None

    return {
        "parsed": parsed,
        "answer": answer,
        "confidence": confidence,
        "flagged": flagged,
        "pages_used": pages_used,
        "tricky": bool(_result_notes(data, pages_used, figures, page_index, next_page_index)),
        "notes": _result_notes(data, pages_used, figures, page_index, next_page_index),
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

    cfg = load_config(args.config)
    if args.bottom_crop_start is not None:
        cfg["bottom_crop_start"] = args.bottom_crop_start
    if args.min_confidence is not None:
        cfg["min_confidence"] = args.min_confidence
    if args.model is not None:
        cfg["model"] = args.model
    if args.figure_mode is not None:
        cfg["figure_mode"] = args.figure_mode

    if args.review:
        run_review_mode(args, cfg)
        return

    # Initialise Anthropic client (reads ANTHROPIC_API_KEY from env).
    client = anthropic.Anthropic()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in %s", input_dir)
        sys.exit(1)

    cache = VisionCache(args.cache)
    corrections = load_corrections(args.corrections)
    combined_path = str(output_dir / "output.tex")
    figures_dir = str(output_dir / "figures")

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

        page_results = []

        page_idx = 0
        while page_idx < n_pages:
            logger.debug("  Page %d / %d", page_idx + 1, n_pages)

            try:
                result = process_page(
                    str(pdf_path),
                    page_idx,
                    page_idx + 1 if page_idx + 1 < n_pages else None,
                    cfg,
                    cache,
                    client,
                    args.force_ocr,
                    figures_dir,
                    Path(fname).stem,
                )
            except Exception as exc:
                logger.error("  Error on page %d of %s: %s", page_idx, fname, exc)
                error_count += 1
                total_pages += 1
                write_review_row(
                    args.review_csv, fname, page_idx, None, 0.0, notes=str(exc)
                )
                page_idx += 1
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
                    notes="answer not detected",
                )
            else:
                successful += 1

            total_pages += max(1, int(result.get("pages_used", 1)))
            source = f"{Path(fname).stem} p{page_idx + 1}"
            if result.get("pages_used", 1) > 1:
                source += f"-{page_idx + result['pages_used']}"
            page_results.append((result["parsed"], result["answer"], source))
            page_idx += max(1, int(result.get("pages_used", 1)))

        if args.combine:
            append_to_combined(page_results, combined_path, fname)
        else:
            out_name = pdf_path.stem + ".tex"
            write_tex_file(page_results, str(output_dir / out_name))

    if args.combine and pdf_files:
        finalise_combined(combined_path)

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
