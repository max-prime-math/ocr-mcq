"""
app.py — Streamlit web interface for OCR-MCQ.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import json
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

import anthropic
from cache import MathpixCache as VisionCache
from latex_writer import render_question
from ocr import extract_page, should_retry_with_next_page
from ocr import should_extract_figures
from parsing import ParsedQuestion
from utils import (
    build_zip_bundle,
    crop_bottom,
    materialise_figures,
    page_count,
    render_page_to_image,
    save_temp_image,
)


def _result_notes(data: dict, pages_used: int, figures: list[dict], next_page_used: bool) -> list[str]:
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
    if next_page_used and answer is None and not solution:
        notes.append("page may need next-page pairing")
    return notes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="OCR-MCQ", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in {
    "results": [],
    "processed": False,
    "tmpdir": None,
    "figures_dir": None,
    "usage_log": [],
    "model_used": None,
    "processing_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Pricing per million tokens (input, output, cache_read, cache_write).
_PRICING = {
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00,  "cache_read": 0.10, "cache_write": 1.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
}


def compute_cost(usage_log: list, model: str) -> dict:
    """Sum token counts and compute USD cost across all API calls in usage_log."""
    p = _PRICING.get(model, _PRICING["claude-haiku-4-5"])
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for u in usage_log:
        totals["input"]       += u.get("input_tokens", 0)
        totals["output"]      += u.get("output_tokens", 0)
        totals["cache_read"]  += u.get("cache_read_input_tokens", 0)
        totals["cache_write"] += u.get("cache_creation_input_tokens", 0)

    cost = (
        totals["input"]         * p["input"]       / 1_000_000
        + totals["output"]      * p["output"]       / 1_000_000
        + totals["cache_read"]  * p["cache_read"]   / 1_000_000
        + totals["cache_write"] * p["cache_write"]  / 1_000_000
    )
    return {**totals, "cost_usd": cost}


def build_tex(results: list, corrections: dict) -> str:
    preamble = (
        "\\documentclass[12pt,addpoints]{exam}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        "\\usepackage{graphicx}\n\n"
        "\\begin{document}\n"
        "\\begin{questions}\n"
    )
    postamble = "\n\\end{questions}\n\\end{document}\n"

    blocks = []
    for r in results:
        if r["error"] or r["parsed"] is None:
            continue
        key = f"{r['fname']}:{r['page']}"
        answer = corrections.get(key, r["answer"])
        source = f"{Path(r['fname']).stem} p{r['page'] + 1}"
        if r.get("page_end", r["page"]) > r["page"]:
            source += f"-{r['page_end'] + 1}"
        blocks.append(render_question(r["parsed"], answer, source))

    return preamble + "\n\n".join(blocks) + postamble


def _progress_value(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, done / total))


def _checkpoint_processing(tmpdir: str, results: list, usage_log: list, model_used: str, figures_dir: str | None) -> None:
    payload = {
        "results": results,
        "usage_log": usage_log,
        "model_used": model_used,
        "figures_dir": figures_dir,
    }
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    with open(Path(tmpdir) / "checkpoint.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("OCR-MCQ")
    st.caption("Multiple-choice PDFs → exam-class LaTeX")
    st.divider()

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        autocomplete="current-password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Starts with sk-ant-. Never saved to disk.",
    )

    model = st.selectbox(
        "Model",
        ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
        help="Haiku: ~$0.003/page. Sonnet: ~$0.013/page.",
    )

    force_ocr = st.checkbox(
        "Force re-process",
        value=False,
        help="Ignore cached results and re-call Claude for every page.",
    )

    figure_mode = st.segmented_control(
        "Figure extraction",
        options=["Off", "Auto", "On"],
        default="Auto",
        help="Off: never crop figures. Auto: detect per question. On: always run figure-aware extraction.",
    )

    st.divider()
    st.caption("Cache is stored in `cache/vision/` so repeated runs are cheap.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("OCR-MCQ")

uploaded_files = st.file_uploader(
    "Drop PDF files here",
    type="pdf",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDFs to get started.")
    st.stop()

if st.button("Process PDFs", type="primary", use_container_width=True):
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
        st.stop()

    tmpdir = tempfile.mkdtemp()
    st.session_state.tmpdir = tmpdir

    client = anthropic.Anthropic(api_key=api_key)
    cache = VisionCache("cache/vision")

    pdf_paths = []
    for uf in uploaded_files:
        dest = os.path.join(tmpdir, uf.name)
        with open(dest, "wb") as fh:
            fh.write(uf.read())
        pdf_paths.append(dest)

    try:
        total_pages = sum(page_count(p) for p in pdf_paths)
    except Exception as exc:
        st.error(f"Could not read PDFs: {exc}")
        st.stop()

    all_results = []
    usage_log = []
    pages_done = 0
    progress = st.progress(0, text="Starting…")
    status = st.empty()
    figures_dir = os.path.join(tmpdir, "figures")
    st.session_state.figures_dir = figures_dir
    st.session_state.processing_error = None

    processing_error = None
    try:
        for pdf_path in pdf_paths:
            fname = Path(pdf_path).name
            n = page_count(pdf_path)
            page_idx = 0

            while page_idx < n:
                status.text(f"{fname} — page {page_idx + 1} of {n}")

                try:
                    primary_dpi = 170
                    fallback_dpi = 240
                    page_images = [render_page_to_image(pdf_path, page_idx, dpi=primary_dpi)]

                    tmp_img = save_temp_image(page_images[0])
                    try:
                        data = extract_page(
                            tmp_img,
                            client=client,
                            cache=cache,
                            force=force_ocr,
                            model=model,
                            usage_out=usage_log,
                            include_figures=False,
                        )
                    finally:
                        Path(tmp_img).unlink(missing_ok=True)

                    mode = (figure_mode or "Auto").lower()
                    wants_figures = mode == "on" or (mode == "auto" and should_extract_figures(data))

                    used_next_page = page_idx + 1 < n and should_retry_with_next_page(data)
                    if used_next_page:
                        page_images = [render_page_to_image(pdf_path, page_idx, dpi=fallback_dpi)]
                        page_images.append(render_page_to_image(pdf_path, page_idx + 1, dpi=fallback_dpi))
                        tmp_img = save_temp_image(page_images[0])
                        tmp_img_2 = save_temp_image(page_images[1])
                        try:
                            data = extract_page(
                                tmp_img,
                                client=client,
                                cache=cache,
                                force=force_ocr,
                                model=model,
                                usage_out=usage_log,
                                second_image_path=tmp_img_2,
                                include_figures=wants_figures,
                            )
                        finally:
                            Path(tmp_img).unlink(missing_ok=True)
                            Path(tmp_img_2).unlink(missing_ok=True)
                    elif wants_figures:
                        page_images = [render_page_to_image(pdf_path, page_idx, dpi=fallback_dpi)]
                        tmp_img = save_temp_image(page_images[0])
                        try:
                            data = extract_page(
                                tmp_img,
                                client=client,
                                cache=cache,
                                force=force_ocr,
                                model=model,
                                usage_out=usage_log,
                                include_figures=True,
                            )
                        finally:
                            Path(tmp_img).unlink(missing_ok=True)

                    pages_used = int(data.get("pages_used") or 1)
                    if used_next_page and (data.get("correct_answer") is not None or data.get("solution")):
                        pages_used = max(pages_used, 2)
                    figures = materialise_figures(
                        data.get("figures", []),
                        page_images[:pages_used],
                        figures_dir,
                        f"{Path(fname).stem}_p{page_idx + 1}",
                    )
                    question_figures = [fig for fig in figures if fig.get("section", "question") == "question"]
                    solution_figures = []
                    question_tables = [tbl for tbl in data.get("tables", []) if tbl.get("section", "question") == "question"]
                    solution_tables = [tbl for tbl in data.get("tables", []) if tbl.get("section") == "solution"]
                    notes = _result_notes(data, pages_used, figures, used_next_page)

                    parsed = ParsedQuestion(
                        question=data.get("question", ""),
                        choices=data.get("choices", {}),
                        solution=data.get("solution"),
                        figures=question_figures,
                        solution_figures=solution_figures,
                        tables=question_tables,
                        solution_tables=solution_tables,
                    )
                    answer = data.get("correct_answer")

                    all_results.append({
                        "fname": fname,
                        "page": page_idx,
                        "page_end": page_idx + pages_used - 1,
                        "parsed": parsed,
                        "answer": answer,
                        "flagged": answer is None,
                        "pdf_path": pdf_path,
                        "pages_used": pages_used,
                        "tricky": bool(notes),
                        "notes": notes,
                        "error": None,
                    })

                except Exception as exc:
                    all_results.append({
                        "fname": fname,
                        "page": page_idx,
                        "page_end": page_idx,
                        "parsed": None,
                        "answer": None,
                        "flagged": True,
                        "pdf_path": pdf_path,
                        "pages_used": 1,
                        "tricky": True,
                        "notes": [str(exc)],
                        "error": str(exc),
                    })

                pages_used = all_results[-1].get("pages_used", 1)
                pages_done += pages_used
                st.session_state.results = all_results
                st.session_state.usage_log = usage_log
                st.session_state.model_used = model
                st.session_state.figures_dir = figures_dir
                _checkpoint_processing(tmpdir, all_results, usage_log, model, figures_dir)
                progress.progress(
                    _progress_value(pages_done, total_pages),
                    text=f"{pages_done} / {total_pages} pages processed",
                )
                page_idx += pages_used
    except Exception as exc:
        processing_error = str(exc)

    progress.empty()
    status.empty()
    st.session_state.results = all_results
    st.session_state.usage_log = usage_log
    st.session_state.model_used = model
    st.session_state.processed = bool(all_results)
    st.session_state.processing_error = processing_error
    if processing_error:
        st.warning("Processing stopped early. Partial results were recovered for review and download.")
    else:
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.processing_error:
    st.warning(f"Recovered partial run after error: {st.session_state.processing_error}")

if not (st.session_state.processed and st.session_state.results):
    st.stop()

results = st.session_state.results
successful = [r for r in results if not r["flagged"] and not r["error"]]
flagged    = [r for r in results if r["flagged"] and not r["error"]]
errors     = [r for r in results if r["error"]]

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("✅ Successful", len(successful))
col2.metric("⚠️ Flagged for review", len(flagged))
col3.metric("❌ Errors", len(errors))

# Token usage & cost summary
if st.session_state.usage_log:
    usage = compute_cost(st.session_state.usage_log, st.session_state.model_used or "claude-haiku-4-5")
    api_calls = len(st.session_state.usage_log)
    cache_hits = len(results) - api_calls

    with st.expander(f"💰 Cost — ${usage['cost_usd']:.4f} for this run", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API calls", api_calls, help="Cache hits made no API call and cost nothing.")
        c2.metric("Cache hits", cache_hits)
        c3.metric("Total tokens", f"{usage['input'] + usage['output'] + usage['cache_read'] + usage['cache_write']:,}")
        c4.metric("Total cost", f"${usage['cost_usd']:.4f}")

        st.caption(
            f"Model: `{st.session_state.model_used}` — "
            f"Input: {usage['input']:,} — "
            f"Output: {usage['output']:,} — "
            f"Cache read: {usage['cache_read']:,} — "
            f"Cache write: {usage['cache_write']:,} tokens"
        )
elif st.session_state.processed:
    st.info("All pages were served from cache — no API calls made, no cost incurred.")

if errors:
    with st.expander("Error details"):
        for r in errors:
            if r.get("page_end", r["page"]) > r["page"]:
                page_label = f"pages {r['page'] + 1}-{r['page_end'] + 1}"
            else:
                page_label = f"page {r['page'] + 1}"
            st.error(f"{r['fname']} {page_label}: {r['error']}")

tricky_results = [r for r in results if r.get("tricky")]
if tricky_results:
    with st.expander(f"Tricky PDFs ({len({r['fname'] for r in tricky_results})})"):
        for r in tricky_results[:50]:
            page_label = f"p{r['page'] + 1}"
            if r.get("page_end", r["page"]) > r["page"]:
                page_label = f"p{r['page'] + 1}-{r['page_end'] + 1}"
            st.write(f"{r['fname']} {page_label}: {', '.join(r.get('notes', []))}")
        if len(tricky_results) > 50:
            st.caption("Showing first 50 tricky results.")

# ---------------------------------------------------------------------------
# Review flagged pages
# ---------------------------------------------------------------------------

corrections = {}

if flagged:
    st.subheader("Review flagged pages")
    st.caption(
        "Claude could not detect a marked answer on these pages. "
        "Select the correct letter before downloading."
    )

    for r in flagged:
        if r.get("page_end", r["page"]) > r["page"]:
            label = f"{r['fname']} — Pages {r['page'] + 1}-{r['page_end'] + 1}"
        else:
            label = f"{r['fname']} — Page {r['page'] + 1}"
        with st.expander(label, expanded=True):
            img_col, form_col = st.columns([1, 1])

            with img_col:
                try:
                    review_page = r.get("page_end", r["page"])
                    img = render_page_to_image(r["pdf_path"], review_page, dpi=150)
                    bottom = crop_bottom(img, 0.5)
                    st.image(bottom, caption="Answer area", use_container_width=True)
                except Exception:
                    st.warning("Could not render page image.")

            with form_col:
                if r["parsed"] and r["parsed"]["question"]:
                    st.markdown(
                        "**Question stem:**\n\n"
                        + r["parsed"]["question"][:300]
                        + ("…" if len(r["parsed"]["question"]) > 300 else "")
                    )

                key = f"{r['fname']}:{r['page']}"
                choice = st.radio(
                    "Correct answer",
                    ["Skip", "A", "B", "C", "D", "E"],
                    horizontal=True,
                    key=f"radio_{key}",
                )
                if choice != "Skip":
                    corrections[key] = choice

# ---------------------------------------------------------------------------
# Build and download LaTeX
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Download")
tex_content = build_tex(results, corrections)
tricky_pdf_paths = sorted(
    {
        (r["pdf_path"], f"tricky_pdfs/{Path(r['pdf_path']).name}")
        for r in results
        if r.get("tricky") or r.get("error")
    }
)
tricky_manifest = "\n".join(
    f"{Path(r['pdf_path']).name}: p{r['page'] + 1}"
    + (f"-{r['page_end'] + 1}" if r.get("page_end", r["page"]) > r["page"] else "")
    + f" :: {', '.join(r.get('notes', [])) or r.get('error', '')}"
    for r in results
    if r.get("tricky") or r.get("error")
)

n_todo = sum(
    1 for r in flagged
    if f"{r['fname']}:{r['page']}" not in corrections
)
if n_todo:
    st.warning(
        f"{n_todo} flagged page(s) still have no answer selected. "
        "They will get a `% TODO` comment in the output."
    )

st.download_button(
    label="⬇️ Download output bundle (.zip)",
    data=build_zip_bundle(
        tex_content,
        st.session_state.get("figures_dir"),
        extra_files=tricky_pdf_paths,
        extra_text_files=[("tricky_pdfs/manifest.txt", tricky_manifest)] if tricky_manifest else None,
    ),
    file_name="output_bundle.zip",
    mime="application/zip",
    type="primary",
    use_container_width=True,
)

with st.expander("Preview LaTeX"):
    preview = tex_content[:4000]
    if len(tex_content) > 4000:
        preview += "\n\n% … (truncated for preview)"
    st.code(preview, language="latex")
