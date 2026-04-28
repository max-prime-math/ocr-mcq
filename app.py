"""
app.py — Streamlit web interface for OCR-MCQ.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

import anthropic
from cache import MathpixCache as VisionCache
from latex_writer import render_question
from ocr import extract_page
from parsing import ParsedQuestion
from utils import crop_bottom, page_count, render_page_to_image, save_temp_image

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
    "usage_log": [],
    "model_used": None,
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

    for pdf_path in pdf_paths:
        fname = Path(pdf_path).name
        n = page_count(pdf_path)

        for page_idx in range(n):
            status.text(f"{fname} — page {page_idx + 1} of {n}")

            try:
                img = render_page_to_image(pdf_path, page_idx, dpi=200)
                tmp_img = save_temp_image(img)
                try:
                    data = extract_page(
                        tmp_img,
                        client=client,
                        cache=cache,
                        force=force_ocr,
                        model=model,
                        usage_out=usage_log,
                    )
                finally:
                    Path(tmp_img).unlink(missing_ok=True)

                parsed = ParsedQuestion(
                    question=data.get("question", ""),
                    choices=data.get("choices", {}),
                    solution=data.get("solution"),
                )
                answer = data.get("correct_answer")

                all_results.append({
                    "fname": fname,
                    "page": page_idx,
                    "parsed": parsed,
                    "answer": answer,
                    "flagged": answer is None,
                    "pdf_path": pdf_path,
                    "error": None,
                })

            except Exception as exc:
                all_results.append({
                    "fname": fname,
                    "page": page_idx,
                    "parsed": None,
                    "answer": None,
                    "flagged": True,
                    "pdf_path": pdf_path,
                    "error": str(exc),
                })

            pages_done += 1
            progress.progress(
                pages_done / total_pages,
                text=f"{pages_done} / {total_pages} pages processed",
            )

    progress.empty()
    status.empty()
    st.session_state.results = all_results
    st.session_state.usage_log = usage_log
    st.session_state.model_used = model
    st.session_state.processed = True
    st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

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
            st.error(f"{r['fname']} page {r['page'] + 1}: {r['error']}")

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
        label = f"{r['fname']} — Page {r['page'] + 1}"
        with st.expander(label, expanded=True):
            img_col, form_col = st.columns([1, 1])

            with img_col:
                try:
                    img = render_page_to_image(r["pdf_path"], r["page"], dpi=150)
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


def build_tex(results: list, corrections: dict) -> str:
    preamble = (
        "\\documentclass[12pt,addpoints]{exam}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n\n"
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
        source = Path(r["fname"]).stem
        blocks.append(render_question(r["parsed"], answer, source))

    return preamble + "\n\n".join(blocks) + postamble


tex_content = build_tex(results, corrections)

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
    label="⬇️ Download output.tex",
    data=tex_content,
    file_name="output.tex",
    mime="text/plain",
    type="primary",
    use_container_width=True,
)

with st.expander("Preview LaTeX"):
    preview = tex_content[:4000]
    if len(tex_content) > 4000:
        preview += "\n\n% … (truncated for preview)"
    st.code(preview, language="latex")
