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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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

    # Save uploaded PDFs to a temp directory so PyMuPDF can read them.
    tmpdir = tempfile.mkdtemp()
    st.session_state.tmpdir = tmpdir

    client = anthropic.Anthropic(api_key=api_key)
    cache = VisionCache("cache/vision")

    # Write all uploaded files to disk first.
    pdf_paths = []
    for uf in uploaded_files:
        dest = os.path.join(tmpdir, uf.name)
        with open(dest, "wb") as fh:
            fh.write(uf.read())
        pdf_paths.append(dest)

    # Count total pages for the progress bar.
    try:
        total_pages = sum(page_count(p) for p in pdf_paths)
    except Exception as exc:
        st.error(f"Could not read PDFs: {exc}")
        st.stop()

    all_results = []
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
                    )
                finally:
                    Path(tmp_img).unlink(missing_ok=True)

                parsed = ParsedQuestion(
                    question=data.get("question", ""),
                    choices=data.get("choices", {}),
                )
                answer = data.get("correct_answer")

                all_results.append(
                    {
                        "fname": fname,
                        "page": page_idx,
                        "parsed": parsed,
                        "answer": answer,
                        "flagged": answer is None,
                        "pdf_path": pdf_path,
                        "error": None,
                    }
                )

            except Exception as exc:
                all_results.append(
                    {
                        "fname": fname,
                        "page": page_idx,
                        "parsed": None,
                        "answer": None,
                        "flagged": True,
                        "pdf_path": pdf_path,
                        "error": str(exc),
                    }
                )

            pages_done += 1
            progress.progress(
                pages_done / total_pages,
                text=f"{pages_done} / {total_pages} pages processed",
            )

    progress.empty()
    status.empty()
    st.session_state.results = all_results
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
        r"\documentclass[12pt,addpoints]{exam}" + "\n"
        r"\usepackage{amsmath,amssymb,amsfonts}" + "\n\n"
        r"\begin{document}" + "\n"
        r"\begin{questions}" + "\n"
    )
    postamble = "\n\\end{questions}\n\\end{document}\n"

    blocks = []
    for r in results:
        if r["error"] or r["parsed"] is None:
            continue
        key = f"{r['fname']}:{r['page']}"
        answer = corrections.get(key, r["answer"])
        blocks.append(render_question(r["parsed"], answer))

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
