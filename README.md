# OCR-MCQ

Converts a folder of multiple-choice PDFs into exam-class LaTeX.
Each PDF page should contain one question with answer choices A–E and a visually marked correct answer.

Uses Claude Vision (Anthropic API) to extract the question, choices, and marked answer in a single API call per page.

---

## Setup

```bash
git clone https://github.com/yourname/ocr-mcq.git
cd ocr-mcq
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Anthropic API key

Sign up at <https://console.anthropic.com>, create an API key, and set it as an environment variable.

**This is separate from your claude.ai subscription** — it's a pay-as-you-go API account billed by token usage.

#### Setting the key safely

The safest approach is to export it directly in your terminal session. It lives only in memory and is never written to disk, so there is nothing for git to commit:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

You will need to re-export it each time you open a new terminal. Alternatively, store it in a `.env` file which is already excluded by `.gitignore`:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
source .env
```

> **Never put your API key in `config.json` or any other file that could be committed.** The `.gitignore` excludes `.env` and `config.json`, but the safest habit is to keep keys out of files entirely. Before every `git push`, run `git status` and `git diff --cached` to confirm nothing sensitive is staged.

### Cost estimate

The default model is `claude-haiku-4-5`, which costs roughly **$0.003 per page** (image tokens + output). The system prompt is cached after the first call, so repeated runs within 5 minutes cost even less.

| Model | Cost / page | Best for |
|-------|-------------|----------|
| `claude-haiku-4-5` | ~$0.003 | Bulk processing (default) |
| `claude-sonnet-4-6` | ~$0.013 | Higher accuracy if needed |

Copy the example config:

```bash
cp config.example.json config.json
```

---

## Web interface

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload PDFs, watch them process, review any flagged pages, and download the `.tex` file — all in the browser. The API key can be entered in the sidebar.

---

## Command-line usage

1. Drop your PDFs into `input_pdfs/`.

2. Run the converter:

```bash
python src/main.py --input input_pdfs --output output_tex --combine
```

This creates `output_tex/output.tex` with all questions merged into one file.
To get one `.tex` file per PDF, omit `--combine`.

3. Check the summary printed at the end:

```
==================================================
  Total pages processed : 40
  Successful            : 37
  Flagged for review    : 3
  Errors                : 0
==================================================
```

4. If pages were flagged, run the interactive review:

```bash
python src/main.py --input input_pdfs --review
```

This shows each flagged page's answer area, tells you what was detected, and lets you type the correct letter.
Corrections are saved to `review/corrections.json`.

5. Re-run step 2 to regenerate the LaTeX with corrections applied.

---

## Output format

```latex
\question
Find the derivative of \(f(x) = x^2 + 3x\).
\begin{choices}
\CorrectChoice \(2x + 3\)
\choice \(x^2 + 3\)
\choice \(2x\)
\choice \(x + 3\)
\choice \(2x^2 + 3\)
\end{choices}
```

If the correct answer could not be determined, a placeholder is inserted:

```latex
% TODO: correct answer not detected
```

---

## Common options

| Flag | What it does |
|------|-------------|
| `--combine` | Merge all questions into one `output.tex` |
| `--force-ocr` | Re-call Claude even if a cached result exists |
| `--model claude-sonnet-4-6` | Use a more capable model |
| `--bottom-crop-start 0.6` | Answer area starts at 60% down (for review display) |
| `--min-confidence 0.7` | Raise the confidence bar for auto-accepting answers |
| `--debug` | Verbose logging |

---

## Running tests

```bash
pytest tests/ -v
```
