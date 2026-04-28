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

Sign up at <https://console.anthropic.com>, create an API key, and export it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

This is separate from your claude.ai subscription — it's a pay-as-you-go API account.

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

## Basic usage

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
