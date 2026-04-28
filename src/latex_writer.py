"""
latex_writer.py — Render parsed questions as exam-class LaTeX.

Produces \\question / \\begin{choices} / \\choice / \\CorrectChoice blocks
that compile directly with the ``exam`` document class.
"""

import logging
from pathlib import Path
from typing import Optional

from parsing import ParsedQuestion, CHOICE_LETTERS

logger = logging.getLogger(__name__)

# Preamble written at the top of every generated .tex file.
_PREAMBLE = r"""\documentclass[12pt,addpoints]{exam}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{enumitem}

\begin{document}
\begin{questions}
"""

_POSTAMBLE = r"""
\end{questions}
\end{document}
"""


def _escape_percent(text: str) -> str:
    """Escape bare % characters that aren't already LaTeX comments."""
    import re
    return re.sub(r"(?<!\\)%", r"\\%", text)


def render_question(
    parsed: ParsedQuestion,
    correct_answer: Optional[str],
) -> str:
    """
    Render one question as a LaTeX ``\\question`` block.

    Args:
        parsed:         Output of ``parse_question``.
        correct_answer: Letter A–E, or None if unknown.

    Returns:
        Multi-line LaTeX string (no trailing newline).
    """
    lines: list[str] = []
    lines.append(r"\question")
    lines.append(parsed["question"])
    lines.append(r"\begin{choices}")

    choices = parsed.get("choices", {})
    for letter in CHOICE_LETTERS:
        text = choices.get(letter)
        if text is None:
            continue
        if correct_answer and letter == correct_answer.upper():
            lines.append(rf"\CorrectChoice {text}")
        else:
            lines.append(rf"\choice {text}")

    if not choices:
        lines.append(r"\choice % TODO: choices not parsed")

    if correct_answer is None:
        lines.append(r"% TODO: correct answer not detected")

    lines.append(r"\end{choices}")

    solution = parsed.get("solution")
    if solution:
        lines.append(r"\begin{solution}")
        lines.append(solution)
        lines.append(r"\end{solution}")

    return "\n".join(lines)


def write_tex_file(
    questions: list[tuple[ParsedQuestion, Optional[str]]],
    output_path: str,
) -> None:
    """
    Write a complete exam-class .tex file for *questions*.

    Args:
        questions:   List of (ParsedQuestion, correct_letter_or_None) tuples.
        output_path: Destination file path (created/overwritten).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    blocks = [render_question(pq, ans) for pq, ans in questions]
    content = _PREAMBLE + "\n\n".join(blocks) + _POSTAMBLE

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Wrote %d question(s) to %s", len(questions), output_path)


def append_to_combined(
    questions: list[tuple[ParsedQuestion, Optional[str]]],
    combined_path: str,
    source_label: str,
) -> None:
    """
    Append questions to a combined .tex file.

    The file is created with a preamble on the first call and extended on
    subsequent calls.  Call ``finalise_combined`` once all PDFs are processed
    to write the closing \\end{document}.

    Args:
        questions:     Questions to append.
        combined_path: Path to the combined output file.
        source_label:  Human-readable label (e.g. PDF filename) for a comment.
    """
    path = Path(combined_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if path.exists() else "w"
    with open(combined_path, mode, encoding="utf-8") as fh:
        if mode == "w":
            fh.write(_PREAMBLE)
        fh.write(f"\n% --- {source_label} ---\n\n")
        for pq, ans in questions:
            fh.write(render_question(pq, ans))
            fh.write("\n\n")


def finalise_combined(combined_path: str) -> None:
    """Write the closing \\end{questions}/\\end{document} to *combined_path*."""
    with open(combined_path, "a", encoding="utf-8") as fh:
        fh.write(_POSTAMBLE)
    logger.info("Finalised combined output: %s", combined_path)
