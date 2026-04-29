"""
latex_writer.py — Render parsed questions as exam-class LaTeX.

Produces \\question / \\begin{choices} / \\choice / \\CorrectChoice blocks
that compile directly with the ``exam`` document class.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from parsing import ParsedQuestion, CHOICE_LETTERS

logger = logging.getLogger(__name__)

# Preamble written at the top of every generated .tex file.
_PREAMBLE = r"""\documentclass[12pt,addpoints]{exam}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{enumitem}
\usepackage{graphicx}

\begin{document}
\begin{questions}
"""

_POSTAMBLE = r"""
\end{questions}
\end{document}
"""


def _escape_percent(text: str) -> str:
    """Escape bare % characters that aren't already LaTeX comments."""
    return re.sub(r"(?<!\\)%", r"\\%", text)


def _count_unescaped_dollars(text: str) -> int:
    return len(re.findall(r"(?<!\\)\$", text))


def _balance_delimited_math(text: str) -> str:
    """
    Repair common OCR delimiter mistakes by appending missing closing tokens.

    This is intentionally conservative: it does not try to rewrite nested
    math, only ensures that unmatched opening delimiters do not break the
    whole document.
    """
    repaired = text

    paren_open = repaired.count(r"\(")
    paren_close = repaired.count(r"\)")
    if paren_open > paren_close:
        repaired += r"\)" * (paren_open - paren_close)

    bracket_open = repaired.count(r"\[")
    bracket_close = repaired.count(r"\]")
    if bracket_open > bracket_close:
        repaired += r"\]" * (bracket_open - bracket_close)

    dollar_count = _count_unescaped_dollars(repaired)
    if dollar_count % 2 == 1:
        repaired += "$"

    return repaired


def _looks_like_bare_math(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(token in stripped for token in (r"\(", r"\[", "$$", "$")):
        return False
    if len(stripped.split()) > 4 and not any(ch in stripped for ch in "=+-*/^_[]{}()\\"):
        return False

    import re
    if re.fullmatch(r"[A-Za-z0-9\\\^_{}\[\]()+\-*/=<>|.,'` :]+", stripped) is None:
        return False

    return any(
        (
            re.search(r"[=^_]", stripped),
            re.search(r"[A-Za-z]\(", stripped),
            re.search(r"\\[A-Za-z]+", stripped),
            re.search(r"\[[^\]]+\]", stripped),
            re.search(r"\d", stripped) and re.search(r"[A-Za-z]", stripped),
        )
    )


def _render_text(text: str) -> str:
    cleaned = _balance_delimited_math(_escape_percent(text))
    if _looks_like_bare_math(cleaned):
        return rf"\({cleaned}\)"
    return cleaned


def _render_figures(lines: list[str], figures: list[dict]) -> None:
    for fig in figures:
        latex_path = fig.get("latex_path")
        if not latex_path:
            continue
        lines.append(r"\begin{center}")
        lines.append(rf"\includegraphics[width=0.65\linewidth]{{{latex_path}}}")
        lines.append(r"\end{center}")
        caption = fig.get("caption")
        if caption:
            lines.append(f"% Figure: {_escape_percent(caption)}")


def _render_tables(lines: list[str], tables: list[dict]) -> None:
    for table in tables:
        latex = table.get("latex")
        if not latex:
            continue
        lines.append(latex)
        caption = table.get("caption")
        if caption:
            lines.append(f"% Table: {_escape_percent(caption)}")


def _by_placement(items: list[dict], placement: str) -> list[dict]:
    return [item for item in items if item.get("placement", "stem") == placement]


def render_question(
    parsed: ParsedQuestion,
    correct_answer: Optional[str],
    source: Optional[str] = None,
) -> str:
    """
    Render one question as a LaTeX ``\\question`` block.

    Args:
        parsed:         Output of ``parse_question``.
        correct_answer: Letter A–E, or None if unknown.
        source:         Optional label (e.g. filename + page) added as a
                        comment directly under the \\question line.

    Returns:
        Multi-line LaTeX string (no trailing newline).
    """
    lines: list[str] = []
    lines.append(r"\question")
    if source:
        lines.append(f"% {source}")
    lines.append(_render_text(parsed["question"]))
    _render_tables(lines, _by_placement(parsed.get("tables", []), "stem"))
    _render_figures(lines, _by_placement(parsed.get("figures", []), "stem"))
    lines.append(r"\begin{choices}")

    choices = parsed.get("choices", {})
    for letter in CHOICE_LETTERS:
        text = choices.get(letter)
        if text is None:
            continue
        if correct_answer and letter == correct_answer.upper():
            lines.append(rf"\CorrectChoice {_render_text(text)}")
        else:
            lines.append(rf"\choice {_render_text(text)}")
        _render_tables(lines, _by_placement(parsed.get("tables", []), letter))
        _render_figures(lines, _by_placement(parsed.get("figures", []), letter))

    if not choices:
        lines.append(r"\choice % TODO: choices not parsed")

    if correct_answer is None:
        lines.append(r"% TODO: correct answer not detected")

    lines.append(r"\end{choices}")

    solution = parsed.get("solution")
    solution_figures = parsed.get("solution_figures", [])
    solution_tables = parsed.get("solution_tables", [])
    if solution or solution_figures or solution_tables:
        lines.append(r"\begin{solution}")
        _render_tables(lines, _by_placement(solution_tables, "stem"))
        _render_figures(lines, _by_placement(solution_figures, "stem"))
        if solution:
            lines.append(_render_text(solution))
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
    blocks = [render_question(pq, ans, src) for pq, ans, src in questions]
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
        for pq, ans, src in questions:
            fh.write(render_question(pq, ans, src))
            fh.write("\n\n")


def finalise_combined(combined_path: str) -> None:
    """Write the closing \\end{questions}/\\end{document} to *combined_path*."""
    with open(combined_path, "a", encoding="utf-8") as fh:
        fh.write(_POSTAMBLE)
    logger.info("Finalised combined output: %s", combined_path)
