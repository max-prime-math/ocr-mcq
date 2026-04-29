"""
parsing.py — Extract question stem and A–E answer choices from OCR text.

Mathpix returns LaTeX-enriched text.  This module splits that text into a
structured dict so the rest of the pipeline never has to touch regexes.

Supported choice label formats (case-insensitive):
    (A)   A.   A)   \\text{(A)}   \\mathbf{(A)}
"""

import logging
import re
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)

# Ordered list of valid choice letters.
CHOICE_LETTERS = ["A", "B", "C", "D", "E"]

# Regex that matches the start of an answer-choice line in various formats.
# Group 1 captures the letter.
_CHOICE_PATTERN = re.compile(
    r"(?:"
    r"\\(?:text|mathbf|mathit|mathrm)\{[(\s]*([A-Ea-e])[)\s]*\}"  # \text{(A)}
    r"|^\(?([A-Ea-e])\)?[.)]\s"                                     # (A)  A.  A)
    r")",
    re.MULTILINE,
)

# Simpler fallback: a line that starts with an isolated letter + delimiter.
_SIMPLE_CHOICE = re.compile(
    r"^[(\[]?([A-Ea-e])[)\].][ \t]+",
    re.MULTILINE,
)


class ParsedQuestion(TypedDict):
    question: str
    choices: dict[str, str]
    solution: str | None
    figures: list["FigureSpec"]


class FigureSpec(TypedDict, total=False):
    page: Literal[1, 2]
    x: float
    y: float
    width: float
    height: float
    caption: str | None
    latex_path: str


def _find_choice_spans(text: str) -> list[tuple[int, str]]:
    """
    Return a list of (char_offset, letter) for each choice label found in
    *text*, in document order.  Deduplicates consecutive hits for the same
    letter.
    """
    spans: list[tuple[int, str]] = []
    seen_letters: set[str] = set()

    for pat in (_CHOICE_PATTERN, _SIMPLE_CHOICE):
        for m in pat.finditer(text):
            letter = (m.group(1) or m.group(2) or "").upper()
            if letter not in CHOICE_LETTERS:
                continue
            if letter not in seen_letters:
                spans.append((m.start(), letter))
                seen_letters.add(letter)

        if len(seen_letters) >= 2:
            break  # found enough with the stricter pattern

    spans.sort(key=lambda x: x[0])
    return spans


def parse_question(ocr_text: str) -> ParsedQuestion:
    """
    Split OCR text into a question stem and five answer choices.

    If fewer than two choice labels are found the entire text is treated as
    the question stem and choices are returned empty — the caller is
    responsible for flagging this as a parse failure.

    Args:
        ocr_text: The ``"text"`` or ``"latex_styled"`` field from Mathpix.

    Returns:
        ParsedQuestion with keys ``"question"`` and ``"choices"``.
    """
    spans = _find_choice_spans(ocr_text)

    if len(spans) < 2:
        logger.warning("Fewer than 2 choice labels found — treating all text as stem.")
        return ParsedQuestion(question=ocr_text.strip(), choices={}, solution=None, figures=[])

    # Everything before the first choice label is the question stem.
    stem_end = spans[0][0]
    question = ocr_text[:stem_end].strip()

    choices: dict[str, str] = {}
    for i, (start, letter) in enumerate(spans):
        # Slice from just after the label to the start of the next label.
        label_match_end = _label_end(ocr_text, start)
        text_start = label_match_end
        text_end = spans[i + 1][0] if i + 1 < len(spans) else len(ocr_text)
        choices[letter] = ocr_text[text_start:text_end].strip()

    # Warn if we got fewer than 5 choices — likely a parse issue.
    missing = [l for l in CHOICE_LETTERS if l not in choices]
    if missing:
        logger.warning("Missing choice(s): %s", missing)

    return ParsedQuestion(question=question, choices=choices, solution=None, figures=[])


def _label_end(text: str, label_start: int) -> int:
    """Return the index of the first character *after* the choice label."""
    # Walk forward past the label pattern.
    for pat in (_CHOICE_PATTERN, _SIMPLE_CHOICE):
        m = pat.match(text, label_start)
        if m:
            return m.end()
    # Fallback: skip 3 characters.
    return label_start + 3
