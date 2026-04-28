"""Tests for parsing.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from parsing import parse_question, CHOICE_LETTERS


SAMPLE_TEXT = r"""Find the derivative of \(f(x) = x^2 + 3x\).

(A) \(2x + 3\)
(B) \(x^2 + 3\)
(C) \(2x\)
(D) \(x + 3\)
(E) \(2x^2 + 3\)
"""

SAMPLE_TEXT_DOT = r"""Which of the following equals \(\lim_{x \to 0} \frac{\sin x}{x}\)?

A. 0
B. 1
C. \(\infty\)
D. undefined
E. \(\pi\)
"""

SAMPLE_TEXT_MATHPIX = r"""Evaluate \(\int_0^1 x\,dx\).

\text{(A)} \(\frac{1}{4}\)
\text{(B)} \(\frac{1}{2}\)
\text{(C)} 1
\text{(D)} 2
\text{(E)} \(\frac{3}{4}\)
"""


def test_parse_paren_labels():
    result = parse_question(SAMPLE_TEXT)
    assert result["question"].startswith("Find the derivative")
    assert set(result["choices"].keys()) == set(CHOICE_LETTERS)
    assert r"2x + 3" in result["choices"]["A"]


def test_parse_dot_labels():
    result = parse_question(SAMPLE_TEXT_DOT)
    assert "lim" in result["question"]
    assert set(result["choices"].keys()) == set(CHOICE_LETTERS)
    assert result["choices"]["B"].strip() == "1"


def test_parse_mathpix_text_labels():
    result = parse_question(SAMPLE_TEXT_MATHPIX)
    assert r"\int" in result["question"]
    assert len(result["choices"]) == 5


def test_parse_no_choices_returns_full_text():
    text = "This is just a paragraph with no choices at all."
    result = parse_question(text)
    assert result["question"] == text.strip()
    assert result["choices"] == {}


def test_parse_partial_choices_warns(caplog):
    import logging
    text = "(A) first choice\n(B) second choice\n"
    with caplog.at_level(logging.WARNING, logger="parsing"):
        result = parse_question(text)
    assert "Missing choice" in caplog.text
    assert "A" in result["choices"]
    assert "B" in result["choices"]
