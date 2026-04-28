"""Tests for latex_writer.py."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latex_writer import render_question, write_tex_file
from parsing import ParsedQuestion


def _make_parsed(q: str = "What is 1+1?", include_all=True) -> ParsedQuestion:
    choices = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"} if include_all else {}
    return ParsedQuestion(question=q, choices=choices)


def test_correct_choice_marker():
    block = render_question(_make_parsed(), correct_answer="B")
    assert r"\CorrectChoice 2" in block
    assert r"\choice 1" in block  # A is wrong
    assert block.count(r"\CorrectChoice") == 1


def test_unknown_answer_inserts_todo():
    block = render_question(_make_parsed(), correct_answer=None)
    assert "TODO: correct answer not detected" in block
    assert r"\CorrectChoice" not in block


def test_all_five_choices_present():
    block = render_question(_make_parsed(), correct_answer="A")
    for letter in ["1", "2", "3", "4", "5"]:
        assert letter in block


def test_write_tex_file_creates_valid_latex(tmp_path):
    out = str(tmp_path / "test.tex")
    questions = [(_make_parsed("Q1?"), "A"), (_make_parsed("Q2?"), None)]
    write_tex_file(questions, out)
    content = Path(out).read_text()
    assert r"\documentclass" in content
    assert r"\begin{questions}" in content
    assert r"\end{document}" in content
    assert content.count(r"\question") == 2


def test_empty_choices_falls_back_gracefully():
    parsed = ParsedQuestion(question="Stem", choices={})
    block = render_question(parsed, correct_answer="A")
    assert "% TODO: choices not parsed" in block
