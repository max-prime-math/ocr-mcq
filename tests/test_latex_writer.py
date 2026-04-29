"""Tests for latex_writer.py."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latex_writer import render_question, write_tex_file
from parsing import ParsedQuestion


def _make_parsed(q: str = "What is 1+1?", include_all=True) -> ParsedQuestion:
    choices = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"} if include_all else {}
    return ParsedQuestion(
        question=q,
        choices=choices,
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )


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
    questions = [(_make_parsed("Q1?"), "A", "test.pdf — page 1"), (_make_parsed("Q2?"), None, "test.pdf — page 2")]
    write_tex_file(questions, out)
    content = Path(out).read_text()
    assert r"\documentclass" in content
    assert r"\begin{questions}" in content
    assert r"\end{document}" in content
    assert content.count(r"\question") == 2


def test_empty_choices_falls_back_gracefully():
    parsed = ParsedQuestion(question="Stem", choices={}, solution=None, figures=[], solution_figures=[], tables=[], solution_tables=[])
    block = render_question(parsed, correct_answer="A")
    assert "% TODO: choices not parsed" in block


def test_choice_math_is_wrapped_when_not_in_math_mode():
    parsed = ParsedQuestion(
        question="Stem",
        choices={
            "A": "f''(g(x))[g'(x)]^2 + f'(g(x))g''(x)",
            "B": "undefined",
            "C": "0",
            "D": "1",
            "E": "2",
        },
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\CorrectChoice \(f''(g(x))[g'(x)]^2 + f'(g(x))g''(x)\)" in block
    assert r"\choice undefined" in block


def test_choice_interval_text_is_wrapped_and_unicode_math_is_normalised():
    parsed = ParsedQuestion(
        question="Stem",
        choices={
            "A": "(−∞, 0.831) and (7.384, ∞)",
            "B": "1 ≤ x ≤ 1.691",
            "C": "plain text",
            "D": "0",
            "E": "1",
        },
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\CorrectChoice \((-\infty, 0.831) and (7.384, \infty)\)" in block
    assert r"\choice \(1 \le x \le 1.691\)" in block
    assert r"\choice plain text" in block


def test_solution_normalises_unicode_math_inside_existing_delimiters():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"The interval is $(−\infty, ∞)$.",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"The interval is $(-\infty, \infty)$." in block


def test_question_wraps_embedded_command_math_inside_prose():
    parsed = ParsedQuestion(
        question=r"What are all values of k for which \int_{-3}^{k} x^2 dx = 0?",
        choices={"A": "-3", "B": "0", "C": "3", "D": "-3 and 3", "E": "-3, 0, and 3"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"What are all values of k for which \(\int_{-3}^{k} x^2 dx = 0\)?" in block


def test_question_table_array_block_is_wrapped_in_display_math():
    parsed = ParsedQuestion(
        question="Use the table.",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[{"latex": r"\begin{array}{|c|c|}\hline x & f(x) \\\hline 1 & 2 \\\hline\end{array}", "placement": "stem"}],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\[\begin{array}{|c|c|}\hline x & f(x) \\\hline 1 & 2 \\\hline\end{array}\]" in block


def test_question_strips_control_chars_and_repairs_igint_token():
    parsed = ParsedQuestion(
        question="The length is \x08igint_1^4 \\sqrt{1+9x^4}\\, dx.",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert "\x08" not in block
    assert r"\(\int_1^4 \sqrt{1+9x^4}\, dx\)" in block


def test_solution_removes_nested_inline_math_inside_dollar_math():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"$x^2 e^x - x^2 - x^3 = \(\frac{x^4}{2\)!} + \(\frac{x^5}{3\)!}$$",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"$x^2 e^x - x^2 - x^3 = \frac{x^4}{2!} + \frac{x^5}{3!}$" in block


def test_solution_dedupes_repeated_inline_math_closers():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"number of barrels = \(\int_0^{24} r(t)\,dt = 3000\)\)",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"number of barrels = \(\int_0^{24} r(t)\,dt = 3000\)" in block


def test_question_wraps_short_inline_equation_in_prose():
    parsed = ParsedQuestion(
        question="A particle moves so that its position is given by x(t) = t^2 - 6t + 5. Find when velocity is zero.",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="C")
    assert r"given by \(x(t) = t^2 - 6t + 5\)." in block


def test_choice_repairs_left_right_mixed_with_inline_delimiters():
    parsed = ParsedQuestion(
        question="Stem",
        choices={
            "A": r"\left(\(\frac{3x^2}{2} + x\right)^5 + C\)",
            "B": "2",
            "C": "3",
            "D": "4",
            "E": "5",
        },
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\CorrectChoice \(\left(\frac{3x^2}{2} + x\right)^5 + C\)" in block


def test_choice_trims_trailing_inline_closer_after_complete_math_block():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": r"\(\left(\frac{3x^2}{2} + x\right)^5 + C\)\)", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\CorrectChoice \(\left(\frac{3x^2}{2} + x\right)^5 + C\)" in block


def test_solution_normalises_triple_dollar_run():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"$x^2 e^x = x^2 + x^3$$$",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"$x^2 e^x = x^2 + x^3$" in block


def test_figures_are_included_in_output():
    parsed = ParsedQuestion(
        question="See the graph.",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[{"latex_path": "figures/sample.png", "placement": "stem"}],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="B")
    assert r"\includegraphics[width=0.65\linewidth]{figures/sample.png}" in block


def test_solution_closes_unmatched_inline_math():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"Differentiate \(x^2 + 1",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"Differentiate \(x^2 + 1\)" in block


def test_solution_closes_unmatched_display_math():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"Compute \[x^2 + 1",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"Compute \[x^2 + 1\]" in block


def test_solution_closes_odd_dollar_math():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=r"The slope is $m = 2x + 1",
        figures=[],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"The slope is $m = 2x + 1$" in block


def test_question_tables_render_before_choices():
    parsed = ParsedQuestion(
        question="Use the table.",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[{"latex": r"\begin{tabular}{cc}1 & 2\end{tabular}", "placement": "stem"}],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    assert r"\begin{tabular}{cc}1 & 2\end{tabular}" in block
    assert block.index(r"\begin{tabular}{cc}1 & 2\end{tabular}") < block.index(r"\begin{choices}")


def test_solution_figures_do_not_render_in_question_block():
    parsed = ParsedQuestion(
        question="Stem",
        choices={"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        solution="See figure.",
        figures=[],
        solution_figures=[{"latex_path": "figures/solution.png", "placement": "stem"}],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    solution_start = block.index(r"\begin{solution}")
    figure_pos = block.index(r"\includegraphics[width=0.65\linewidth]{figures/solution.png}")
    assert figure_pos > solution_start


def test_choice_specific_figure_renders_under_that_choice():
    parsed = ParsedQuestion(
        question="Pick the matching diagram.",
        choices={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D", "E": "Option E"},
        solution=None,
        figures=[{"latex_path": "figures/choice_b.png", "placement": "B"}],
        solution_figures=[],
        tables=[],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="C")
    choice_b = block.index(r"\choice Option B")
    fig = block.index(r"\includegraphics[width=0.65\linewidth]{figures/choice_b.png}")
    choice_c = block.index(r"\CorrectChoice Option C")
    assert choice_b < fig < choice_c


def test_choice_specific_table_renders_under_that_choice():
    parsed = ParsedQuestion(
        question="Choose the correct table.",
        choices={"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta", "E": "Epsilon"},
        solution=None,
        figures=[],
        solution_figures=[],
        tables=[{"latex": r"\begin{tabular}{c}42\end{tabular}", "placement": "D"}],
        solution_tables=[],
    )
    block = render_question(parsed, correct_answer="A")
    choice_d = block.index(r"\choice Delta")
    table = block.index(r"\begin{tabular}{c}42\end{tabular}")
    choice_e = block.index(r"\choice Epsilon")
    assert choice_d < table < choice_e
