"""Tests for grading and the quiz runner (keypress input mocked)."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from roger import quiz as quiz_module
from roger.grader import grade_answer, has_passed, score_answers
from roger.models import QuizAnswer
from roger.quiz import run_quiz
from tests.conftest import make_question

# --- grader -------------------------------------------------------------------


def test_grade_answer_correct_and_case_insensitive() -> None:
    question = make_question(correct="B")
    assert grade_answer(question, "B")
    assert grade_answer(question, "b")
    assert grade_answer(question, " b ")


def test_grade_answer_wrong_or_missing() -> None:
    question = make_question(correct="B")
    assert not grade_answer(question, "A")
    assert not grade_answer(question, None)
    assert not grade_answer(question, "")


def test_score_answers() -> None:
    q = make_question()
    answers = [
        QuizAnswer(question=q, user_answer="B", is_correct=True, time_taken_secs=1.0),
        QuizAnswer(question=q, user_answer="A", is_correct=False, time_taken_secs=1.0),
        QuizAnswer(question=q, user_answer="B", is_correct=True, time_taken_secs=1.0),
    ]
    assert score_answers(answers) == 2


def test_has_passed_threshold() -> None:
    assert has_passed(3, 5, pass_threshold=3)
    assert not has_passed(2, 5, pass_threshold=3)
    # Threshold clamps to quiz size: 2/2 with threshold 3 still passes.
    assert has_passed(2, 2, pass_threshold=3)
    assert has_passed(0, 0, pass_threshold=3)  # empty quiz never fails


# --- run_quiz -----------------------------------------------------------------


def _run(questions, keys: list[str], monkeypatch: pytest.MonkeyPatch):
    pressed = iter(keys)
    monkeypatch.setattr(quiz_module, "collect_keypress", lambda: next(pressed))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100)
    result = run_quiz(questions, session_type="quiz", pass_threshold=2, console=console)
    return result, buffer.getvalue()


def test_run_quiz_all_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = [
        make_question(node_id="n1", text="Q1?", correct="A"),
        make_question(node_id="n2", text="Q2?", correct="C"),
    ]
    result, output = _run(questions, ["A", "C"], monkeypatch)

    assert result.score == 2
    assert result.total == 2
    assert result.passed
    assert result.session_type == "quiz"
    assert result.weak_nodes == []
    assert result.duration_secs >= 0
    assert "✓ Correct" in output
    assert "Quiz passed" in output


def test_run_quiz_records_wrong_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = [
        make_question(node_id="n1", text="Q1?", correct="A"),
        make_question(node_id="n2", text="Q2?", correct="B"),
        make_question(node_id="n3", text="Q3?", correct="C"),
    ]
    result, output = _run(questions, ["A", "D", "D"], monkeypatch)

    assert result.score == 1
    assert result.total == 3
    assert not result.passed  # threshold 2, scored 1
    assert result.weak_nodes == ["n2", "n3"]
    assert [a.user_answer for a in result.answers] == ["A", "D", "D"]
    assert [a.is_correct for a in result.answers] == [True, False, False]
    assert "✗ Incorrect" in output
    assert "Quiz failed" in output
    assert "roger quiz --module" in output  # weak-area tip shown


def test_run_quiz_shows_question_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    question = make_question(node_id="payments.charge", text="What does charge return?")
    _, output = _run([question], ["B"], monkeypatch)

    assert "Question 1 of 1" in output
    assert "payments.charge" in output
    assert "What does charge return?" in output
    for option_text in question.options.values():
        assert option_text in output
    assert question.explanation in output


def test_run_quiz_empty_question_list(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _run([], [], monkeypatch)
    assert result.total == 0
    assert result.score == 0
    assert result.passed  # nothing to fail


def test_run_quiz_header_and_summary_use_display_names(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = [make_question(node_id="pkg_module_do_work_slug", text="Q1?", correct="A")]
    pressed = iter(["B"])  # wrong on purpose so the weak list renders
    monkeypatch.setattr(quiz_module, "collect_keypress", lambda: next(pressed))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    run_quiz(
        questions,
        session_type="quiz",
        pass_threshold=1,
        console=console,
        node_names={"pkg_module_do_work_slug": "do_work (src/module.py)"},
    )
    output = buffer.getvalue()
    assert "do_work (src/module.py)" in output       # header + weak list
    assert "pkg_module_do_work_slug" not in output   # slug never shown


def test_node_display_names_builds_labels() -> None:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("pkg_do_work", display="do_work", file="src/module.py")
    graph.add_node("bare")
    questions = [
        make_question(node_id="pkg_do_work", text="Q1?"),
        make_question(node_id="bare", text="Q2?"),
        make_question(node_id="not_in_graph", text="Q3?"),
    ]
    names = quiz_module.node_display_names(graph, questions)
    assert names == {"pkg_do_work": "do_work (src/module.py)", "bare": "bare"}


def test_run_quiz_renders_snippet_and_escapes_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    question = make_question(node_id="n1", text="Which is the real line?", correct="A")
    question.snippet = "if items[0] == sentinel:\n    ________________"
    question.options = {"A": "return [x] or None", "B": "raise KeyError", "C": "pass", "D": "break"}
    pressed = iter(["A"])
    monkeypatch.setattr(quiz_module, "collect_keypress", lambda: next(pressed))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100)
    run_quiz([question], session_type="quiz", pass_threshold=1, console=console)
    output = buffer.getvalue()
    assert "items[0] == sentinel" in output   # snippet shown, brackets survive
    assert "________________" in output
    assert "return [x] or None" in output     # option markup not eaten by Rich


# --- web quiz ------------------------------------------------------------------


def test_language_for_file() -> None:
    assert quiz_module.language_for_file("src/app/main.py") == "python"
    assert quiz_module.language_for_file("pkg/broker.go") == "go"
    assert quiz_module.language_for_file("src/Cart.tsx") == "typescript"
    assert quiz_module.language_for_file("Makefile") == "text"


def test_render_quiz_html_and_record_roundtrip(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from roger import webquiz

    monkeypatch.chdir(tmp_path)
    questions = [
        make_question(node_id="n1", text="Q1?", correct="B"),
        make_question(node_id="n2", text="Q2 with </script> inside?", correct="A"),
    ]
    questions[0].snippet = "def f():\n    return [1] < [2]"
    questions[0].language = "python"

    page = webquiz.render_quiz_html(
        questions, session_type="quiz", pass_threshold=1,
        node_names={"n1": "f (src/f.py)"},
    )
    html = page.read_text(encoding="utf-8")
    assert "quiz-data" in html and "language-python" not in html  # set client-side
    assert "<\\/script>" in html          # snippet cannot break the embed tag
    assert "f (src/f.py)" in html
    assert webquiz.PENDING_PATH.is_file()

    # Round-trip: answers B (right), B (wrong) → 1/2, threshold 1 → passed.
    result = webquiz.record_answer_code("BB")
    assert (result.score, result.total, result.passed) == (1, 2, True)
    assert [a.is_correct for a in result.answers] == [True, False]
    assert not webquiz.PENDING_PATH.is_file()  # consumed


def test_record_rejects_bad_codes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from roger import webquiz

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        webquiz.record_answer_code("AB")  # no pending session at all

    webquiz.render_quiz_html(
        [make_question(text="Q?")], session_type="quiz", pass_threshold=1
    )
    with pytest.raises(ValueError):
        webquiz.record_answer_code("ABX")  # wrong length / invalid letter


def test_quiz_template_file_matches_embedded_copy() -> None:
    from pathlib import Path

    from roger.webquiz import EMBEDDED_TEMPLATE

    template = Path(__file__).resolve().parent.parent / "templates" / "quiz.html.jinja"
    assert template.read_text(encoding="utf-8") == EMBEDDED_TEMPLATE
