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


# --- snippet language mapping ---------------------------------------------------


def test_language_for_file() -> None:
    assert quiz_module.language_for_file("src/app/main.py") == "python"
    assert quiz_module.language_for_file("pkg/broker.go") == "go"
    assert quiz_module.language_for_file("src/Cart.tsx") == "typescript"
    assert quiz_module.language_for_file("Makefile") == "text"


# --- streaming ----------------------------------------------------------------


def test_question_stream_delivers_in_order_and_prefetches() -> None:
    questions = [make_question(node_id=f"n{i}", text=f"Q{i}?") for i in range(4)]
    stream = quiz_module.QuestionStream(iter(questions), prefetch=2)
    assert [q.question for q in stream] == ["Q0?", "Q1?", "Q2?", "Q3?"]


def test_question_stream_error_only_when_nothing_delivered() -> None:
    def empty_failing():
        raise ValueError("nothing worked")
        yield  # pragma: no cover

    with pytest.raises(ValueError):
        list(quiz_module.QuestionStream(empty_failing()))

    def partial():
        yield make_question(text="Q1?")
        raise ValueError("later failure")

    # Partial delivery: the stream just ends early, no exception.
    delivered = list(quiz_module.QuestionStream(partial()))
    assert len(delivered) == 1


def test_run_quiz_accepts_streaming_input(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = [
        make_question(node_id="n1", text="Q1?", correct="A"),
        make_question(node_id="n2", text="Q2?", correct="B"),
    ]
    pressed = iter(["A", "B"])
    monkeypatch.setattr(quiz_module, "collect_keypress", lambda: next(pressed))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100)
    result = run_quiz(
        quiz_module.QuestionStream(iter(questions)),
        session_type="quiz",
        pass_threshold=2,
        console=console,
        total=5,  # stream planned 5 but delivered 2 — grade what was asked
    )
    assert (result.score, result.total, result.passed) == (2, 2, True)
    assert "Question 1 of 5" in buffer.getvalue()


def test_markdown_snippets_render_as_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    question = make_question(node_id="docs/x.md", text="Which value belongs?", correct="A")
    question.snippet = "## Levels\n\n| Level | Content |\n|---|---|\n| L0 | claim |\n| L2 | chunk |"
    question.language = "markdown"
    pressed = iter(["A"])
    monkeypatch.setattr(quiz_module, "collect_keypress", lambda: next(pressed))
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=100)
    run_quiz([question], session_type="quiz", pass_threshold=1, console=console)
    output = buffer.getvalue()
    assert "Levels" in output and "chunk" in output
    assert "|---|" not in output        # the raw pipe-table syntax is gone
    assert "## Levels" not in output    # headings rendered, not shown raw


# --- the app layer (session blueprint + launcher policy) -------------------------


def test_quiz_blueprint_mixes_categories(monkeypatch) -> None:
    from roger import session
    from roger.config import Config

    doc_q = make_question(node_id="doc", text="Doc opener?")
    code_qs = [make_question(node_id=f"c{i}", text=f"Code {i}?") for i in range(3)]
    design_q = make_question(node_id="__design__", text="Design closer?")

    monkeypatch.setattr(session, "doc_questions", lambda **kw: [doc_q])
    monkeypatch.setattr(session, "iter_questions", lambda *a, **kw: iter(code_qs))
    monkeypatch.setattr(session, "get_design_questions", lambda *a, **kw: [design_q])
    monkeypatch.setattr(session, "DESIGN_NODE_ID", "__design__")

    stream, names, total = session.quiz_blueprint(
        __import__("networkx").DiGraph(), Config(), count=5,
        node_picker=lambda graph, n: [f"c{i}" for i in range(n)],
    )
    questions = list(stream)
    assert total == 5
    assert questions[0].node_id == "doc"          # instant opener leads
    assert questions[-1].node_id == "__design__"  # design closes
    assert {q.node_id for q in questions} == {"doc", "c0", "c1", "c2", "__design__"}
    assert names["__design__"] == "system design (module map)"


def test_streamlit_remedy_detects_environment(monkeypatch) -> None:
    import sys

    from roger import cli

    monkeypatch.setattr(sys, "prefix", "/Users/x/.local/pipx/venvs/roger-cli")
    assert cli._streamlit_missing_remedy() == "pipx inject roger-cli streamlit"


def test_ensure_streamlit_errors_for_non_tty(monkeypatch) -> None:
    import importlib.util

    import typer

    from roger import cli

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        cli.importlib.util, "find_spec",
        lambda name, *a: None if name == "streamlit" else real_find_spec(name, *a),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        "typer.confirm", lambda *a, **k: pytest.fail("prompted a non-TTY caller")
    )
    with pytest.raises(typer.Exit) as excinfo:
        cli._ensure_streamlit()
    assert excinfo.value.exit_code == 1


# --- the app's visual system (design-review contract) ----------------------------


def test_style_is_fully_local() -> None:
    # The privacy promise: no web fonts, no CDNs, no URLs of any kind.
    from roger.style import STYLE, THEME_ENV

    assert "http" not in STYLE.lower()
    assert "url(" not in STYLE.lower()
    assert "@import" not in STYLE.lower()
    assert THEME_ENV["STREAMLIT_THEME_BASE"] == "light"  # deliberately pinned


def test_style_covers_the_design_states() -> None:
    from roger.style import STYLE

    for marker in (
        "stateok", "stateno", "stateans", "statemut",   # graded option rows
        "st-key-primary", "st-key-next", "st-key-link",
        "turnuser", "turnroger", "rog-thinking", "rog-why",
        "rog-badge", "rog-file", "stButtonGroup",       # v2: badge, code weld, nav
        "rog-ego", "rog-hoplink", "rog-edge", "rog-sub",  # explore + subtitle
        "st-key-nb", "st-key-hop", "stSelectbox",
    ):
        assert marker in STYLE, marker
    # help= tooltips wrap buttons in extra spans — a child combinator here
    # silently unstyles every button that has a tooltip.
    assert "div.stButton > button" not in STYLE


def test_app_has_the_explore_view() -> None:
    from pathlib import Path

    source = Path("roger/app.py").read_text(encoding="utf-8")
    assert '"Quiz", "Ask", "Explore"' in source
    assert "_explore_view" in source
    assert "explain_data" in source and "path_data" in source


def test_app_env_pins_theme_and_privacy(monkeypatch) -> None:
    from roger import cli

    monkeypatch.setenv("GRAPHIFY_FORCE", "1")
    env = cli._app_env()
    assert env["STREAMLIT_SERVER_HEADLESS"] == "true"
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert env["STREAMLIT_THEME_BASE"] == "light"
    assert env["STREAMLIT_THEME_PRIMARY_COLOR"] == "#C1683F"
    assert "GRAPHIFY_FORCE" not in env  # footgun scrubbed, as everywhere


def test_table_snippets_are_valid_markdown_tables() -> None:
    # Field regression: table snippets lacked the |---| separator row, so
    # every renderer collapsed the table into one paragraph of pipes.
    import random

    from roger.docs import DocSection, table_questions

    text = (
        "| Command | What it does |\n"
        "| --- | --- |\n"
        "| alpha | does a |\n"
        "| beta | does b |\n"
        "| gamma | does c |\n"
        "| delta | does d |\n"
    )
    section = DocSection(file="README.md", heading="Commands", text=text)
    questions = table_questions([section], "medium", random.Random(7), limit=1)
    assert questions, "expected a table question from a 4-row table"
    lines = questions[0].snippet.splitlines()
    assert lines[0].startswith("| ")                      # header
    assert set(lines[1].replace("|", "").strip()) <= {"-", " "}  # separator row
    assert len(lines) >= 5
