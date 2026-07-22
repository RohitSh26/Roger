"""Terminal quiz runner: Rich display, single-keypress answers, immediate feedback."""

from __future__ import annotations

import sys
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from roger.grader import grade_answer, has_passed, score_answers
from roger.models import Question, QuizAnswer, QuizResult

VALID_KEYS = ("A", "B", "C", "D")


def collect_keypress() -> str:
    """Capture single keypress A/B/C/D without requiring Enter.

    Falls back to line input when stdin is not a TTY (tests, pipes).
    """
    while True:
        key = _read_key().upper()
        if key in VALID_KEYS:
            return key


def _read_key() -> str:
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed while waiting for an answer")
        return line.strip()[:1]

    if sys.platform == "win32":  # pragma: no cover - not exercised on darwin CI
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if char == "\x03":  # Ctrl+C in raw mode
        raise KeyboardInterrupt
    return char


def _show_question(console: Console, question: Question, index: int, total: int) -> None:
    body_lines = [question.question, ""]
    for key in VALID_KEYS:
        body_lines.append(f"  [bold]{key}[/bold]) {question.options[key]}")
    console.print(
        Panel(
            "\n".join(body_lines),
            title=f"Question {index} of {total} | {question.node_id}",
            title_align="left",
            border_style="cyan",
        )
    )


def _show_feedback(console: Console, question: Question, is_correct: bool, score: int, asked: int) -> None:
    if is_correct:
        console.print("[bold green]✓ Correct[/bold green]")
    else:
        console.print(
            f"[bold red]✗ Incorrect[/bold red] — correct answer: "
            f"[bold]{question.correct}[/bold]) {question.options[question.correct]}"
        )
    if question.explanation:
        console.print(f"[dim]{question.explanation}[/dim]")
    console.print(f"[dim]Score so far: {score}/{asked}[/dim]\n")


def _show_summary(console: Console, result: QuizResult) -> None:
    lines = [
        f"Score: [bold]{result.score}/{result.total}[/bold]",
        f"Time: {result.duration_secs:.0f}s",
    ]
    if result.weak_nodes:
        lines.append("")
        lines.append("Review these:")
        lines.extend(f"  • {node_id}" for node_id in dict.fromkeys(result.weak_nodes))
        lines.append("")
        lines.append("Run 'roger quiz --module <path>' to focus on weak areas")
    style = "green" if result.passed else "red"
    title = "Quiz passed" if result.passed else "Quiz failed"
    console.print(Panel("\n".join(lines), title=title, border_style=style))


def run_quiz(
    questions: list[Question],
    session_type: str,
    pass_threshold: int = 3,
    module_scope: Optional[str] = None,
    console: Optional[Console] = None,
) -> QuizResult:
    """Display each question, collect answers, show feedback, return the result."""
    console = console or Console()
    answers: list[QuizAnswer] = []
    quiz_start = time.monotonic()

    for index, question in enumerate(questions, start=1):
        _show_question(console, question, index, len(questions))
        console.print("[dim]Answer (A/B/C/D):[/dim] ", end="")

        question_start = time.monotonic()
        user_answer = collect_keypress()
        time_taken = time.monotonic() - question_start
        console.print(user_answer)

        is_correct = grade_answer(question, user_answer)
        answers.append(
            QuizAnswer(
                question=question,
                user_answer=user_answer,
                is_correct=is_correct,
                time_taken_secs=time_taken,
            )
        )
        _show_feedback(console, question, is_correct, score_answers(answers), len(answers))

    score = score_answers(answers)
    total = len(questions)
    result = QuizResult(
        session_type=session_type,
        answers=answers,
        score=score,
        total=total,
        passed=has_passed(score, total, pass_threshold),
        commit_hash=None,
        module_scope=module_scope,
        duration_secs=time.monotonic() - quiz_start,
    )
    _show_summary(console, result)
    return result
