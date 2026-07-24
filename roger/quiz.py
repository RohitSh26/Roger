"""Terminal quiz runner: Rich display, single-keypress answers, immediate feedback."""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Iterable, Iterator, Optional, Union

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from roger.grader import grade_answer, has_passed, score_answers
from roger.models import Question, QuizAnswer, QuizResult

VALID_KEYS = ("A", "B", "C", "D")

_LANGUAGE_BY_EXT = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".go": "go", ".java": "java",
    ".rb": "ruby", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cc": "cpp", ".cs": "csharp", ".sh": "bash", ".php": "php",
    ".kt": "kotlin", ".swift": "swift", ".scala": "scala", ".sql": "sql",
    ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
}


def language_for_file(file: str) -> str:
    """Highlighting language for a file path, or 'text' when unknown."""
    dot = file.rfind(".")
    return _LANGUAGE_BY_EXT.get(file[dot:].lower(), "text") if dot != -1 else "text"


class QuestionStream:
    """Iterate questions produced on a background thread.

    While the developer answers question N, question N+1 is generated
    behind the scenes; a small buffer keeps the pipeline a step ahead
    without generating the whole session up front. Mirrors
    iter_questions' error contract: a source error surfaces only if
    nothing was delivered — otherwise the stream simply ends early.
    """

    _DONE = object()

    def __init__(self, source: Iterable[Question], prefetch: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        self._error: Optional[BaseException] = None
        self._delivered_any = False
        self._thread = threading.Thread(target=self._fill, args=(iter(source),), daemon=True)
        self._thread.start()

    def _fill(self, source: Iterator[Question]) -> None:
        try:
            for item in source:
                self._queue.put(item)
        except BaseException as exc:  # surfaced on the consumer side
            self._error = exc
        finally:
            self._queue.put(self._DONE)

    def __iter__(self) -> "QuestionStream":
        return self

    def __next__(self) -> Question:
        item = self._queue.get()
        if item is self._DONE:
            if self._error is not None and not self._delivered_any:
                raise self._error
            raise StopIteration
        self._delivered_any = True
        return item


def node_display_names(graph, questions_or_ids: Iterable) -> dict[str, str]:
    """Readable 'name (file)' labels for the quiz header and summary.

    Accepts Questions or bare node ids — streaming callers know their node
    ids before any question exists.
    """
    names: dict[str, str] = {}
    for entry in questions_or_ids:
        node_id = entry.node_id if isinstance(entry, Question) else str(entry)
        if node_id not in graph.nodes:
            continue
        attrs = graph.nodes[node_id]
        name = str(attrs.get("display") or node_id)
        file = str(attrs.get("file") or "")
        names[node_id] = f"{name} ({file})" if file else name
    return names


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


def _show_question(
    console: Console, question: Question, index: int, total: int, node_label: str
) -> None:
    # Text renderables take code literally — no markup escaping worries —
    # and Syntax gives real highlighting plus line numbers in the terminal.
    parts: list = [Text(question.question), Text("")]
    if question.snippet:
        parts.append(
            Syntax(
                question.snippet,
                question.language or "text",
                theme="ansi_dark",
                line_numbers=True,
                word_wrap=True,
            )
        )
        parts.append(Text(""))
    options = Text()
    for key in VALID_KEYS:
        options.append(f"  {key}) ", style="bold")
        options.append(f"{question.options[key]}\n")
    parts.append(options)
    console.print(
        Panel(
            Group(*parts),
            title=f"Question {index} of {total} | {node_label}",
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
            f"[bold]{question.correct}[/bold]) {escape(question.options[question.correct])}"
        )
    if question.explanation:
        console.print(f"[dim]{escape(question.explanation)}[/dim]")
    console.print(f"[dim]Score so far: {score}/{asked}[/dim]\n")


def _show_summary(
    console: Console, result: QuizResult, node_names: dict[str, str]
) -> None:
    lines = [
        f"Score: [bold]{result.score}/{result.total}[/bold]",
        f"Time: {result.duration_secs:.0f}s",
    ]
    if result.weak_nodes:
        lines.append("")
        lines.append("Review these:")
        lines.extend(
            f"  • {node_names.get(node_id, node_id)}"
            for node_id in dict.fromkeys(result.weak_nodes)
        )
        lines.append("")
        lines.append("Run 'roger quiz --module <path>' to focus on weak areas")
    style = "green" if result.passed else "red"
    title = "Quiz passed" if result.passed else "Quiz failed"
    console.print(Panel("\n".join(lines), title=title, border_style=style))


def run_quiz(
    questions: Union[list[Question], Iterable[Question]],
    session_type: str,
    pass_threshold: int = 3,
    module_scope: Optional[str] = None,
    console: Optional[Console] = None,
    node_names: Optional[dict[str, str]] = None,
    total: Optional[int] = None,
) -> QuizResult:
    """Display each question, collect answers, show feedback, return the result.

    Accepts a list or a (possibly streaming) iterable; pass `total` for the
    "Question X of N" header when the input has no len(). node_names maps
    node ids to human-readable labels; ids are what gets recorded.
    """
    console = console or Console()
    node_names = node_names or {}
    answers: list[QuizAnswer] = []
    quiz_start = time.monotonic()
    if total is None and hasattr(questions, "__len__"):
        total = len(questions)  # type: ignore[arg-type]

    for index, question in enumerate(questions, start=1):
        _show_question(
            console,
            question,
            index,
            total if total is not None else index,
            node_names.get(question.node_id, question.node_id),
        )
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
    answered = len(answers)  # a stream may end early; grade what was asked
    result = QuizResult(
        session_type=session_type,
        answers=answers,
        score=score,
        total=answered,
        passed=has_passed(score, answered, pass_threshold),
        commit_hash=None,
        module_scope=module_scope,
        duration_secs=time.monotonic() - quiz_start,
    )
    _show_summary(console, result, node_names)
    return result
