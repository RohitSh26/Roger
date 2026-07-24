"""Roger CLI: Typer app and all Phase 1 command definitions.

Phase 1 commands: init, quiz, guard [install|uninstall].
(Flags like --module/--difficulty, plus ask/chat/report/status, come in later phases.)
"""

from __future__ import annotations

import importlib.util
import random
import shutil
import subprocess
import webbrowser
from pathlib import Path

import requests
import typer
from rich.console import Console

from roger.config import CONFIG_PATH, ROGER_DIR, Config, load_config, write_default_config
from roger.exceptions import (
    CacheError,
    GraphNotFoundError,
    ModelNotRegisteredError,
    OllamaNotRunningError,
)
from roger.docs import doc_questions
from roger.generator import generate_questions, interleave_questions, iter_questions
from roger.graph import get_god_nodes, get_quizzable_nodes, load_graph
from roger.hooks.pre_commit import install_hook, run_guard, uninstall_hook
from roger.llm.local import DEFAULT_MODEL, MODELFILE_CONTENT
from roger.quiz import QuestionStream, node_display_names, run_quiz
from roger.storage import init_dbs, record_session
from roger.webquiz import record_answer_code, render_quiz_html

app = typer.Typer(
    name="roger",
    help="Quiz yourself on your own codebase before you commit.",
    no_args_is_help=True,
)
guard_app = typer.Typer(help="Pre-commit quiz guard.", invoke_without_command=True)
app.add_typer(guard_app, name="guard")

console = Console()
err_console = Console(stderr=True)


def _fail(message: str) -> None:
    # markup=False: error text can contain literal [model]-style TOML section
    # names, which Rich would otherwise swallow as markup tags.
    err_console.print(str(message), markup=False)
    raise typer.Exit(code=1)


def _ensure_modelfile() -> Path:
    """Return a Modelfile path, materializing the embedded copy if needed.

    A checkout's local/Modelfile (cwd) wins so it stays user-editable; wheel
    installs don't ship that file, so init writes the embedded content to
    .roger/Modelfile instead.
    """
    checkout = Path("local/Modelfile")
    if checkout.exists():
        return checkout
    target = ROGER_DIR / "Modelfile"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(MODELFILE_CONTENT, encoding="utf-8")
    return target


def _ensure_model(config: Config) -> None:
    """Make the configured model usable.

    Default model → register from the Modelfile. Custom model → only verify
    it is already pulled: running `ollama create` here would re-point the
    user's model tag at the MiniCPM base, silently destroying it.
    """
    if config.model.local == DEFAULT_MODEL:
        modelfile = _ensure_modelfile()
        console.print(
            f"Registering model '{config.model.local}' (downloads ~1.15 GB on first run)…"
        )
        try:
            subprocess.run(
                ["ollama", "create", config.model.local, "-f", str(modelfile)], check=True
            )
        except subprocess.CalledProcessError as exc:
            _fail(f"✗ Roger: ollama create failed: {exc}")
        return

    probe = subprocess.run(
        ["ollama", "show", config.model.local], capture_output=True, text=True
    )
    if probe.returncode != 0:
        _fail(
            f"✗ Roger: custom model '{config.model.local}' is not in Ollama.\n"
            f"  Pull it with: ollama pull {config.model.local}\n"
            '  Or set local = "roger-local" under [model] in .roger/config.toml'
        )
    console.print(f"Using custom model '{config.model.local}' (already in Ollama).")


@app.command()
def init() -> None:
    """Bootstrap Roger: graphify build, Ollama model registration, config, databases."""
    config = load_config()

    # 1. graphify installed?
    if importlib.util.find_spec("graphify") is None:
        _fail(
            "✗ Roger: graphify is not installed.\n"
            "  Install it with: pip install graphifyy"
        )

    # 2. Build the knowledge graph. --code-only keeps graphify on its local
    #    AST path: doc/image extraction needs a cloud LLM key, which Roger's
    #    local-only constraint forbids.
    console.print("Building knowledge graph with graphify…")
    try:
        subprocess.run(["graphify", "./", "--code-only"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _fail(f"✗ Roger: graphify failed: {exc}")
    graph_path = Path(config.graph.path)
    if not graph_path.exists():
        _fail(
            f"✗ Roger: graphify finished but {config.graph.path} was not created.\n"
            "  Check graphify's output for errors."
        )

    # 3. Ollama installed?
    if shutil.which("ollama") is None:
        _fail(
            "✗ Roger: Ollama is not installed.\n"
            "  Install it from: https://ollama.ai"
        )

    # 4. Ollama running?
    try:
        requests.get(config.ollama.url, timeout=2).raise_for_status()
    except requests.RequestException:
        _fail(
            "✗ Roger: Ollama is not running.\n"
            "  Start it with: ollama serve\n"
            "  First-time setup: roger init"
        )

    # 5. Register the default model, or verify a user-configured one.
    _ensure_model(config)

    # 6-8. .roger/ directory, default config, databases.
    ROGER_DIR.mkdir(parents=True, exist_ok=True)
    write_default_config(CONFIG_PATH)
    init_dbs()

    # 9. Success summary.
    graph = load_graph(config.graph.path)
    console.print()
    console.print(
        f"✓ Graph built: {config.graph.path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    model_note = " (MiniCPM5-1B)" if config.model.local == DEFAULT_MODEL else " (custom)"
    console.print(f"✓ Model ready: {config.model.local}{model_note}")
    console.print(f"✓ Config: {CONFIG_PATH}")
    console.print()
    console.print("Next steps:")
    console.print("  roger quiz          — quiz yourself on this repo")
    console.print("  roger guard install — set up pre-commit hook")
    console.print("  roger ask '...'     — ask a question about the codebase")


def _pick_quiz_nodes(graph, count: int, god_node_weight: bool) -> list[str]:
    """Choose nodes for a whole-repo quiz: up to half god nodes, rest random.

    Only quiz-worthy nodes are considered — real code involved in call
    edges, not doc stubs, entry markers, or (preferably) test helpers.
    """
    candidates = get_quizzable_nodes(graph) or sorted(graph.nodes)
    if len(candidates) <= count:
        return candidates

    picked: list[str] = []
    if god_node_weight:
        quizzable = set(candidates)
        # Sample from a wider god pool instead of always leading with the
        # same top nodes — repeat sessions should not repeat a fixed opener.
        god_pool = [n for n in get_god_nodes(graph, top_n=count * 4) if n in quizzable]
        god_share = max(1, count // 2)
        picked.extend(random.sample(god_pool, min(god_share, len(god_pool))))
    remaining = [n for n in candidates if n not in set(picked)]
    picked.extend(random.sample(remaining, min(count - len(picked), len(remaining))))
    random.shuffle(picked)
    return picked


@app.command()
def quiz(
    web: bool = typer.Option(
        False, "--web", help="Take the quiz in the browser (highlighted code, no server)."
    ),
) -> None:
    """Quiz yourself on this repo (whole repo, config defaults)."""
    config = load_config()
    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        _fail(str(exc))
        return

    if graph.number_of_nodes() == 0:
        _fail("✗ Roger: the knowledge graph is empty. Rebuild it with: roger init")

    count = config.quiz.questions_per_session
    difficulty = config.quiz.default_difficulty

    # Docs contribute ~a third of a session when the repo has quizzable
    # docs (ADRs, tables, prose) — constructed instantly, no LLM.
    doc_qs = (
        doc_questions(count=max(1, count // 3), difficulty=difficulty, paths=config.docs.paths)
        if config.docs.enabled
        else []
    )
    code_count = max(1, count - len(doc_qs))
    node_ids = _pick_quiz_nodes(graph, code_count, config.graph.god_node_weight)
    names = node_display_names(graph, node_ids)

    if web:
        # The page is a static file, so it needs every question up front.
        console.print(f"Generating {count} {difficulty} questions…")
        try:
            questions = generate_questions(
                node_ids, graph, difficulty=difficulty, count=code_count, config=config
            )
        except (OllamaNotRunningError, ModelNotRegisteredError, CacheError, ValueError) as exc:
            _fail(str(exc))
            return
        questions = questions + doc_qs
        random.shuffle(questions)
        if not questions:
            _fail("✗ Roger: could not generate any questions for this repo.")
        page = render_quiz_html(
            questions,
            session_type="quiz",
            pass_threshold=config.quiz.pass_threshold,
            node_names=names,
        )
        console.print(f"✓ Quiz ready: {page}")
        console.print("  Answer in the browser, then run the 'roger record' command it shows.")
        webbrowser.open(page.resolve().as_uri())
        return

    # Terminal mode streams: the first question appears as soon as it is
    # ready, and the next one generates while the developer answers. Doc
    # questions (instant) are woven between the streamed code questions.
    console.print(f"Preparing {count} {difficulty} questions — the rest generate as you answer…")
    stream = QuestionStream(
        interleave_questions(
            iter_questions(node_ids, graph, difficulty=difficulty, count=code_count, config=config),
            doc_qs,
        )
    )
    try:
        result = run_quiz(
            stream,
            session_type="quiz",
            pass_threshold=config.quiz.pass_threshold,
            node_names=names,
            total=count,
        )
    except (OllamaNotRunningError, ModelNotRegisteredError, CacheError, ValueError) as exc:
        _fail(str(exc))
        return

    if result.total == 0:
        _fail("✗ Roger: could not generate any questions for this repo.")
    try:
        record_session(result)
    except CacheError as exc:
        err_console.print(f"⚠ Roger: quiz finished but history was not saved: {exc}")


@app.command()
def record(code: str) -> None:
    """Record a finished web quiz session (the page shows the answer code)."""
    try:
        result = record_answer_code(code)
    except ValueError as exc:
        _fail(f"✗ Roger: {exc}")
        return
    try:
        record_session(result)
    except CacheError as exc:
        _fail(f"✗ Roger: session graded but history was not saved: {exc}")
    verdict = "passed" if result.passed else "failed"
    console.print(f"✓ Recorded: {result.score}/{result.total} — {verdict}.")


@guard_app.callback()
def guard(ctx: typer.Context) -> None:
    """Run the guard quiz on staged files (or use install/uninstall)."""
    if ctx.invoked_subcommand is None:
        run_guard()


@guard_app.command("install")
def guard_install() -> None:
    """Write the pre-commit hook to .git/hooks/pre-commit."""
    try:
        install_hook()
    except (FileNotFoundError, FileExistsError, OSError) as exc:
        _fail(f"✗ Roger: {exc}")
    console.print("✓ Roger: pre-commit hook installed.")
    console.print("  Skip once with: ROGER_SKIP=1 git commit …")


@guard_app.command("uninstall")
def guard_uninstall() -> None:
    """Remove the Roger pre-commit hook."""
    try:
        removed = uninstall_hook()
    except OSError as exc:
        _fail(f"✗ Roger: {exc}")
        return
    if removed:
        console.print("✓ Roger: pre-commit hook removed.")
    else:
        console.print("Roger: no Roger-installed pre-commit hook found; nothing removed.")


if __name__ == "__main__":
    app()
