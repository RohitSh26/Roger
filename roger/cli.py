"""Roger CLI: Typer app and all Phase 1 command definitions.

Phase 1 commands: init, quiz, guard [install|uninstall].
(Flags like --module/--difficulty, plus ask/chat/report/status, come in later phases.)
"""

from __future__ import annotations

import importlib.util
import random
import shutil
import subprocess
from pathlib import Path

import requests
import typer
from rich.console import Console

from roger.config import CONFIG_PATH, ROGER_DIR, load_config, write_default_config
from roger.exceptions import (
    CacheError,
    GraphNotFoundError,
    ModelNotRegisteredError,
    OllamaNotRunningError,
)
from roger.generator import generate_questions
from roger.graph import get_god_nodes, load_graph
from roger.hooks.pre_commit import install_hook, run_guard, uninstall_hook
from roger.quiz import run_quiz
from roger.storage import init_dbs, record_session

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
    err_console.print(str(message))
    raise typer.Exit(code=1)


def _find_modelfile() -> Path:
    """Locate local/Modelfile: target repo first, then Roger's own checkout."""
    candidates = [
        Path("local/Modelfile"),
        Path(__file__).resolve().parent.parent / "local" / "Modelfile",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    _fail(
        "✗ Roger: local/Modelfile not found.\n"
        "  Reinstall roger-cli or run init from the Roger repository."
    )
    raise AssertionError("unreachable")


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

    # 5. Register the model.
    modelfile = _find_modelfile()
    console.print(f"Registering model '{config.model.local}' (downloads ~1.15 GB on first run)…")
    try:
        subprocess.run(
            ["ollama", "create", config.model.local, "-f", str(modelfile)], check=True
        )
    except subprocess.CalledProcessError as exc:
        _fail(f"✗ Roger: ollama create failed: {exc}")

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
    console.print(f"✓ Model ready: {config.model.local} (MiniCPM5-1B)")
    console.print(f"✓ Config: {CONFIG_PATH}")
    console.print()
    console.print("Next steps:")
    console.print("  roger quiz          — quiz yourself on this repo")
    console.print("  roger guard install — set up pre-commit hook")
    console.print("  roger ask '...'     — ask a question about the codebase")


def _pick_quiz_nodes(graph, count: int, god_node_weight: bool) -> list[str]:
    """Choose nodes for a whole-repo quiz: up to half god nodes, rest random."""
    all_nodes = sorted(graph.nodes)
    if len(all_nodes) <= count:
        return all_nodes

    picked: list[str] = []
    if god_node_weight:
        picked.extend(get_god_nodes(graph, top_n=max(1, count // 2)))
    remaining = [n for n in all_nodes if n not in set(picked)]
    picked.extend(random.sample(remaining, min(count - len(picked), len(remaining))))
    return picked


@app.command()
def quiz() -> None:
    """Quiz yourself on this repo (whole repo, config defaults)."""
    config = load_config()
    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        _fail(str(exc))
        return

    if graph.number_of_nodes() == 0:
        _fail("✗ Roger: the knowledge graph is empty. Rebuild it with: roger init")

    node_ids = _pick_quiz_nodes(
        graph, config.quiz.questions_per_session, config.graph.god_node_weight
    )
    console.print(
        f"Generating {config.quiz.questions_per_session} "
        f"{config.quiz.default_difficulty} questions…"
    )
    try:
        questions = generate_questions(
            node_ids,
            graph,
            difficulty=config.quiz.default_difficulty,
            count=config.quiz.questions_per_session,
            config=config,
        )
    except (OllamaNotRunningError, ModelNotRegisteredError, CacheError) as exc:
        _fail(str(exc))
        return

    if not questions:
        _fail("✗ Roger: could not generate any questions for this repo.")

    result = run_quiz(
        questions, session_type="quiz", pass_threshold=config.quiz.pass_threshold
    )
    try:
        record_session(result)
    except CacheError as exc:
        err_console.print(f"⚠ Roger: quiz finished but history was not saved: {exc}")


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
