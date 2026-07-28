"""Roger CLI: Typer app and all Phase 1 command definitions.

Phase 1 commands: init, quiz, guard [install|uninstall].
(Flags like --module/--difficulty, plus ask/chat/report/status, come in later phases.)
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import random
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.panel import Panel

from roger.ask import answer_question, context_pack

from roger.config import (
    CONFIG_PATH,
    ROGER_DIR,
    Config,
    load_config,
    normalize_provider,
    save_config,
    write_default_config,
)
from roger.exceptions import (
    CacheError,
    CloudBackendError,
    GraphNotFoundError,
    ModelNotRegisteredError,
    OllamaNotRunningError,
)
from roger.llm.azure import API_KEY_ENV as AZURE_API_KEY_ENV
from roger.llm.azure import ensure_ready as azure_ensure_ready
from roger.docs import doc_questions
from roger import freshness
from roger.freshness import is_source_file
from roger.generator import (
    generate_questions,
    interleave_questions,
    iter_questions,
)
from roger.llm.router import DESIGN_NODE_ID, get_design_questions
from roger.graph import get_god_nodes, get_quizzable_nodes, load_graph
from roger.hooks.pre_commit import install_hook, run_guard, uninstall_hook
from roger.llm.local import DEFAULT_MODEL, MODELFILE_CONTENT
from roger.quiz import QuestionStream, node_display_names, run_quiz
from roger.storage import init_dbs, record_session
from roger.webquiz import record_answer_code, render_ask_html, render_quiz_html

app = typer.Typer(
    name="roger",
    help="Quiz yourself on your own codebase before you commit.",
)
guard_app = typer.Typer(help="Pre-commit quiz guard.", invoke_without_command=True)
app.add_typer(guard_app, name="guard")


@app.callback(invoke_without_command=True)
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    # TTY gate: scripts, pipes, and CI keep today's exact behavior
    # (help + exit 2). The one-word experience exists only for humans.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo(ctx.get_help())
        raise typer.Exit(code=2)
    _default_flow()

console = Console()
err_console = Console(stderr=True)


def _fail(message: str) -> None:
    # markup=False: error text can contain literal [model]-style TOML section
    # names, which Rich would otherwise swallow as markup tags.
    err_console.print(str(message), markup=False)
    raise typer.Exit(code=1)


def _load_config() -> Config:
    try:
        return load_config()
    except ValueError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


def _anchor_repo_root() -> Optional[Path]:
    """Run from the repo root regardless of cwd — every Roger path
    (.roger/, graphify-out/) is repo-relative, and a quiz started from
    src/utils/ must not create a second Roger world there."""
    top = freshness.repo_root()
    if top is not None and top != Path.cwd():
        os.chdir(top)
    return top


def _default_flow() -> None:
    """Bare `roger`: the whole product behind one word."""
    top = _anchor_repo_root()
    if top is None:
        _fail("✗ Roger quizzes you on a git repository — cd into one and run 'roger'.")
        return
    config = _load_config()
    if not Path(config.graph.path).exists():
        _offer_setup(top, config)
    quiz(web=False, count=0)


def _offer_setup(top: Path, config: Config) -> None:
    """First run: one keypress, full honesty about size and downloads."""
    notes = []
    listed = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    source_count = sum(1 for f in listed if is_source_file(f))
    if source_count > 5_000:
        notes.append(
            f"large repo (~{source_count:,} source files) — the first index may take several minutes"
        )
    if config.model.provider != "azure-anthropic" and config.model.local == DEFAULT_MODEL:
        registered = subprocess.run(
            ["ollama", "show", DEFAULT_MODEL], capture_output=True, check=False
        ).returncode == 0
        if not registered:
            notes.append("downloads the ~1.15 GB local model if not already cached")
    detail = f" ({'; '.join(notes)})" if notes else ""
    if not typer.confirm(f"First time here — set up Roger for '{top.name}'?{detail}", default=True):
        raise typer.Exit(code=0)
    _run_init(config)


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
    user's model tag at the MiniCPM base, silently destroying it. Azure
    provider → verify configuration; no Ollama involvement at all.
    """
    if config.model.provider == "azure-anthropic":
        try:
            azure_ensure_ready(config)
        except CloudBackendError as exc:
            _fail(str(exc))
        console.print(
            f"Using Azure Foundry Anthropic deployment "
            f"'{config.model.azure_deployment}' — prompts leave this machine."
        )
        return

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
    """Bootstrap Roger: graphify build, model registration, config, databases."""
    _anchor_repo_root()
    _run_init(_load_config())


def _run_init(config: Config) -> None:
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
        subprocess.run([freshness.graphify_executable(), "./", "--code-only"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        _fail(f"✗ Roger: graphify failed: {exc}")
    graph_path = Path(config.graph.path)
    if not graph_path.exists():
        _fail(
            f"✗ Roger: graphify finished but {config.graph.path} was not created.\n"
            "  Check graphify's output for errors."
        )

    # 3+4. Ollama installed and running? (Skipped entirely on the Azure
    # provider — nothing local to install.)
    if config.model.provider != "azure-anthropic":
        if shutil.which("ollama") is None:
            _fail(
                "✗ Roger: Ollama is not installed.\n"
                "  Install it from: https://ollama.ai"
            )
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
    if config.model.provider == "azure-anthropic":
        console.print(f"✓ Model ready: {config.model.azure_deployment} (Azure Foundry)")
    else:
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
    count: int = typer.Option(
        0,
        "--count",
        "-n",
        help="Questions this session (default: questions_per_session from .roger/config.toml).",
    ),
) -> None:
    """Quiz yourself on this repo (whole repo, config defaults)."""
    _anchor_repo_root()
    config = _load_config()
    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        _fail(str(exc))
        return

    if graph.number_of_nodes() == 0:
        _fail("✗ Roger: the knowledge graph is empty. Rebuild it with: roger init")

    # Self-freshening: if source changed since the graph was built, refresh
    # in the background while this session runs on the loaded graph. A
    # previously failed refresh surfaces as exactly one line, once.
    failure_note = freshness.pending_failure_note(config.graph.path)
    if failure_note:
        console.print(f"[dim]{failure_note}[/dim]")
    freshness.maybe_refresh_in_background(config.graph.path)

    # Session size: --count flag wins, else the config value — the
    # docs/code/design split below scales automatically from it.
    count = count if count > 0 else config.quiz.questions_per_session
    difficulty = config.quiz.default_difficulty

    # Session blueprint — even split across question categories:
    # docs (instant, no LLM) open the session, code questions stream in the
    # middle, one system-design question closes it (generated in the
    # background while earlier questions are answered).
    doc_qs = (
        doc_questions(count=max(1, count // 3), difficulty=difficulty, paths=config.docs.paths)
        if config.docs.enabled
        else []
    )
    design_share = 1 if count >= 4 else 0
    code_count = max(1, count - len(doc_qs) - design_share)
    # iter_questions orders cache-hits first internally.
    node_ids = _pick_quiz_nodes(graph, code_count, config.graph.god_node_weight)
    names = node_display_names(graph, node_ids)
    names[DESIGN_NODE_ID] = "system design (module map)"

    if web:
        # The page is a static file, so it needs every question up front.
        console.print(f"Generating {count} {difficulty} questions…")
        try:
            questions = generate_questions(
                node_ids, graph, difficulty=difficulty, count=code_count, config=config
            )
        except (OllamaNotRunningError, ModelNotRegisteredError, CacheError, CloudBackendError, ValueError) as exc:
            _fail(str(exc))
            return
        random.shuffle(questions)
        questions = doc_qs + questions  # instant openers first, like the terminal
        if design_share:
            questions += get_design_questions(graph, difficulty, design_share, config)
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
    backend = (
        f"Azure Foundry '{config.model.azure_deployment}'"
        if config.model.provider == "azure-anthropic"
        else f"Ollama '{config.model.local}'"
    )
    console.print(
        f"Preparing {count} {difficulty} questions via {backend} — "
        "the rest generate as you answer…"
    )

    def _design_tail():
        if design_share:
            yield from get_design_questions(graph, difficulty, design_share, config)

    stream = QuestionStream(
        itertools.chain(
            interleave_questions(
                iter_questions(
                    node_ids, graph, difficulty=difficulty, count=code_count, config=config
                ),
                doc_qs,
            ),
            _design_tail(),
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
    except (OllamaNotRunningError, ModelNotRegisteredError, CacheError, CloudBackendError, ValueError) as exc:
        _fail(str(exc))
        return

    if result.total == 0:
        _fail("✗ Roger: could not generate any questions for this repo.")
    result.commit_hash = freshness.head_commit()
    try:
        record_session(result)
    except CacheError as exc:
        err_console.print(f"⚠ Roger: quiz finished but history was not saved: {exc}")


@app.command()
def record(code: str) -> None:
    """Record a finished web quiz session (the page shows the answer code)."""
    _anchor_repo_root()
    try:
        result = record_answer_code(code)
    except ValueError as exc:
        _fail(f"✗ Roger: {exc}")
        return
    result.commit_hash = freshness.head_commit()
    try:
        record_session(result)
    except CacheError as exc:
        _fail(f"✗ Roger: session graded but history was not saved: {exc}")
    verdict = "passed" if result.passed else "failed"
    console.print(f"✓ Recorded: {result.score}/{result.total} — {verdict}.")


@app.command()
def update(
    background: bool = typer.Option(False, "--background", hidden=True),
) -> None:
    """Refresh the knowledge graph from the current code (fast, no LLM)."""
    _anchor_repo_root()
    config = _load_config()
    if not freshness.acquire_lock():
        if background:
            raise typer.Exit(code=0)
        _fail("✗ Roger: a graph update is already running — try again in a moment.")
    try:
        result = freshness.run_update(config.graph.path)
        if background:
            return
        if result.outcome == "ok":
            delta = result.nodes_after - result.nodes_before
            console.print(
                f"✓ Graph updated in {result.duration_secs:.0f}s — "
                f"{result.nodes_after:,} nodes ({'+' if delta >= 0 else ''}{delta:,})."
            )
        elif result.outcome == "shrink_refused":
            # The ONE verb finishes its own job: explain, confirm, rebuild —
            # never send the user to another tool's --force flag.
            console.print(
                "The index refused to shrink — code was deleted since the last "
                "build (protection against a bad parse wiping the graph)."
            )
            if typer.confirm("Rebuild the graph to match the current code?", default=True):
                result = freshness.run_update(config.graph.path, force=True)
                if result.outcome == "ok":
                    console.print(
                        f"✓ Graph rebuilt in {result.duration_secs:.0f}s — "
                        f"{result.nodes_after:,} nodes."
                    )
                else:
                    _fail(f"✗ Roger: rebuild failed:\n{result.detail}")
        else:
            _fail(f"✗ Roger: graph update failed:\n{result.detail}")
    finally:
        freshness.release_lock()


@app.command()
def use(
    provider: str = typer.Argument(..., help='Backend: "ollama" or "azure"'),
    endpoint: str = typer.Option("", "--endpoint", help="Azure Foundry endpoint URL"),
    deployment: str = typer.Option("", "--deployment", help="Azure Claude deployment name"),
    model: str = typer.Option("", "--model", help="Ollama model name (e.g. qwen2.5:7b)"),
) -> None:
    """Switch the generation backend — no TOML editing required.

    Examples:
      roger use azure --endpoint https://acme.services.ai.azure.com/anthropic --deployment claude-x
      roger use ollama --model qwen2.5:7b-instruct-q4_K_M
      roger use ollama
    """
    _anchor_repo_root()
    config = _load_config()
    try:
        target = normalize_provider(provider)
    except ValueError as exc:
        _fail(str(exc))
        return
    config.model.provider = target

    if target == "azure-anthropic":
        if endpoint:
            config.model.azure_endpoint = endpoint
        if deployment:
            config.model.azure_deployment = deployment
        if not config.model.azure_endpoint or not config.model.azure_deployment:
            _fail(
                "✗ Roger: the Azure backend needs an endpoint and a deployment "
                "(first time only):\n"
                "  roger use azure --endpoint https://<resource>.services.ai.azure.com/anthropic "
                "--deployment <name>"
            )
        save_config(config)
        console.print(
            f"✓ Backend: Azure Foundry Anthropic — deployment "
            f"'{config.model.azure_deployment}'. Applies to quiz, quiz --web, guard, and ask."
        )
        if not os.environ.get(AZURE_API_KEY_ENV):
            console.print(
                f"  One more step: export {AZURE_API_KEY_ENV}=…  (environment only, never in files)"
            )
    else:
        if model:
            config.model.local = model
        save_config(config)
        console.print(
            f"✓ Backend: Ollama — model '{config.model.local}'. "
            "Applies to quiz, quiz --web, guard, and ask."
        )


AGENT_SNIPPET_START = "<!-- roger:start -->"
AGENT_SNIPPET_END = "<!-- roger:end -->"
AGENT_SNIPPET = f"""{AGENT_SNIPPET_START}
## Roger — token-efficient code context

Before exploring this codebase with grep/find or by reading whole files, run:

    roger context "<your question>" --budget 2000

It returns the relevant source (complete functions), the team's recorded
decisions (ADRs, contracts), and call relationships in one budgeted, cited
pack — typically far fewer tokens than raw file reading. For a direct
answered question with citations, use `roger ask "<question>"` instead.
{AGENT_SNIPPET_END}"""

agent_app = typer.Typer(help="Teach coding agents to use Roger (no MCP, no server).")
app.add_typer(agent_app, name="agent")


@app.command()
def context(
    question: str,
    budget: int = typer.Option(
        2000, "--budget", help="Token budget for the pack (approximate)."
    ),
) -> None:
    """Print a cited context pack for a question — built for coding agents.

    Zero LLM calls: instant, free, and works without any model backend.
    """
    _anchor_repo_root()
    config = _load_config()
    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        _fail(str(exc))
        return
    freshness.maybe_refresh_in_background(config.graph.path)
    # Plain stdout — agents consume this; no panels, no colors.
    typer.echo(context_pack(question, graph, config, budget_tokens=budget))


def _install_snippet(path: Path) -> str:
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if AGENT_SNIPPET_START in text and AGENT_SNIPPET_END in text:
            start = text.index(AGENT_SNIPPET_START)
            end = text.index(AGENT_SNIPPET_END) + len(AGENT_SNIPPET_END)
            path.write_text(text[:start] + AGENT_SNIPPET + text[end:], encoding="utf-8")
            return "updated"
        path.write_text(text.rstrip() + "\n\n" + AGENT_SNIPPET + "\n", encoding="utf-8")
        return "appended"
    path.write_text(AGENT_SNIPPET + "\n", encoding="utf-8")
    return "created"


@agent_app.command("install")
def agent_install() -> None:
    """Write the Roger instructions into the files coding agents read."""
    if _anchor_repo_root() is None:
        _fail("✗ Roger: run this inside a git repository.")
    outcome = _install_snippet(Path("AGENTS.md"))
    console.print(f"✓ AGENTS.md {outcome} — OpenCode, Codex, Copilot CLI, and Aider read it.")
    copilot_path = Path(".github/copilot-instructions.md")
    copilot_path.parent.mkdir(parents=True, exist_ok=True)
    outcome = _install_snippet(copilot_path)
    console.print(
        f"✓ .github/copilot-instructions.md {outcome} — GitHub Copilot in VS Code reads it."
    )
    if Path("CLAUDE.md").exists():
        outcome = _install_snippet(Path("CLAUDE.md"))
        console.print(f"✓ CLAUDE.md {outcome} — Claude Code reads it.")
    console.print("  Agents will now run 'roger context' before grepping and reading files.")


@agent_app.command("uninstall")
def agent_uninstall() -> None:
    """Remove the Roger instructions from AGENTS.md and CLAUDE.md."""
    _anchor_repo_root()
    removed = 0
    for name in ("AGENTS.md", "CLAUDE.md", ".github/copilot-instructions.md"):
        path = Path(name)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if AGENT_SNIPPET_START not in text:
            continue
        start = text.index(AGENT_SNIPPET_START)
        end = text.index(AGENT_SNIPPET_END) + len(AGENT_SNIPPET_END)
        cleaned = (text[:start] + text[end:]).strip()
        if cleaned:
            path.write_text(cleaned + "\n", encoding="utf-8")
        else:
            path.unlink()
        removed += 1
        console.print(f"✓ Roger section removed from {name}.")
    if not removed:
        console.print("Roger: no agent instructions found; nothing removed.")


@app.command()
def ask(
    question: str,
    web: bool = typer.Option(
        False, "--web", help="Render the answer in the browser (markdown, highlighted code)."
    ),
) -> None:
    """Ask a question about the codebase — answered from graph, source, and docs."""
    _anchor_repo_root()
    config = _load_config()
    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        _fail(str(exc))
        return

    with console.status("[dim]Reading the codebase…[/dim]"):
        try:
            answer, sources = answer_question(question, graph, config)
        except (OllamaNotRunningError, ModelNotRegisteredError, CloudBackendError, ValueError) as exc:
            _fail(str(exc))
            return

    if web:
        page = render_ask_html(question, answer, sources)
        console.print(f"✓ Answer ready: {page}")
        webbrowser.open(page.resolve().as_uri())
        return

    title = question if len(question) <= 76 else question[:73] + "…"
    console.print(
        Panel(RichMarkdown(answer), title=escape(title), title_align="left", border_style="cyan")
    )
    if sources:
        console.print("[dim]Grounded in: " + escape(" · ".join(sources)) + "[/dim]")


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
