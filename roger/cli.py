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
import time
import webbrowser
from pathlib import Path
from typing import Optional

from roger import activity, embeddings

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
    _maybe_offer_semantic(config)
    quiz(web=False, count=0)


def _maybe_offer_semantic(config: Config, reoffer: bool = False) -> None:
    """The once-ever smarter-search question — human TTY flows only.

    Capability-sensed: the model being present IS enablement; only a
    refusal is recorded (per machine, not per repo). Never asked in agent
    paths (context/ask) or inside the guard hook.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    if config.model.provider == "azure-anthropic":
        return  # no Ollama requirement on these repos; keyword-only silently
    if embeddings.embed_prompt_declined() and not reoffer:
        return
    if embeddings.model_digest(config) is not None:
        # Already enabled by presence — but a fresh repo on an enabled
        # machine still needs its index built once.
        _ensure_semantic_index(config)
        return
    from roger.llm.local import is_ollama_running

    if not is_ollama_running(config.ollama.url):
        return
    if typer.confirm(
        "Enable smarter search? Finds code by meaning, not just keywords "
        "(~270 MB one-time download)", default=True
    ):
        embeddings.clear_declined()
        try:
            subprocess.run(
                ["ollama", "pull", embeddings.EMBED_MODEL], check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print(
                "[dim]Download didn't finish — roger doctor can retry later; "
                "everything keeps working with keyword search.[/dim]"
            )
            return
        freshness.maybe_refresh_in_background(config.graph.path, force=True)
        console.print("[dim]Smarter search enabled — the index builds in the background.[/dim]")
    else:
        embeddings.record_declined()


def _ensure_semantic_index(config: Config) -> bool:
    """Self-heal an unhealthy vector index: missing (fresh clone, repo
    first touched by an agent), interrupted mid-build (closed laptop,
    stopped container), stale after a model change, or partially covered.
    Presence of the embed model is the consent; the build runs in the
    background. Returns True if a build was started."""
    if not embeddings.needs_refresh(config):
        return False
    return freshness.maybe_refresh_in_background(config.graph.path, force=True)


def _refresh_semantic(config: Config) -> Optional[dict]:
    """Run the index refresh after a graph update; never fails the update."""
    try:
        return embeddings.refresh_index(load_graph(config.graph.path), config)
    except Exception:  # noqa: BLE001 - semantic layer must never block updates
        return None


def _print_semantic_result(stats: Optional[dict]) -> None:
    """One honest line about what the update did to the smarter-search
    index — it always ran; it should never run invisibly."""
    if stats is None:
        console.print("[dim]• Smarter search: off (keyword-only) — 'roger doctor' to enable.[/dim]")
    elif stats["embedded"]:
        console.print(
            f"✓ Smarter search: re-indexed {stats['embedded']} changed function(s) "
            f"({stats['with_vec']:,} of {stats['cards']:,} indexed)."
        )
    else:
        console.print(
            f"✓ Smarter search: index already current ({stats['cards']:,} functions)."
        )


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


def _default_model_registered() -> bool:
    return (
        subprocess.run(
            ["ollama", "show", DEFAULT_MODEL], capture_output=True, check=False
        ).returncode
        == 0
    )


def _ensure_model_ready(config: Config) -> None:
    """Lazy model install: the download happens the first time it's needed.

    Setup no longer pays the ~1.15 GB default-model download — quizzes and
    questions are the first moment the LLM matters, so the one-keypress
    offer lives here. Already installed → silent. Azure and custom models
    are user-managed; their existing error paths already name the remedy.
    Non-TTY callers get the standard error, never a prompt.
    """
    if config.model.provider == "azure-anthropic" or config.model.local != DEFAULT_MODEL:
        return
    from roger.llm.local import MODEL_NOT_REGISTERED_MSG, is_ollama_running

    if not is_ollama_running(config.ollama.url):
        return  # generation raises OllamaNotRunningError with its remedy
    if _default_model_registered():
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        _fail(MODEL_NOT_REGISTERED_MSG.format(model=DEFAULT_MODEL))
        return
    if not typer.confirm(
        "Roger's AI model isn't installed yet (~1.15 GB, one time). Download now?",
        default=True,
    ):
        console.print(
            "[dim]No download — run this again whenever you're ready. "
            "'roger context' works without it.[/dim]"
        )
        raise typer.Exit(code=0)
    _ensure_model(config)


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

    # 5. Verify a user-configured model or the Azure backend — both are
    #    fast checks. The default local model is deliberately NOT
    #    downloaded here: it installs (one keypress) the first time a quiz
    #    or question needs it, so setup stays fast and context-only users
    #    never pay ~1.15 GB for a model they don't use.
    if config.model.provider == "azure-anthropic" or config.model.local != DEFAULT_MODEL:
        _ensure_model(config)

    # 6-8. .roger/ directory, default config, databases.
    ROGER_DIR.mkdir(parents=True, exist_ok=True)
    write_default_config(CONFIG_PATH)
    init_dbs()

    # Machine already has the embed model (smarter search enabled) → this
    # new repo's index builds now, in the background, so the first
    # question — human's or agent's — can already match by meaning.
    semantic_started = _ensure_semantic_index(config)

    # 9. Success summary.
    graph = load_graph(config.graph.path)
    console.print()
    console.print(
        f"✓ Graph built: {config.graph.path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    if config.model.provider == "azure-anthropic":
        console.print(f"✓ Model ready: {config.model.azure_deployment} (Azure Foundry)")
    elif config.model.local != DEFAULT_MODEL:
        console.print(f"✓ Model ready: {config.model.local} (custom)")
    elif _default_model_registered():
        console.print(f"✓ Model ready: {config.model.local} (already in Ollama)")
    else:
        console.print(
            "• AI model: downloads on your first quiz or question (~1.15 GB, one time)"
        )
    if semantic_started:
        console.print("✓ Smarter search: index building in the background")
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

    # The LLM's first actual use — install the model now if it never was.
    _ensure_model_ready(config)

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
        # Embed step rides the same flow; never fails the graph update.
        stats = _refresh_semantic(config) if result.outcome == "ok" else None
        if background:
            return
        if result.outcome == "ok":
            delta = result.nodes_after - result.nodes_before
            console.print(
                f"✓ Graph updated in {result.duration_secs:.0f}s — "
                f"{result.nodes_after:,} nodes ({'+' if delta >= 0 else ''}{delta:,})."
            )
            _print_semantic_result(stats)
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
                    _print_semantic_result(_refresh_semantic(config))
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
## Roger — token-efficient code context (use this FIRST)

ALWAYS run this BEFORE any grep/find/file-reading when exploring this codebase:

    roger context "<your question>" --budget 2000

It returns the relevant source (complete functions), the team's recorded
decisions (ADRs, contracts), and call relationships in one budgeted, cited
pack — typically far fewer tokens than raw file reading. If the pack misses,
re-query ONCE with likely code terms (e.g. throttle → rate limit, backoff)
before falling back to grep.

TRUST CONTRACT — the pack's code blocks are VERBATIM source, mechanically
extracted (no AI generates them). Therefore:
- For read-only questions: answer directly from the pack. Do NOT re-read
  files whose relevant lines the pack already shows — that read returns
  byte-identical text and only wastes tokens.
- DO read the actual files when you are about to MODIFY code, when a claim
  lacks a citation, when citations conflict, or when the pack says it was
  truncated.
{AGENT_SNIPPET_END}"""

agent_app = typer.Typer(help="Teach coding agents to use Roger (no MCP, no server).")
app.add_typer(agent_app, name="agent")


@app.command()
def doctor() -> None:
    """Check this environment and print the fix for anything wrong.

    Run this first whenever Roger misbehaves on a new machine, container,
    or Codespace — it knows the common failure modes and their one-line
    remedies.
    """
    checks: list[tuple[str, str, str]] = []  # (status, finding, remedy)

    def check(ok: Optional[bool], good: str, bad: str, remedy: str = "") -> None:
        if ok is True:
            checks.append(("ok", good, ""))
        elif ok is False:
            checks.append(("fail", bad, remedy))
        else:
            checks.append(("warn", bad, remedy))

    # Environment
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    in_container = Path("/.dockerenv").exists() or os.environ.get("REMOTE_CONTAINERS")
    checks.append(("ok", f"Python {py}" + (" (inside a container)" if in_container else ""), ""))
    check(
        True if importlib.util.find_spec("setuptools") is not None else None,
        "setuptools present (legacy package builds will work)",
        "setuptools missing — if a package ever builds from source here "
        "(common in containers), it can fail with \"No module named 'distutils'\"",
        "pip install --upgrade pip setuptools wheel",
    )
    check(
        importlib.util.find_spec("graphify") is not None,
        "graphify installed",
        "graphify missing (needed to build/update the code map)",
        "pip install graphifyy",
    )

    # Repository + graph
    top = freshness.repo_root()
    check(top is not None, f"git repository: {top.name if top else ''}",
          "not inside a git repository", "cd into your project and re-run")
    if top is not None:
        os.chdir(top)
        config = _load_config()
        graph_exists = Path(config.graph.path).exists()
        check(graph_exists, "code map present (graphify-out/graph.json)",
              "no code map yet", "run: roger   (first-run setup builds it)")
        if graph_exists:
            stale = freshness.stale_source_files(config.graph.path)
            check(
                not stale,
                "code map is current",
                f"code map is behind ({len(stale)} changed source file(s))",
                "roger update   (quiz sessions also refresh it automatically)",
            )

        # Backend (matters for ask/quiz; roger context needs none)
        if config.model.provider == "azure-anthropic":
            check(bool(os.environ.get(AZURE_API_KEY_ENV)),
                  f"Azure backend configured ('{config.model.azure_deployment}')",
                  f"Azure selected but {AZURE_API_KEY_ENV} is not set",
                  f"export {AZURE_API_KEY_ENV}=…")
        else:
            from roger.llm.local import is_ollama_running

            check(
                is_ollama_running(config.ollama.url) or None,
                f"Ollama reachable ({config.ollama.url})",
                f"Ollama not reachable at {config.ollama.url} — quizzes and "
                "'roger ask' need it; 'roger context' works without it",
                "open the Ollama app (or: ollama serve)",
            )
            if config.model.local == DEFAULT_MODEL and is_ollama_running(config.ollama.url):
                if _default_model_registered():
                    checks.append(("ok", f"AI model installed ({DEFAULT_MODEL})", ""))
                else:
                    checks.append(
                        ("ok",
                         "AI model not downloaded yet — installs on your first "
                         "quiz or question (~1.15 GB, one keypress)", "")
                    )

        # Agent integration
        check(
            Path("AGENTS.md").exists() or Path(".github/copilot-instructions.md").exists() or None,
            "agent instructions installed",
            "no agent instructions in this repo (fine unless you use AI agents)",
            "roger agent install",
        )

        # Retrieval mode (semantic is optional; keyword-only is never wrong).
        # Doctor is the remedy surface: an unhealthy index isn't just
        # reported — the rebuild starts right here.
        status = embeddings.index_status(config)
        if status["mode"] == "semantic+keyword":
            checks.append(
                ("ok", f"smarter search active ({status['with_vec']}/{status['cards']} indexed)", "")
            )
        else:
            line = f"smarter search: {status.get('reason', 'off')}"
            if _ensure_semantic_index(config):
                line += " — rebuilding in the background now"
            elif not status.get("model_present"):
                line = f"search: keyword-only ({status.get('reason', '')})"
            checks.append(("ok", line, ""))

    icons = {"ok": "[green]✓[/green]", "warn": "[yellow]⚠[/yellow]", "fail": "[red]✗[/red]"}
    failed = False
    for status, finding, remedy in checks:
        console.print(f"{icons[status]} {escape(finding)}")
        if remedy:
            console.print(f"   → {escape(remedy)}")
        failed = failed or status == "fail"
    if failed:
        raise typer.Exit(code=1)

    # The repentance door: a machine that declined smarter search can
    # change its mind here — doctor is already the remedy surface.
    if top is not None and embeddings.embed_prompt_declined():
        _maybe_offer_semantic(_load_config(), reoffer=True)


@app.command("log")
def show_log(
    limit: int = typer.Option(20, "--limit", "-l", help="How many recent events to show."),
) -> None:
    """Show what was recently asked of Roger — including by your AI agents.

    The log lives locally at .roger/activity.log (JSONL). It records
    machinery, never people: what was requested, tokens served, duration.
    """
    _anchor_repo_root()
    events = activity.read_recent(limit)
    if not events:
        console.print("No activity yet — it appears here once roger context/ask are used.")
        return
    from rich.table import Table

    table = Table(box=None, header_style="bold")
    table.add_column("when")
    table.add_column("command")
    table.add_column("caller")
    table.add_column("question")
    table.add_column("tokens", justify="right")
    table.add_column("ms", justify="right")
    for event in events:
        table.add_row(
            str(event.get("ts", "")),
            str(event.get("command", "")),
            str(event.get("caller", "")),
            escape(str(event.get("question", ""))[:60]),
            str(event.get("tokens_served", "—")),
            str(event.get("duration_ms", "—")),
        )
    console.print(table)


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
    # Freshness first; if nothing needed refreshing, self-heal a missing
    # vector index so the NEXT agent call gets semantic matching. This
    # call still answers immediately (keyword-only if the index isn't
    # ready) — agents are never made to wait on an index build.
    if not freshness.maybe_refresh_in_background(config.graph.path):
        _ensure_semantic_index(config)
    # Plain stdout — agents consume this; no panels, no colors.
    started = time.monotonic()
    pack = context_pack(question, graph, config, budget_tokens=budget)
    activity.log_event(
        "context",
        question=question[:200],
        budget=budget,
        tokens_served=len(pack) // 4,
        matched="No matching code" not in pack,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    typer.echo(pack)


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

    # The human is setting up agents right now — the one moment to make
    # sure their context packs get semantic matching from the first call.
    config = _load_config()
    status = embeddings.index_status(config)
    if status["mode"] == "semantic+keyword":
        console.print("✓ Smarter search is on — packs match by meaning as well as keywords.")
    elif _ensure_semantic_index(config):
        console.print(
            "✓ Smarter search index building in the background — "
            "packs match by meaning once it finishes (a minute or two)."
        )
    else:
        _maybe_offer_semantic(config)


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

    _ensure_model_ready(config)
    started = time.monotonic()
    with console.status("[dim]Reading the codebase…[/dim]"):
        try:
            answer, sources = answer_question(question, graph, config)
        except (OllamaNotRunningError, ModelNotRegisteredError, CloudBackendError, ValueError) as exc:
            _fail(str(exc))
            return
    activity.log_event(
        "ask",
        question=question[:200],
        sources=len(sources),
        duration_ms=int((time.monotonic() - started) * 1000),
    )

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
