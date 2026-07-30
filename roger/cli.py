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
from roger.session import quiz_blueprint
from roger.storage import init_dbs

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
    if not embeddings.offer_appropriate(config, reoffer):
        # Includes the enabled-machine case — where a fresh repo may still
        # need its first index build.
        embeddings.self_heal_index(config, config.graph.path)
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


def _refresh_semantic(config: Config, live: bool = True) -> Optional[embeddings.IndexRefresh]:
    """Refresh the smarter-search index IN THE FOREGROUND and narrate every
    outcome — success, partial, model unreachable, or crash. Field lesson:
    a swallowed exception here once looped a machine forever between
    'interrupted' (doctor) and 'off' (update) with an empty update.log.
    Failures must be printed, never converted into a misleading state.

    live=True renders a Rich progress line; live=False prints plain lines,
    which the background updater's stdout redirect lands in .roger/update.log.
    """

    def say(message: str) -> None:
        if live:
            console.print(message)
        else:
            from rich.markup import render as _render  # strip markup for logs

            print(_render(message).plain, flush=True)

    from roger.llm.local import is_ollama_running

    if not is_ollama_running(config.ollama.url):
        say("• Smarter search: Ollama isn't reachable right now — index left as-is.")
        return None
    if embeddings.model_digest(config, timeout=8) is None:
        say("• Smarter search: off (keyword-only) — 'roger doctor' can enable it.")
        return None
    try:
        graph = load_graph(config.graph.path)
        if live:
            with console.status("[dim]Smarter search: checking the index…[/dim]") as spinner:
                stats = embeddings.refresh_index(
                    graph, config,
                    progress=lambda done, total: spinner.update(
                        f"[dim]Smarter search: embedding {done:,} of {total:,} "
                        "changed functions…[/dim]"
                    ),
                )
        else:
            stats = embeddings.refresh_index(
                graph, config,
                progress=lambda done, total: print(
                    f"smarter search: embedded {done}/{total}", flush=True
                ),
            )
    except Exception as exc:  # noqa: BLE001 - report it; never block the update
        say(
            f"⚠ Smarter search: index refresh failed — "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        )
        return None
    if stats is None:
        say("• Smarter search: embedding model vanished mid-refresh — index left as-is.")
    elif stats.with_vec < stats.cards:
        say(
            f"⚠ Smarter search: indexed {stats.with_vec:,} of {stats.cards:,} — "
            "embedding stopped early (Ollama hiccup?). 'roger update' finishes it."
        )
    elif stats.embedded:
        say(
            f"✓ Smarter search: re-indexed {stats.embedded} changed function(s) "
            f"({stats.with_vec:,} of {stats.cards:,} indexed)."
        )
    else:
        say(f"✓ Smarter search: index already current ({stats.cards:,} functions).")
    return stats


def _rebuild_index_now(config: Config) -> bool:
    """Doctor's TTY path: fix the index right here, on screen, with
    progress — no 'watch it somewhere else'. Ctrl-C safe (builds resume).
    Returns True when the index came out fully healthy."""
    if not freshness.acquire_lock():
        return False
    try:
        stats = _refresh_semantic(config, live=True)
        return stats is not None and stats.with_vec >= stats.cards
    finally:
        freshness.release_lock()


def _watch_background_update(config: Config) -> None:
    """Attach to a running background update and show it move — the honest
    answer to 'is the index actually building?'."""
    with console.status("[dim]working…[/dim]") as spinner:
        while freshness.lock_held():
            progress = embeddings.index_progress()
            if progress and progress[1]:
                spinner.update(
                    f"[dim]smarter search: {progress[0]:,} of {progress[1]:,} "
                    "functions embedded…[/dim]"
                )
            time.sleep(1)
    final = embeddings.index_status(config)
    if final.mode == "semantic+keyword":
        console.print(
            f"✓ Background update finished — smarter search index current "
            f"({final.cards:,} functions)."
        )
    else:
        console.print(f"• Background update finished — smarter search: {final.reason}.")


def _semantic_doctor_advice(
    config: Config, status: embeddings.IndexStatus
) -> tuple[str, str]:
    """(finding, remedy) for a degraded index on a machine that has the
    embed model. Never leaves the user guessing between four states:
    running / just started / failed / paused — each names its next step,
    and 'roger update' is always the one safe verb (it attaches to a
    running build instead of restarting it)."""
    line = f"smarter search: {status.reason}"
    if freshness.lock_held():
        return line, "watch it live: roger update   (safe — attaches, never restarts)"
    if embeddings.self_heal_index(config, config.graph.path):
        return (
            line + " — rebuild started in the background just now",
            "watch it live: roger update — or re-run roger doctor for a snapshot",
        )
    state = freshness.read_state()
    if state.get("outcome") in ("failed", "shrink_refused"):
        return (
            line + " — the last background attempt failed",
            "run: roger update   (retries in the foreground and shows the error; "
            "details also in .roger/update.log)",
        )
    return (
        line + " — retry is paused after a recent attempt",
        "run: roger update   (rebuilds right now, with progress)",
    )


def _note_index_build() -> None:
    """One dim line when an index build is running during a quiz — the quiz
    works fine meanwhile, but invisible background work is banned."""
    if not freshness.lock_held():
        return
    progress = embeddings.index_progress()
    if progress and progress[1] and progress[0] < progress[1]:
        console.print(
            f"[dim]Smarter search index building in the background "
            f"({progress[0]:,} of {progress[1]:,} functions) — "
            "the quiz works fine meanwhile.[/dim]"
        )


def _print_semantic_result(stats: Optional[embeddings.IndexRefresh]) -> None:
    """One honest line about what the update did to the smarter-search
    index — it always ran; it should never run invisibly."""
    if stats is None:
        console.print("[dim]• Smarter search: off (keyword-only) — 'roger doctor' to enable.[/dim]")
    elif stats.embedded:
        console.print(
            f"✓ Smarter search: re-indexed {stats.embedded} changed function(s) "
            f"({stats.with_vec:,} of {stats.cards:,} indexed)."
        )
    else:
        console.print(
            f"✓ Smarter search: index already current ({stats.cards:,} functions)."
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
    semantic_started = embeddings.self_heal_index(config, config.graph.path)

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


APP_PATH = Path(__file__).parent / "app.py"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _streamlit_missing_remedy() -> str:
    """The right install command for THIS environment — a raw pip failure
    under pipx/uv or a PEP 668 managed Python is exactly the hiccup the
    Simplicity Doctrine forbids."""
    import sysconfig

    if "pipx" in sys.prefix:
        return "pipx inject roger-cli streamlit"
    if (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists():
        return f"{sys.executable} -m pip install --user streamlit"
    return f"{sys.executable} -m pip install streamlit"


def _ensure_streamlit() -> None:
    """Lazy app-dependency install: one keypress, live progress, never in
    the base install (agents and CI don't pay for a GUI)."""
    if importlib.util.find_spec("streamlit") is not None:
        return
    remedy = _streamlit_missing_remedy()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        _fail(
            "✗ Roger: the app needs Streamlit (not installed).\n"
            f"  Install it with: {remedy}"
        )
        return
    if not typer.confirm(
        "The Roger app needs Streamlit (~250 MB of Python packages, one time). "
        "Install now?", default=True,
    ):
        console.print("[dim]No install — the terminal quiz and ask work as always.[/dim]")
        raise typer.Exit(code=0)
    if "pip install" not in remedy:
        _fail(f"✗ Roger: this Python is managed externally — install manually:\n  {remedy}")
        return
    # No capture: pip's own progress streams to the terminal (doctrine:
    # anything that doesn't finish in seconds must show progress).
    result = subprocess.run(remedy.split(), check=False)
    if result.returncode != 0 or importlib.util.find_spec("streamlit") is None:
        _fail(f"✗ Roger: Streamlit install failed — try manually:\n  {remedy}")


@app.command("app")
def app_command() -> None:
    """Open the Roger app in your browser — quiz and ask, all local.

    A foreground process on 127.0.0.1 only: it starts when you run this,
    stops when you press Ctrl-C, and never talks to anything but your
    local Ollama.
    """
    top = _anchor_repo_root()
    if top is None:
        _fail("✗ Roger: run this inside a git repository.")
        return
    config = _load_config()
    if not Path(config.graph.path).exists():
        _offer_setup(top, config)
    _ensure_model_ready(config)
    _ensure_streamlit()

    port = _free_port()
    env = freshness._scrubbed_env()
    env.update(
        # Suppress Streamlit's first-run email prompt and its usage
        # telemetry per-process — never by writing the user's global
        # ~/.streamlit config. Headless also stops Streamlit opening the
        # browser itself; Roger owns that moment.
        STREAMLIT_SERVER_HEADLESS="true",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
    )
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_PATH),
            "--server.address=127.0.0.1", f"--server.port={port}",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    import socket

    for _ in range(120):  # first launch imports streamlit: allow ~30s
        if proc.poll() is not None:
            _fail("✗ Roger: the app failed to start — run 'roger doctor'.")
            return
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.25)
    webbrowser.open(url)
    console.print(f"✓ Roger app running at {url} — press Ctrl-C here to stop it.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        console.print("Stopped.")


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
        False, "--web", help="Deprecated: opens the Roger app (roger app)."
    ),
    count: int = typer.Option(
        0,
        "--count",
        "-n",
        help="Questions this session (default: questions_per_session from .roger/config.toml).",
    ),
) -> None:
    """Quiz yourself on this repo (whole repo, config defaults)."""
    if web:
        # One-release shim: the static page (and its answer-code dance)
        # was replaced by the Roger app.
        console.print("The browser quiz is now the Roger app — starting it.")
        app_command()
        return
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
    _note_index_build()

    # The LLM's first actual use — install the model now if it never was.
    _ensure_model_ready(config)

    # Session size: --count flag wins, else the config value — the
    # docs/code/design split scales automatically from it (see session.py).
    count = count if count > 0 else config.quiz.questions_per_session

    stream_iter, names, total = quiz_blueprint(
        graph, config, count,
        lambda g_, n: _pick_quiz_nodes(g_, n, config.graph.god_node_weight),
    )

    backend = (
        f"Azure Foundry '{config.model.azure_deployment}'"
        if config.model.provider == "azure-anthropic"
        else f"Ollama '{config.model.local}'"
    )
    console.print(
        f"Preparing {total} {config.quiz.default_difficulty} questions via {backend} — "
        "the rest generate as you answer…"
    )

    stream = QuestionStream(stream_iter)
    try:
        result = run_quiz(
            stream,
            session_type="quiz",
            pass_threshold=config.quiz.pass_threshold,
            node_names=names,
            total=total,
        )
    except (OllamaNotRunningError, ModelNotRegisteredError, CacheError, CloudBackendError, ValueError) as exc:
        _fail(str(exc))
        return

    if result.total == 0:
        _fail("✗ Roger: could not generate any questions for this repo.")


@app.command(hidden=True)
def record(code: str = typer.Argument("")) -> None:
    """(Retired) The web quiz now grades itself — see 'roger app'."""
    console.print(
        "The answer-code flow is gone: the web quiz was replaced by the Roger "
        "app, which grades as you click. Start it with: roger app"
    )


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
        console.print("An update is already running in the background — watching it:")
        _watch_background_update(config)
        return
    try:
        result = freshness.run_update(config.graph.path)
        # Embed step rides the same flow; never fails the graph update.
        if background:
            if result.outcome == "ok":
                _refresh_semantic(config, live=False)
            return
        if result.outcome == "ok":
            delta = result.nodes_after - result.nodes_before
            console.print(
                f"✓ Graph updated in {result.duration_secs:.0f}s — "
                f"{result.nodes_after:,} nodes ({'+' if delta >= 0 else ''}{delta:,})."
            )
            _refresh_semantic(config, live=True)
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
                    _refresh_semantic(config)
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
## Roger — MANDATORY first step for codebase questions

RULE 1 — Roger FIRST, always. For ANY question about this codebase ("how
does X work", "where is Y handled", "what calls Z"), your FIRST action is:

    roger context "<your question>" --budget 2000

Never grep, search, or open files as a first step. One Roger call returns
the relevant source (complete functions), the team's recorded decisions
(ADRs, contracts), and call relationships in one budgeted, cited pack —
it typically replaces 5-7 tool calls and thousands of tokens.

RULE 2 — if the pack answers the question, STOP. Its code blocks are
VERBATIM source, mechanically extracted seconds ago — no AI wrote or
summarized them. Re-reading a cited file returns byte-identical text;
doing so is pure token waste. Answer from the pack and cite its sources.

RULE 3 — if the pack misses, re-query ONCE with likely code vocabulary
(e.g. "throttle" → rate limit, backoff, sleep). Only after that second
miss may you fall back to grep and file reading.

RULE 4 — the ONLY reasons to open a file the pack already cites: you are
about to MODIFY that code, a claim lacks a citation, citations conflict,
or the pack says it was truncated.
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
    semantic_degraded = False
    semantic_fix_now = False

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
        # reported — the rebuild starts right here, and the message answers
        # the three questions a degraded state raises: is it running, how
        # do I watch it, and can I break it (no).
        status = embeddings.index_status(config)
        if status.mode == "semantic+keyword":
            checks.append(
                ("ok", f"smarter search active ({status.with_vec}/{status.cards} indexed)", "")
            )
        elif not status.model_present:
            checks.append(("ok", f"search: keyword-only ({status.reason})", ""))
        elif (
            sys.stdin.isatty() and sys.stdout.isatty() and not freshness.lock_held()
        ):
            # A human is looking at a broken index: fix it HERE, on screen,
            # with progress — not "watch it somewhere else in the background".
            checks.append(("warn", f"smarter search: {status.reason} — fixing it below", ""))
            semantic_fix_now = True
        else:
            line, remedy = _semantic_doctor_advice(config, status)
            checks.append(("warn", line, remedy))
            semantic_degraded = True

    icons = {"ok": "[green]✓[/green]", "warn": "[yellow]⚠[/yellow]", "fail": "[red]✗[/red]"}
    failed = False
    for check_status, finding, remedy in checks:
        console.print(f"{icons[check_status]} {escape(finding)}")
        if remedy:
            console.print(f"   → {escape(remedy)}")
        failed = failed or check_status == "fail"
    if semantic_fix_now:
        console.print()
        console.print("Rebuilding the smarter-search index now (Ctrl-C is safe — it resumes):")
        if _rebuild_index_now(_load_config()):
            console.print("[green]✓[/green] smarter search is healthy again.")
    if semantic_degraded:
        console.print(
            "[dim]While the index rebuilds, everything keeps working — quizzes, "
            "ask, and agent packs just use keyword search. No Roger command can "
            "interrupt the build.[/dim]"
        )
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
        embeddings.self_heal_index(config, config.graph.path)
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


# One list drives install AND uninstall — adding an agent file target is a
# one-line change. only_if_exists: never create that file, only amend it.
AGENT_TARGETS: list[tuple[str, str, bool]] = [
    ("AGENTS.md", "OpenCode, Codex, Copilot CLI, and Aider read it", False),
    (".github/copilot-instructions.md", "GitHub Copilot in VS Code reads it", False),
    ("CLAUDE.md", "Claude Code reads it", True),
]


@agent_app.command("install")
def agent_install() -> None:
    """Write the Roger instructions into the files coding agents read."""
    if _anchor_repo_root() is None:
        _fail("✗ Roger: run this inside a git repository.")
    for name, audience, only_if_exists in AGENT_TARGETS:
        path = Path(name)
        if only_if_exists and not path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome = _install_snippet(path)
        console.print(f"✓ {name} {outcome} — {audience}.")
    console.print("  Agents will now run 'roger context' before grepping and reading files.")

    # The human is setting up agents right now — the one moment to make
    # sure their context packs get semantic matching from the first call.
    config = _load_config()
    status = embeddings.index_status(config)
    if status.mode == "semantic+keyword":
        console.print("✓ Smarter search is on — packs match by meaning as well as keywords.")
    elif embeddings.self_heal_index(config, config.graph.path):
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
    for name, _, _ in AGENT_TARGETS:
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
        False, "--web", help="Deprecated: use the Roger app (roger app)."
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
        console.print(
            "[dim]--web was replaced by the Roger app (roger app) — "
            "showing the answer here:[/dim]"
        )

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
