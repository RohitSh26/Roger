"""Pre-commit guard: quiz on staged changes before a commit lands.

Installed at .git/hooks/pre-commit. Skips are logged, never hidden:
ROGER_SKIP=1 logs to history.db and exits 0; `git commit --no-verify`
bypasses the hook at the git level (not observable, so not logged).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from roger.config import load_config
from roger.exceptions import (
    CloudBackendError,
    GraphNotFoundError,
    ModelNotRegisteredError,
    OllamaNotRunningError,
)
from roger import freshness
from roger.docs import doc_questions
from roger.freshness import is_source_file
from roger.generator import interleave_questions, iter_questions
from roger.graph import normalize_path, get_changed_nodes, get_quizzable_nodes, load_graph
from roger.quiz import QuestionStream, node_display_names, run_quiz
from roger.storage import record_session, record_skip

HOOK_PATH = Path(".git/hooks/pre-commit")


def uncovered_source_files(staged_files: list[str], graph) -> list[str]:
    """Staged source files with no node in the graph — invisible to the quiz."""
    graph_files = {
        normalize_path(str(attrs.get("file") or ""))
        for _, attrs in graph.nodes(data=True)
    }
    return [
        f
        for f in staged_files
        if is_source_file(f) and normalize_path(f) not in graph_files
    ]
HOOK_MARKER = "# installed by roger"
HOOK_SCRIPT = f"""#!/bin/sh
{HOOK_MARKER}
roger guard
"""


def run_guard() -> None:
    """Full guard flow. Exits 0 on pass/skip, 1 on a blocking failure."""
    if os.environ.get("ROGER_SKIP"):
        _log_skip(reason="ROGER_SKIP env var")
        sys.exit(0)

    try:
        config = load_config()
    except ValueError as exc:
        print(exc)
        print("  Fix .roger/config.toml, or skip once with: ROGER_SKIP=1 git commit ...")
        sys.exit(1)
    if not config.guard.enabled:
        sys.exit(0)

    staged_files = _get_staged_files()
    if not staged_files:
        sys.exit(0)

    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
        if freshness.lock_held():
            # A background refresh may be mid-write — retry once rather than
            # letting it silently disable the gate with a misleading error.
            time.sleep(2)
            try:
                graph = load_graph(config.graph.path)
            except GraphNotFoundError:
                print("Roger: graph refresh in progress — guard skipped this once.")
                sys.exit(0)
        else:
            # Roger isn't set up in this repo — warn but never block the commit.
            print(exc)
            sys.exit(0)

    changed_nodes = get_changed_nodes(graph, staged_files)
    # Tests stay in scope for guard (the developer changed them), but doc
    # stubs and entry markers make meaningless questions.
    quizzable = set(get_quizzable_nodes(graph, exclude_tests=False))
    changed_nodes = [n for n in changed_nodes if n in quizzable]

    # Changed docs are quiz material too: you edited the ADR, prove you
    # know what it says.
    staged_docs = [f for f in staged_files if f.lower().endswith((".md", ".markdown"))]
    doc_qs = (
        doc_questions(files=staged_docs, count=2, difficulty=config.guard.difficulty)
        if staged_docs and config.docs.enabled
        else []
    )
    # Staged source files the graph has never seen: the newest code, which
    # is exactly what guard exists to check. Silent passes hide that; say it.
    uncovered = uncovered_source_files(staged_files, graph)
    if uncovered and not changed_nodes:
        duration = freshness.last_update_duration()
        estimate = f" (~{duration:.0f}s)" if duration else ""
        print(
            f"⚠ Roger: {len(uncovered)} staged source file(s) aren't in the "
            f"knowledge graph yet — run 'roger update'{estimate} to include them."
        )
        if config.guard.require_coverage:
            print("  Blocking: [guard] require_coverage = true in .roger/config.toml.")
            sys.exit(1)
        if not doc_qs:
            sys.exit(0)

    if not changed_nodes and not doc_qs:
        sys.exit(0)

    # Stream: the first question appears as soon as it is ready; the next
    # generates while the developer answers — commit-time waiting shrinks
    # to a single generation.
    code_count = max(0, config.quiz.questions_per_session - len(doc_qs))
    stream = QuestionStream(
        interleave_questions(
            iter_questions(
                changed_nodes,
                graph,
                difficulty=config.guard.difficulty,
                count=code_count,
                config=config,
            )
            if changed_nodes and code_count
            else iter(()),
            doc_qs,
        )
    )
    try:
        result = run_quiz(
            stream,
            session_type="guard",
            pass_threshold=config.quiz.pass_threshold,
            node_names=node_display_names(graph, changed_nodes),
            total=config.quiz.questions_per_session,
        )
    except ModelNotRegisteredError:
        # Lazy model install means "never downloaded" is a normal state.
        # A missing download is Roger's gap, not the developer's — never
        # block a commit over it.
        print(
            "⚠ Roger: AI model not installed yet — letting this commit through.\n"
            "  Run 'roger' once to set it up (one keypress)."
        )
        sys.exit(0)
    except (OllamaNotRunningError, CloudBackendError, ValueError) as exc:
        print(exc)
        print("  Skip this quiz once with: ROGER_SKIP=1 git commit ...")
        sys.exit(1)

    if result.total == 0:
        sys.exit(0)
    result.commit_hash = freshness.head_commit()  # the parent of the commit being made
    record_session(result)

    if result.passed:
        print("✓ Roger: quiz passed. Proceeding with commit.")
        sys.exit(0)
    if config.guard.block_on_fail:
        print(f"✗ Roger: {result.score}/{result.total} — use --no-verify to skip.")
        sys.exit(1)
    print(f"⚠ Roger: {result.score}/{result.total} — warning only (block_on_fail=false).")
    sys.exit(0)


def _get_staged_files() -> list[str]:
    """subprocess: git diff --cached --name-only"""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _log_skip(reason: str) -> None:
    """Record a skip event to history.db."""
    record_skip(reason=reason, session_type="guard")


def _resolved_hooks_dir() -> Path:
    """The real hooks dir: honors worktrees and core.hooksPath."""
    proc = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(
            "Not inside a git repository — cd into one and run 'roger guard install'."
        )
    return (Path.cwd() / proc.stdout.strip()).resolve()


def install_hook(hook_path: Path | None = None) -> None:
    """Write the pre-commit hook and chmod +x (worktree/core.hooksPath aware)."""
    if hook_path is None:
        hook_path = _resolved_hooks_dir() / "pre-commit"
    if hook_path.exists() and HOOK_MARKER not in hook_path.read_text(encoding="utf-8"):
        manager = subprocess.run(
            ["git", "config", "core.hooksPath"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        managed = f" (managed hooks directory: core.hooksPath={manager})" if manager else ""
        raise FileExistsError(
            f"A pre-commit hook not installed by Roger already exists at "
            f"{hook_path}{managed}. Add 'roger guard' to it manually, then rerun."
        )
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def uninstall_hook(hook_path: Path | None = None) -> bool:
    """Remove the pre-commit hook if it was installed by Roger.

    Returns True if a Roger hook was removed, False if none was found.
    """
    if hook_path is None:
        hook_path = _resolved_hooks_dir() / "pre-commit"
    if not hook_path.exists():
        return False
    if HOOK_MARKER not in hook_path.read_text(encoding="utf-8"):
        return False
    hook_path.unlink()
    return True
