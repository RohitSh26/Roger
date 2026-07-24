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
from pathlib import Path

from roger.config import load_config
from roger.exceptions import GraphNotFoundError, ModelNotRegisteredError, OllamaNotRunningError
from roger.docs import doc_questions
from roger.generator import interleave_questions, iter_questions, order_cache_first
from roger.graph import get_changed_nodes, get_quizzable_nodes, load_graph
from roger.quiz import QuestionStream, node_display_names, run_quiz
from roger.storage import record_session, record_skip

HOOK_PATH = Path(".git/hooks/pre-commit")
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

    config = load_config()
    if not config.guard.enabled:
        sys.exit(0)

    staged_files = _get_staged_files()
    if not staged_files:
        sys.exit(0)

    try:
        graph = load_graph(config.graph.path)
    except GraphNotFoundError as exc:
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
    if not changed_nodes and not doc_qs:
        sys.exit(0)

    # Stream: the first question appears as soon as it is ready; the next
    # generates while the developer answers — commit-time waiting shrinks
    # to a single generation.
    code_count = max(0, config.quiz.questions_per_session - len(doc_qs))
    changed_nodes = order_cache_first(changed_nodes, graph, config.guard.difficulty)
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
    except (OllamaNotRunningError, ModelNotRegisteredError, ValueError) as exc:
        print(exc)
        print("  Skip this quiz once with: ROGER_SKIP=1 git commit ...")
        sys.exit(1)

    if result.total == 0:
        sys.exit(0)
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


def install_hook(hook_path: Path = HOOK_PATH) -> None:
    """Write the hook script to .git/hooks/pre-commit and chmod +x."""
    git_dir = hook_path.parent.parent
    if not git_dir.exists():
        raise FileNotFoundError(
            "No .git directory found — run 'roger guard install' from the repo root."
        )
    if hook_path.exists() and HOOK_MARKER not in hook_path.read_text(encoding="utf-8"):
        raise FileExistsError(
            f"A pre-commit hook not installed by Roger already exists at {hook_path}. "
            "Remove or merge it manually, then re-run 'roger guard install'."
        )
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def uninstall_hook(hook_path: Path = HOOK_PATH) -> bool:
    """Remove .git/hooks/pre-commit if it was installed by Roger.

    Returns True if a Roger hook was removed, False if none was found.
    """
    if not hook_path.exists():
        return False
    if HOOK_MARKER not in hook_path.read_text(encoding="utf-8"):
        return False
    hook_path.unlink()
    return True
