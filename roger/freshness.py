"""Graph freshness: staleness detection, safe updates, background refresh.

Design reviewed and mandated:
- Staleness is content-based (changed SOURCE files since built_at_commit,
  plus dirty source files) — never raw commit distance, so it stays silent
  when nothing graphify cares about changed.
- Updates never pass --force and always scrub GRAPHIFY_FORCE from the
  child environment; force is only ever applied interactively in the CLI
  after an explicit confirmation.
- A single-writer lock (.roger/update.lock) means concurrent sessions skip
  rather than pile up; failure memory (.roger/update-state.json) means a
  doomed update is never respawned for the same repo fingerprint, and its
  failure is surfaced exactly once.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROGER_DIR = Path(".roger")
LOCK_PATH = ROGER_DIR / "update.lock"
STATE_PATH = ROGER_DIR / "update-state.json"
UPDATE_LOG_PATH = ROGER_DIR / "update.log"
LOCK_STALE_SECS = 10 * 60

# One shared definition of "source file" for guard coverage checks and
# staleness detection — the two must never disagree.
SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rb", ".rs",
    ".c", ".h", ".cpp", ".cc", ".cs", ".sh", ".php", ".kt", ".swift",
    ".scala", ".sql",
}


def is_source_file(path: str) -> bool:
    dot = path.rfind(".")
    return dot != -1 and path[dot:].lower() in SOURCE_EXTENSIONS


def _git(*args: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def repo_root() -> Optional[Path]:
    top = _git("rev-parse", "--show-toplevel")
    return Path(top) if top else None


def head_commit() -> Optional[str]:
    return _git("rev-parse", "HEAD")


def built_at_commit(graph_path: str) -> Optional[str]:
    try:
        data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("built_at_commit")
    return str(value) if value else None


def dirty_source_files() -> list[str]:
    porcelain = _git("status", "--porcelain")
    if not porcelain:
        return []
    files = []
    for line in porcelain.splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if is_source_file(path):
            files.append(path)
    return files


def stale_source_files(graph_path: str) -> list[str]:
    """Source files changed since the graph was built (committed or dirty).

    Empty list means fresh — or that staleness can't be determined (shallow
    clone, missing built_at_commit), in which case silence is the mandated
    degradation.
    """
    changed: list[str] = []
    base = built_at_commit(graph_path)
    if base:
        diff = _git("diff", "--name-only", f"{base}..HEAD")
        if diff is not None:  # None = base not in history (shallow) → degrade
            changed.extend(f for f in diff.splitlines() if is_source_file(f))
    changed.extend(dirty_source_files())
    return sorted(set(changed))


def repo_fingerprint(graph_path: str) -> str:
    """Identity of 'the repo in its current state' for failure memory."""
    dirty = "\n".join(dirty_source_files())
    head = head_commit() or "no-head"
    digest = hashlib.sha256(dirty.encode("utf-8")).hexdigest()[:16]
    return f"{head}:{digest}"


# --- single-writer lock -----------------------------------------------------


def acquire_lock() -> bool:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _lock_is_stale():
            LOCK_PATH.unlink(missing_ok=True)
            return acquire_lock()
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "started": time.time()}, f)
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def lock_held() -> bool:
    return LOCK_PATH.exists() and not _lock_is_stale()


def _lock_is_stale() -> bool:
    try:
        info = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if time.time() - float(info.get("started", 0)) > LOCK_STALE_SECS:
        return True
    pid = int(info.get("pid", 0))
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


# --- update state (failure memory) --------------------------------------------


def read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def last_update_duration() -> Optional[float]:
    duration = read_state().get("duration_secs")
    return float(duration) if duration else None


def pending_failure_note(graph_path: str) -> Optional[str]:
    """The one-shot line about a failed background refresh, or None.

    Shown once per failure fingerprint — a single sentence naming the one
    verb, never repeated nagging.
    """
    state = read_state()
    if state.get("outcome") not in ("shrink_refused", "failed"):
        return None
    if state.get("fingerprint") != repo_fingerprint(graph_path):
        return None
    if state.get("surfaced"):
        return None
    state["surfaced"] = True
    write_state(state)
    if state.get("outcome") == "shrink_refused":
        return (
            "Couldn't refresh the graph in the background (code was deleted "
            "since the last index) — 'roger update' will fix it."
        )
    return "Couldn't refresh the graph in the background — 'roger update' will retry it."


# --- running updates ------------------------------------------------------------


@dataclass
class UpdateResult:
    outcome: str  # "ok" | "shrink_refused" | "failed"
    duration_secs: float
    nodes_before: int
    nodes_after: int
    detail: str = ""


def graphify_executable() -> str:
    """graphify installs beside roger in the same environment — resolve it
    there first so Roger works when invoked by absolute path (no activated
    venv, hooks, agents)."""
    sibling = Path(sys.executable).parent / "graphify"
    return str(sibling) if sibling.exists() else "graphify"


def _scrubbed_env() -> dict:
    # GRAPHIFY_FORCE=1 exported once in some past shell must never turn a
    # Roger-initiated update into a silent force-overwrite.
    env = dict(os.environ)
    env.pop("GRAPHIFY_FORCE", None)
    return env


def _node_count(graph_path: str) -> int:
    try:
        return len(json.loads(Path(graph_path).read_text(encoding="utf-8")).get("nodes", []))
    except (OSError, json.JSONDecodeError):
        return 0


def run_update(graph_path: str, force: bool = False) -> UpdateResult:
    """Run `graphify update .` synchronously. Caller holds the lock.

    Never passes --force unless the caller explicitly obtained user consent
    (the interactive `roger update` path is the only one that ever does).
    """
    before = _node_count(graph_path)
    started = time.monotonic()
    cmd = [graphify_executable(), "update", "."]
    if force:
        cmd.append("--force")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=_scrubbed_env()
        )
    except FileNotFoundError:
        result = UpdateResult(
            "failed", 0.0, before, before, "graphify executable not found"
        )
        write_state(
            {
                "fingerprint": repo_fingerprint(graph_path),
                "outcome": result.outcome,
                "duration_secs": 0,
                "surfaced": False,
            }
        )
        return result
    duration = time.monotonic() - started
    after = _node_count(graph_path)

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or (not force and after < before and "force" in output.lower()):
        outcome = "shrink_refused" if "force" in output.lower() else "failed"
        result = UpdateResult(outcome, duration, before, after, output[-400:])
    elif not force and after < before:
        # graphify declined to write (shrink protection) — graph unchanged.
        result = UpdateResult("shrink_refused", duration, before, after, output[-400:])
    else:
        result = UpdateResult("ok", duration, before, after)

    write_state(
        {
            "fingerprint": repo_fingerprint(graph_path),
            "outcome": result.outcome,
            "duration_secs": round(duration, 1),
            "nodes_before": before,
            "nodes_after": after,
            "surfaced": False,
        }
    )
    return result


def maybe_refresh_in_background(graph_path: str) -> bool:
    """Spawn a detached background refresh if warranted. Returns True if spawned.

    Warranted = source files changed since the graph was built, no update
    already running, and this exact repo state hasn't already failed.
    """
    if not stale_source_files(graph_path):
        return False
    if lock_held():
        return False
    state = read_state()
    if (
        state.get("outcome") in ("shrink_refused", "failed")
        and state.get("fingerprint") == repo_fingerprint(graph_path)
    ):
        return False  # never respawn a doomed update for the same state

    UPDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(UPDATE_LOG_PATH, "a")
    subprocess.Popen(
        [sys.executable, "-m", "roger.cli", "update", "--background"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        env=_scrubbed_env(),
    )
    return True
