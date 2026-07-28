"""Local activity log: what was asked of Roger, by whom-shaped caller, at what cost.

Observability with the project's privacy stance intact: the log is a local
JSONL file inside .roger/ — machinery metrics (what was requested, tokens
served, duration), never centralized, never a person-level dashboard. Teams
that want it in their own observability stack can ship the file themselves;
Roger sends nothing anywhere.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ACTIVITY_PATH = Path(".roger/activity.log")
_ROTATE_BYTES = 2_000_000  # keep the current file bounded; one prior kept


def log_event(command: str, **fields) -> None:
    """Append one JSONL line. Never raises — observability must not break use."""
    try:
        ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ACTIVITY_PATH.exists() and ACTIVITY_PATH.stat().st_size > _ROTATE_BYTES:
            ACTIVITY_PATH.replace(ACTIVITY_PATH.with_suffix(".log.1"))
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": command,
            # Non-TTY stdout means a program (agent, script, pipe) consumed
            # the output — the honest signal for "was this an agent?".
            "caller": "human" if sys.stdout.isatty() else "agent/script",
            **fields,
        }
        with ACTIVITY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_recent(limit: int = 20) -> list[dict]:
    """Most recent events, newest first."""
    try:
        lines = ACTIVITY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in reversed(lines[-limit:]):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
