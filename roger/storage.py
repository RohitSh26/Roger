"""SQLite storage: .roger/cache.db (question cache) and .roger/history.db (quiz history).

Connections are opened and closed per-function — no persistent connection.
All storage failures raise CacheError with a descriptive message.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from roger.config import ROGER_DIR
from roger.exceptions import CacheError
from roger.models import Question, QuizResult

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS question_cache (
    hash            TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    questions_json  TEXT NOT NULL,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version   TEXT
);
"""

HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_type    TEXT NOT NULL,
    commit_hash     TEXT,
    module_scope    TEXT,
    score           INTEGER,
    total           INTEGER,
    passed          BOOLEAN,
    skipped         BOOLEAN DEFAULT FALSE,
    skip_reason     TEXT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_secs   INTEGER
);

CREATE TABLE IF NOT EXISTS quiz_answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES quiz_sessions(id),
    node_id         TEXT NOT NULL,
    question        TEXT NOT NULL,
    user_answer     TEXT,
    correct_answer  TEXT,
    is_correct      BOOLEAN,
    difficulty      TEXT
);
"""


def get_db_path(db_name: str) -> str:
    """Returns .roger/cache.db or .roger/history.db"""
    return str(ROGER_DIR / db_name)


def _connect(db_name: str, schema: str) -> sqlite3.Connection:
    """Open a connection, ensuring .roger/ and the schema exist."""
    path = Path(get_db_path(db_name))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.executescript(schema)
        return conn
    except (OSError, sqlite3.Error) as exc:
        raise CacheError(f"Could not open {path}: {exc}") from exc


ROGER_GITIGNORE = """# maintained by Roger — machine-local files, never commit these
vectors.db
activity.log
update.log
update-state.json
update.lock
quiz-pending.json
history.db
Modelfile
"""


def ensure_roger_gitignore() -> None:
    """Make the right git behavior automatic: cache.db and config.toml stay
    shareable; machine-local files can never be committed by accident."""
    try:
        path = ROGER_DIR / ".gitignore"
        if not path.exists() or path.read_text(encoding="utf-8") != ROGER_GITIGNORE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ROGER_GITIGNORE, encoding="utf-8")
    except OSError:
        pass


def init_dbs() -> None:
    """Create both databases and their tables (used by `roger init`)."""
    _connect("cache.db", CACHE_SCHEMA).close()
    _connect("history.db", HISTORY_SCHEMA).close()
    ensure_roger_gitignore()


# --- cache.db ---------------------------------------------------------------

def get_cached_questions(hash: str) -> Optional[list[Question]]:
    """Return cached questions for a code hash, or None on cache miss."""
    try:
        with closing(_connect("cache.db", CACHE_SCHEMA)) as conn:
            row = conn.execute(
                "SELECT questions_json FROM question_cache WHERE hash = ?", (hash,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise CacheError(f"Cache lookup failed for hash {hash[:12]}…: {exc}") from exc

    if row is None:
        return None
    try:
        return [Question(**q) for q in json.loads(row["questions_json"])]
    except (json.JSONDecodeError, TypeError) as exc:
        raise CacheError(f"Corrupt cache entry for hash {hash[:12]}…: {exc}") from exc


def cache_questions(
    hash: str,
    node_id: str,
    difficulty: str,
    questions: list[Question],
    model_version: str,
) -> None:
    """Store generated questions under a code hash (replaces any prior entry)."""
    payload = json.dumps([asdict(q) for q in questions])
    try:
        with closing(_connect("cache.db", CACHE_SCHEMA)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO question_cache "
                "(hash, node_id, difficulty, questions_json, model_version) "
                "VALUES (?, ?, ?, ?, ?)",
                (hash, node_id, difficulty, payload, model_version),
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise CacheError(f"Cache write failed for node {node_id}: {exc}") from exc


# --- history.db -------------------------------------------------------------

def record_session(result: QuizResult) -> int:
    """Record a completed quiz session. Returns the new session_id."""
    try:
        with closing(_connect("history.db", HISTORY_SCHEMA)) as conn:
            cursor = conn.execute(
                "INSERT INTO quiz_sessions "
                "(session_type, commit_hash, module_scope, score, total, passed, "
                " skipped, duration_secs) "
                "VALUES (?, ?, ?, ?, ?, ?, FALSE, ?)",
                (
                    result.session_type,
                    result.commit_hash,
                    result.module_scope,
                    result.score,
                    result.total,
                    result.passed,
                    int(result.duration_secs),
                ),
            )
            conn.commit()
            session_id = cursor.lastrowid
    except sqlite3.Error as exc:
        raise CacheError(f"Could not record quiz session: {exc}") from exc

    if session_id is None:  # pragma: no cover - sqlite always sets lastrowid here
        raise CacheError("Could not record quiz session: no row id returned")
    record_answers(session_id, result.answers)
    return session_id


def record_answers(session_id: int, answers: list) -> None:
    """Record per-question answers for a session (list of QuizAnswer)."""
    rows = [
        (
            session_id,
            a.question.node_id,
            a.question.question,
            a.user_answer,
            a.question.correct,
            a.is_correct,
            a.question.difficulty,
        )
        for a in answers
    ]
    try:
        with closing(_connect("history.db", HISTORY_SCHEMA)) as conn:
            conn.executemany(
                "INSERT INTO quiz_answers "
                "(session_id, node_id, question, user_answer, correct_answer, "
                " is_correct, difficulty) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise CacheError(f"Could not record answers for session {session_id}: {exc}") from exc


def record_skip(reason: str, session_type: str = "guard") -> int:
    """Record a skipped guard run (ROGER_SKIP). Returns the session_id."""
    try:
        with closing(_connect("history.db", HISTORY_SCHEMA)) as conn:
            cursor = conn.execute(
                "INSERT INTO quiz_sessions (session_type, skipped, skip_reason) "
                "VALUES (?, TRUE, ?)",
                (session_type, reason),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)
    except sqlite3.Error as exc:
        raise CacheError(f"Could not record skip: {exc}") from exc
