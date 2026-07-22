"""Tests for roger/storage.py — run against a tmp cwd so .roger/ is isolated."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from roger import storage
from roger.models import QuizAnswer, QuizResult
from tests.conftest import make_question

pytestmark = pytest.mark.usefixtures("in_tmp_repo")


def test_get_db_path() -> None:
    assert storage.get_db_path("cache.db") == str(Path(".roger") / "cache.db")


def test_init_dbs_creates_files() -> None:
    storage.init_dbs()
    assert Path(".roger/cache.db").exists()
    assert Path(".roger/history.db").exists()


def test_cache_miss_returns_none() -> None:
    assert storage.get_cached_questions("deadbeef" * 8) is None


def test_cache_roundtrip() -> None:
    questions = [make_question(), make_question(text="Second question?", correct="A")]
    storage.cache_questions("abc123", "payments.process_payment", "medium", questions, "roger-local")

    cached = storage.get_cached_questions("abc123")
    assert cached == questions  # dataclass equality covers every field


def test_cache_replace_overwrites() -> None:
    storage.cache_questions("k1", "n1", "medium", [make_question()], "roger-local")
    replacement = [make_question(text="Newer question?", difficulty="hard")]
    storage.cache_questions("k1", "n1", "hard", replacement, "roger-local")
    assert storage.get_cached_questions("k1") == replacement


def test_corrupt_cache_entry_raises_cache_error() -> None:
    storage.cache_questions("k2", "n1", "medium", [make_question()], "roger-local")
    with sqlite3.connect(storage.get_db_path("cache.db")) as conn:
        conn.execute("UPDATE question_cache SET questions_json = 'not json' WHERE hash = 'k2'")
        conn.commit()
    with pytest.raises(storage.CacheError):
        storage.get_cached_questions("k2")


def _make_result(score: int = 2, total: int = 3) -> QuizResult:
    q1 = make_question(node_id="payments.charge", text="Q1?")
    q2 = make_question(node_id="db.connect", text="Q2?")
    q3 = make_question(node_id="payments.charge", text="Q3?")
    answers = [
        QuizAnswer(question=q1, user_answer="B", is_correct=True, time_taken_secs=3.0),
        QuizAnswer(question=q2, user_answer="A", is_correct=False, time_taken_secs=5.0),
        QuizAnswer(question=q3, user_answer="B", is_correct=True, time_taken_secs=2.0),
    ]
    return QuizResult(
        session_type="quiz",
        answers=answers,
        score=score,
        total=total,
        passed=True,
        commit_hash=None,
        module_scope="src/payments",
        duration_secs=10.4,
    )


def test_record_session_and_history() -> None:
    session_id = storage.record_session(_make_result())
    assert session_id >= 1

    history = storage.get_history()
    assert len(history) == 1
    row = history[0]
    assert row["session_type"] == "quiz"
    assert row["score"] == 2
    assert row["total"] == 3
    assert row["passed"]
    assert not row["skipped"]
    assert row["module_scope"] == "src/payments"
    assert row["duration_secs"] == 10


def test_record_session_records_answers() -> None:
    session_id = storage.record_session(_make_result())
    with sqlite3.connect(storage.get_db_path("history.db")) as conn:
        rows = conn.execute(
            "SELECT node_id, user_answer, is_correct FROM quiz_answers "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    assert rows == [
        ("payments.charge", "B", 1),
        ("db.connect", "A", 0),
        ("payments.charge", "B", 1),
    ]


def test_get_weak_nodes() -> None:
    storage.record_session(_make_result())
    weak = storage.get_weak_nodes()
    assert weak == [{"node_id": "db.connect", "wrong_count": 1, "total_count": 1}]


def test_record_skip_and_skip_history() -> None:
    storage.record_skip("ROGER_SKIP env var")
    skips = storage.get_skip_history()
    assert len(skips) == 1
    assert skips[0]["session_type"] == "guard"
    assert skips[0]["skip_reason"] == "ROGER_SKIP env var"

    # Skips must not pollute regular history scoring fields.
    history = storage.get_history()
    assert history[0]["skipped"]


def test_get_score_trend_includes_recent_session() -> None:
    storage.record_session(_make_result())
    trend = storage.get_score_trend(days=30)
    assert len(trend) == 1
    assert trend[0]["sessions"] == 1
    assert trend[0]["avg_pct"] == pytest.approx(2 / 3 * 100)


def test_history_limit() -> None:
    for _ in range(5):
        storage.record_session(_make_result())
    assert len(storage.get_history(limit=3)) == 3
