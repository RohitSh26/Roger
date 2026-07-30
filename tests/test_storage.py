"""Tests for roger/storage.py — run against a tmp cwd so .roger/ is isolated."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from roger import storage
from tests.conftest import make_question

pytestmark = pytest.mark.usefixtures("in_tmp_repo")


def test_get_db_path() -> None:
    assert storage.get_db_path("cache.db") == str(Path(".roger") / "cache.db")


def test_init_dbs_creates_files() -> None:
    storage.init_dbs()
    assert Path(".roger/cache.db").exists()


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


def test_provider_aliases_normalize(tmp_path) -> None:
    from pathlib import Path

    from roger.config import load_config

    for raw in ("azure", "Azure-Anthropic", "azure_anthropic", "FOUNDRY"):
        cfg_path = tmp_path / f"{raw.lower()}.toml"
        cfg_path.write_text(f'[model]\nprovider = "{raw}"\n', encoding="utf-8")
        assert load_config(Path(cfg_path)).model.provider == "azure-anthropic", raw
    ollama_path = tmp_path / "ollama.toml"
    ollama_path.write_text('[model]\nprovider = "local"\n', encoding="utf-8")
    assert load_config(Path(ollama_path)).model.provider == "ollama"


def test_unknown_provider_is_a_hard_error(tmp_path) -> None:
    from pathlib import Path

    from roger.config import load_config

    cfg = tmp_path / "bad.toml"
    cfg.write_text('[model]\nprovider = "azurr"\n', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(Path(cfg))
    assert "azurr" in str(excinfo.value)
    assert "azure-anthropic" in str(excinfo.value)


def test_misplaced_keys_warn_instead_of_vanishing(tmp_path, capsys) -> None:
    from pathlib import Path

    from roger.config import load_config

    cfg = tmp_path / "misplaced.toml"
    cfg.write_text(
        'provider = "azure-anthropic"\n\n[ollama]\nazure_endpoint = "https://x"\n',
        encoding="utf-8",
    )
    config = load_config(Path(cfg))
    err = capsys.readouterr().err
    assert "did you mean to put it under [model]?" in err
    assert "belongs under [model]" in err
    assert config.model.provider == "ollama"  # defaults, but loudly


def test_save_config_roundtrips_azure_settings(tmp_path) -> None:
    from pathlib import Path

    from roger.config import Config, ModelConfig, load_config, save_config

    config = Config(
        model=ModelConfig(
            provider="azure-anthropic",
            azure_endpoint="https://acme.services.ai.azure.com/anthropic",
            azure_deployment="claude-x",
        )
    )
    path = Path(tmp_path / "config.toml")
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.model.provider == "azure-anthropic"
    assert loaded.model.azure_endpoint == "https://acme.services.ai.azure.com/anthropic"
    assert loaded.model.azure_deployment == "claude-x"
    assert loaded.quiz.questions_per_session == 5     # untouched sections survive
    assert loaded.docs.paths == ["docs", "README.md"]  # lists round-trip


# --- freshness: locks, state memory, safe updates ----------------------------------


def test_is_source_file() -> None:
    from roger import freshness

    assert freshness.is_source_file("src/app/main.py")
    assert freshness.is_source_file("pkg/broker.GO".lower())
    assert not freshness.is_source_file("docs/adr/0001.md")
    assert not freshness.is_source_file("Makefile")


def test_lock_lifecycle_and_stale_break() -> None:
    import json
    import time

    from roger import freshness

    assert freshness.acquire_lock()
    assert freshness.lock_held()
    assert not freshness.acquire_lock()   # second writer skips
    freshness.release_lock()
    assert not freshness.lock_held()

    # Dead-pid lock is stale and gets broken.
    freshness.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    freshness.LOCK_PATH.write_text(json.dumps({"pid": 99999999, "started": time.time()}))
    assert not freshness.lock_held()
    assert freshness.acquire_lock()
    freshness.release_lock()


def test_failure_memory_never_respawns_and_surfaces_once(monkeypatch) -> None:
    from roger import freshness

    monkeypatch.setattr(freshness, "repo_fingerprint", lambda p: "fp1")
    monkeypatch.setattr(freshness, "stale_source_files", lambda p: ["a.py"])
    freshness.write_state(
        {"fingerprint": "fp1", "outcome": "shrink_refused", "surfaced": False}
    )

    assert freshness.maybe_refresh_in_background("graph.json") is False  # doomed → no respawn
    note = freshness.pending_failure_note("graph.json")
    assert note is not None and "roger update" in note
    assert freshness.pending_failure_note("graph.json") is None  # once, not nagging


def test_run_update_scrubs_force_env_and_detects_shrink(monkeypatch) -> None:

    from roger import freshness

    monkeypatch.setenv("GRAPHIFY_FORCE", "1")
    seen = {}

    class FakeProc:
        returncode = 0
        stdout = "refusing to shrink graph; pass --force to override"
        stderr = ""

    def fake_run(cmd, capture_output, text, check, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return FakeProc()

    counts = iter([10, 4])
    monkeypatch.setattr(freshness.subprocess, "run", fake_run)
    monkeypatch.setattr(freshness, "_node_count", lambda p: next(counts))
    monkeypatch.setattr(freshness, "repo_fingerprint", lambda p: "fp")

    result = freshness.run_update("graph.json")
    assert "GRAPHIFY_FORCE" not in seen["env"]     # env footgun scrubbed
    assert "--force" not in seen["cmd"]            # never forced without consent
    assert result.outcome == "shrink_refused"
    assert freshness.read_state()["outcome"] == "shrink_refused"


def test_stale_source_files_is_content_based(monkeypatch, tmp_path) -> None:
    import json as json_module

    from roger import freshness

    graph = tmp_path / "graph.json"
    graph.write_text(json_module.dumps({"built_at_commit": "abc", "nodes": []}))

    def fake_git(*args):
        joined = " ".join(args)
        if joined.startswith("diff"):
            return "src/a.py\ndocs/readme.md"
        if joined.startswith("status"):
            return " M src/b.py\n?? notes.txt"
        return "headhash"

    monkeypatch.setattr(freshness, "_git", fake_git)
    assert freshness.stale_source_files(str(graph)) == ["src/a.py", "src/b.py"]


# --- activity log (local observability) ---------------------------------------------


def test_activity_log_roundtrip_and_caller_detection() -> None:
    from roger import activity

    activity.log_event("context", question="how is auth checked?", tokens_served=812)
    activity.log_event("ask", question="why L0-L3?", sources=4)
    events = activity.read_recent(10)
    assert len(events) == 2
    assert events[0]["command"] == "ask"          # newest first
    assert events[1]["tokens_served"] == 812
    # Under pytest stdout is not a TTY — recorded as an agent/script caller.
    assert events[0]["caller"] == "agent/script"


def test_activity_log_rotates_and_never_raises(monkeypatch) -> None:
    from roger import activity

    activity.ACTIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    activity.ACTIVITY_PATH.write_text("x" * 2_100_000, encoding="utf-8")
    activity.log_event("context", question="q")
    assert activity.ACTIVITY_PATH.with_suffix(".log.1").exists()
    assert len(activity.read_recent(5)) == 1

    # Unwritable path must never break the actual command.
    monkeypatch.setattr(activity, "ACTIVITY_PATH", activity.ACTIVITY_PATH / "impossible" / "x.log")
    activity.log_event("context", question="q")  # no exception
