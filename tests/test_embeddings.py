"""Semantic retrieval: index lifecycle, fusion, consent, and every silent fallback.

No Ollama required — the embed call and model sensing are faked. The fake
embedder maps text to axis-aligned vectors by keyword so cosine ranking is
fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roger import ask, embeddings
from roger.config import Config
from roger.storage import init_dbs


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fresh matrix cache and a per-test machine-state file for every test."""
    monkeypatch.setattr(embeddings, "_matrix_cache", (None, None, None))
    monkeypatch.setattr(embeddings, "MACHINE_STATE_PATH", tmp_path / "machine-state.json")


def fake_embed_factory():
    """Deterministic 3-dim embeddings: charge→x, refund→y, else z."""

    def fake_embed(texts, config, timeout):
        out = []
        for text in texts:
            low = text.lower()
            if "charge" in low:
                out.append([1.0, 0.0, 0.0])
            elif "refund" in low:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out

    return fake_embed


@pytest.fixture
def indexed(graph, in_tmp_repo, monkeypatch: pytest.MonkeyPatch):
    """A built vector index over the synthetic graph, digest 'digest-a'."""
    monkeypatch.setattr(embeddings, "model_digest", lambda config: "digest-a")
    monkeypatch.setattr(embeddings, "_embed", fake_embed_factory())
    stats = embeddings.refresh_index(graph, Config())
    return graph, stats


# --- index lifecycle ---------------------------------------------------------


def test_refresh_index_embeds_every_candidate(indexed) -> None:
    _, stats = indexed
    assert stats is not None
    assert stats["cards"] == 11  # all synthetic nodes are identifier-shaped .py code
    assert stats["with_vec"] == stats["cards"]
    assert stats["embedded"] == stats["cards"]


def test_refresh_index_is_incremental(indexed, monkeypatch) -> None:
    graph, _ = indexed
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats["embedded"] == 0  # nothing changed → no work


def test_refresh_index_reembeds_on_model_change(indexed, monkeypatch) -> None:
    graph, _ = indexed
    monkeypatch.setattr(embeddings, "model_digest", lambda config: "digest-b")
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats["embedded"] == stats["cards"]


def test_refresh_index_drops_deleted_nodes(indexed) -> None:
    graph, _ = indexed
    graph.remove_node("payments.refund")
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats["cards"] == 10
    assert "payments.refund" not in embeddings.load_card_texts()


def test_refresh_index_without_model_is_none(graph, in_tmp_repo, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config: None)
    assert embeddings.refresh_index(graph, Config()) is None
    assert not embeddings.VECTORS_PATH.exists()


# --- query-time ranking and fallbacks --------------------------------------------


def test_semantic_rank_finds_by_meaning(indexed) -> None:
    ranked = embeddings.semantic_rank("charge the customer's card", Config())
    assert ranked is not None
    assert ranked[0] == "payments.charge"


def test_semantic_rank_without_index_is_none(in_tmp_repo) -> None:
    assert embeddings.semantic_rank("anything", Config()) is None


def test_semantic_rank_digest_mismatch_hard_disables(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config: "digest-CHANGED")
    assert embeddings.semantic_rank("charge", Config()) is None


def test_semantic_rank_embed_failure_is_silent(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_embed", lambda texts, config, timeout: None)
    assert embeddings.semantic_rank("charge", Config()) is None


def test_semantic_rank_ollama_gone_is_silent(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config: None)
    assert embeddings.semantic_rank("charge", Config()) is None


# --- fusion (retrieve_nodes) -----------------------------------------------------


def test_retrieve_falls_back_to_keyword_exactly(graph, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "semantic_rank", lambda *a, **k: None)
    keyword = ask.find_relevant_nodes(graph, "how is a payment refunded")
    assert ask.retrieve_nodes(graph, "how is a payment refunded") == keyword


def test_retrieve_fuses_both_channels_rrf(graph, monkeypatch) -> None:
    monkeypatch.setattr(
        ask, "find_relevant_nodes", lambda *a, **k: ["auth.login", "db.connect"]
    )
    monkeypatch.setattr(
        embeddings, "semantic_rank", lambda *a, **k: ["api.gateway", "auth.login"]
    )
    fused = ask.retrieve_nodes(graph, "nothing exact here")
    # auth.login appears in both channels → outranks either single-channel hit
    assert fused[0] == "auth.login"
    assert set(fused) == {"auth.login", "db.connect", "api.gateway"}


def test_exact_name_pin_beats_semantic_similarity(graph, monkeypatch) -> None:
    # Semantic channel is convinced the answer is refund; the developer
    # literally typed "charge" — the literal name must win.
    monkeypatch.setattr(
        embeddings,
        "semantic_rank",
        lambda *a, **k: ["payments.refund", "payments.notify", "payments.charge"],
    )
    fused = ask.retrieve_nodes(graph, "what does charge do")
    assert fused[0] == "payments.charge"


def test_card_texts_enrich_keyword_channel(graph, monkeypatch) -> None:
    # Words that appear only in the card (docstring/body head) now score.
    monkeypatch.setattr(
        embeddings,
        "load_card_texts",
        lambda: {"db.connect": "maintains a pooled keepalive connection"},
    )
    hits = ask.find_relevant_nodes(graph, "where is the pooled keepalive handled")
    assert "db.connect" in hits


# --- consent state (per machine, refusal-only) --------------------------------


def test_declined_state_roundtrip() -> None:
    assert not embeddings.embed_prompt_declined()
    embeddings.record_declined()
    assert embeddings.embed_prompt_declined()
    embeddings.clear_declined()
    assert not embeddings.embed_prompt_declined()
    # Only ever a refusal marker — acceptance stores nothing.
    assert "true" not in embeddings.MACHINE_STATE_PATH.read_text().lower()


def test_offer_never_prompts_outside_tty(monkeypatch) -> None:
    from roger import cli

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        "typer.confirm", lambda *a, **k: pytest.fail("prompted a non-TTY caller")
    )
    cli._maybe_offer_semantic(Config())


def test_offer_respects_prior_refusal(monkeypatch) -> None:
    from roger import cli

    embeddings.record_declined()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        embeddings, "model_digest",
        lambda config: pytest.fail("sensed capability after refusal"),
    )
    cli._maybe_offer_semantic(Config())


# --- the never-commit guarantee ---------------------------------------------------


def test_roger_gitignore_ships_with_init(in_tmp_repo) -> None:
    init_dbs()
    ignore = Path(".roger/.gitignore").read_text(encoding="utf-8")
    for name in ("vectors.db", "activity.log", "history.db", "update.lock"):
        assert name in ignore
    # Shareable files must stay shareable.
    assert "cache.db" not in ignore
    assert "config.toml" not in ignore
