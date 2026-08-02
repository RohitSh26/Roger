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
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-a")
    monkeypatch.setattr(embeddings, "_embed", fake_embed_factory())
    stats = embeddings.refresh_index(graph, Config())
    return graph, stats


# --- index lifecycle ---------------------------------------------------------


def test_refresh_index_embeds_every_candidate(indexed) -> None:
    _, stats = indexed
    assert stats is not None
    assert stats.cards == 11  # all synthetic nodes are identifier-shaped .py code
    assert stats.with_vec == stats.cards
    assert stats.embedded == stats.cards


def test_refresh_index_is_incremental(indexed, monkeypatch) -> None:
    graph, _ = indexed
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats.embedded == 0  # nothing changed → no work


def test_refresh_index_reembeds_on_model_change(indexed, monkeypatch) -> None:
    graph, _ = indexed
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-b")
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats.embedded == stats.cards


def test_refresh_index_drops_deleted_nodes(indexed) -> None:
    graph, _ = indexed
    graph.remove_node("payments.refund")
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats.cards == 10
    assert "payments.refund" not in embeddings.load_card_texts()


def test_refresh_index_without_model_is_none(graph, in_tmp_repo, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: None)
    assert embeddings.refresh_index(graph, Config()) is None
    assert not embeddings.VECTORS_PATH.exists()


def test_refresh_writes_cards_upfront_for_real_progress(graph, in_tmp_repo, monkeypatch) -> None:
    # Embedding fails entirely → zero vectors, but every card row (and its
    # text) exists, so done/total progress and keyword enrichment are real.
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-a")
    monkeypatch.setattr(embeddings, "_embed", lambda texts, config, timeout: None)
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None
    assert (stats.cards, stats.with_vec, stats.embedded) == (11, 0, 0)
    assert len(embeddings.load_card_texts()) == 11
    assert embeddings.index_progress() == (0, 11)


def test_refresh_resumes_vectorless_rows(graph, in_tmp_repo, monkeypatch) -> None:
    # Rows written upfront by an interrupted build must be re-embedded even
    # though their content hash already matches.
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-a")
    monkeypatch.setattr(embeddings, "_embed", lambda texts, config, timeout: None)
    embeddings.refresh_index(graph, Config())
    monkeypatch.setattr(embeddings, "_embed", fake_embed_factory())
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None and stats.embedded == 11
    assert embeddings.index_progress() == (11, 11)


def test_progress_callback_reports_done_of_total(graph, in_tmp_repo, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-a")
    monkeypatch.setattr(embeddings, "_embed", fake_embed_factory())
    ticks: list[tuple[int, int]] = []
    embeddings.refresh_index(
        graph, Config(), batch_size=4, progress=lambda d, t: ticks.append((d, t))
    )
    assert ticks == [(4, 11), (8, 11), (11, 11)]


# --- query-time ranking and fallbacks --------------------------------------------


def test_semantic_rank_finds_by_meaning(indexed) -> None:
    ranked = embeddings.semantic_rank("charge the customer's card", Config())
    assert ranked is not None
    assert ranked[0] == "payments.charge"


def test_semantic_rank_without_index_is_none(in_tmp_repo) -> None:
    assert embeddings.semantic_rank("anything", Config()) is None


def test_semantic_rank_digest_mismatch_hard_disables(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-CHANGED")
    assert embeddings.semantic_rank("charge", Config()) is None


def test_semantic_rank_embed_failure_is_silent(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_embed", lambda texts, config, timeout: None)
    assert embeddings.semantic_rank("charge", Config()) is None


def test_semantic_rank_ollama_gone_is_silent(indexed, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: None)
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
    # Refusal suppresses the PROMPT forever — but not capability sensing:
    # a user who later pulls the model themselves gets semantic search
    # without being re-asked (presence is enablement).
    from roger import cli

    embeddings.record_declined()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: None)
    monkeypatch.setattr(
        "typer.confirm", lambda *a, **k: pytest.fail("prompted after refusal")
    )
    cli._maybe_offer_semantic(Config())


# --- index self-heal (enabled machine, repo without an index yet) -----------------


def _heal_setup(monkeypatch, tmp_path, *, model=True):
    from roger import cli, freshness

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        embeddings, "model_digest", lambda config, timeout=2: "digest-a" if model else None
    )
    spawned: list[dict] = []
    monkeypatch.setattr(
        freshness, "maybe_refresh_in_background",
        lambda path, force=False: spawned.append({"force": force}) or True,
    )
    return cli, spawned


def _write_index(meta: dict, cards: list[tuple[str, bool]]) -> None:
    import time
    from contextlib import closing

    with closing(embeddings._connect()) as conn:
        defaults = {"embed_version": "1", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        for key, value in {**defaults, **meta}.items():
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        for node_id, has_vec in cards:
            conn.execute(
                "INSERT OR REPLACE INTO cards (node_id, content_hash, card_text, vec) "
                "VALUES (?, ?, ?, ?)",
                (node_id, "h", "text", b"\x00\x00\x80?" if has_vec else None),
            )
        conn.commit()


def test_self_heal_builds_missing_index(monkeypatch, tmp_path) -> None:
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is True
    assert spawned == [{"force": True}]


def test_self_heal_noop_when_index_healthy(monkeypatch, tmp_path) -> None:
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    _write_index({"digest": "digest-a"}, [("a", True), ("b", True)])
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is False
    assert spawned == []


def test_self_heal_resumes_interrupted_build(monkeypatch, tmp_path) -> None:
    # A killed build leaves cards but no meta (meta is written on finish) —
    # this was the stuck "index rebuilding (model changed)" state.
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    _write_index({}, [("a", True), ("b", False)])
    with __import__("contextlib").closing(embeddings._connect()) as conn:
        conn.execute("DELETE FROM meta")
        conn.commit()
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is True
    assert spawned == [{"force": True}]


def test_self_heal_rebuilds_on_model_change_after_cooldown(monkeypatch, tmp_path) -> None:
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    _write_index(
        {"digest": "digest-OLD", "updated_at": "2000-01-01T00:00:00"}, [("a", True)]
    )
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is True


def test_self_heal_honors_cooldown_after_recent_attempt(monkeypatch, tmp_path) -> None:
    # Persistently failing embeds must not turn every call into a rebuild.
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    _write_index({"digest": "digest-a"}, [("a", True), ("b", False)])  # partial, fresh
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is False
    assert spawned == []


def test_self_heal_noop_without_model(monkeypatch, tmp_path) -> None:
    cli, spawned = _heal_setup(monkeypatch, tmp_path, model=False)
    assert embeddings.self_heal_index(Config(), "graphify-out/graph.json") is False
    assert spawned == []


def test_status_names_interrupted_build_honestly(monkeypatch, tmp_path) -> None:
    from roger import freshness

    _heal_setup(monkeypatch, tmp_path)
    _write_index({}, [("a", True), ("b", False)])
    with __import__("contextlib").closing(embeddings._connect()) as conn:
        conn.execute("DELETE FROM meta")
        conn.commit()
    monkeypatch.setattr(freshness, "lock_held", lambda: False)
    status = embeddings.index_status(Config())
    assert status.reason == "an index build was interrupted"
    monkeypatch.setattr(freshness, "lock_held", lambda: True)
    status = embeddings.index_status(Config())
    assert "index building now" in status.reason


def test_offer_self_heals_on_enabled_machine(monkeypatch, tmp_path) -> None:
    # Model present + no index: the consent flow must not prompt (presence
    # IS consent) but must quietly start the repo's first index build.
    cli, spawned = _heal_setup(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        "typer.confirm", lambda *a, **k: pytest.fail("prompted an enabled machine")
    )
    cli._maybe_offer_semantic(Config())
    assert spawned == [{"force": True}]


# --- the never-commit guarantee ---------------------------------------------------


def test_roger_gitignore_ships_with_init(in_tmp_repo) -> None:
    init_dbs()
    ignore = Path(".roger/.gitignore").read_text(encoding="utf-8")
    for name in ("vectors.db", "activity.log", "history.db", "update.lock"):
        assert name in ignore
    # Shareable files must stay shareable.
    assert "cache.db" not in ignore
    assert "config.toml" not in ignore


# --- doctor advice for a degraded index (the "what do I do now?" answer) ---------


def _advice(monkeypatch, *, lock=False, heals=False, outcome=None):
    from roger import cli, freshness

    monkeypatch.setattr(freshness, "lock_held", lambda: lock)
    monkeypatch.setattr(embeddings, "self_heal_index", lambda c, p: heals)
    monkeypatch.setattr(
        freshness, "read_state", lambda: {"outcome": outcome} if outcome else {}
    )
    status = embeddings.IndexStatus("keyword-only", True, "an index build was interrupted")
    return cli._semantic_doctor_advice(Config(), status)


def test_doctor_advice_build_running(monkeypatch) -> None:
    line, remedy = _advice(monkeypatch, lock=True)
    assert "attaches, never restarts" in remedy


def test_doctor_advice_rebuild_just_started(monkeypatch) -> None:
    line, remedy = _advice(monkeypatch, heals=True)
    assert "started in the background just now" in line
    assert "roger update" in remedy


def test_doctor_advice_names_a_failed_attempt(monkeypatch) -> None:
    line, remedy = _advice(monkeypatch, outcome="failed")
    assert "failed" in line
    assert "update.log" in remedy


def test_doctor_advice_cooldown_offers_foreground(monkeypatch) -> None:
    line, remedy = _advice(monkeypatch)
    assert "paused" in line
    assert "roger update" in remedy


# --- honest refresh narration (field bug: silent failure looked like 'off') ------


def _narrate(monkeypatch, capsys, *, ollama=True, digest="d", crash=None, stats="ok"):
    import roger.llm.local as local_mod

    from roger import cli
    from roger.embeddings import IndexRefresh

    monkeypatch.setattr(local_mod, "is_ollama_running", lambda url=None: ollama)
    monkeypatch.setattr(
        embeddings, "model_digest", lambda config, timeout=2: digest
    )
    if crash is not None:
        monkeypatch.setattr(
            cli, "load_graph", lambda p: (_ for _ in ()).throw(crash)
        )
    else:
        monkeypatch.setattr(cli, "load_graph", lambda p: object())
        results = {
            "ok": IndexRefresh(cards=10, embedded=2, with_vec=10),
            "partial": IndexRefresh(cards=10, embedded=3, with_vec=7),
        }
        monkeypatch.setattr(
            embeddings, "refresh_index", lambda *a, **k: results[stats]
        )
    out = cli._refresh_semantic(Config(), live=True)
    return out, capsys.readouterr().out


def test_refresh_narrates_ollama_down(monkeypatch, capsys) -> None:
    result, out = _narrate(monkeypatch, capsys, ollama=False)
    assert result is None and "Ollama isn't reachable" in out


def test_refresh_narrates_model_absent(monkeypatch, capsys) -> None:
    result, out = _narrate(monkeypatch, capsys, digest=None)
    assert result is None and "off (keyword-only)" in out


def test_refresh_narrates_crash_with_real_error(monkeypatch, capsys) -> None:
    # THE field bug: a crash here used to print a misleading 'off' line and
    # leave update.log empty. The actual exception must reach the screen.
    result, out = _narrate(monkeypatch, capsys, crash=RuntimeError("vectors.db corrupt"))
    assert result is None
    assert "RuntimeError" in out and "vectors.db corrupt" in out


def test_refresh_narrates_partial_coverage(monkeypatch, capsys) -> None:
    result, out = _narrate(monkeypatch, capsys, stats="partial")
    assert "7" in out and "10" in out and "stopped early" in out


def test_refresh_narrates_success(monkeypatch, capsys) -> None:
    result, out = _narrate(monkeypatch, capsys, stats="ok")
    assert result is not None and "re-indexed 2 changed function(s)" in out


def test_rebuild_index_now_holds_the_lock(monkeypatch, tmp_path) -> None:
    from roger import cli, freshness
    from roger.embeddings import IndexRefresh

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli, "_refresh_semantic",
        lambda config, live=True: calls.append(freshness.lock_held())
        or IndexRefresh(cards=5, embedded=5, with_vec=5),
    )
    assert cli._rebuild_index_now(Config()) is True
    assert calls == [True]          # refresh ran WITH the lock held
    assert not freshness.lock_held()  # and released it after


def test_one_poisoned_node_never_kills_the_build(indexed, monkeypatch) -> None:
    # Field regression: an IndexError building ONE card aborted the whole
    # refresh, looping the machine between 'interrupted' and 'failed'.
    graph, _ = indexed
    real_card_text = embeddings.card_text

    def poisoned(node):
        if node["id"] == "payments.charge":
            raise IndexError("list index out of range")
        return real_card_text(node)

    monkeypatch.setattr(embeddings, "card_text", poisoned)
    monkeypatch.setattr(embeddings, "model_digest", lambda config, timeout=2: "digest-b")
    stats = embeddings.refresh_index(graph, Config())
    assert stats is not None
    assert stats.skipped == 1
    assert stats.embedded == 10  # everyone else still made it


# --- interface packs (vertical-stack contract briefings) --------------------------


def _relgraph():
    import networkx as nx

    graph = nx.DiGraph()
    for node_id in ("api.handle", "core.rank", "core.store", "util.hub", "docs.stub"):
        graph.add_node(node_id, display=node_id.split(".")[-1], file=f"src/{node_id.replace('.', '/')}.py")
    graph.add_edge("api.handle", "core.rank", relation="calls")
    graph.add_edge("api.handle", "util.hub", relation="references")
    graph.add_edge("core.store", "util.hub", relation="references")
    graph.add_edge("api.handle", "docs.stub", relation="rationale_for")  # junk relation
    return graph


def test_seam_ranks_by_anchor_multiplicity_and_whitelists(monkeypatch) -> None:
    graph = _relgraph()
    seam = ask._seam_nodes(graph, ["api.handle", "core.store"])
    ids = [n for n, _ in seam]
    assert ids[0] == "util.hub"          # touched by BOTH anchors → first
    assert "core.rank" in ids            # calls relation admitted
    assert "docs.stub" not in ids        # rationale_for excluded


def test_seam_caps_per_anchor(monkeypatch) -> None:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("a", display="a", file="src/a.py")
    for i in range(20):
        graph.add_node(f"n{i}", display=f"n{i}", file=f"src/n{i}.py")
        graph.add_edge("a", f"n{i}", relation="calls")
    seam = ask._seam_nodes(graph, ["a"], per_anchor_cap=6)
    assert len(seam) == 6


def test_interface_pack_budget_fills_whole_cards(graph, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "semantic_rank", lambda *a, **k: None)
    monkeypatch.setattr(embeddings, "load_card_texts", lambda: {})
    pack = ask.interface_pack("how is a payment charged and refunded", graph, Config())
    assert "# Roger interfaces:" in pack
    assert "VERBATIM" in pack
    assert "bodies are omitted" in pack.lower()


def test_interface_pack_no_match_says_so(graph, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "semantic_rank", lambda *a, **k: None)
    monkeypatch.setattr(embeddings, "load_card_texts", lambda: {})
    pack = ask.interface_pack("zzz qqq xyzzy", graph, Config())
    assert "No matching code found" in pack


def test_rare_terms_outrank_common_ones() -> None:
    # IDF weighting: 'frobnicate' (matches 1 node) must dominate 'widget'
    # (matches many). Flat scoring tied them and sorted alphabetically,
    # which is how keyword-rich harness classes outranked production code.
    import networkx as nx

    graph = nx.DiGraph()
    for i in range(5):
        graph.add_node(f"common{i}", display=f"widget_{i}", file=f"src/w{i}.py")
    graph.add_node("special", display="frobnicate", file="src/f.py")
    hits = ask.find_relevant_nodes(graph, "widget frobnicate", top=3)
    assert hits[0] == "special"


def test_interface_tail_anchors_need_semantic_endorsement(graph, monkeypatch) -> None:
    anchors = ["payments.process_payment", "payments.validate_card",
               "payments.charge", "payments.refund", "payments.notify"]
    monkeypatch.setattr(ask, "retrieve_nodes", lambda *a, **k: list(anchors))
    monkeypatch.setattr(
        ask, "score_relevant_nodes",
        lambda *a, **k: {n: 8.0 for n in anchors} | {"payments.process_payment": 10.0},
    )
    # Only refund is meaning-endorsed; notify is word-coincidence.
    monkeypatch.setattr(embeddings, "semantic_rank", lambda *a, **k: ["payments.refund"])
    monkeypatch.setattr(embeddings, "load_card_texts", lambda: {})
    pack = ask.interface_pack("some question", graph, Config())
    assert "## refund" in pack or "refund (" in pack       # tail + endorsed → card
    card_section = pack.split("More contracts")[0]
    assert "notify" not in card_section                    # tail, unendorsed → gated
    assert "notify" in pack                                # …but visible in the index


# --- relational questions: complete, deterministic, zero-LLM ---------------------


def test_relational_what_calls_is_complete_and_deterministic(graph) -> None:
    first = ask.relational_facts("what calls check_token?", graph)
    second = ask.relational_facts("what calls check_token?", graph)
    assert first is not None and first == second  # byte-identical every run
    answer, sources = first
    # ALL three callers, not a sample (fixture: process_payment, login, logout)
    assert "`payments.process_payment`" in answer
    assert "`auth.login`" in answer
    assert "`auth.logout`" in answer
    assert "Complete list from the code graph" in answer
    assert "src/auth/token.py" in sources


def test_relational_reverse_direction(graph) -> None:
    result = ask.relational_facts("what does login call?", graph)
    assert result is not None
    answer, _ = result
    assert "`auth.hash_password`" in answer and "`auth.check_token`" in answer


def test_relational_ignores_analytical_questions(graph) -> None:
    assert ask.relational_facts("how does charge work?", graph) is None
    assert ask.relational_facts("why do we retry payments twice?", graph) is None


def test_relational_unknown_name_falls_through(graph) -> None:
    assert ask.relational_facts("what calls FluxCapacitor?", graph) is None


def test_relational_answer_needs_no_backend(graph, monkeypatch) -> None:
    # 'what calls X' must answer even with Ollama completely dead.
    from roger.llm import router

    monkeypatch.setattr(
        router, "ensure_backend",
        lambda config: pytest.fail("relational answers must not need a backend"),
    )
    monkeypatch.setattr(ask, "ensure_backend", router.ensure_backend)
    answer, sources = ask.answer_question("who uses charge?", graph, Config())
    assert "`payments.process_payment`" in answer  # calls charge in the fixture


def test_context_pack_leads_with_graph_facts(graph, monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "semantic_rank", lambda *a, **k: None)
    monkeypatch.setattr(embeddings, "load_card_texts", lambda: {})
    pack = ask.context_pack("what calls check_token?", graph, Config())
    assert "## Graph facts (complete)" in pack
    assert pack.index("Graph facts") < pack.index("### Code:")  # facts lead


# --- explain / path: zero-LLM graph verbs for agents ------------------------------


def test_explain_lists_every_connection(graph) -> None:
    result = ask.explain_symbol(graph, "check_token")
    assert result is not None
    assert "File:   src/auth/token.py" in result
    assert "Degree: 3" in result
    # all three callers appear as incoming edges
    for name in ("payments.process_payment", "auth.login", "auth.logout"):
        assert f"<-- {name}" in result


def test_explain_unknown_symbol_is_none(graph) -> None:
    assert ask.explain_symbol(graph, "FluxCapacitor") is None


def test_snippet_never_attributes_another_symbols_code(tmp_path) -> None:
    # Graphify records a start LINE. Edit the file without `roger update`
    # and that line drifts onto a DIFFERENT function — printing it under
    # the original header, inside a pack that promises VERBATIM source,
    # is a confidently-cited wrong answer. Re-find it or say nothing.
    from roger.graph import get_source_snippet

    source = tmp_path / "mod.py"
    source.write_text(
        "def alpha():\n    return 'a'\n\n\ndef beta():\n    return 'b'\n",
        encoding="utf-8",
    )
    attrs = {"file": "mod.py", "display": "beta()", "source_location": "L5"}
    assert "def beta" in get_source_snippet(attrs, repo_root=tmp_path)

    # Two lines inserted above: the recorded line now points into alpha().
    source.write_text(
        "import os\nimport sys\n"
        "def alpha():\n    return 'a'\n\n\ndef beta():\n    return 'b'\n",
        encoding="utf-8",
    )
    drifted = get_source_snippet(attrs, repo_root=tmp_path)
    assert "def alpha" not in drifted      # never another symbol's body
    assert "def beta" in drifted           # re-found by name

    gone = {"file": "mod.py", "display": "deleted_fn()", "source_location": "L5"}
    assert get_source_snippet(gone, repo_root=tmp_path) == ""  # honest absence


def test_retrieval_survives_an_index_that_outlived_its_nodes(graph, monkeypatch) -> None:
    # vectors.db is a separate artifact on a separate clock: it can name
    # nodes this graph no longer has (deleted code, or a graph rebuilt
    # while the index lagged). That raised a raw KeyError mid-answer.
    from roger import embeddings
    from roger.config import Config

    monkeypatch.setattr(
        embeddings, "semantic_rank",
        lambda *a, **k: ["ghost_node_from_an_older_graph", "payments.charge"],
    )
    nodes = ask.retrieve_nodes(graph, "how are payments charged?", Config())
    assert "ghost_node_from_an_older_graph" not in nodes
    assert all(n in graph for n in nodes)


def test_weak_match_warning_only_fires_for_alien_questions(graph) -> None:
    # Honest scope: this catches questions whose vocabulary the repo does
    # not contain at all. It deliberately does NOT claim to catch
    # plausible-but-wrong matches — measured, those are lexically
    # indistinguishable from good ones.
    from roger.config import Config

    alien = ask.context_pack("kubernetes ingress TLS termination", graph, Config())
    assert "Weak match" in alien
    real = ask.context_pack("what charges the card?", graph, Config())
    assert "Weak match" not in real


def test_context_caps_open_up_for_cloud_backends() -> None:
    # Local: trimmed to the small num_ctx window. Cloud: 100k+ windows —
    # the 40-line class clip made honest models report truncation instead
    # of answering (field-hit), so azure providers get whole classes.
    from roger.config import Config, ModelConfig

    local_budget, local_matches, local_lines, _ = ask._context_caps(Config())
    assert local_lines == 40 and local_matches == 3

    for provider in ("azure-anthropic", "azure-foundry"):
        budget, matches, lines, chars = ask._context_caps(
            Config(model=ModelConfig(provider=provider))
        )
        assert budget > local_budget
        assert matches > local_matches
        assert lines >= 200 and chars >= 10_000


def test_explain_data_mirrors_explain_symbol(graph) -> None:
    # The app renders from the structured form — it must agree with the
    # text verb on resolution and edge coverage.
    nodes = ask.explain_data(graph, "check_token")
    assert len(nodes) == 1
    node = nodes[0]
    assert node["file"] == "src/auth/token.py"
    assert node["degree"] == 3
    incoming = {e["display"] for e in node["incoming"]}
    assert incoming == {"payments.process_payment", "auth.login", "auth.logout"}
    assert all(e["relation"] for e in node["incoming"])
    assert ask.explain_data(graph, "FluxCapacitor") == []


def test_path_data_mirrors_connection_path(graph) -> None:
    result = ask.path_data(graph, "notify", "hash_password")
    assert "error" not in result
    assert len(result["links"]) == len(result["hops"]) - 1
    displays = [h["display"] for h in result["hops"]]
    assert "payments.process_payment" in displays and "auth.login" in displays
    assert all(isinstance(link["forward"], bool) for link in result["links"])
    assert "not in the graph" in ask.path_data(graph, "notify", "Nonexistent")["error"]


def test_path_map_svg_is_self_contained_and_deterministic(graph) -> None:
    # The app's constellation map: pure inline SVG (the offline promise —
    # graphify's own HTML exports pull D3 from a CDN, ours must not), and
    # byte-identical across runs so Streamlit reruns don't jitter.
    from roger import codemap

    result = ask.path_data(graph, "notify", "hash_password")
    svg = codemap.path_map_svg(graph, result["hops"], result["links"])
    assert svg == codemap.path_map_svg(graph, result["hops"], result["links"])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "http" not in svg and "<script" not in svg
    for hop in result["hops"]:
        assert hop["display"] in svg  # route nodes are labeled
    assert svg.count("stroke-width=\"2.2\"") == len(result["links"])


def test_path_uses_undirected_connectivity(graph) -> None:
    # notify ← process_payment → check_token ← login → hash_password:
    # NO directed path exists — a directed search would say "no path".
    result = ask.connection_path(graph, "notify", "hash_password")
    assert result is not None
    assert "hops" in result
    assert "payments.process_payment" in result and "auth.login" in result


def test_path_unknown_endpoint_says_which(graph) -> None:
    result = ask.connection_path(graph, "notify", "Nonexistent")
    assert "Nonexistent" in result and "not in the graph" in result


def test_graph_verbs_need_no_backend(graph, monkeypatch) -> None:
    from roger.llm import router

    monkeypatch.setattr(
        router, "ensure_backend",
        lambda config: pytest.fail("graph verbs must never need a backend"),
    )
    assert ask.explain_symbol(graph, "charge") is not None
    assert "hops" in ask.connection_path(graph, "gateway", "connect")
