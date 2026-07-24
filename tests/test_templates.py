"""Tests for Tier 0 template questions — zero external dependencies."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from roger import templates
from roger.graph import get_node
from roger.models import Question


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


def _assert_valid_mcq(question: Question) -> None:
    assert sorted(question.options) == ["A", "B", "C", "D"]
    assert len(set(question.options.values())) == 4  # all options distinct
    assert question.correct in question.options
    assert question.difficulty == "simple"
    assert question.tier == 0
    assert question.explanation


def test_caller_question(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.process_payment")
    question = templates.caller_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] == "api.gateway"
    # Distractors must not be actual callers.
    wrong = [v for k, v in question.options.items() if k != question.correct]
    assert "api.gateway" not in wrong


def test_caller_question_no_callers_returns_none(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "api.gateway")  # entry point: nothing calls it
    assert templates.caller_question(node, graph, rng=rng) is None


def test_dependency_question(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.process_payment")
    question = templates.dependency_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] in node["callees"]
    wrong = [v for k, v in question.options.items() if k != question.correct]
    assert not set(wrong) & set(node["callees"])


def test_dependency_question_no_callees_returns_none(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "db.connect")
    assert templates.dependency_question(node, graph, rng=rng) is None


def test_module_question(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.charge")
    question = templates.module_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] == "payments"
    wrong = set(question.options.values()) - {"payments"}
    assert wrong == {"auth", "db", "api"}  # the only other communities


def test_return_type_question(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.process_payment")
    question = templates.return_type_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] == "Receipt"


def test_location_question(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.charge")
    question = templates.location_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] == "src/payments/charge.py"


def test_small_community_falls_back_to_full_graph(graph: nx.DiGraph, rng: random.Random) -> None:
    # db.connect is alone in its community — distractors must come from the
    # full graph instead of the (empty) community pool.
    node = get_node(graph, "db.connect")
    question = templates.location_question(node, graph, rng=rng)
    assert question is not None
    _assert_valid_mcq(question)
    assert question.options[question.correct] == "src/db/conn.py"


def test_build_from_graph_full_node(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "payments.process_payment")
    questions = templates.build_from_graph(node, graph, rng=rng)
    assert 1 <= len(questions) <= 5
    assert len(questions) == 5  # this node has callers, callees, and all metadata
    for question in questions:
        _assert_valid_mcq(question)
        assert question.node_id == "payments.process_payment"


def test_build_from_graph_skips_unavailable_templates(graph: nx.DiGraph, rng: random.Random) -> None:
    node = get_node(graph, "db.connect")  # no callees
    questions = templates.build_from_graph(node, graph, rng=rng)
    assert questions  # still produces module/location/return/caller questions
    texts = " ".join(q.question for q in questions)
    assert "directly call" not in texts


def test_templates_are_deterministic_with_seeded_rng(graph: nx.DiGraph) -> None:
    node = get_node(graph, "payments.process_payment")
    first = templates.build_from_graph(node, graph, rng=random.Random(7))
    second = templates.build_from_graph(node, graph, rng=random.Random(7))
    assert first == second


def test_module_question_skips_numeric_communities(graph: nx.DiGraph, rng: random.Random) -> None:
    # Real graphify emits integer Leiden ids — a "which module?" question over
    # bare numbers is meaningless and must be skipped.
    node = get_node(graph, "payments.charge")
    node["community"] = "229"
    assert templates.module_question(node, graph, rng=rng) is None


def test_display_values_disambiguate_same_label(graph: nx.DiGraph) -> None:
    graph.add_node("mod_a.save", display="save", file="src/mod_a/store.py")
    graph.add_node("mod_b.save", display="save", file="src/mod_b/cache.py")
    values = templates._display_values(graph, ["mod_a.save", "mod_b.save", "db.connect"])
    assert values == ["save (store.py)", "save (cache.py)", "db.connect"]


def test_questions_use_display_names(graph: nx.DiGraph, rng: random.Random) -> None:
    # Give the target and its caller graphify-style labels; questions must
    # show the labels, never the slug-like ids.
    graph.nodes["payments.validate_card"]["display"] = "validate_card"
    graph.nodes["payments.process_payment"]["display"] = "process_payment"
    node = get_node(graph, "payments.validate_card")
    question = templates.caller_question(node, graph, rng=rng)
    assert question is not None
    assert "validate_card()" in question.question
    assert question.options[question.correct] == "process_payment"


# --- doc questions (constructed, zero LLM) ------------------------------------------


ADR_TEXT = """\
# {n}. {title}

## Context

{context} This paragraph gives enough context body to pass the length
threshold used by the section splitter and the ADR parser alike.

## Decision

We will {decision}.

## Consequences

Everything changes accordingly, and follow-up work lands in later records.
"""


@pytest.fixture
def docs_repo(tmp_path):
    adr = tmp_path / "docs" / "adr"
    adr.mkdir(parents=True)
    specs = [
        ("Use Postgres for the registry", "The registry needs transactional writes."),
        ("Ship a remote MCP developer setup", "Developers need a one-command setup."),
        ("Adopt nightly batch builds", "Streaming would add operational burden."),
        ("Keep services self-contained", "Shared libraries created lockstep deploys."),
    ]
    for i, (title, context) in enumerate(specs):
        (adr / f"000{i}-x.md").write_text(
            ADR_TEXT.format(n=i, title=title, context=context, decision=title.lower()),
            encoding="utf-8",
        )
    contract = tmp_path / "docs" / "contracts.md"
    contract.write_text(
        "# Evidence contract\n\n## Levels\n\n"
        "The table below defines what is served at each evidence level and why.\n\n"
        "| Level | Content | Served via |\n|---|---|---|\n"
        "| L0 | One-line claim | card |\n| L1 | Card summary | card |\n"
        "| L2 | Raw chunk | open_evidence |\n| L3 | Expanded neighborhood | open_evidence |\n",
        encoding="utf-8",
    )
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "agent.md").write_text("# never quiz this\n" + "x" * 200)
    return tmp_path


def test_discover_doc_files_excludes_dot_dirs(docs_repo) -> None:
    from roger import docs

    files = docs.discover_doc_files(["docs", ".claude"], repo_root=docs_repo)
    names = {f.name for f in files}
    assert "contracts.md" in names and "0000-x.md" in names
    assert "agent.md" not in names  # dotted dirs never quizzed


def test_adr_questions_use_other_decisions_as_distractors(docs_repo) -> None:
    from roger import docs

    files = docs.discover_doc_files(["docs"], repo_root=docs_repo)
    questions = docs.adr_questions(files, "medium", random.Random(1), repo_root=docs_repo)
    assert questions
    q = questions[0]
    all_titles = {
        "Use Postgres for the registry", "Ship a remote MCP developer setup",
        "Adopt nightly batch builds", "Keep services self-contained",
    }
    assert set(q.options.values()) <= all_titles     # every option is a real decision
    assert q.options[q.correct] in all_titles
    assert q.tier == 0 and q.language == "markdown"
    assert q.snippet  # the context is shown


def test_table_questions_quiz_real_cells(docs_repo) -> None:
    from roger import docs

    files = docs.discover_doc_files(["docs"], repo_root=docs_repo)
    sections = []
    for f in files:
        rel = str(f.relative_to(docs_repo))
        sections.extend(docs.split_sections(rel, f.read_text(encoding="utf-8")))
    questions = docs.table_questions(sections, "medium", random.Random(2))
    assert questions
    q = questions[0]
    cells = {"One-line claim", "Card summary", "Raw chunk", "Expanded neighborhood",
             "card", "open_evidence"}
    assert q.options[q.correct] in cells


def test_doc_questions_end_to_end_mixes_formats(docs_repo) -> None:
    from roger import docs

    questions = docs.doc_questions(
        count=3, difficulty="medium", paths=["docs"], repo_root=docs_repo,
        rng=random.Random(3),
    )
    assert 1 <= len(questions) <= 3
    assert all(q.tier == 0 for q in questions)


def test_doc_questions_files_restriction(docs_repo) -> None:
    from roger import docs

    questions = docs.doc_questions(
        count=3, difficulty="medium", repo_root=docs_repo,
        files=["docs/contracts.md"], rng=random.Random(4),
    )
    assert questions
    assert all(q.node_id == "docs/contracts.md" for q in questions)


def test_doc_questions_empty_repo_is_silent(tmp_path) -> None:
    from roger import docs

    assert docs.doc_questions(count=3, repo_root=tmp_path, rng=random.Random(5)) == []
