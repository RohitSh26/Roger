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
