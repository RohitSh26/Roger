"""Tests for roger/graph.py — all against the synthetic fixture graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from roger import graph as g
from roger.exceptions import GraphNotFoundError


def test_load_graph_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GraphNotFoundError) as excinfo:
        g.load_graph(str(tmp_path / "nope" / "graph.json"))
    assert "roger init" in str(excinfo.value)


def test_load_graph_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "graph.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphNotFoundError):
        g.load_graph(str(bad))


def test_load_graph_returns_directed_graph(graph: nx.DiGraph) -> None:
    assert graph.is_directed()
    assert graph.number_of_nodes() == 11
    assert graph.number_of_edges() == 11


def test_get_node_includes_callers_and_callees(graph: nx.DiGraph) -> None:
    node = g.get_node(graph, "payments.process_payment")
    assert node["id"] == "payments.process_payment"
    assert node["file"] == "src/payments/processor.py"
    assert node["community"] == "payments"
    assert node["returns"] == "Receipt"
    assert node["callers"] == ["api.gateway"]
    assert node["callees"] == [
        "auth.check_token",
        "payments.charge",
        "payments.notify",
        "payments.validate_card",
    ]


def test_get_node_unknown_id_raises(graph: nx.DiGraph) -> None:
    with pytest.raises(KeyError):
        g.get_node(graph, "does.not.exist")


def test_get_subgraph_covers_callers_and_callees(graph: nx.DiGraph) -> None:
    sub = g.get_subgraph(graph, "payments.charge", hops=1)
    # Undirected 1-hop: the node, its callers, and its callees.
    assert set(sub.nodes) == {
        "payments.charge",
        "payments.process_payment",
        "payments.refund",
        "db.connect",
    }


def test_get_god_nodes_orders_by_degree(graph: nx.DiGraph) -> None:
    top = g.get_god_nodes(graph, top_n=3)
    assert top[0] == "payments.process_payment"  # degree 5
    assert set(top).issubset(set(graph.nodes))
    assert len(top) == 3


def test_get_community_nodes(graph: nx.DiGraph) -> None:
    payments = g.get_community_nodes(graph, "payments")
    assert len(payments) == 5
    assert all(n.startswith("payments.") for n in payments)
    assert g.get_community_nodes(graph, "ghosts") == []


def test_get_changed_nodes_maps_files(graph: nx.DiGraph) -> None:
    changed = g.get_changed_nodes(
        graph, ["src/payments/charge.py", "./src/db/conn.py", "README.md"]
    )
    assert changed == ["db.connect", "payments.charge"]


def test_get_nodes_by_path_prefix_and_exact(graph: nx.DiGraph) -> None:
    assert g.get_nodes_by_path(graph, "src/payments") == [
        "payments.charge",
        "payments.notify",
        "payments.process_payment",
        "payments.refund",
        "payments.validate_card",
    ]
    assert g.get_nodes_by_path(graph, "src/db/conn.py") == ["db.connect"]
    assert g.get_nodes_by_path(graph, "src/nonexistent") == []
    # Prefix must respect path boundaries: src/pay should not match src/payments.
    assert g.get_nodes_by_path(graph, "src/pay") == []


def test_serialize_subgraph_is_deterministic(graph: nx.DiGraph) -> None:
    sub = g.get_subgraph(graph, "payments.process_payment", hops=1)
    first = g.serialize_subgraph(sub)
    second = g.serialize_subgraph(g.get_subgraph(graph, "payments.process_payment", hops=1))
    assert first == second
    assert "payments.process_payment" in first
    assert "->" in first


def test_god_nodes_from_report(report_file: Path) -> None:
    assert g.get_god_node_ids_from_report(str(report_file)) == [
        "payments.process_payment",
        "auth.check_token",
    ]


def test_god_nodes_from_missing_report(tmp_path: Path) -> None:
    assert g.get_god_node_ids_from_report(str(tmp_path / "GRAPH_REPORT.md")) == []


def test_surprise_edges_from_report(report_file: Path) -> None:
    assert g.get_surprise_edges(str(report_file)) == [
        ("payments.process_payment", "auth.check_token")
    ]


def test_surprise_edges_missing_report(tmp_path: Path) -> None:
    assert g.get_surprise_edges(str(tmp_path / "GRAPH_REPORT.md")) == []


def test_query_graph_for_ask_matches_descriptions(graph: nx.DiGraph) -> None:
    context = g.query_graph_for_ask(graph, "What charges the card via the gateway?")
    assert "payments.charge" in context


def test_query_graph_for_ask_no_match(graph: nx.DiGraph) -> None:
    assert g.query_graph_for_ask(graph, "zzz qqq xyzzy") == ""
