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


# --- real graphify schema (undirected serialization, source_file/label, relations) ---

REAL_SCHEMA_DATA = {
    "directed": False,  # graphify serializes undirected, but edge direction is semantic
    "multigraph": False,
    "graph": {},
    "hyperedges": [],  # extra graphify keys must not break loading
    "nodes": [
        {"id": "app_main", "label": "main.py", "source_file": "app/main.py", "community": 3},
        {"id": "app_helper", "label": "helper", "source_file": "app/helper.py", "community": 3},
        {"id": "app_config", "label": "config", "source_file": "app/config.py", "community": 7},
    ],
    "links": [
        {"source": "app_main", "target": "app_helper", "relation": "calls"},
        {"source": "app_main", "target": "app_config", "relation": "contains"},
        {"source": "app_helper", "target": "app_config", "relation": "references"},
    ],
}


@pytest.fixture
def real_schema_graph(tmp_path: Path) -> nx.DiGraph:
    import json

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(REAL_SCHEMA_DATA), encoding="utf-8")
    return g.load_graph(str(path))


def test_real_schema_loads_directed_without_symmetrizing(real_schema_graph) -> None:
    assert real_schema_graph.is_directed()
    # 3 links must stay 3 edges — not doubled into 6.
    assert real_schema_graph.number_of_edges() == 3
    assert real_schema_graph.has_edge("app_main", "app_helper")
    assert not real_schema_graph.has_edge("app_helper", "app_main")


def test_real_schema_normalizes_attributes(real_schema_graph) -> None:
    node = g.get_node(real_schema_graph, "app_main")
    assert node["file"] == "app/main.py"       # from source_file
    assert node["description"] == "main.py"    # from label
    assert node["community"] == "3"            # int → str


def test_real_schema_only_call_edges_count(real_schema_graph) -> None:
    main = g.get_node(real_schema_graph, "app_main")
    assert main["callees"] == ["app_helper"]   # 'contains' edge excluded
    config = g.get_node(real_schema_graph, "app_config")
    assert config["callers"] == []             # contains/references are not calls
    helper = g.get_node(real_schema_graph, "app_helper")
    assert helper["callers"] == ["app_main"]


def test_real_schema_file_queries_use_normalized_paths(real_schema_graph) -> None:
    assert g.get_changed_nodes(real_schema_graph, ["app/helper.py"]) == ["app_helper"]
    assert g.get_nodes_by_path(real_schema_graph, "app") == [
        "app_config", "app_helper", "app_main",
    ]


def test_serialize_subgraph_respects_max_chars(graph: nx.DiGraph) -> None:
    sub = g.get_subgraph(graph, "payments.process_payment", hops=1)
    full = g.serialize_subgraph(sub)
    capped = g.serialize_subgraph(sub, max_chars=120)
    assert len(capped) < len(full)
    assert "omitted" in capped
    # Uncapped output (the cache-hash input) must never carry the marker.
    assert "omitted" not in full


def test_get_quizzable_nodes_filters_noise(tmp_path: Path) -> None:
    import json

    data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "app_worker", "label": "process_job", "source_file": "app/worker.py"},
            {"id": "app_helper", "label": "retry", "source_file": "app/helper.py"},
            {"id": "docs_rationale_7", "label": "rationale_7", "source_file": "docs/x.md"},
            {"id": "run_sh__entry", "label": "entry", "source_file": "run.sh"},
            {"id": "lonely", "label": "orphan", "source_file": "app/orphan.py"},
            {"id": "tests_util", "label": "fake_broker", "source_file": "tests/util.py"},
        ],
        "links": [
            {"source": "app_worker", "target": "app_helper", "relation": "calls"},
            {"source": "app_worker", "target": "docs_rationale_7", "relation": "calls"},
            {"source": "run_sh__entry", "target": "app_worker", "relation": "calls"},
            {"source": "tests_util", "target": "app_worker", "relation": "calls"},
            {"source": "app_helper", "target": "lonely", "relation": "references"},
        ],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    graph = g.load_graph(str(path))

    # rationale/_entry stubs and call-less nodes are never quizzable;
    # test helpers drop out when production code exists.
    assert g.get_quizzable_nodes(graph) == ["app_helper", "app_worker"]
    assert g.get_quizzable_nodes(graph, exclude_tests=False) == [
        "app_helper", "app_worker", "tests_util",
    ]


def test_get_quizzable_nodes_falls_back_to_tests_only_repo(tmp_path: Path) -> None:
    import json

    data = {
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [
            {"id": "t1", "label": "fake_a", "source_file": "tests/a.py"},
            {"id": "t2", "label": "fake_b", "source_file": "tests/b.py"},
        ],
        "links": [{"source": "t1", "target": "t2", "relation": "calls"}],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    graph = g.load_graph(str(path))
    assert g.get_quizzable_nodes(graph) == ["t1", "t2"]  # better than nothing


def test_serialize_subgraph_labels_mode_hides_slugs(real_schema_graph) -> None:
    text = g.serialize_subgraph(real_schema_graph, labels=True)
    assert "main.py calls helper" in text
    assert "main.py contains config" in text
    assert "app_main" not in text  # slugs never reach the prompt


def test_get_source_snippet_reads_window(tmp_path: Path) -> None:
    src = tmp_path / "app" / "worker.py"
    src.parent.mkdir()
    src.write_text("\n".join(f"line {i}" for i in range(1, 101)), encoding="utf-8")

    attrs = {"file": "app/worker.py", "source_location": "L10"}
    snippet = g.get_source_snippet(attrs, max_lines=5, repo_root=tmp_path)
    # Window of 5 from a 100-line file — cut, and honestly marked as such.
    assert snippet.splitlines() == [
        "line 10", "line 11", "line 12", "line 13", "line 14", g.TRUNCATION_MARKER,
    ]

    ranged = g.get_source_snippet(
        {"file": "app/worker.py", "source_location": "L10-L12"}, repo_root=tmp_path
    )
    assert ranged.splitlines() == ["line 10", "line 11", "line 12"]


def test_get_source_snippet_missing_file_or_location(tmp_path: Path) -> None:
    assert g.get_source_snippet({"file": "gone.py", "source_location": "L1"}, repo_root=tmp_path) == ""
    assert g.get_source_snippet({"source_location": "L1"}, repo_root=tmp_path) == ""
    # No/garbled location falls back to the top of the file — and a cut
    # this tight is honestly marked as truncated.
    src = tmp_path / "x.py"
    src.write_text("first\nsecond\n", encoding="utf-8")
    snippet = g.get_source_snippet({"file": "x.py"}, max_lines=1, repo_root=tmp_path)
    assert snippet.splitlines()[0] == "first"
    assert snippet.endswith(g.TRUNCATION_MARKER)


PYTHON_SOURCE = '''\
import os


class Cache:
    def get_or_compute(self, key, compute):
        """Return the cached value, computing it once on miss."""
        if key in self.store:
            return self.store[key]
        value = compute(key)
        if value is not None:
            self.store[key] = value
        return value

    def clear(self):
        self.store = {}
'''


def test_snippet_extracts_complete_python_function(tmp_path: Path) -> None:
    (tmp_path / "cache.py").write_text(PYTHON_SOURCE, encoding="utf-8")
    attrs = {"file": "cache.py", "source_location": "L5"}  # def get_or_compute
    snippet = g.get_source_snippet(attrs, repo_root=tmp_path)
    assert snippet.startswith("def get_or_compute")
    assert "return value" in snippet          # reaches the end of the function
    assert "def clear" not in snippet          # stops before the next method
    assert g.TRUNCATION_MARKER not in snippet  # complete block → no marker


def test_snippet_extracts_complete_class_block(tmp_path: Path) -> None:
    (tmp_path / "cache.py").write_text(PYTHON_SOURCE, encoding="utf-8")
    attrs = {"file": "cache.py", "source_location": "L4"}  # class Cache
    snippet = g.get_source_snippet(attrs, repo_root=tmp_path)
    assert snippet.startswith("class Cache:")
    assert "def clear" in snippet              # whole class, both methods


def test_snippet_extracts_brace_delimited_block(tmp_path: Path) -> None:
    js = "function ranked(hits) {\n  const best = {};\n  for (const h of hits) {\n    best[h.id] = h;\n  }\n  return best;\n}\n\nfunction other() {\n  return 1;\n}\n"
    (tmp_path / "rank.js").write_text(js, encoding="utf-8")
    snippet = g.get_source_snippet({"file": "rank.js", "source_location": "L1"}, repo_root=tmp_path)
    assert snippet.startswith("function ranked")
    assert snippet.rstrip().endswith("}")
    assert "function other" not in snippet


def test_snippet_marks_truncation_visibly(tmp_path: Path) -> None:
    body = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(200)) + "\nprint('after')\n"
    (tmp_path / "big.py").write_text(body, encoding="utf-8")
    snippet = g.get_source_snippet(
        {"file": "big.py", "source_location": "L1"}, max_lines=20, repo_root=tmp_path
    )
    assert snippet.endswith(g.TRUNCATION_MARKER)
    assert len(snippet.splitlines()) <= 21     # 20 code lines + marker


def test_snippet_char_budget_cuts_on_line_boundary(tmp_path: Path) -> None:
    body = "def wide():\n" + "\n".join("    y = " + "a" * 100 for _ in range(50))
    (tmp_path / "wide.py").write_text(body, encoding="utf-8")
    snippet = g.get_source_snippet(
        {"file": "wide.py", "source_location": "L1"}, max_chars=500, repo_root=tmp_path
    )
    assert snippet.endswith(g.TRUNCATION_MARKER)
    for line in snippet.splitlines()[:-1]:
        assert line == "def wide():" or line.endswith("a")  # no mid-line cuts


# --- generalization guardrails: Roger must work on any repo, any language ---------


@pytest.mark.parametrize(
    ("file", "is_test"),
    [
        ("src/payments/charge.py", False),
        ("tests/test_charge.py", True),          # Python
        ("pkg/broker/broker_test.go", True),      # Go
        ("pkg/broker/broker.go", False),
        ("src/cart/cart.spec.ts", True),          # JS/TS spec
        ("src/cart/Cart.test.tsx", True),         # JS/TS test
        ("src/cart/Cart.tsx", False),
        ("src/main/java/App.java", False),
        ("src/test/java/AppTest.java", True),     # Java (dir + suffix)
        ("lib/__tests__/util.js", True),          # Jest convention
        ("app/models/protester.rb", False),       # 'test' inside a word ≠ test file
    ],
)
def test_test_file_detection_across_languages(file: str, is_test: bool) -> None:
    assert g._looks_like_test_file(file) is is_test


GO_SOURCE = """\
package broker

func RankHits(hits []Hit, top int) []Hit {
	best := map[string]Hit{}
	for _, h := range hits {
		cur, ok := best[h.ID]
		if !ok || h.Score > cur.Score {
			best[h.ID] = h
		}
	}
	return sortHits(best, top)
}

func sortHits(m map[string]Hit, top int) []Hit {
	return nil
}
"""

TS_SOURCE = """\
export function rankHits(hits: Hit[], top: number): Hit[] {
  const best = new Map<string, Hit>();
  for (const h of hits) {
    const cur = best.get(h.id);
    if (cur === undefined || h.score > cur.score) {
      best.set(h.id, h);
    }
  }
  return [...best.values()].slice(0, top);
}

export function other(): number {
  return 1;
}
"""


@pytest.mark.parametrize(
    ("filename", "source", "must_contain", "must_not_contain"),
    [
        ("rank.go", GO_SOURCE, "return sortHits(best, top)", "func sortHits"),
        ("rank.ts", TS_SOURCE, ".slice(0, top)", "export function other"),
    ],
)
def test_block_extraction_works_across_languages(
    tmp_path: Path, filename: str, source: str, must_contain: str, must_not_contain: str
) -> None:
    (tmp_path / filename).write_text(source, encoding="utf-8")
    snippet = g.get_source_snippet(
        {"file": filename, "source_location": "L3" if filename.endswith(".go") else "L1"},
        repo_root=tmp_path,
    )
    assert must_contain in snippet       # reaches the true end of the block
    assert must_not_contain not in snippet  # stops before the next declaration
