"""Graphify integration: load graph.json and answer all graph queries.

Graphify serializes a NetworkX graph as node-link JSON. Node attributes
observed in graphify output include: description, file, community, returns.
Callers/callees are derived from in/out edges rather than stored attributes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from roger.exceptions import GraphNotFoundError

GRAPH_PATH = "graphify-out/graph.json"
REPORT_PATH = "graphify-out/GRAPH_REPORT.md"

_GRAPH_MISSING_MSG = (
    "✗ Roger: No knowledge graph found at graphify-out/graph.json\n"
    "  Build it with: roger init\n"
    "  Or update it with: roger update"
)


def load_graph(path: str = GRAPH_PATH) -> nx.DiGraph:
    """Load graphify output into a NetworkX directed graph."""
    graph_file = Path(path)
    if not graph_file.exists():
        raise GraphNotFoundError(_GRAPH_MISSING_MSG)
    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphNotFoundError(
            f"✗ Roger: Could not read knowledge graph at {path}: {exc}\n"
            "  Rebuild it with: roger init"
        ) from exc

    # Graphify may serialize edges under "links" (node-link default) or "edges".
    edge_key = "links" if "links" in data else "edges"
    try:
        graph = nx.node_link_graph(data, edges=edge_key)
    except TypeError:  # networkx < 3.4 uses link= instead of edges=
        graph = nx.node_link_graph(data, link=edge_key)

    if not graph.is_directed():
        graph = nx.DiGraph(graph)
    return graph


def get_node(graph: nx.DiGraph, node_id: str) -> dict:
    """Return node attributes: id, description, callers, callees, returns, community, file."""
    if node_id not in graph:
        raise KeyError(f"Node not in graph: {node_id}")
    attrs = dict(graph.nodes[node_id])
    attrs["id"] = node_id
    attrs["callers"] = sorted(graph.predecessors(node_id))
    attrs["callees"] = sorted(graph.successors(node_id))
    return attrs


def get_subgraph(graph: nx.DiGraph, node_id: str, hops: int = 1) -> nx.DiGraph:
    """Return 1-hop neighborhood of node_id — used as LLM context."""
    return nx.ego_graph(graph, node_id, radius=hops, undirected=True)


def get_god_nodes(graph: nx.DiGraph, top_n: int = 10) -> list[str]:
    """Return highest-degree node IDs. These are highest priority for quizzing."""
    ranked = sorted(graph.degree, key=lambda pair: (-pair[1], pair[0]))
    return [node_id for node_id, _degree in ranked[:top_n]]


def get_community_nodes(graph: nx.DiGraph, community: str) -> list[str]:
    """Return all node IDs in a named Leiden community."""
    return sorted(
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("community") == community
    )


def _normalize_path(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def get_changed_nodes(graph: nx.DiGraph, changed_files: list[str]) -> list[str]:
    """Map a list of changed file paths (from git diff) to graph node IDs."""
    changed = {_normalize_path(p) for p in changed_files}
    return sorted(
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if _normalize_path(str(attrs.get("file", ""))) in changed
    )


def get_nodes_by_path(graph: nx.DiGraph, path: str) -> list[str]:
    """Return all node IDs whose file attribute matches path or starts with path."""
    prefix = _normalize_path(path).rstrip("/")
    matches = []
    for node_id, attrs in graph.nodes(data=True):
        file = _normalize_path(str(attrs.get("file", "")))
        if file == prefix or file.startswith(prefix + "/"):
            matches.append(node_id)
    return sorted(matches)


def serialize_subgraph(subgraph: nx.DiGraph) -> str:
    """Serialize a subgraph to a compact text format for LLM prompt injection.

    Output is deterministic (sorted nodes/edges) — it also feeds the cache hash.
    """
    lines = []
    for node_id in sorted(subgraph.nodes):
        attrs = subgraph.nodes[node_id]
        desc = attrs.get("description", "")
        file = attrs.get("file", "")
        community = attrs.get("community", "")
        returns = attrs.get("returns", "")
        lines.append(f"- {node_id} [file={file} community={community} returns={returns}] {desc}")
    for src, dst in sorted(subgraph.edges):
        lines.append(f"  {src} -> {dst}")
    return "\n".join(lines)


def get_god_node_ids_from_report(report_path: str = REPORT_PATH) -> list[str]:
    """Parse GRAPH_REPORT.md to extract named god nodes — used for quiz weighting."""
    report = Path(report_path)
    if not report.exists():
        return []
    god_nodes: list[str] = []
    in_section = False
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            in_section = "god node" in line.lower()
            continue
        if in_section:
            god_nodes.extend(re.findall(r"`([^`]+)`", line))
    return god_nodes


def get_surprise_edges(report_path: str = REPORT_PATH) -> list[tuple]:
    """Parse GRAPH_REPORT.md to extract surprise edges — direct quiz material."""
    report = Path(report_path)
    if not report.exists():
        return []
    edges: list[tuple] = []
    in_section = False
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            in_section = "surprise" in line.lower()
            continue
        if in_section:
            match = re.search(r"`?([\w./:]+)`?\s*(?:->|→)\s*`?([\w./:]+)`?", line)
            if match:
                edges.append((match.group(1), match.group(2)))
    return edges


def query_graph_for_ask(graph: nx.DiGraph, question: str) -> str:
    """Natural language graph query for roger ask.

    Keyword-match question terms against node descriptions (and IDs).
    Return serialized subgraph of top matching nodes.
    """
    terms = {t.lower() for t in re.findall(r"\w+", question) if len(t) > 2}
    if not terms:
        return ""

    scores: dict[str, int] = {}
    for node_id, attrs in graph.nodes(data=True):
        haystack = f"{node_id} {attrs.get('description', '')}".lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scores[node_id] = score

    top = sorted(scores, key=lambda n: (-scores[n], n))[:5]
    if not top:
        return ""

    combined: set[str] = set()
    for node_id in top:
        combined.update(get_subgraph(graph, node_id, hops=1).nodes)
    return serialize_subgraph(graph.subgraph(combined))
