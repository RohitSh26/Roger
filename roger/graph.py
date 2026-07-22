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
    # Graphify writes "directed": false but its links carry semantic
    # source→target relations (calls, imports, contains). Force a directed
    # load so each link stays one edge in its original direction instead of
    # being symmetrized into a caller/callee-scrambling double edge.
    if not data.get("directed", False):
        data = {**data, "directed": True}
    try:
        graph = nx.node_link_graph(data, edges=edge_key)
    except TypeError:  # networkx < 3.4 uses link= instead of edges=
        graph = nx.node_link_graph(data, link=edge_key)

    _normalize_node_attrs(graph)
    return graph


def _normalize_node_attrs(graph: nx.DiGraph) -> None:
    """Map graphify's real attribute names onto the ones Roger queries.

    Real graphify nodes carry source_file/label and an integer community;
    Roger reads file/description and string communities. Missing attributes
    (e.g. returns) stay absent — templates skip questions they can't build.
    """
    for node_id, attrs in graph.nodes(data=True):
        if "file" not in attrs and "source_file" in attrs:
            attrs["file"] = attrs["source_file"]
        if "description" not in attrs and "label" in attrs:
            attrs["description"] = str(attrs["label"])
        if attrs.get("community") is not None:
            attrs["community"] = str(attrs["community"])
        # Human-readable name for questions and prompts — developers should
        # see make_broker_deps, never the underscore slug it hashes to.
        attrs["display"] = str(attrs.get("label") or "").strip() or str(node_id)


# Graphify edge relations that mean "A calls B". Other relations (contains,
# references, imports…) must not masquerade as call edges in quiz questions.
# Edges with no relation attribute (plain graphs) count as calls.
CALL_RELATIONS = {"calls", "indirect_call"}


def _is_call_edge(edge_attrs: dict) -> bool:
    relation = edge_attrs.get("relation")
    return relation is None or relation in CALL_RELATIONS


def get_node(graph: nx.DiGraph, node_id: str) -> dict:
    """Return node attributes: id, description, callers, callees, returns, community, file."""
    if node_id not in graph:
        raise KeyError(f"Node not in graph: {node_id}")
    attrs = dict(graph.nodes[node_id])
    attrs["id"] = node_id
    attrs.setdefault("display", node_id)
    attrs["callers"] = sorted(
        u for u, _, d in graph.in_edges(node_id, data=True) if _is_call_edge(d)
    )
    attrs["callees"] = sorted(
        v for _, v, d in graph.out_edges(node_id, data=True) if _is_call_edge(d)
    )
    return attrs


def get_subgraph(graph: nx.DiGraph, node_id: str, hops: int = 1) -> nx.DiGraph:
    """Return 1-hop neighborhood of node_id — used as LLM context."""
    return nx.ego_graph(graph, node_id, radius=hops, undirected=True)


def get_god_nodes(graph: nx.DiGraph, top_n: int = 10) -> list[str]:
    """Return highest-degree node IDs. These are highest priority for quizzing."""
    ranked = sorted(graph.degree, key=lambda pair: (-pair[1], pair[0]))
    return [node_id for node_id, _degree in ranked[:top_n]]


# Graphify emits bookkeeping nodes developers should never be quizzed on:
# doc-derived rationale_N stubs and shell __entry markers.
_JUNK_NODE_RE = re.compile(r"(?:rationale_\d+|__?entry)$")


def _looks_like_test_file(file: str) -> bool:
    name = file.rsplit("/", 1)[-1]
    return (
        any(segment in ("test", "tests") for segment in file.split("/"))
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def get_quizzable_nodes(graph: nx.DiGraph, exclude_tests: bool = True) -> list[str]:
    """Node IDs worth quizzing a developer on.

    Keeps nodes involved in at least one real call edge and drops graph
    noise (rationale_N doc stubs, __entry markers). With exclude_tests,
    prefers production code over test helpers — falling back to the full
    set if the repo is nothing but tests.
    """
    in_calls: set[str] = set()
    for src, dst, data in graph.edges(data=True):
        if _is_call_edge(data):
            in_calls.add(src)
            in_calls.add(dst)

    picked = []
    for node_id, attrs in graph.nodes(data=True):
        if node_id not in in_calls:
            continue
        label = str(attrs.get("label") or "")
        if _JUNK_NODE_RE.search(str(node_id)) or _JUNK_NODE_RE.search(label):
            continue
        picked.append(node_id)

    if exclude_tests:
        non_test = [
            n for n in picked
            if not _looks_like_test_file(str(graph.nodes[n].get("file", "")))
        ]
        if non_test:
            return sorted(non_test)
    return sorted(picked)


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


def serialize_subgraph(
    subgraph: nx.DiGraph, max_chars: int | None = None, labels: bool = False
) -> str:
    """Serialize a subgraph to a compact text format.

    labels=False (the cache-hash form) is deterministic and id-based —
    hashing must pass max_chars=None for full fidelity. labels=True is the
    prompt form: human-readable names and spelled-out relations, so the
    model never sees underscore slugs to parrot back. Prompts also pass a
    budget: god nodes can have hundreds of neighbors, and an uncapped
    serialization blows past the model's context window.
    """
    lines = []
    if labels:
        for node_id in sorted(subgraph.nodes):
            attrs = subgraph.nodes[node_id]
            name = str(attrs.get("display") or node_id)
            desc = str(attrs.get("description", ""))
            suffix = f" — {desc}" if desc and desc != name else ""
            lines.append(f"- {name} ({attrs.get('file', '')}){suffix}")
        for src, dst, data in sorted(subgraph.edges(data=True), key=lambda e: (e[0], e[1])):
            relation = str(data.get("relation") or "calls")
            src_name = str(subgraph.nodes[src].get("display") or src)
            dst_name = str(subgraph.nodes[dst].get("display") or dst)
            lines.append(f"  {src_name} {relation} {dst_name}")
    else:
        for node_id in sorted(subgraph.nodes):
            attrs = subgraph.nodes[node_id]
            desc = attrs.get("description", "")
            file = attrs.get("file", "")
            community = attrs.get("community", "")
            returns = attrs.get("returns", "")
            lines.append(
                f"- {node_id} [file={file} community={community} returns={returns}] {desc}"
            )
        for src, dst in sorted(subgraph.edges):
            lines.append(f"  {src} -> {dst}")

    if max_chars is None:
        return "\n".join(lines)

    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max_chars:
            break
        kept.append(line)
        used += len(line) + 1
    omitted = len(lines) - len(kept)
    if omitted:
        kept.append(f"… (+{omitted} more neighbors/edges omitted)")
    return "\n".join(kept)


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
