"""Constellation SVG for the app's Explore view — the path drawn as a
glowing route through the faint surrounding graph.

Everything is computed here, in Python: NetworkX lays out the
neighborhood (spring layout, deterministic seed, path nodes pinned along
the horizontal axis), and the result is hand-built inline SVG. No D3, no
CDN, no JavaScript — graphify's own HTML exports pull D3/Mermaid from the
network, which the app's offline promise rules out.

Kept free of Streamlit imports so it stays unit-testable (streamlit is an
optional extra).
"""

from __future__ import annotations

import html

import networkx as nx

# Palette mirrors roger/style.py's dark code panel.
_BG = "#1F1C18"
_FAINT_EDGE = "rgba(232,225,213,.09)"
_FAINT_NODE = "rgba(232,225,213,.30)"
_ROUTE = "#C1683F"          # clay — the one accent
_ROUTE_SOFT = "rgba(193,104,63,.30)"
_LABEL = "#E8E1D5"
_REL = "#A39A8C"

MAX_BACKGROUND_NODES = 46
MAX_BACKGROUND_EDGES = 150


def _neighborhood(graph: nx.DiGraph, path_ids: list[str]) -> nx.Graph:
    """Path nodes plus a deterministic sample of their neighbors, as an
    undirected subgraph (direction is drawn on labels, not layout)."""
    keep: list[str] = list(path_ids)
    seen = set(keep)
    undirected = graph.to_undirected(as_view=True)
    for node in path_ids:
        for other in sorted(undirected.neighbors(node)):
            if other not in seen and str(graph.nodes[other].get("display", "")):
                seen.add(other)
                keep.append(other)
            if len(keep) >= len(path_ids) + MAX_BACKGROUND_NODES:
                break
        if len(keep) >= len(path_ids) + MAX_BACKGROUND_NODES:
            break
    return nx.Graph(undirected.subgraph(keep))


def _positions(
    sub: nx.Graph, path_ids: list[str], width: int, height: int
) -> dict[str, tuple[float, float]]:
    """Spring layout with the path pinned left-to-right on a gentle zigzag
    (the route reads as a route; the rest of the graph drapes around it)."""
    init: dict[str, tuple[float, float]] = {}
    m = len(path_ids)
    for i, node in enumerate(path_ids):
        x = i / max(m - 1, 1)
        init[node] = (x, 0.36 if i % 2 == 0 else 0.64)
    pos = nx.spring_layout(
        sub, pos=init, fixed=path_ids if m > 0 else None, seed=7,
        k=0.28, iterations=60,
    )
    out: dict[str, tuple[float, float]] = {}
    for node, (x, y) in pos.items():
        x = min(max(float(x), -0.06), 1.06)
        y = min(max(float(y), -0.06), 1.06)
        out[node] = (58 + x * (width - 116), 34 + y * (height - 82))
    return out


def path_map_svg(
    graph: nx.DiGraph,
    hops: list[dict],
    links: list[dict],
    width: int = 900,
    height: int = 380,
) -> str:
    """The whole picture for a connection: faint neighborhood, glowing
    route, mono labels on the route nodes, relation on each route edge."""
    path_ids = [h["id"] for h in hops]
    sub = _neighborhood(graph, path_ids)
    pos = _positions(sub, path_ids, width, height)
    on_route = set(path_ids)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Connection map" '
        f'style="width:100%;height:auto;display:block">',
        f'<rect width="{width}" height="{height}" rx="13" fill="{_BG}"/>',
    ]

    background_edges = sorted(
        (min(a, b), max(a, b)) for a, b in sub.edges()
        if not (a in on_route and b in on_route)
    )[:MAX_BACKGROUND_EDGES]
    for a, b in background_edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        parts.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{_FAINT_EDGE}" stroke-width="1"/>'
        )
    for node in sorted(sub.nodes()):
        if node in on_route:
            continue
        x, y = pos[node]
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.5" fill="{_FAINT_NODE}"/>')

    for i, link in enumerate(links):
        (x1, y1), (x2, y2) = pos[path_ids[i]], pos[path_ids[i + 1]]
        parts.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{_ROUTE}" stroke-width="2.2" stroke-linecap="round"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 9
        arrow = "→" if link["forward"] else "←"
        label = html.escape(f'{arrow} {link["relation"]}')
        parts.append(
            f'<text x="{mx:.0f}" y="{my:.0f}" text-anchor="middle" '
            f'font-family="ui-monospace,Menlo,monospace" font-size="11" '
            f'font-style="italic" fill="{_REL}">{label}</text>'
        )

    for i, hop in enumerate(hops):
        x, y = pos[hop["id"]]
        above = y >= height / 2
        ty = y - 16 if above else y + 26
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="9" fill="{_ROUTE_SOFT}"/>')
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4.5" fill="{_ROUTE}"/>')
        anchor = "start" if i == 0 else ("end" if i == len(hops) - 1 else "middle")
        tx = x - 10 if i == 0 else (x + 10 if i == len(hops) - 1 else x)
        parts.append(
            f'<text x="{tx:.0f}" y="{ty:.0f}" text-anchor="{anchor}" '
            f'font-family="ui-monospace,Menlo,monospace" font-size="12.5" '
            f'font-weight="600" fill="{_LABEL}">{html.escape(hop["display"])}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)
