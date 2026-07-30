"""roger ask — grounded Q&A over the code graph, real source, and docs.

The question is keyword-matched against code nodes AND documentation
sections; the winning source excerpts and doc excerpts become the model's
entire world. Answers must cite what they were grounded in, and the CLI
shows the sources so the developer can verify.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from roger import graph as g
from roger.config import Config
from roger.docs import DocSection, _md_excerpt, discover_doc_files, split_sections
from roger.freshness import is_source_file
from roger.graph import candidate_code_nodes, looks_like_test_file
from roger.llm.router import chat_with_model, ensure_backend
from roger.quiz import language_for_file

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "how",
    "why", "does", "do", "did", "when", "where", "who", "in", "of", "to",
    "for", "on", "and", "or", "not", "it", "this", "that", "with", "from",
    "can", "could", "should", "would", "we", "our", "you", "about",
    "via", "using", "into", "any", "all",
}


def _variants(term: str) -> set[str]:
    """Cheap stemming: charges→charge, ranking→rank, implemented→implement."""
    variants = {term}
    if term.endswith("s"):
        variants.add(term[:-1])
    if term.endswith("ing") and len(term) > 5:
        variants.add(term[:-3])
    if term.endswith("ed") and len(term) > 4:
        variants.add(term[:-2])
    return variants


def _hits(term: str, haystack: str) -> bool:
    return any(v in haystack for v in _variants(term))

MAX_CODE_MATCHES = 3
MAX_DOC_MATCHES = 2
_SNIPPET_CHARS = 3_000

ASK_PROMPT = """\
You are Roger, a codebase assistant. Answer the developer's question using
ONLY the repository material below.

QUESTION: {question}

REPOSITORY MATERIAL:
{context}

RULES:
- Answer in concise markdown: short paragraphs, bullet lists where they
  help, `code spans` for identifiers, fenced blocks only for real code.
- Ground every claim in the material above and cite the file or document
  inline, like (services/x/y.py) or (docs/adr/0003.md).
- If the material does not answer the question, say so plainly and name
  what is missing — never invent behavior or refer to this prompt.
"""


_SHORT_IDENTIFIER_RE = re.compile(r"[a-z]\d")  # L0, v2 — meaningful despite length


def _terms(question: str) -> list[str]:
    terms = []
    for token in re.findall(r"\w+", question):
        low = token.lower()
        if low in _STOPWORDS:
            continue
        if len(low) > 2 or _SHORT_IDENTIFIER_RE.fullmatch(low):
            terms.append(low)
    return terms


def retrieve_nodes(
    graph, question: str, config: Optional[Config] = None, top: int = MAX_CODE_MATCHES
) -> list[str]:
    """Final code retrieval: keyword ∪ semantic channels, rank-fused.

    Two independent channels so a question sharing no words with its answer
    can still arrive via meaning. Fused with Reciprocal Rank Fusion (scale-
    free — keyword integers and cosine bands don't share a scale). An exact
    identifier match is pinned ahead of fusion: semantic similarity must
    never displace a literal name hit. No index/model → exactly the keyword
    ranking, bit-identical to keyword-only Roger.
    """
    from roger import embeddings

    keyword = find_relevant_nodes(graph, question, top=30)
    semantic = embeddings.semantic_rank(question, config or Config(), top=30)
    if semantic and not any(t.startswith("test") for t in _terms(question)):
        # Same rule as the keyword channel: tests never answer "how does X
        # work" — similarity must not smuggle them back in.
        semantic = [
            n for n in semantic
            if not looks_like_test_file(str(graph.nodes[n].get("file") or ""))
        ]

    if not semantic:
        fused = keyword
    else:
        scores: dict[str, float] = {}
        for rank, node_id in enumerate(keyword):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (60 + rank)
        for rank, node_id in enumerate(semantic):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (60 + rank)
        fused = sorted(scores, key=lambda n: (-scores[n], n))

    # Exact-name pin: a term that IS the identifier wins outright. Both the
    # qualified form and the bare tail count — a developer types "charge",
    # the graph knows "payments.charge".
    terms = set(_terms(question))

    def _is_named(node_id: str) -> bool:
        display = str(graph.nodes[node_id].get("display", node_id))
        names = set()
        for raw in (display, node_id):
            name = raw.lower().removesuffix("()")
            names.add(name)
            names.add(name.rsplit(".", 1)[-1])
        return bool(names & terms)

    pinned = [node_id for node_id in fused if _is_named(node_id)]
    ordered = pinned + [n for n in fused if n not in pinned]
    return ordered[:top]


def find_relevant_nodes(graph, question: str, top: int = MAX_CODE_MATCHES) -> list[str]:
    """Keyword channel: node ids best matching the question by words."""
    terms = _terms(question)
    if not terms:
        return []
    from roger import embeddings

    card_texts = embeddings.load_card_texts()  # docstring-aware when index exists
    scores: dict[str, int] = {}
    for node_id in candidate_code_nodes(graph):
        attrs = graph.nodes[node_id]
        display = str(attrs.get("display") or node_id)
        file = str(attrs.get("file") or "")
        # Production code answers "how does X work"; descriptive test names
        # are keyword-rich sentences that always out-score real code, so
        # tests are excluded unless the question is literally about tests.
        if looks_like_test_file(file) and not any(t.startswith("test") for t in terms):
            continue
        names = f"{node_id} {display}".lower()
        prose = (
            f"{attrs.get('description', '')} {file} "
            f"{card_texts.get(node_id, '')}".lower()
        )
        score = sum(3 for t in terms if _hits(t, names)) + sum(
            1 for t in terms if _hits(t, prose)
        )
        if score > 0:
            scores[node_id] = score
    return sorted(scores, key=lambda n: (-scores[n], n))[:top]


def find_relevant_sections(
    question: str,
    paths: Optional[list[str]] = None,
    repo_root: Path = Path("."),
    top: int = MAX_DOC_MATCHES,
) -> list[DocSection]:
    """Doc sections best matching the question (heading hits weigh more)."""
    terms = _terms(question)
    if not terms:
        return []
    scored: list[tuple[int, DocSection]] = []
    for path in discover_doc_files(paths, repo_root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
        for section in split_sections(rel, text):
            body = section.text.lower()
            heading = section.heading.lower()
            score = sum(3 for t in terms if _hits(t, heading))
            score += sum(min(body.count(t) + body.count(t[:-1] if t.endswith("s") else t), 5) for t in terms)
            if score >= 2:
                scored.append((score, section))
    scored.sort(key=lambda pair: -pair[0])
    return [section for _, section in scored[:top]]


def build_context(
    graph, question: str, config: Config, max_chars: Optional[int] = None
) -> tuple[str, list[str]]:
    """(material for the prompt, human-readable source labels)."""
    budget = max_chars or max(6_000, (config.ollama.num_ctx - 1_200) * 5 // 2)
    blocks: list[str] = []
    sources: list[str] = []
    used = 0

    for node_id in retrieve_nodes(graph, question, config):
        node = g.get_node(graph, node_id)
        name = str(node.get("display") or node_id)
        file = str(node.get("file") or "")
        header = (
            f"### Code: {name} ({file})\n"
            f"Called by: {', '.join(node['callers'][:8]) or 'none'} | "
            f"Calls: {', '.join(node['callees'][:8]) or 'none'}"
        )
        snippet = g.get_source_snippet(node, max_lines=40)[:_SNIPPET_CHARS]
        block = header + (
            f"\n```{language_for_file(file)}\n{snippet}\n```" if snippet else ""
        )
        if used + len(block) > budget:
            break
        blocks.append(block)
        sources.append(f"{name} ({file})" if file else name)
        used += len(block)

    for section in find_relevant_sections(question, config.docs.paths):
        block = f"### Doc: {section.file} § {section.heading}\n{_md_excerpt(section.text, 40)}"
        if used + len(block) > budget:
            break
        blocks.append(block)
        sources.append(f"{section.file} § {section.heading}")
        used += len(block)

    return "\n\n".join(blocks), sources


def context_pack(
    question: str,
    graph,
    config: Optional[Config] = None,
    budget_tokens: int = 2_000,
) -> str:
    """A cited context pack for coding agents — the briefing without the answer.

    Zero LLM calls: Roger retrieves and assembles (complete source blocks,
    doc/ADR excerpts, call facts); the agent does its own reasoning. Output
    is plain markdown on stdout, capped at a token budget (~4 chars/token).
    """
    config = config or Config()
    max_chars = max(2_000, budget_tokens * 4)
    context, sources = build_context(graph, question, config, max_chars=max_chars - 400)

    # Truncation without navigation is lossy; an index makes it precise.
    # List matches that were NOT expanded so the caller can re-query
    # narrowly instead of guessing what the budget cut.
    expanded = len([s for s in sources if "§" not in s])
    more_lines = []
    for node_id in retrieve_nodes(graph, question, config, top=8)[expanded:]:
        attrs = graph.nodes[node_id]
        more_lines.append(
            f"- {attrs.get('display', node_id)} ({attrs.get('file', '')})"
        )
    more_section = (
        "\n\n## More matches (not expanded)\n"
        + "\n".join(more_lines)
        + '\n\nTo expand one, re-run with its name: roger context "<name> <your question>"'
        if more_lines
        else ""
    )
    if not context:
        return (
            f"# Roger context: {question}\n\n"
            "No matching code or docs found. Try naming a function, class, "
            "file, or doc topic — or the graph may need `roger update`."
        )
    # Self-declared provenance: agents fact-check unverified summaries (as
    # they should) — so state plainly that these excerpts are mechanical
    # copies, making a re-read of the same lines provably redundant.
    provenance = (
        "> Provenance: code and doc excerpts below are VERBATIM file contents, "
        "mechanically extracted just now — not AI-generated summaries. "
        "Re-reading the cited lines returns identical text.\n\n"
    )
    pack = (
        f"# Roger context: {question}\n\n{provenance}{context}{more_section}"
        "\n\n## Sources\n" + "\n".join(f"- {s}" for s in sources)
    )
    if len(pack) > max_chars:
        kept: list[str] = []
        used = 0
        for line in pack.splitlines():
            if used + len(line) + 1 > max_chars - 80:
                break
            kept.append(line)
            used += len(line) + 1
        kept.append("")
        kept.append("*(truncated at budget — narrow the question or raise --budget)*")
        pack = "\n".join(kept)
    return pack


def _seam_nodes(
    graph, anchors: list[str], per_anchor_cap: int = 6
) -> list[tuple[str, int]]:
    """Boundary nodes 1 hop from the anchors over interface relations,
    ranked by how many distinct anchors touch them.

    Measured on the real graph: uncapped 1-hop from 12 anchors reached far
    nodes (far past any budget) because hub nodes have very high degree. Per-anchor,
    prefer specific neighbors (low degree) over hubs; globally, a node
    touched by 3 anchors is signal while one touched by 1 is noise.
    """
    anchor_set = set(anchors)
    touched: dict[str, set[str]] = {}
    for anchor in anchors:
        neighbors: list[tuple[int, str]] = []
        for src, dst, data in graph.in_edges(anchor, data=True):
            if data.get("relation") in g.INTERFACE_RELATIONS and src not in anchor_set:
                neighbors.append((graph.degree(src), src))
        for src, dst, data in graph.out_edges(anchor, data=True):
            if data.get("relation") in g.INTERFACE_RELATIONS and dst not in anchor_set:
                neighbors.append((graph.degree(dst), dst))
        for _, node_id in sorted(set(neighbors))[:per_anchor_cap]:
            touched.setdefault(node_id, set()).add(anchor)
    ranked = sorted(touched.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [(node_id, len(anchors_)) for node_id, anchors_ in ranked]


def _relation_lines(graph, node_id: str, limit: int = 6) -> str:
    """Per-relation, direction-aware relationship summary for one node."""
    by_verb: dict[str, list[str]] = {}
    for _, dst, data in graph.out_edges(node_id, data=True):
        verbs = g.RELATION_VERBS.get(str(data.get("relation")))
        if verbs:
            by_verb.setdefault(verbs[0], []).append(
                str(graph.nodes[dst].get("display", dst))
            )
    for src, _, data in graph.in_edges(node_id, data=True):
        verbs = g.RELATION_VERBS.get(str(data.get("relation")))
        if verbs:
            by_verb.setdefault(verbs[1], []).append(
                str(graph.nodes[src].get("display", src))
            )
    parts = []
    for verb in sorted(by_verb):
        names = sorted(set(by_verb[verb]))
        shown = ", ".join(names[:limit])
        more = f" (+{len(names) - limit})" if len(names) > limit else ""
        parts.append(f"{verb}: {shown}{more}")
    return " | ".join(parts)


def _interface_card(graph, node_id: str) -> Optional[str]:
    """One contract card: signature + docstring line + relationships.
    File nodes render as a module line, never a fake signature; nodes
    whose file is gone render as name + relationships only."""
    attrs = graph.nodes[node_id]
    display = str(attrs.get("display") or node_id)
    file = str(attrs.get("file") or "")
    relations = _relation_lines(graph, node_id)
    if display == Path(file).name and file:
        head = f"module {file}"
        doc = str(attrs.get("description") or "")
        body = f" — {doc[:120]}" if doc and doc != display else ""
        card = f"{head}{body}"
    else:
        contract = g.get_interface(attrs)
        if contract:
            language = language_for_file(file)
            card = f"{display} ({file})\n```{language}\n{contract}\n```"
        elif file:
            card = f"{display} ({file})"
        else:
            return None  # sourceless external — appears in others' relations
    if relations:
        card += f"\n{relations}"
    return card


def interface_pack(
    question: str,
    graph,
    config: Optional[Config] = None,
    budget_tokens: int = 2_000,
) -> str:
    """Contracts, not contents: the interfaces relevant to one task.

    Built for vertical-stack agent work — agent N+1 building on the layer
    below needs signatures, docstrings, and relationships, not bodies.
    Zero LLM calls. Budget is FILLED card by card, never truncated
    mid-card: a dropped load-bearing contract is worse than a shorter
    list, so the cut happens at card boundaries with the remainder named.
    """
    config = config or Config()
    max_chars = max(2_000, budget_tokens * 4)
    anchors = retrieve_nodes(graph, question, config, top=10)
    if not anchors:
        return (
            f"# Roger interfaces: {question}\n\n"
            "No matching code found. Try naming a function, class, or file — "
            "or the graph may need `roger update`."
        )
    seam = _seam_nodes(graph, anchors)

    header = (
        f"# Roger interfaces: {question}\n\n"
        "> Contracts only — signatures, docstrings, and relationships, "
        "extracted VERBATIM from source just now (not AI-generated). "
        "Bodies are omitted by design; `roger context` expands full code.\n"
    )
    # Doc context is budgeted BEFORE the cards — appending it after would
    # overflow the cap (field-measured 8.6k against an 8k budget).
    doc_note = ""
    sections = find_relevant_sections(question, config.docs.paths, top=1)
    if sections:
        section = sections[0]
        doc_note = (
            f"\n\n### Design context: {section.file} § {section.heading}\n"
            + _md_excerpt(section.text, 12)
        )

    blocks: list[str] = []
    skipped: list[str] = []
    used = len(header) + len(doc_note) + 400  # 400 ≈ the More-contracts index
    for node_id in anchors + [n for n, _ in seam]:
        card = _interface_card(graph, node_id)
        if card is None:
            continue
        entry = f"\n## {card}" if not card.startswith("module ") else f"\n## {card}"
        if used + len(entry) > max_chars - 200:
            skipped.append(str(graph.nodes[node_id].get("display", node_id)))
            continue
        blocks.append(entry)
        used += len(entry)
    more = (
        "\n\n### More contracts (not expanded)\n"
        + ", ".join(skipped[:20])
        + "\n\nNarrow the question or raise --budget to expand them."
        if skipped
        else ""
    )
    return header + "".join(blocks) + doc_note + more


def answer_question(question: str, graph, config: Optional[Config] = None) -> tuple[str, list[str]]:
    """Answer a question about the repo. Returns (markdown answer, sources)."""
    config = config or Config()
    ensure_backend(config)
    context, sources = build_context(graph, question, config)
    if not context:
        return (
            "I couldn't find anything in the code graph or the docs matching "
            "that question. Try naming a function, class, file, or document "
            "topic — or rebuild the graph with `roger update` if the code is new.",
            [],
        )
    answer = chat_with_model(ASK_PROMPT.format(question=question, context=context), config)
    return answer.strip(), sources
