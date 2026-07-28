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
from roger.llm.router import chat_with_model, ensure_backend
from roger.quiz import language_for_file

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "how",
    "why", "does", "do", "did", "when", "where", "who", "in", "of", "to",
    "for", "on", "and", "or", "not", "it", "this", "that", "with", "from",
    "can", "could", "should", "would", "we", "our", "you", "about",
    "via", "using", "into", "any", "all",
}


def _hits(term: str, haystack: str) -> bool:
    """Substring hit with cheap plural-stemming ('charges' finds 'charge')."""
    return term in haystack or (term.endswith("s") and term[:-1] in haystack)

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
_IDENTIFIER_NAME_RE = re.compile(r"[A-Za-z_][\w.]*(\(\))?$")


def _terms(question: str) -> list[str]:
    terms = []
    for token in re.findall(r"\w+", question):
        low = token.lower()
        if low in _STOPWORDS:
            continue
        if len(low) > 2 or _SHORT_IDENTIFIER_RE.fullmatch(low):
            terms.append(low)
    return terms


def find_relevant_nodes(graph, question: str, top: int = MAX_CODE_MATCHES) -> list[str]:
    """Node ids best matching the question — names weigh more than prose."""
    terms = _terms(question)
    if not terms:
        return []
    scores: dict[str, int] = {}
    for node_id, attrs in graph.nodes(data=True):
        # Sentence-label nodes (doc-derived) make noisy code matches — the
        # docs matcher covers that material properly.
        if not _IDENTIFIER_NAME_RE.fullmatch(str(attrs.get("display") or node_id)):
            continue
        names = f"{node_id} {attrs.get('display', '')}".lower()
        prose = f"{attrs.get('description', '')} {attrs.get('file', '')}".lower()
        score = sum(3 for t in terms if _hits(t, names)) + sum(
            1 for t in terms if _hits(t, prose)
        )
        if score:
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

    for node_id in find_relevant_nodes(graph, question):
        node = g.get_node(graph, node_id)
        name = str(node.get("display") or node_id)
        file = str(node.get("file") or "")
        header = (
            f"### Code: {name} ({file})\n"
            f"Called by: {', '.join(node['callers'][:8]) or 'none'} | "
            f"Calls: {', '.join(node['callees'][:8]) or 'none'}"
        )
        snippet = g.get_source_snippet(node, max_lines=60)[:_SNIPPET_CHARS]
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
    pack = f"# Roger context: {question}\n\n{provenance}{context}\n\n## Sources\n" + "\n".join(
        f"- {s}" for s in sources
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
