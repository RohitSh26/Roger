"""Question generation orchestration: caching, tier routing, selection."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable, Iterator, Optional

import networkx as nx

from roger import graph as g
from roger.config import Config
from roger.llm.router import get_questions as get_questions_from_llm
from roger.models import Question
from roger.storage import cache_questions, get_cached_questions

# Bump when question generation changes materially (prompt style, filters).
# The version feeds the cache key, so everyone's stale-style questions
# regenerate automatically — no manual cache clearing across a team.
QUESTION_STYLE_VERSION = 14


def _model_id(config: Config) -> str:
    if config.model.provider == "azure-anthropic":
        return f"azure:{config.model.azure_deployment}"
    return f"ollama:{config.model.local}"


def hash_node(
    node: dict,
    subgraph: nx.DiGraph,
    snippet: str = "",
    difficulty: str = "",
    model: str = "",
) -> str:
    """SHA-256 cache key: what the developer's questions were built from.

    Includes only prompt-relevant, stable inputs: identity/signature facts,
    call relationships, the EXACT rendered source snippet, difficulty, and
    the generating model. Deliberately excludes `community` — Leiden ids
    renumber on every re-cluster, and including them would mass-orphan the
    cache on every background graph refresh.
    """
    stable = {
        "id": node.get("id"),
        "display": node.get("display"),
        "file": node.get("file"),
        "returns": node.get("returns"),
        "callers": node.get("callers", []),
        "callees": node.get("callees", []),
    }
    payload = "\n".join(
        [
            f"style-v{QUESTION_STYLE_VERSION}",
            f"difficulty={difficulty}",
            f"model={model}",
            json.dumps(stable, sort_keys=True, default=str),
            hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
            g.serialize_subgraph(subgraph),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_questions(
    all_questions: list[Question],
    count: int,
    god_node_ids: list[str],
) -> list[Question]:
    """Select `count` questions from the pool.

    Weight toward questions about god nodes, and ensure variety by
    round-robining across nodes rather than exhausting one node first.
    """
    by_node: dict[str, list[Question]] = {}
    for question in all_questions:
        by_node.setdefault(question.node_id, []).append(question)

    god_set = set(god_node_ids)
    # God nodes first (in god-list order), then the rest in pool order.
    ordered_nodes = [n for n in god_node_ids if n in by_node]
    ordered_nodes += [n for n in by_node if n not in god_set]

    selected: list[Question] = []
    seen_texts: set[str] = set()
    while len(selected) < count:
        progressed = False
        for node_id in ordered_nodes:
            if not by_node[node_id] or len(selected) >= count:
                continue
            question = by_node[node_id].pop(0)
            if question.question in seen_texts:  # dedupe identical questions
                continue
            seen_texts.add(question.question)
            selected.append(question)
            progressed = True
        if not progressed:
            break
    return selected


def iter_questions(
    node_ids: list[str],
    graph: nx.DiGraph,
    difficulty: str = "medium",
    count: int = 5,
    config: Optional[Config] = None,
    rng: Optional[random.Random] = None,
) -> Iterator[Question]:
    """Yield up to `count` questions lazily, as each becomes ready.

    For each node: hash the node + 1-hop subgraph, hit the cache, and on a
    miss route to Tier 0/1 and cache the result. One question per node is
    yielded first (variety), then leftovers fill the remainder. Because
    generation happens on demand, a UI can show question 1 while question 2
    generates behind the scenes. Node order is the caller's priority order
    (god nodes first for whole-repo quizzes).

    A node the model can't handle is skipped; if NO node yields anything,
    the last error is raised.
    """
    config = config or Config()
    rng = rng or random.Random()
    # Ask each node for a small batch rather than `count` apiece: the model
    # must fit its JSON inside num_predict tokens, and long identifiers make
    # 5-question responses truncate mid-array. A couple per node still gives
    # a pool ~2x the session size.
    per_node = min(5, max(2, -(-count // len(node_ids)))) if node_ids else count

    yielded = 0
    seen_texts: set[str] = set()
    leftovers: list[Question] = []
    last_error: Exception | None = None

    # Prepare every node once — snippet read, key, cache lookup — then order
    # cache hits first so early questions are instant while cold nodes
    # generate during answering. The SAME snippet string feeds both the key
    # and the prompt (no read-twice TOCTOU window).
    prepared = _prepare_nodes(node_ids, graph, difficulty, config)

    for item in prepared:
        if yielded >= count:
            break
        batch = item["batch"]
        if batch is None:
            try:
                batch = get_questions_from_llm(
                    item["node"], graph, difficulty, per_node,
                    config=config, snippet=item["snippet"],
                )
            except ValueError as exc:
                # One node the model can't write valid questions for must not
                # kill the whole quiz — skip it and quiz on the rest.
                last_error = exc
                continue
            cache_questions(
                item["key"], item["node_id"], difficulty, batch, _model_id(config)
            )

        fresh = [q for q in batch if q.question not in seen_texts]
        if not fresh:
            continue
        # A cached node always serves the same batch — shuffle so repeat
        # sessions don't repeat the identical question from it every time.
        rng.shuffle(fresh)
        seen_texts.add(fresh[0].question)
        yielded += 1
        yield fresh[0]
        for extra in fresh[1:]:
            seen_texts.add(extra.question)
            leftovers.append(extra)

    for question in leftovers:
        if yielded >= count:
            break
        yielded += 1
        yield question

    if yielded == 0 and last_error is not None:
        raise last_error


def _prepare_nodes(
    node_ids: list[str], graph: nx.DiGraph, difficulty: str, config: Config
) -> list[dict]:
    """One pass per node: snippet, cache key, cached batch. Cache hits first."""
    model = _model_id(config)
    prepared: list[dict] = []
    for node_id in node_ids:
        node = g.get_node(graph, node_id)
        subgraph = g.get_subgraph(graph, node_id, hops=1)
        snippet = g.get_source_snippet(node)
        key = hash_node(node, subgraph, snippet=snippet, difficulty=difficulty, model=model)
        cached = get_cached_questions(key)
        batch = (
            [q for q in cached if q.difficulty == difficulty] if cached is not None else []
        )
        prepared.append(
            {
                "node_id": node_id,
                "node": node,
                "snippet": snippet,
                "key": key,
                "batch": batch or None,
            }
        )
    prepared.sort(key=lambda item: item["batch"] is None)  # stable: hits first
    return prepared


def interleave_questions(
    stream: Iterable[Question], extras: list[Question]
) -> Iterator[Question]:
    """Weave pre-built questions (docs — instant) between streamed ones.

    Leads with a pre-built question when one exists, so the session's first
    question appears with zero generation wait.
    """
    remaining = list(extras)
    if remaining:
        yield remaining.pop(0)
    for question in stream:
        yield question
        if remaining:
            yield remaining.pop(0)
    yield from remaining


def order_cache_first(
    node_ids: list[str],
    graph: nx.DiGraph,
    difficulty: str,
    config: Optional[Config] = None,
) -> list[str]:
    """Order nodes so cache hits come first (iter_questions now does this
    internally; kept as a public helper)."""
    prepared = _prepare_nodes(node_ids, graph, difficulty, config or Config())
    return [item["node_id"] for item in prepared]


def generate_questions(
    node_ids: list[str],
    graph: nx.DiGraph,
    difficulty: str = "medium",
    count: int = 5,
    config: Optional[Config] = None,
) -> list[Question]:
    """Materialized form of iter_questions — for the web page and callers
    that need every question up front."""
    return list(iter_questions(node_ids, graph, difficulty, count, config))
