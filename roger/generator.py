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
QUESTION_STYLE_VERSION = 11


def hash_node(node: dict, subgraph: nx.DiGraph) -> str:
    """SHA-256 of node attributes + serialized subgraph. Cache key.

    Stable: dict keys are sorted and the subgraph serializer is deterministic,
    so identical code always produces an identical hash.
    """
    canonical_node = json.dumps(node, sort_keys=True, default=str)
    payload = (
        f"style-v{QUESTION_STYLE_VERSION}\n"
        + canonical_node
        + "\n"
        + g.serialize_subgraph(subgraph)
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

    for node_id in node_ids:
        if yielded >= count:
            break
        node = g.get_node(graph, node_id)
        subgraph = g.get_subgraph(graph, node_id, hops=1)
        node_hash = hash_node(node, subgraph)

        cached = get_cached_questions(node_hash)
        batch: list[Question] = []
        if cached is not None:
            batch = [q for q in cached if q.difficulty == difficulty]
        if not batch:
            try:
                batch = get_questions_from_llm(node, graph, difficulty, per_node, config=config)
            except ValueError as exc:
                # One node the model can't write valid questions for must not
                # kill the whole quiz — skip it and quiz on the rest.
                last_error = exc
                continue
            cache_questions(node_hash, node_id, difficulty, batch, config.model.local)

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


def interleave_questions(
    stream: Iterable[Question], extras: list[Question]
) -> Iterator[Question]:
    """Weave pre-built questions (docs — instant) between streamed ones."""
    remaining = list(extras)
    for question in stream:
        yield question
        if remaining:
            yield remaining.pop(0)
    yield from remaining


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
