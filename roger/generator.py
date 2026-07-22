"""Question generation orchestration: caching, tier routing, selection."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import networkx as nx

from roger import graph as g
from roger.config import Config
from roger.llm.router import get_questions as get_questions_from_llm
from roger.models import Question
from roger.storage import cache_questions, get_cached_questions


def hash_node(node: dict, subgraph: nx.DiGraph) -> str:
    """SHA-256 of node attributes + serialized subgraph. Cache key.

    Stable: dict keys are sorted and the subgraph serializer is deterministic,
    so identical code always produces an identical hash.
    """
    canonical_node = json.dumps(node, sort_keys=True, default=str)
    payload = canonical_node + "\n" + g.serialize_subgraph(subgraph)
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


def generate_questions(
    node_ids: list[str],
    graph: nx.DiGraph,
    difficulty: str = "medium",
    count: int = 5,
    config: Optional[Config] = None,
) -> list[Question]:
    """Main entry point for question generation.

    For each node: hash the node + 1-hop subgraph, hit the cache, and on a
    miss route to Tier 0/1 and cache the result. Then select `count`
    questions from the pool, weighted toward god nodes.
    """
    config = config or Config()
    pool: list[Question] = []

    for node_id in node_ids:
        node = g.get_node(graph, node_id)
        subgraph = g.get_subgraph(graph, node_id, hops=1)
        node_hash = hash_node(node, subgraph)

        cached = get_cached_questions(node_hash)
        if cached is not None:
            matching = [q for q in cached if q.difficulty == difficulty]
            if matching:
                pool.extend(matching)
                continue

        questions = get_questions_from_llm(node, graph, difficulty, count, config=config)
        cache_questions(node_hash, node_id, difficulty, questions, config.model.local)
        pool.extend(questions)

    god_node_ids = g.get_god_nodes(graph) if config.graph.god_node_weight else []
    return select_questions(pool, count, god_node_ids)
