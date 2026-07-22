"""Tier 0 question generation: templates over graph metadata, zero LLM calls.

Distractors are sampled from nodes in the same Leiden community as the target
node (plausible but wrong). Communities with fewer than 3 usable candidates
fall back to sampling from the full graph. Options are always shuffled.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import networkx as nx

from roger.models import Question

OPTION_KEYS = ("A", "B", "C", "D")


def _community_pool(node: dict, graph: nx.DiGraph) -> list[str]:
    """Other node IDs in the target node's community."""
    community = node.get("community")
    return sorted(
        n
        for n, attrs in graph.nodes(data=True)
        if n != node["id"] and attrs.get("community") == community
    )


def _pick_distractors(
    preferred: list[str],
    fallback: list[str],
    exclude: set[str],
    rng: random.Random,
    needed: int = 3,
) -> Optional[list[str]]:
    """Sample distractors from `preferred`, topping up from `fallback` if short."""
    primary = [c for c in preferred if c not in exclude]
    if len(primary) >= needed:
        return rng.sample(primary, needed)
    secondary = [c for c in fallback if c not in exclude and c not in primary]
    pool = primary + rng.sample(secondary, min(needed - len(primary), len(secondary)))
    return pool if len(pool) == needed else None


def _make_question(
    node: dict,
    text: str,
    correct_value: str,
    distractors: list[str],
    explanation: str,
    rng: random.Random,
) -> Question:
    """Assemble a shuffled 4-option MCQ."""
    values = [correct_value, *distractors]
    rng.shuffle(values)
    options = dict(zip(OPTION_KEYS, values))
    correct_key = next(k for k, v in options.items() if v == correct_value)
    return Question(
        node_id=node["id"],
        question=text,
        options=options,
        correct=correct_key,
        explanation=explanation,
        difficulty="simple",
        tier=0,
    )


def caller_question(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> Optional[Question]:
    """Which of the following calls `{node}()`?"""
    rng = rng or random.Random()
    callers = node.get("callers") or []
    if not callers:
        return None
    correct = rng.choice(callers)
    distractors = _pick_distractors(
        preferred=_community_pool(node, graph),
        fallback=sorted(graph.nodes),
        exclude={node["id"], *callers},
        rng=rng,
    )
    if distractors is None:
        return None
    return _make_question(
        node,
        f"Which of the following calls `{node['id']}()`?",
        correct,
        distractors,
        f"`{correct}` is a direct caller of `{node['id']}` in the code graph.",
        rng,
    )


def dependency_question(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> Optional[Question]:
    """What does `{node}()` directly call?"""
    rng = rng or random.Random()
    callees = node.get("callees") or []
    if not callees:
        return None
    correct = rng.choice(callees)
    distractors = _pick_distractors(
        preferred=_community_pool(node, graph),
        fallback=sorted(graph.nodes),
        exclude={node["id"], *callees},
        rng=rng,
    )
    if distractors is None:
        return None
    return _make_question(
        node,
        f"What does `{node['id']}()` directly call?",
        correct,
        distractors,
        f"`{node['id']}` calls `{correct}` directly, per the code graph.",
        rng,
    )


def module_question(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> Optional[Question]:
    """Which module/layer does `{node}` belong to?"""
    rng = rng or random.Random()
    community = node.get("community")
    if not community:
        return None
    all_communities = sorted(
        {
            str(attrs["community"])
            for _, attrs in graph.nodes(data=True)
            if attrs.get("community")
        }
    )
    distractors = _pick_distractors(
        preferred=all_communities, fallback=[], exclude={str(community)}, rng=rng
    )
    if distractors is None:
        return None
    return _make_question(
        node,
        f"Which module/layer does `{node['id']}` belong to?",
        str(community),
        distractors,
        f"`{node['id']}` is part of the `{community}` community in the code graph.",
        rng,
    )


def return_type_question(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> Optional[Question]:
    """What does `{node}()` return?"""
    rng = rng or random.Random()
    returns = node.get("returns")
    if not returns:
        return None
    community_returns = sorted(
        {
            str(graph.nodes[n]["returns"])
            for n in _community_pool(node, graph)
            if graph.nodes[n].get("returns")
        }
    )
    all_returns = sorted(
        {
            str(attrs["returns"])
            for _, attrs in graph.nodes(data=True)
            if attrs.get("returns")
        }
    )
    distractors = _pick_distractors(
        preferred=community_returns, fallback=all_returns, exclude={str(returns)}, rng=rng
    )
    if distractors is None:
        return None
    return _make_question(
        node,
        f"What does `{node['id']}()` return?",
        str(returns),
        distractors,
        f"`{node['id']}` returns `{returns}`, per the code graph.",
        rng,
    )


def location_question(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> Optional[Question]:
    """In which file is `{node}` defined?"""
    rng = rng or random.Random()
    file = node.get("file")
    if not file:
        return None
    community_files = sorted(
        {
            str(graph.nodes[n]["file"])
            for n in _community_pool(node, graph)
            if graph.nodes[n].get("file")
        }
    )
    all_files = sorted(
        {str(attrs["file"]) for _, attrs in graph.nodes(data=True) if attrs.get("file")}
    )
    distractors = _pick_distractors(
        preferred=community_files, fallback=all_files, exclude={str(file)}, rng=rng
    )
    if distractors is None:
        return None
    return _make_question(
        node,
        f"In which file is `{node['id']}` defined?",
        str(file),
        distractors,
        f"`{node['id']}` is defined in `{file}`.",
        rng,
    )


_TEMPLATES: tuple[Callable[..., Optional[Question]], ...] = (
    caller_question,
    dependency_question,
    module_question,
    return_type_question,
    location_question,
)


def build_from_graph(
    node: dict, graph: nx.DiGraph, rng: Optional[random.Random] = None
) -> list[Question]:
    """Generate 1-5 simple questions from graph node metadata (Tier 0)."""
    rng = rng or random.Random()
    questions = []
    for template in _TEMPLATES:
        question = template(node, graph, rng=rng)
        if question is not None:
            questions.append(question)
    return questions
