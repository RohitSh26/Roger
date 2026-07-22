"""Tier routing: simple → Tier 0 templates, medium/hard → Tier 1 local Ollama."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from roger import graph as g
from roger.config import Config
from roger.exceptions import OllamaNotRunningError
from roger.llm import local
from roger.models import Question
from roger.templates import OPTION_KEYS, build_from_graph

DIFFICULTIES = ("simple", "medium", "hard")

PROMPT_TEMPLATE = """\
Given the following code graph context, generate {count} quiz questions
at {difficulty} difficulty level.

GRAPH CONTEXT:
Node: {node_id}
Description: {description}
File: {file}
Module/Community: {community}
Called by: {callers}
Calls: {callees}
Returns: {returns}

Neighboring nodes:
{serialized_subgraph}

DIFFICULTY GUIDE:
- medium: test understanding of what this code does and how it connects to neighbors
- hard: test failure modes, edge cases, architectural trade-offs, design intent

Generate exactly {count} questions. Respond with JSON only, no other text:
{{
  "questions": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct": "B",
      "explanation": "One or two sentences explaining why the answer is correct."
    }}
  ]
}}
"""


def build_prompt(node: dict, graph: nx.DiGraph, difficulty: str, count: int) -> str:
    """Fill the question-generation prompt with node + 1-hop subgraph context."""
    subgraph = g.get_subgraph(graph, node["id"], hops=1)
    return PROMPT_TEMPLATE.format(
        count=count,
        difficulty=difficulty,
        node_id=node["id"],
        description=node.get("description", ""),
        file=node.get("file", ""),
        community=node.get("community", ""),
        callers=", ".join(node.get("callers", [])) or "none",
        callees=", ".join(node.get("callees", [])) or "none",
        returns=node.get("returns", "") or "unknown",
        serialized_subgraph=g.serialize_subgraph(subgraph),
    )


def parse_questions(raw: dict, node_id: str, difficulty: str, tier: int) -> list[Question]:
    """Validate the model's JSON and build Question objects, skipping malformed items."""
    items = raw.get("questions")
    if not isinstance(items, list):
        raise ValueError(f"Local model response has no 'questions' list: {raw!r}")

    questions = []
    for item in items:
        if not isinstance(item, dict):
            continue
        options = item.get("options")
        correct = item.get("correct")
        if (
            not isinstance(options, dict)
            or sorted(options) != sorted(OPTION_KEYS)
            or correct not in OPTION_KEYS
            or not item.get("question")
        ):
            continue
        questions.append(
            Question(
                node_id=node_id,
                question=str(item["question"]),
                options={k: str(v) for k, v in options.items()},
                correct=str(correct),
                explanation=str(item.get("explanation", "")),
                difficulty=difficulty,
                tier=tier,
            )
        )
    if not questions:
        raise ValueError(f"Local model returned no valid questions for node {node_id}")
    return questions


def get_questions(
    node: dict,
    graph: nx.DiGraph,
    difficulty: str,
    count: int,
    config: Optional[Config] = None,
) -> list[Question]:
    """Route to the correct tier for one node's questions.

    Tier 0: difficulty == 'simple' → templates (zero LLM)
    Tier 1: difficulty == 'medium' or 'hard' → local Ollama
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty {difficulty!r}; expected one of {DIFFICULTIES}")

    if difficulty == "simple":
        return build_from_graph(node, graph)

    config = config or Config()
    if not local.is_ollama_running(config.ollama.url):
        raise OllamaNotRunningError(local.OLLAMA_NOT_RUNNING_MSG)

    prompt = build_prompt(node, graph, difficulty, count)
    raw = local.call_local(prompt, model=config.model.local, base_url=config.ollama.url)
    return parse_questions(raw, node_id=node["id"], difficulty=difficulty, tier=1)
