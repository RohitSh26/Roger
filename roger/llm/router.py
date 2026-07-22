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


DEFAULT_NUM_CTX = 8192
MAX_LISTED_NEIGHBORS = 15


def _subgraph_char_budget(num_ctx: int) -> int:
    """Prompt budget for the serialized neighborhood, derived from num_ctx.

    Reserve ~2K tokens for the output (num_predict 1024), the system prompt,
    and the template; code identifiers tokenize at roughly 2.5 chars/token.
    At the default 8192 this yields ~15K chars — god nodes with hundreds of
    neighbors otherwise produce 200K+ char prompts Ollama rejects outright.
    """
    return max(4_000, (num_ctx - 2_048) * 5 // 2)


def _cap_list(items: list[str], limit: int = MAX_LISTED_NEIGHBORS) -> str:
    if not items:
        return "none"
    shown = ", ".join(items[:limit])
    hidden = len(items) - limit
    return f"{shown} (+{hidden} more)" if hidden > 0 else shown


def build_prompt(
    node: dict,
    graph: nx.DiGraph,
    difficulty: str,
    count: int,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """Fill the question-generation prompt with node + 1-hop subgraph context."""
    subgraph = g.get_subgraph(graph, node["id"], hops=1)
    return PROMPT_TEMPLATE.format(
        count=count,
        difficulty=difficulty,
        node_id=node["id"],
        description=node.get("description", ""),
        file=node.get("file", ""),
        community=node.get("community", ""),
        callers=_cap_list(node.get("callers", [])),
        callees=_cap_list(node.get("callees", [])),
        returns=node.get("returns", "") or "unknown",
        serialized_subgraph=g.serialize_subgraph(
            subgraph, max_chars=_subgraph_char_budget(num_ctx)
        ),
    )


def _normalize_options(options: object) -> Optional[dict[str, str]]:
    """Coerce model option variants (list form, lowercase keys) to {A..D: text}."""
    if isinstance(options, list) and len(options) >= 4:
        return {key: str(v) for key, v in zip(OPTION_KEYS, options)}
    if isinstance(options, dict):
        upper = {str(k).strip().upper(): str(v) for k, v in options.items()}
        if all(key in upper for key in OPTION_KEYS):
            return {key: upper[key] for key in OPTION_KEYS}
    return None


def _normalize_correct(item: dict, options: dict[str, str]) -> Optional[str]:
    """Coerce 'correct' variants ('b', 'B)', full option text, 'answer' key)."""
    raw = item.get("correct", item.get("answer"))
    if raw is None:
        return None
    text = str(raw).strip()
    letter = text[:1].upper()
    if letter in OPTION_KEYS and (len(text) <= 2 or text[1] in ").:. "):
        return letter
    for key, value in options.items():  # model repeated the option text verbatim
        if value.strip() == text:
            return key
    return letter if letter in OPTION_KEYS else None


def parse_questions(raw: dict, node_id: str, difficulty: str, tier: int) -> list[Question]:
    """Validate the model's JSON and build Question objects, skipping malformed items."""
    items = raw.get("questions")
    if not isinstance(items, list):
        raise ValueError(f"Local model response has no 'questions' list: {raw!r}")

    questions = []
    for item in items:
        if not isinstance(item, dict) or not item.get("question"):
            continue
        options = _normalize_options(item.get("options"))
        if options is None:
            continue
        correct = _normalize_correct(item, options)
        if correct is None:
            continue
        questions.append(
            Question(
                node_id=node_id,
                question=str(item["question"]),
                options=options,
                correct=correct,
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

    prompt = build_prompt(node, graph, difficulty, count, num_ctx=config.ollama.num_ctx)
    # A 1B model at temperature 0.7 occasionally emits an unusable shape;
    # a fresh sample usually fixes it, so retry before giving up.
    last_error: Exception = ValueError("no attempts made")
    for _ in range(3):
        try:
            raw = local.call_local(
                prompt,
                model=config.model.local,
                base_url=config.ollama.url,
                num_ctx=config.ollama.num_ctx,
            )
            return parse_questions(raw, node_id=node["id"], difficulty=difficulty, tier=1)
        except ValueError as exc:
            last_error = exc
    raise last_error
