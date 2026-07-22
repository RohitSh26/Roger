"""Tier routing: simple → Tier 0 templates, medium/hard → Tier 1 local Ollama."""

from __future__ import annotations

import re
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
You are quizzing a developer on code they work with. Using the context below,
generate {count} multiple-choice questions at {difficulty} difficulty.

CODE CONTEXT:
Name: {name}
File: {file}
Called by: {callers}
Calls: {callees}
{returns_line}Related code (name (file), then relationships):
{serialized_subgraph}

DIFFICULTY GUIDE:
- medium: test understanding of what this code does and how it connects to the related code
- hard: test failure modes, edge cases, architectural trade-offs, design intent

RULES:
- Write like a developer talking about code. Never use the words "node",
  "neighboring", "graph", "community", or "metadata" in questions or options.
- Refer to code only by the names shown above (e.g. {name}), never by
  underscore identifiers or internal ids.
- Ask about what the code does, why it exists, and how it interacts with the
  related code — never about this prompt's structure or its fields.
- The question must not give away the answer: a developer who never read this
  code should not be able to guess it from wording alone. Never restate the
  code's name as the correct option.
- All four options must be plausible statements about the code. Exactly one
  is correct.

EXAMPLES:
- BAD: "What does {name} do?" with the correct option "It is {name}." —
  self-answering, tests nothing.
- GOOD: "What happens in {name} when its dependency is unavailable?"
- GOOD: "Why does {name} call the code it calls, instead of doing that work itself?"
- GOOD: "Which change would break the callers of {name}?"

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
    """Fill the question-generation prompt with node + 1-hop subgraph context.

    Everything is rendered with human-readable names — the model mirrors the
    prompt's vocabulary, so slugs or graph jargon here become slugs and
    "neighboring node" questions in the quiz.
    """
    subgraph = g.get_subgraph(graph, node["id"], hops=1)

    def names(node_ids: list[str]) -> list[str]:
        return [str(graph.nodes[n].get("display") or n) for n in node_ids]

    returns = node.get("returns")
    return PROMPT_TEMPLATE.format(
        count=count,
        difficulty=difficulty,
        name=str(node.get("display") or node["id"]),
        file=node.get("file", ""),
        callers=_cap_list(names(node.get("callers", []))),
        callees=_cap_list(names(node.get("callees", []))),
        returns_line=f"Returns: {returns}\n" if returns else "",
        serialized_subgraph=g.serialize_subgraph(
            subgraph, max_chars=_subgraph_char_budget(num_ctx), labels=True
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


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _is_giveaway(question_text: str, options: dict[str, str], correct: str, subject: Optional[str]) -> bool:
    """True for self-answering questions a non-reader could guess.

    Catches the two shapes developers complained about: the correct option
    restating the subject's name back ("What is usage()?" → "usage"), and
    the correct answer appearing verbatim inside the question text.
    """
    correct_norm = _norm_text(options[correct])
    if not correct_norm:
        return True
    if len(correct_norm) >= 12 and correct_norm in _norm_text(question_text):
        return True
    if subject:
        subject_tokens = {t for t in _norm_text(subject).split() if len(t) > 2}
        if subject_tokens and subject_tokens <= set(correct_norm.split()):
            return True
    return False


def parse_questions(
    raw: dict,
    node_id: str,
    difficulty: str,
    tier: int,
    subject: Optional[str] = None,
) -> list[Question]:
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
        if _is_giveaway(str(item["question"]), options, correct, subject):
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
            return parse_questions(
                raw,
                node_id=node["id"],
                difficulty=difficulty,
                tier=1,
                subject=str(node.get("display") or node["id"]),
            )
        except ValueError as exc:
            last_error = exc
    raise last_error
