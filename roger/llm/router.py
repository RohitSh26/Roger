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
You are writing a code-comprehension quiz for a developer who works in this
codebase, in the style of a professional certification exam: scenario-based,
testing real understanding, never trivia.

CODE CONTEXT:
Name: {name}
File: {file}
Called by: {callers}
Calls: {callees}
{returns_line}{source_block}Related code (name (file), then relationships):
{serialized_subgraph}

THE MEMORY RULE — the most important rule:
Never ask the developer to recall structural facts from memory (who calls
what, file names, signatures). If a question needs such a fact, state the
fact inside the question. Test only what a developer who once understood
this code would still know a month later: purpose, behavior, design intent,
and consequences.

QUESTION TYPES — write {count} questions, each a different type; skip any
type this context cannot support:
1. Behavior: what does {name} do in a specific situation visible in its
   SOURCE (a particular input, a branch, an early return, an error path)?
2. Purpose: what problem does {name} solve for the code that uses it?
   Wrong options: the real purposes of other related code.
3. Design: why is {name} written the way it is (a guard, a delegation, an
   ordering, a data structure) — what would go wrong without it?
4. Consequence: state a structural fact in the question ("{name} is used
   by ..."), then ask what a proposed change would mean — the fact is
   given, the judgment is tested.
5. Ripple (hard difficulty only): given the dependencies shown, where would
   a failure in one of {name}'s collaborators surface, and why there?

DIFFICULTY: {difficulty} — medium prefers types 1, 2 and 4; hard prefers
types 3 and 5 with design trade-offs.

RULES — every question must pass all five:
- Grounded: the correct answer is provable from the SOURCE and CODE CONTEXT
  above. Never invent runtime behavior, error messages, or values that are
  not shown.
- Cover test: a developer who understands this code can answer before
  reading the options.
- No giveaways: the question never contains or paraphrases its own answer,
  and the correct option never merely restates the name {name} — a
  self-answering question is worthless.
- Honest options: wrong options are realistic-sounding claims about this
  code that the SOURCE rules out. All four options share the same
  grammatical form and similar length. Never use "all/none of the above".
- Developer voice: plain code-review language. Never say "node", "graph",
  "community", "neighboring", or refer to this prompt.

{example_block}Respond with JSON only, no other text:
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
MAX_SOURCE_CHARS = 4_000


def _subgraph_char_budget(num_ctx: int) -> int:
    """Prompt budget for the serialized neighborhood, derived from num_ctx.

    Reserve ~2.5K tokens for the output (num_predict 1024), the system prompt,
    and the exam-style template; code identifiers tokenize at roughly
    2.5 chars/token. At the default 8192 this yields ~14K chars — god nodes
    with hundreds of neighbors otherwise produce 200K+ char prompts Ollama
    rejects outright.
    """
    return max(4_000, (num_ctx - 2_560) * 5 // 2)


def _cap_list(items: list[str], limit: int = MAX_LISTED_NEIGHBORS) -> str:
    if not items:
        return "none"
    shown = ", ".join(items[:limit])
    hidden = len(items) - limit
    return f"{shown} (+{hidden} more)" if hidden > 0 else shown


def _example_block(name: str, caller_names: list[str], callee_names: list[str]) -> str:
    """A worked question shape built from the node's own relationships.

    Small models copy worked examples verbatim — a fixed example about
    foreign code produced quizzes about that foreign code. Built from the
    real names, copying the example still yields a grounded question. Only
    the question shape is shown, never answer mechanics, so nothing here
    can leak an answer format into question text.
    """
    if callee_names:
        return (
            "WORKED EXAMPLE of a good question shape (real names, vary the angle):\n"
            f'"Why does {name} hand part of its work to {callee_names[0]} instead\n'
            'of doing it inline?"\n\n'
        )
    if caller_names:
        return (
            "WORKED EXAMPLE of a good question shape (real names, vary the angle):\n"
            f'"What does {caller_names[0]} rely on {name} to take care of?"\n\n'
        )
    return ""


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

    name = str(node.get("display") or node["id"])
    caller_names = names(node.get("callers", []))
    callee_names = names(node.get("callees", []))
    returns = node.get("returns")

    # Real source makes the difference between comprehension questions and
    # structure trivia. It gets first claim on the context budget; the
    # serialized neighborhood absorbs whatever remains.
    snippet = g.get_source_snippet(node)[:MAX_SOURCE_CHARS]
    source_block = (
        f"SOURCE (excerpt from {node.get('file', '')} {node.get('source_location', '')}):\n"
        f"{snippet}\n\n"
        if snippet
        else ""
    )
    subgraph_budget = max(3_000, _subgraph_char_budget(num_ctx) - len(source_block))

    return PROMPT_TEMPLATE.format(
        count=count,
        difficulty=difficulty,
        name=name,
        file=node.get("file", ""),
        callers=_cap_list(caller_names),
        callees=_cap_list(callee_names),
        returns_line=f"Returns: {returns}\n" if returns else "",
        source_block=source_block,
        serialized_subgraph=g.serialize_subgraph(
            subgraph, max_chars=subgraph_budget, labels=True
        ),
        example_block=_example_block(name, caller_names, callee_names),
    )


def _clean_option(value: object) -> str:
    return str(value).strip().rstrip(";,").strip()


def _normalize_options(options: object) -> Optional[dict[str, str]]:
    """Coerce model option variants (list form, lowercase keys) to {A..D: text}."""
    if isinstance(options, list) and len(options) >= 4:
        return {key: _clean_option(v) for key, v in zip(OPTION_KEYS, options)}
    if isinstance(options, dict):
        upper = {str(k).strip().upper(): _clean_option(v) for k, v in options.items()}
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


# Quiz-metadata vocabulary leaking into a question text means the model
# copied the answer or the option list into the question itself.
_META_MARKER_RE = re.compile(
    r"(?i)(\boptions?\s*:|\bcorrect\s*:|\banswer\s+is\b|\bright\s+answer\b|\bwrong\s+option)"
)


def _is_giveaway(question_text: str, options: dict[str, str], correct: str, subject: Optional[str]) -> bool:
    """True for self-answering questions a non-reader could guess.

    Catches the two shapes developers complained about: the correct option
    restating the subject's name back ("What is usage()?" → "usage"), and
    the correct answer appearing verbatim inside the question text.
    """
    if _META_MARKER_RE.search(question_text):
        return True
    correct_norm = _norm_text(options[correct])
    if not correct_norm:
        return True
    if len(correct_norm) >= 12 and correct_norm in _norm_text(question_text):
        return True
    if subject:
        subject_tokens = {t for t in _norm_text(subject).split() if len(t) > 2}
        if subject_tokens and subject_tokens <= set(correct_norm.split()):
            return True
    # The classic amateur-quiz tell: the correct option is far longer and more
    # detailed than every distractor. (Floor of 20 chars so short name-style
    # options are never rejected.)
    distractor_lens = [len(_norm_text(v)) for k, v in options.items() if k != correct]
    if (
        distractor_lens
        and len(correct_norm) >= 20
        and len(correct_norm) > 2 * max(distractor_lens)
    ):
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
