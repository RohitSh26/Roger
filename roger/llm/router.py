"""Tier routing: simple → Tier 0 templates, medium/hard → Tier 1 local Ollama."""

from __future__ import annotations

import random
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

The developer answering will see the same SOURCE excerpt beside each
question, so questions may refer to it directly ("this function", "the
early return", "the loop at the end").

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
6. Trace: pick a small concrete input and ask what this code returns or does
   with it — every option must be a concrete value or outcome, and the
   SOURCE must fully determine the answer.

DIFFICULTY: {difficulty} — medium prefers types 1, 2, 4 and 6; hard prefers
types 3, 5 and 6 with design trade-offs.

RULES — every question must pass all of these:
- Grounded: the correct answer is provable from the SOURCE and CODE CONTEXT
  above. Never invent runtime behavior, error messages, or values that are
  not shown.
- In scope: ask only about {name} and code visible in SOURCE. Never ask what
  a caller or callee does internally — neither you nor the developer can see
  its code here. Callers and callees may only appear as stated facts.
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
MAX_DISPLAY_SNIPPET_LINES = 24  # what the developer sees beside the question

CLOZE_PROMPT = """\
One line of real code has been removed from this excerpt of {file}:

{blanked}

The removed line is:
{real_line}

Write three alternative lines that would look plausible in that spot but are
NOT what the code does — the kind of line a developer who only skimmed this
code might believe. Match the style. Respond with JSON only, no other text:
{{"alternatives": ["...", "...", "..."]}}
"""


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
    snippet: Optional[str] = None,
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
    if snippet is None:
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

# Code identifiers a question refers to: `backticked` names and call-style
# tokens like _run_case().
_CODE_REF_RE = re.compile(r"`([^`]+)`|\b([A-Za-z_][\w.]*)\s*\(\)")


def _referenced_code_names(question_text: str) -> set[str]:
    names = set()
    for backticked, called in _CODE_REF_RE.findall(question_text):
        raw = (backticked or called).strip().strip("`")
        raw = raw.removesuffix("()").strip()
        if raw:
            names.add(raw.split(".")[-1].strip("_() "))
    return {n for n in names if n}


def _is_out_of_scope(question_text: str, subject: Optional[str], snippet: str) -> bool:
    """True if the question asks about code the developer cannot see.

    The model knows callers/callees by name only; a question about what one
    of them does internally is unanswerable from the shown snippet — the
    exact 'asks about _run_case() while showing FakeSearchClient' failure.
    """
    if not snippet:
        return False
    subject_base = ""
    if subject:
        subject_base = subject.strip().removesuffix("()").split(".")[-1].strip("_() ")
    for name in _referenced_code_names(question_text):
        if subject_base and (name in subject_base or subject_base in name):
            continue
        if name not in snippet:
            return True
    return False


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
    snippet: str = "",
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
        if _is_out_of_scope(str(item["question"]), subject, snippet):
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


_CLOZE_SKIP_PREFIXES = (
    "#", "//", "/*", "*", '"""', "'''",
    "def ", "class ", "import ", "from ", "func ", "fn ", "function ",
)


def _pick_cloze_line(lines: list[str], rng: Optional[random.Random] = None) -> Optional[int]:
    """Index of a line worth blanking: real logic, not signatures or comments."""
    rng = rng or random.Random()
    candidates = []
    for index, line in enumerate(lines):
        if index == 0:  # usually the definition line — blanking it names nothing
            continue
        stripped = line.strip()
        if len(stripped) < 10 or stripped.startswith(_CLOZE_SKIP_PREFIXES):
            continue
        if "(" in stripped or "=" in stripped or stripped.startswith(("return", "raise", "yield")):
            candidates.append(index)
    return rng.choice(candidates) if candidates else None


def build_cloze_question(
    node: dict,
    snippet: str,
    difficulty: str,
    config: Config,
    rng: Optional[random.Random] = None,
) -> Optional[Question]:
    """Fill-in-the-blank over real source: correctness holds by construction.

    We remove a real line, so the right answer is ground truth we control —
    the model only invents the plausible-but-wrong alternatives. Returns
    None when the snippet has no blankable line or the model can't produce
    three distinct alternatives.
    """
    rng = rng or random.Random()
    lines = snippet.splitlines()[:MAX_DISPLAY_SNIPPET_LINES]
    index = _pick_cloze_line(lines, rng)
    if index is None:
        return None
    real_line = lines[index]
    indent = real_line[: len(real_line) - len(real_line.lstrip())]
    blanked_lines = [*lines]
    blanked_lines[index] = f"{indent}________________________________"
    blanked = "\n".join(blanked_lines)

    try:
        raw = local.call_local(
            CLOZE_PROMPT.format(file=node.get("file", ""), blanked=blanked, real_line=real_line),
            model=config.model.local,
            base_url=config.ollama.url,
            num_ctx=config.ollama.num_ctx,
        )
    except ValueError:
        return None

    # Models answer either {"alternatives": [...]} or a bare JSON list.
    alternatives = raw if isinstance(raw, list) else (
        raw.get("alternatives") if isinstance(raw, dict) else None
    )
    if not isinstance(alternatives, list):
        return None
    real_norm = " ".join(real_line.split())
    distractors: list[str] = []
    for alt in alternatives:
        text = " ".join(str(alt).split())
        if text and text != real_norm and text not in distractors:
            distractors.append(text)
    if len(distractors) < 3:
        return None

    name = str(node.get("display") or node["id"])
    values = [real_norm, *distractors[:3]]
    rng.shuffle(values)
    options = dict(zip(("A", "B", "C", "D"), values))
    correct = next(k for k, v in options.items() if v == real_norm)
    return Question(
        node_id=node["id"],
        question=f"One line in `{name}` is blanked out below. Which is the real line?",
        options=options,
        correct=correct,
        explanation=f"That is the actual line in {node.get('file', 'the source')}.",
        difficulty=difficulty,
        tier=1,
        snippet=blanked,
    )


# Textual mutations for spot-the-alteration questions. Plain substring pairs
# applied once — no AST, language-agnostic, plausible by design.
_MUTATION_RULES: tuple[tuple[str, str], ...] = (
    (" == ", " != "),
    (" != ", " == "),
    (" >= ", " <= "),
    (" <= ", " >= "),
    (" and ", " or "),
    (" or ", " and "),
    ("&&", "||"),
    ("||", "&&"),
    ("max(", "min("),
    ("min(", "max("),
    ("True", "False"),
    ("False", "True"),
    (" += ", " -= "),
    (" -= ", " += "),
    (" + ", " - "),
    (" - ", " + "),
)


def _mutate_line(line: str, rng: random.Random) -> Optional[str]:
    rules = list(_MUTATION_RULES)
    rng.shuffle(rules)
    for old, new in rules:
        if old in line:
            return line.replace(old, new, 1)
    return None


def build_mutant_question(
    node: dict,
    snippet: str,
    difficulty: str,
    rng: Optional[random.Random] = None,
) -> Optional[Question]:
    """Spot-the-alteration over real source — zero LLM calls, truth by construction.

    One line is silently altered (operator flip, and/or swap, boundary
    change); the developer must recognize which shown line is not what
    their code really does. The answer key cannot be wrong: we made the
    alteration ourselves.
    """
    rng = rng or random.Random()
    lines = snippet.splitlines()[:MAX_DISPLAY_SNIPPET_LINES]

    mutable = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 10 or stripped.startswith(_CLOZE_SKIP_PREFIXES):
            continue
        mutated = _mutate_line(line, rng)
        if mutated is not None and mutated != line:
            mutable.append((index, mutated))
    if not mutable:
        return None
    index, mutated_line = rng.choice(mutable)

    shown = [*lines]
    shown[index] = mutated_line
    correct_norm = " ".join(mutated_line.split())

    distractor_pool = []
    for j, line in enumerate(shown):
        if j == index:
            continue
        stripped = line.strip()
        if len(stripped) < 10 or stripped.startswith(_CLOZE_SKIP_PREFIXES):
            continue
        norm = " ".join(stripped.split())
        if norm != correct_norm and norm not in distractor_pool:
            distractor_pool.append(norm)
    if len(distractor_pool) < 3:
        return None

    name = str(node.get("display") or node["id"])
    values = [correct_norm, *rng.sample(distractor_pool, 3)]
    rng.shuffle(values)
    options = dict(zip(("A", "B", "C", "D"), values))
    correct = next(k for k, v in options.items() if v == correct_norm)
    return Question(
        node_id=node["id"],
        question=(
            f"One line in the code below was altered from the real implementation "
            f"of `{name}`. Which shown line is NOT what the real code does?"
        ),
        options=options,
        correct=correct,
        explanation=f"The real code reads: {lines[index].strip()}",
        difficulty=difficulty,
        tier=0,  # constructed, no LLM involved
        snippet="\n".join(shown),
    )


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

    snippet = g.get_source_snippet(node)[:MAX_SOURCE_CHARS]
    display_snippet = "\n".join(snippet.splitlines()[:MAX_DISPLAY_SNIPPET_LINES])

    # Construction-grounded questions when the source supports them — their
    # correct answers are real lines we removed or altered ourselves,
    # immune to model hallucination.
    cloze = build_cloze_question(node, snippet, difficulty, config) if snippet else None
    mutant = build_mutant_question(node, snippet, difficulty) if snippet else None
    constructed = [q for q in (cloze, mutant) if q is not None]
    llm_count = max(1, count - len(constructed))

    prompt = build_prompt(
        node, graph, difficulty, llm_count, num_ctx=config.ollama.num_ctx, snippet=snippet
    )
    # A small model occasionally emits an unusable shape; a fresh sample
    # usually fixes it, so retry before giving up.
    last_error: Exception = ValueError("no attempts made")
    for _ in range(3):
        try:
            raw = local.call_local(
                prompt,
                model=config.model.local,
                base_url=config.ollama.url,
                num_ctx=config.ollama.num_ctx,
            )
            questions = parse_questions(
                raw,
                node_id=node["id"],
                difficulty=difficulty,
                tier=1,
                subject=str(node.get("display") or node["id"]),
                snippet=snippet,
            )
            for question in questions:
                question.snippet = display_snippet
            combined = questions + constructed
            # Shuffle so downstream round-robin selection surfaces a mix of
            # formats, not always the same kind from every node.
            random.shuffle(combined)
            return combined
        except ValueError as exc:
            last_error = exc
    if constructed:  # the LLM failed but the constructed questions are solid
        return constructed
    raise last_error
