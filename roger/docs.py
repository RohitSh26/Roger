"""Doc-based quiz questions — constructed from markdown structure, zero LLM.

The documents already contain the answers (ADR decisions, table cells, the
sentences themselves), so questions are assembled by extraction: the correct
option is copied from the doc, the wrong options are real content from
elsewhere in the corpus. Instant, deterministic-per-seed, hallucination-free.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from roger.models import Question

DEFAULT_DOC_PATHS = ["docs", "README.md"]
MAX_DOC_FILES = 500
_SNIPPET_LINES = 18

# Never quiz on tool configs, vendored trees, or generated agent files.
_EXCLUDED_SEGMENTS = {"node_modules", "vendor", "dist", "build", "target"}


@dataclass
class DocSection:
    file: str      # repo-relative path
    heading: str
    text: str


def discover_doc_files(
    paths: Optional[list[str]] = None, repo_root: Path = Path(".")
) -> list[Path]:
    """Markdown files under the configured paths, minus noise directories."""
    found: list[Path] = []
    for entry in paths or DEFAULT_DOC_PATHS:
        root = repo_root / entry
        if root.is_file() and root.suffix.lower() in (".md", ".markdown"):
            found.append(root)
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            parts = path.relative_to(repo_root).parts
            if any(p.startswith(".") or p in _EXCLUDED_SEGMENTS for p in parts):
                continue
            found.append(path)
            if len(found) >= MAX_DOC_FILES:
                return found
    return found


def split_sections(file: str, text: str) -> list[DocSection]:
    """Split a markdown document on headings; drop trivial sections."""
    sections: list[DocSection] = []
    heading = "(top)"
    lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            body = "\n".join(lines).strip()
            if len(body) >= 80:
                sections.append(DocSection(file=file, heading=heading, text=body))
            heading = line.lstrip("#").strip()
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if len(body) >= 80:
        sections.append(DocSection(file=file, heading=heading, text=body))
    return sections


def _make_mcq(
    node_id: str,
    text: str,
    correct: str,
    distractors: list[str],
    explanation: str,
    difficulty: str,
    snippet: str,
    rng: random.Random,
) -> Question:
    values = [correct, *distractors]
    rng.shuffle(values)
    options = dict(zip(("A", "B", "C", "D"), values))
    key = next(k for k, v in options.items() if v == correct)
    return Question(
        node_id=node_id,
        question=text,
        options=options,
        correct=key,
        explanation=explanation,
        difficulty=difficulty,
        tier=0,
        snippet=snippet,
        language="markdown",
    )


# --- ADR decision-match -------------------------------------------------------

_ADR_TITLE_RE = re.compile(r"^#\s+(?:\d+\.?\s*)?(.+)$", re.M)
_ADR_CONTEXT_RE = re.compile(
    r"^#{2,3}\s+(?:Context(?: and Problem Statement)?|Problem)\s*\n(.*?)(?=\n#{1,3}\s|\Z)",
    re.S | re.M | re.I,
)
_ADR_DECISION_RE = re.compile(r"^#{2,3}\s+Decision(?:\s+Outcome)?\b", re.M | re.I)


def _parse_adr(path: Path, repo_root: Path) -> Optional[tuple[str, str, str]]:
    """(file, title, context) if this markdown looks like an ADR."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not _ADR_DECISION_RE.search(text):
        return None
    title = _ADR_TITLE_RE.search(text)
    context = _ADR_CONTEXT_RE.search(text)
    if not title or not context or len(context.group(1).strip()) < 80:
        return None
    file = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
    return file, title.group(1).strip(), context.group(1).strip()


def adr_questions(
    doc_files: list[Path],
    difficulty: str,
    rng: random.Random,
    repo_root: Path = Path("."),
    limit: int = 2,
) -> list[Question]:
    """Show an ADR's Context, ask which decision the team recorded.

    Wrong options are other ADRs' real titles — plausible (same team, same
    voice), wrong by construction.
    """
    adrs = [a for a in (_parse_adr(p, repo_root) for p in doc_files) if a]
    if len(adrs) < 4:
        return []
    questions: list[Question] = []
    for file, title, context in rng.sample(adrs, len(adrs)):
        if len(questions) >= limit:
            break
        # Skip when the context gives the title away verbatim.
        if title.lower() in context.lower():
            continue
        others = [t for f, t, _ in adrs if t != title]
        if len(others) < 3:
            continue
        snippet = "\n".join(context.splitlines()[:_SNIPPET_LINES])
        questions.append(
            _make_mcq(
                node_id=file,
                text=(
                    "Your team recorded this context in a decision record (shown "
                    "below). Which decision did the team make for it?"
                ),
                correct=title,
                distractors=rng.sample(others, 3),
                explanation=f"That is the decision recorded in {file}.",
                difficulty=difficulty,
                snippet=snippet,
                rng=rng,
            )
        )
    return questions


# --- doc cloze ------------------------------------------------------------------

_LINE_OK_RE = re.compile(r"[A-Za-z].*[A-Za-z]")


def _candidate_lines(section: DocSection) -> list[str]:
    out = []
    for line in section.text.splitlines():
        stripped = line.strip().lstrip("-*>0123456789. ").strip()
        if (
            40 <= len(stripped) <= 200
            and _LINE_OK_RE.search(stripped)
            and not stripped.startswith(("|", "```", "#", "[", "!"))
        ):
            out.append(stripped)
    return out


def doc_cloze_questions(
    sections: list[DocSection],
    difficulty: str,
    rng: random.Random,
    limit: int = 2,
) -> list[Question]:
    """Blank a real statement in a doc section; distractors are real
    statements from other sections of the corpus."""
    by_section = [(s, _candidate_lines(s)) for s in sections]
    usable = [(s, lines) for s, lines in by_section if lines]
    if len(usable) < 2:
        return []

    questions: list[Question] = []
    for section, lines in rng.sample(usable, len(usable)):
        if len(questions) >= limit:
            break
        target = rng.choice(lines)
        others = [
            line
            for other, other_lines in usable
            if other is not section
            for line in other_lines
            if line != target
        ]
        others = list(dict.fromkeys(others))
        if len(others) < 3:
            continue
        blanked = section.text.replace(target, "________________________________", 1)
        snippet = "\n".join(blanked.splitlines()[:_SNIPPET_LINES])
        if "____" not in snippet:  # the blank fell outside the shown excerpt
            continue
        questions.append(
            _make_mcq(
                node_id=section.file,
                text=(
                    f'One statement in "{section.heading}" ({section.file}) is '
                    "blanked out below. Based on the surrounding documentation, "
                    "which statement belongs there?"
                ),
                correct=target,
                distractors=rng.sample(others, 3),
                explanation=f"That is the actual statement in {section.file}.",
                difficulty=difficulty,
                snippet=snippet,
                rng=rng,
            )
        )
    return questions


# --- table lookup ------------------------------------------------------------------


def _tables(section: DocSection) -> list[list[list[str]]]:
    tables, current = [], []
    for line in section.text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [t for t in tables if len(t) >= 5]  # header + at least 4 data rows


def table_questions(
    sections: list[DocSection],
    difficulty: str,
    rng: random.Random,
    limit: int = 2,
) -> list[Question]:
    """Quiz a table cell; distractors are the other rows' cells in the
    same column — homogeneous by construction."""
    questions: list[Question] = []
    with_tables = [(s, t) for s in sections for t in _tables(s)]
    rng.shuffle(with_tables)
    for section, table in with_tables:
        if len(questions) >= limit:
            break
        header, rows = table[0], table[1:]
        columns = [c for c in range(1, len(header)) if header[c]]
        if not columns:
            continue
        column = rng.choice(columns)
        candidates = [r for r in rows if len(r) > column and r[0] and r[column]]
        values = list(dict.fromkeys(r[column] for r in candidates))
        if len(candidates) < 1 or len(values) < 4:
            continue
        row = rng.choice(candidates)
        distractors = rng.sample([v for v in values if v != row[column]], 3)
        snippet = "\n".join(
            "| " + " | ".join(r) + " |" for r in [header] + rows[:_SNIPPET_LINES]
        )
        questions.append(
            _make_mcq(
                node_id=section.file,
                text=(
                    f'Per the table under "{section.heading}" in {section.file}: '
                    f"what is the {header[column]} for “{row[0]}”?"
                ),
                correct=row[column],
                distractors=distractors,
                explanation=f"That is the {header[column]} recorded for {row[0]}.",
                difficulty=difficulty,
                snippet=snippet,
                rng=rng,
            )
        )
    return questions


# --- entry point ---------------------------------------------------------------------


def doc_questions(
    count: int,
    difficulty: str = "medium",
    paths: Optional[list[str]] = None,
    repo_root: Path = Path("."),
    files: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> list[Question]:
    """Up to `count` doc questions, mixed across formats.

    `files` restricts to specific markdown paths (guard mode: quiz on the
    docs the developer just changed). No LLM, no cache — construction is
    instant and a fresh rng varies the questions every session.
    """
    rng = rng or random.Random()
    if files is not None:
        doc_files = [repo_root / f for f in files if (repo_root / f).is_file()]
    else:
        doc_files = discover_doc_files(paths, repo_root)
    if not doc_files:
        return []

    sections: list[DocSection] = []
    for path in doc_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
        sections.extend(split_sections(file, text))

    pool = (
        adr_questions(doc_files, difficulty, rng, repo_root)
        + doc_cloze_questions(sections, difficulty, rng)
        + table_questions(sections, difficulty, rng)
    )
    rng.shuffle(pool)
    return pool[:count]
