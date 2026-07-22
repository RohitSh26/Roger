# Roger — Codebase Understanding & Quiz CLI

## Summary

Roger is an open-source CLI tool that prevents cognitive debt in AI-assisted development.
As developers use AI coding agents to write more code faster, they risk shipping code they
don't understand. Roger intercepts that risk by generating quizzes from the actual codebase
— using a knowledge graph (via Graphify) and a local LLM (via Ollama) — and presenting them
to developers before they commit or when they return to unfamiliar code.

Three modes:
- `roger quiz` — explore any repo or module interactively, rebuild mental models
- `roger guard` — pre-commit hook that quizzes on staged changes before a commit lands
- `roger ask` — conversational Q&A backed by the knowledge graph

Everything runs locally. No cloud LLM, no API keys, no token costs. Fully private.

---

## Goals

- Give developers a "speed regulator" for AI-generated code: fast output with maintained
  understanding
- Generate quiz questions from real codebase structure, not documentation or hand-crafted
  content
- Work across all major programming languages via Graphify + Tree-sitter
- Run entirely locally with zero ongoing cost (Ollama + MiniCPM5-1B)
- Integrate naturally into git workflow via pre-commit hook with a non-punitive override
- Track developer understanding over time via a local HTML dashboard
- Be fully open source, installable via pip, and usable on any repo

---

## Scope

### In Scope
- Python CLI tool (Typer-based), installable via `pip install roger-cli`
- Three command groups: `quiz`, `guard`, `ask`
- Graphify integration for knowledge graph construction
- Ollama integration with MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF
- Three-tier question generation (described below)
- Multiple choice questions (MCQ) as primary format — 4 options, one correct
- SQLite-backed question cache keyed by code hash (shareable across team via git)
- SQLite-backed quiz history with per-node tracking
- HTML dashboard (static file, generated locally, no server)
- Pre-commit hook with skip/override mechanism that logs but never hard-blocks
- Quiz scoping: by module path, by date range, by author, by PR number
- Configurable via `.roger/config.toml`

### Out of Scope (for now)
- Cloud LLM of any kind — deliberately excluded, local only
- Web server or hosted dashboard
- Team shared backend / analytics server
- IDE plugins
- CI/CD integration beyond pre-commit hook

---

## Architecture Overview

```
Developer
  └── Roger CLI (Typer)
        ├── roger init      → graphify setup + ollama model registration
        ├── roger quiz      → graph load → question generation → quiz runner
        ├── roger guard     → git diff → quiz on staged nodes → pass/fail
        ├── roger ask       → graph query → LLM answer
        ├── roger chat      → multi-turn interactive Q&A session
        ├── roger report    → generate + open HTML dashboard
        └── roger update    → incremental graphify rebuild

Knowledge Layer
  └── graphify (pip: graphifyy)
        → graphify-out/graph.json   (NetworkX graph: nodes, edges, communities)
        → graphify-out/GRAPH_REPORT.md  (god nodes, surprise edges)

Question Generation (Three Tiers)
  └── Tier 0: Graph templates     (simple questions, zero LLM calls)
  └── Tier 1: Local Ollama LLM    (medium + hard questions, local MiniCPM5-1B)
  (No Tier 2 cloud LLM — everything stays local)

Storage Layer
  └── .roger/cache.db      (question cache: hash → questions)
  └── .roger/history.db    (quiz sessions + per-answer records)
  └── .roger/report.html   (generated dashboard)
  └── .roger/config.toml   (user config)
```

---

## Dependencies

| Package      | PyPI Name    | Purpose                                        |
|--------------|--------------|------------------------------------------------|
| graphify     | `graphifyy`  | Knowledge graph (Tree-sitter + NetworkX + Leiden) |
| networkx     | `networkx`   | Loading and querying graph.json                |
| typer        | `typer[all]` | CLI framework                                  |
| rich         | `rich`       | Terminal UI: quiz display, colors, panels      |
| requests     | `requests`   | Ollama API calls                               |
| gitpython    | `gitpython`  | Git diff extraction for guard mode             |
| jinja2       | `jinja2`     | HTML dashboard templating                      |
| tomli        | `tomli`      | Config parsing (Python < 3.11 compat)          |

**External tools (not pip — user must install):**
- **Ollama**: https://ollama.ai — must be installed and running (`ollama serve`)
- **Model**: `hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0`
  - Q8_0 is recommended (~1.15 GB). Q4_K_M (~688 MB) for low-RAM machines.
  - Ollama loads it automatically from HuggingFace on first `roger init`

---

## Project Structure

```
roger/
├── roger/
│   ├── __init__.py
│   ├── cli.py                  # Typer app, all command definitions
│   ├── config.py               # .roger/config.toml loading + defaults
│   ├── graph.py                # Graphify integration, subgraph queries
│   ├── generator.py            # Question generation orchestration (all tiers)
│   ├── templates.py            # Tier 0: template-based questions from graph metadata
│   ├── quiz.py                 # Quiz runner: display, collect answers, score
│   ├── grader.py               # MCQ grading logic
│   ├── storage.py              # SQLite: cache.db + history.db
│   ├── report.py               # HTML dashboard generator
│   ├── hooks/
│   │   ├── __init__.py
│   │   └── pre_commit.py       # Pre-commit hook logic
│   └── llm/
│       ├── __init__.py
│       ├── local.py            # Ollama API client + thinking-block stripping
│       └── router.py           # Tier routing logic
├── local/
│   └── Modelfile               # Ollama model definition
├── templates/
│   └── report.html.jinja       # Dashboard Jinja2 template
├── tests/
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_graph.py
│   ├── test_quiz.py
│   ├── test_storage.py
│   └── test_templates.py
├── CLAUDE.md                   # Claude Code instructions
├── ROGER_SPEC.md              # This file
├── README.md
└── pyproject.toml
```

---

## Module Details

### roger/cli.py

Entry point. Typer app with three command groups.

```python
# All commands to implement:

roger init
# Bootstraps everything:
# 1. pip check: graphifyy installed?
# 2. Run: graphify ./ (builds graphify-out/)
# 3. ollama check: installed and running?
# 4. ollama create roger-local -f local/Modelfile
# 5. mkdir .roger/
# 6. Write .roger/config.toml with defaults
# 7. Print success + next steps

roger quiz
roger quiz --module <path>            # e.g. src/payments
roger quiz --since "3 months ago"     # quiz on nodes changed since date
roger quiz --author me                # quiz on nodes you authored
roger quiz --pr <number>             # quiz on diff vs main branch
roger quiz --difficulty [simple|medium|hard]   # default: medium
roger quiz --count <n>               # number of questions (default: 5)

roger guard install                   # writes .git/hooks/pre-commit
roger guard uninstall
roger guard                           # manual run on staged files

roger ask "<question>"                # single Q&A
roger chat                            # multi-turn interactive Q&A

roger report                          # generate .roger/report.html and open it
roger update                          # re-run graphify incrementally
```

---

### roger/graph.py

Loads `graphify-out/graph.json` into NetworkX. All graph queries go through here.

```python
import networkx as nx
import json
from pathlib import Path

def load_graph(path: str = "graphify-out/graph.json") -> nx.DiGraph:
    """Load graphify output into a NetworkX directed graph."""

def get_node(graph: nx.DiGraph, node_id: str) -> dict:
    """Return node attributes: id, description, callers, callees, returns, community, file."""

def get_subgraph(graph: nx.DiGraph, node_id: str, hops: int = 1) -> nx.DiGraph:
    """Return 1-hop neighborhood of node_id — used as LLM context."""

def get_god_nodes(graph: nx.DiGraph, top_n: int = 10) -> list[str]:
    """Return highest-degree node IDs. These are highest priority for quizzing."""

def get_community_nodes(graph: nx.DiGraph, community: str) -> list[str]:
    """Return all node IDs in a named Leiden community."""

def get_changed_nodes(graph: nx.DiGraph, changed_files: list[str]) -> list[str]:
    """Map a list of changed file paths (from git diff) to graph node IDs."""

def get_nodes_by_path(graph: nx.DiGraph, path: str) -> list[str]:
    """Return all node IDs whose file attribute matches path or starts with path."""

def serialize_subgraph(subgraph: nx.DiGraph) -> str:
    """Serialize a subgraph to a compact text format for LLM prompt injection."""

def get_god_node_ids_from_report(report_path: str = "graphify-out/GRAPH_REPORT.md") -> list[str]:
    """Parse GRAPH_REPORT.md to extract named god nodes — used for quiz weighting."""

def get_surprise_edges(report_path: str = "graphify-out/GRAPH_REPORT.md") -> list[tuple]:
    """Parse GRAPH_REPORT.md to extract surprise edges — direct quiz material."""

def query_graph_for_ask(graph: nx.DiGraph, question: str) -> str:
    """
    Natural language graph query for roger ask.
    Keyword-match question terms against node descriptions.
    Return serialized subgraph of top matching nodes.
    """
```

---

### roger/generator.py

Orchestrates question generation. Handles caching, tier routing, and deduplication.

```python
from roger.storage import get_cached_questions, cache_questions
from roger.templates import build_from_graph
from roger.llm.router import get_questions_from_llm
from roger import graph as g
import hashlib, json

def generate_questions(
    node_ids: list[str],
    graph: nx.DiGraph,
    difficulty: str = "medium",
    count: int = 5
) -> list[Question]:
    """
    Main entry point for question generation.

    For each node_id:
      1. Compute hash of node + 1-hop subgraph
      2. Check cache.db for this hash
      3. Cache hit → return cached questions
      4. Cache miss → route to Tier 0 or Tier 1
      5. Store result in cache
    
    Select `count` questions from all generated, weighted toward god nodes.
    """

def hash_node(node: dict, subgraph: nx.DiGraph) -> str:
    """SHA-256 of node attributes + serialized subgraph. Cache key."""

def select_questions(
    all_questions: list[Question],
    count: int,
    god_node_ids: list[str]
) -> list[Question]:
    """
    Select `count` questions from pool.
    Weight toward questions about god nodes.
    Ensure variety: not all from same node.
    """
```

---

### roger/templates.py — Tier 0

Zero LLM calls. Generates simple MCQ questions purely from graph metadata.

```python
def build_from_graph(node: dict, graph: nx.DiGraph) -> list[Question]:
    """
    Generate 1-5 simple questions from graph node metadata.
    Only runs for difficulty='simple'.
    Distractor selection: sample from other nodes in same community.
    """

# Template functions — each returns a Question or None if data unavailable:

def caller_question(node: dict, graph: nx.DiGraph) -> Question | None:
    """
    Template: "Which of the following calls `{node_name}()`?"
    Correct: random choice from node['callers']
    Distractors: 3 nodes from same community that are NOT callers
    """

def dependency_question(node: dict, graph: nx.DiGraph) -> Question | None:
    """
    Template: "What does `{node_name}()` directly call?"
    Correct: random choice from node['callees']
    Distractors: 3 other callees from graph not in node['callees']
    """

def module_question(node: dict, graph: nx.DiGraph) -> Question | None:
    """
    Template: "Which module/layer does `{node_name}` belong to?"
    Correct: node['community']
    Distractors: 3 other community names from graph
    """

def return_type_question(node: dict, graph: nx.DiGraph) -> Question | None:
    """
    Template: "What does `{node_name}()` return?"
    Correct: node['returns']
    Distractors: 3 other return types from nodes in same community
    """

def location_question(node: dict, graph: nx.DiGraph) -> Question | None:
    """
    Template: "In which file is `{node_name}` defined?"
    Correct: node['file']
    Distractors: 3 other files from same community
    """
```

---

### roger/llm/local.py — Tier 1

Ollama API client. Handles MiniCPM5-1B's thinking blocks.

```python
import requests, re, json

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "roger-local"

def is_ollama_running() -> bool:
    try:
        return requests.get(OLLAMA_BASE, timeout=2).ok
    except Exception:
        return False

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by MiniCPM5 thinking mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def call_local(prompt: str, model: str = DEFAULT_MODEL) -> dict:
    """
    Call Ollama, strip thinking blocks, parse JSON response.
    Raises OllamaNotRunningError if Ollama is not available.
    Raises ValueError if response is not valid JSON after stripping.
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=60
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    content = strip_thinking(content)
    return json.loads(content)
```

**Prompt template for question generation:**

```
Given the following code graph context, generate {count} quiz questions
at {difficulty} difficulty level.

GRAPH CONTEXT:
Node: {node.id}
Description: {node.description}
File: {node.file}
Module/Community: {node.community}
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
```

---

### roger/llm/router.py

Routes to the correct tier based on difficulty.

```python
from roger.llm import local
from roger.templates import build_from_graph
from roger.exceptions import OllamaNotRunningError

def get_questions(
    node: dict,
    graph: nx.DiGraph,
    difficulty: str,
    count: int
) -> list[Question]:
    """
    Tier 0: difficulty == 'simple' → templates (zero LLM)
    Tier 1: difficulty == 'medium' or 'hard' → local Ollama

    If Ollama is not running and difficulty requires LLM,
    raise OllamaNotRunningError with a helpful setup message.
    """
    if difficulty == "simple":
        return build_from_graph(node, graph)

    if not local.is_ollama_running():
        raise OllamaNotRunningError(
            "Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "First-time setup: roger init"
        )

    prompt = build_prompt(node, graph, difficulty, count)
    raw = local.call_local(prompt)
    return parse_questions(raw, node_id=node["id"], difficulty=difficulty, tier=1)
```

---

### roger/quiz.py

Terminal UI. Uses Rich for display. Keypress-based answer collection (no Enter needed).

```python
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

def run_quiz(questions: list[Question], session_type: str) -> QuizResult:
    """
    Display each question, collect answer, show immediate feedback, return result.
    
    Display format per question:
    - Panel header: "Question 2 of 5 | src/payments/processor.py"
    - Question text
    - Options A B C D on separate lines
    - Single keypress input (use readchar or similar)
    - Immediate: ✓ Correct / ✗ Incorrect + explanation
    - Running score shown after each answer
    
    End summary:
    - Score: X/5
    - Time taken
    - Which nodes to review (ones answered wrong)
    - "Run 'roger quiz --module <path>' to focus on weak areas"
    """

def collect_keypress() -> str:
    """Capture single keypress A/B/C/D without requiring Enter."""
```

---

### roger/storage.py

Two SQLite databases under `.roger/`.

**cache.db schema:**
```sql
CREATE TABLE IF NOT EXISTS question_cache (
    hash            TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    questions_json  TEXT NOT NULL,
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version   TEXT
);
```

**history.db schema:**
```sql
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_type    TEXT NOT NULL,      -- 'guard' | 'quiz' | 'ask'
    commit_hash     TEXT,
    module_scope    TEXT,
    score           INTEGER,
    total           INTEGER,
    passed          BOOLEAN,
    skipped         BOOLEAN DEFAULT FALSE,
    skip_reason     TEXT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_secs   INTEGER
);

CREATE TABLE IF NOT EXISTS quiz_answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES quiz_sessions(id),
    node_id         TEXT NOT NULL,
    question        TEXT NOT NULL,
    user_answer     TEXT,
    correct_answer  TEXT,
    is_correct      BOOLEAN,
    difficulty      TEXT
);
```

```python
# Key functions:

def get_db_path(db_name: str) -> str:
    """Returns .roger/cache.db or .roger/history.db"""

def get_cached_questions(hash: str) -> list[Question] | None
def cache_questions(hash: str, node_id: str, difficulty: str, questions: list[Question], model_version: str)

def record_session(result: QuizResult) -> int      # returns session_id
def record_answers(session_id: int, answers: list[QuizAnswer])

def get_history(limit: int = 50) -> list[dict]
def get_weak_nodes(limit: int = 10) -> list[dict]  # nodes most often answered wrong
def get_skip_history() -> list[dict]
def get_score_trend(days: int = 30) -> list[dict]  # for dashboard chart
```

---

### roger/hooks/pre_commit.py

The pre-commit hook logic. Installed at `.git/hooks/pre-commit`.

```python
import os, sys
from roger.graph import load_graph, get_changed_nodes
from roger.generator import generate_questions
from roger.quiz import run_quiz
from roger.storage import record_session

def run_guard():
    """
    Full guard flow:
    1. Check ROGER_SKIP env var → log + exit 0 if set
    2. Get staged files (git diff --cached --name-only)
    3. Load graph, map files to node IDs
    4. If no nodes found → exit 0 (nothing graphify knows about changed)
    5. Generate questions for changed nodes
    6. Run quiz
    7. Record result
    8. Exit 0 if passed, exit 1 if failed (unless block_on_fail=false in config)
    """

    if os.environ.get("ROGER_SKIP"):
        _log_skip(reason="ROGER_SKIP env var")
        sys.exit(0)

    staged_files = _get_staged_files()
    if not staged_files:
        sys.exit(0)

    graph = load_graph()
    changed_nodes = get_changed_nodes(graph, staged_files)
    if not changed_nodes:
        sys.exit(0)

    config = load_config()
    questions = generate_questions(
        changed_nodes,
        graph,
        difficulty=config.guard.difficulty,
        count=config.quiz.questions_per_session
    )

    result = run_quiz(questions, session_type="guard")
    record_session(result)

    if result.passed:
        print("✓ Roger: quiz passed. Proceeding with commit.")
        sys.exit(0)
    else:
        if config.guard.block_on_fail:
            print(f"✗ Roger: {result.score}/{result.total} — use --no-verify to skip.")
            sys.exit(1)
        else:
            print(f"⚠ Roger: {result.score}/{result.total} — warning only (block_on_fail=false).")
            sys.exit(0)

def _get_staged_files() -> list[str]:
    """subprocess: git diff --cached --name-only"""

def _log_skip(reason: str):
    """Record a skip event to history.db."""

def install_hook():
    """Write the hook script to .git/hooks/pre-commit and chmod +x."""

def uninstall_hook():
    """Remove .git/hooks/pre-commit if it was installed by Roger."""
```

**Hook script written to `.git/hooks/pre-commit`:**
```bash
#!/bin/sh
roger guard
```

---

### roger/report.py

Generates static HTML dashboard from quiz history.

```python
from jinja2 import Environment, FileSystemLoader
import webbrowser

def generate_report(output_path: str = ".roger/report.html"):
    """
    Pull data from history.db and render report.html.jinja.
    
    Dashboard sections:
    - Score trend line chart (Chart.js, CDN)
    - Recent sessions table (date, mode, scope, score, commit hash)
    - Weak spots: nodes most often answered wrong, with links to graph
    - Skip history (date, reason)
    - Per-module breakdown (which modules have lowest scores)
    """

def open_report(path: str = ".roger/report.html"):
    """Open in default browser: webbrowser.open(path)"""
```

---

### roger/config.py

Loads `.roger/config.toml`. Provides typed defaults via dataclasses.

```python
from dataclasses import dataclass, field
from pathlib import Path
import tomllib  # stdlib 3.11+; fall back to tomli

ROGER_DIR = Path(".roger")
CONFIG_PATH = ROGER_DIR / "config.toml"

@dataclass
class ModelConfig:
    local: str = "roger-local"

@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    num_ctx: int = 8192

@dataclass
class QuizConfig:
    default_difficulty: str = "medium"
    questions_per_session: int = 5
    pass_threshold: int = 3

@dataclass
class GuardConfig:
    enabled: bool = True
    difficulty: str = "medium"
    block_on_fail: bool = True

@dataclass
class GraphConfig:
    path: str = "graphify-out/graph.json"
    god_node_weight: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    quiz: QuizConfig = field(default_factory=QuizConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

def load_config() -> Config:
    """Load from .roger/config.toml if it exists, else return defaults."""
```

**Default `.roger/config.toml` written by `roger init`:**
```toml
[model]
local = "roger-local"

[ollama]
url = "http://localhost:11434"
num_ctx = 8192

[quiz]
default_difficulty = "medium"
questions_per_session = 5
pass_threshold = 3

[guard]
enabled = true
difficulty = "medium"
block_on_fail = true

[graph]
path = "graphify-out/graph.json"
god_node_weight = true
```

---

## Ollama Setup

### local/Modelfile

```Dockerfile
FROM hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
PARAMETER num_predict 1024

SYSTEM """
You are a code comprehension quiz generator embedded in the Roger developer tool.
You receive structured code graph context and output quiz questions as JSON.

Rules:
- Output valid JSON only. No prose, no markdown fences, no preamble.
- Never include your reasoning in the final output.
- Questions must be answerable from the provided graph context alone.
- Multiple choice: exactly one correct answer and three plausible distractors.
- The explanation field must be one or two sentences maximum.
"""
```

### roger init flow (step by step)

```
1. Verify graphifyy is installed: importlib.util.find_spec("graphify")
   If not: print install instructions, exit 1

2. Run graphify on current directory:
   subprocess.run(["graphify", "./"], check=True)
   Wait for graphify-out/graph.json to exist

3. Verify ollama is installed:
   shutil.which("ollama")
   If not: print install URL, exit 1

4. Verify ollama is running:
   GET http://localhost:11434
   If not running: print "ollama serve" instruction, exit 1

5. Register model:
   subprocess.run(["ollama", "create", "roger-local", "-f", "local/Modelfile"])

6. Create .roger/ directory

7. Write .roger/config.toml with defaults

8. Initialize .roger/cache.db and .roger/history.db (CREATE TABLE IF NOT EXISTS)

9. Print success summary:
   "✓ Graph built: graphify-out/graph.json ({N} nodes, {E} edges)"
   "✓ Model ready: roger-local (MiniCPM5-1B)"
   "✓ Config: .roger/config.toml"
   ""
   "Next steps:"
   "  roger quiz          — quiz yourself on this repo"
   "  roger guard install — set up pre-commit hook"
   "  roger ask '...'     — ask a question about the codebase"
```

---

## Data Models

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Question:
    node_id: str
    question: str
    options: dict[str, str]     # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct: str                # "A" | "B" | "C" | "D"
    explanation: str
    difficulty: str
    tier: int                   # 0 = template, 1 = local LLM

@dataclass
class QuizAnswer:
    question: Question
    user_answer: Optional[str]
    is_correct: bool
    time_taken_secs: float

@dataclass
class QuizResult:
    session_type: str           # "guard" | "quiz" | "ask"
    answers: list[QuizAnswer]
    score: int
    total: int
    passed: bool
    commit_hash: Optional[str]
    module_scope: Optional[str]
    duration_secs: float

    @property
    def weak_nodes(self) -> list[str]:
        return [a.question.node_id for a in self.answers if not a.is_correct]
```

---

## CLI Reference (complete)

```bash
# Bootstrap
roger init                            # full setup: graphify + ollama model

# Quiz mode
roger quiz                            # whole repo, medium difficulty, 5 questions
roger quiz --module src/payments      # scoped to directory
roger quiz --since "3 months ago"     # nodes changed since date
roger quiz --author me                # nodes you authored (via git blame)
roger quiz --pr 142                   # diff vs main branch
roger quiz --difficulty simple        # simple | medium | hard
roger quiz --count 10                 # override question count

# Guard mode
roger guard install                   # writes .git/hooks/pre-commit
roger guard uninstall                 # removes hook
roger guard                           # manual run on staged changes

# Ask / chat mode
roger ask "What calls normalize_query?"
roger ask "Why does PaymentProcessor depend on NotificationService?"
roger chat                            # multi-turn interactive session

# Dashboard
roger report                          # generate .roger/report.html and open

# Maintenance
roger update                          # re-run graphify (incremental)
roger status                          # show: graph age, model status, cache size
```

**Skip mechanism for guard:**
```bash
ROGER_SKIP=1 git commit -m "wip"     # skip quiz, logged silently
git commit --no-verify                # bypass entire hook (git native)
```

---

## Three-Tier Question Generation Summary

| Tier | Trigger | LLM | Cost | Quality |
|------|---------|-----|------|---------|
| 0 | difficulty = simple | None | Free | Structural facts only |
| 1 | difficulty = medium or hard | Ollama MiniCPM5-1B | Free (local) | Understanding + reasoning |

All questions at all difficulty levels are generated locally. No cloud LLM is used at any point.

**Question types generated:**

| Type | Example | Tier |
|------|---------|------|
| Caller | "Which module calls `get_or_compute()`?" | 0 |
| Dependency | "What does `EmbeddingCache` call directly?" | 0 |
| Module | "Which layer does `RetryHandler` belong to?" | 0 |
| Return type | "What does `normalize_query()` return?" | 0 |
| Purpose | "What is the responsibility of `PaymentProcessor`?" | 1 |
| Behavior | "What happens when the Redis key is not found?" | 1 |
| Failure mode | "What breaks if the upstream service returns 429?" | 1 |
| Design intent | "Why does this function use a TTL cache instead of memoization?" | 1 |
| Surprise edge | "Why does `AuthService` have a dependency on `ReportingModule`?" | 1 |

---

## Build Phases

### Phase 1 — MVP (build this first)
- [ ] `roger init` — graphify + ollama model setup
- [ ] `roger quiz` — whole repo quiz, medium difficulty
- [ ] `roger guard install` + pre-commit hook execution
- [ ] Tier 0 template questions (simple difficulty)
- [ ] Tier 1 local LLM questions (medium + hard)
- [ ] SQLite question cache + hash-based lookup
- [ ] SQLite history recording
- [ ] Terminal quiz UI with Rich (MCQ, keypress, feedback)
- [ ] Basic config loading from `.roger/config.toml`

### Phase 2 — Polish
- [ ] `roger quiz --module` / `--since` / `--difficulty` / `--count` flags
- [ ] `roger report` — HTML dashboard with Chart.js score trend
- [ ] `roger ask` — single Q&A mode
- [ ] God-node weighting in question selection
- [ ] Surprise-edge questions (from GRAPH_REPORT.md)
- [ ] Better Rich UI: panels, progress bars, color-coded results
- [ ] `roger status` command

### Phase 3 — Full Feature
- [ ] `roger chat` — multi-turn interactive Q&A
- [ ] `roger quiz --pr` — PR review mode (diff vs main)
- [ ] `roger quiz --author` — git blame attribution scoping
- [ ] Weak spot tracking and targeted re-quiz suggestions
- [ ] `roger update` — incremental graphify rebuild
- [ ] Shareable `.roger/cache.db` team workflow documentation

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "roger-cli"
version = "0.1.0"
description = "Codebase understanding quiz tool for AI-assisted development"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "typer[all]>=0.12.0",
    "rich>=13.0.0",
    "networkx>=3.0",
    "requests>=2.31.0",
    "gitpython>=3.1.0",
    "jinja2>=3.1.0",
    "graphifyy>=0.1.0",
    "tomli>=2.0.0; python_version < '3.11'",
]

[project.scripts]
roger = "roger.cli:app"

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

---

## Open Source Notes

- **License**: MIT
- **PyPI name**: `roger-cli` (verify availability before publishing)
- **One-paragraph pitch for README**:
  > "Roger is a speed regulator for AI-assisted development. It uses a knowledge graph of your codebase and a fully local LLM to quiz you on the code before you commit — keeping your understanding in sync with your output. No cloud, no API keys, no tokens. Just you and your code."
- README must have a demo GIF of the terminal quiz as the first thing after the pitch
- The `.roger/cache.db` file can be committed to the repo so teammates share the question pool — document this explicitly as a team workflow feature
