# CLAUDE.md — Roger Project Instructions

This file gives Claude Code the context it needs to build the Roger project correctly.
Read ROGER_SPEC.md first for the full technical spec. This file covers decisions,
preferences, and things not to second-guess.

---

## What This Project Is

Roger is a CLI tool that quizzes developers on their own codebase before they commit.
It uses Graphify (a knowledge graph library) to understand the repo structure and a local
Ollama LLM (MiniCPM5-1B) to generate questions. Full details in ROGER_SPEC.md.

---

## Decisions Already Made — Do Not Revisit

**No cloud LLM.** Everything runs locally via Ollama. Do not add OpenAI, Anthropic, or
any other cloud model as a dependency, fallback, or optional integration. If Ollama is
not running, raise a clear error with setup instructions.

**Graphify is the only graph/parsing layer.** Do not add tree-sitter, AST parsing, or
any other code parsing. Graphify (pip: `graphifyy`) handles all of that via its own
Tree-sitter + NetworkX pipeline. Roger reads `graphify-out/graph.json`.

**MiniCPM5-1B is the LLM.** Model: `hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0`.
Registered via the Modelfile at `local/Modelfile` as `roger-local`. The Modelfile
content is also embedded in `roger/llm/local.py` (MODELFILE_CONTENT, kept in sync by a
test) because wheel installs don't ship `local/Modelfile`; `roger init` writes it to
`.roger/Modelfile` when no checkout copy exists.

**Three tiers, not two.** Tier 0 (graph templates, no LLM) handles simple questions.
Tier 1 (local Ollama) handles medium and hard. There is no Tier 2.

**MCQ only for now.** Multiple choice with 4 options. No free-text answers in Phase 1.

**Typer + Rich for CLI.** Use Typer for command definitions. Use Rich for all terminal
output: panels, color, progress. Do not use Click directly or plain print() for UI.

**SQLite only.** No PostgreSQL, no Redis, no other database. Two SQLite files:
`.roger/cache.db` and `.roger/history.db`.

**Static HTML dashboard.** `roger report` generates a `.roger/report.html` file and
opens it in the browser. No Flask, no FastAPI, no live server.

---

## Build Order

Build Phase 1 first. Do not start Phase 2 or 3 until Phase 1 is complete and working.

**Phase 1 order:**
1. `pyproject.toml` + project scaffold
2. `roger/config.py` — config loading with defaults
3. `roger/graph.py` — load and query graph.json
4. `roger/storage.py` — SQLite init + cache + history functions
5. `roger/templates.py` — Tier 0 question templates
6. `roger/llm/local.py` — Ollama client + thinking-block stripping
7. `roger/llm/router.py` — tier routing
8. `roger/generator.py` — orchestration + caching
9. `roger/quiz.py` — terminal quiz UI
10. `roger/grader.py` — MCQ grading
11. `roger/hooks/pre_commit.py` — guard logic
12. `roger/cli.py` — wire everything into Typer commands
13. `roger init` command — full bootstrap flow
14. Tests

---

## Key Technical Details

### Graphify graph.json format
Graphify outputs a NetworkX graph serialized as JSON. Load it with:
```python
import networkx as nx
G = nx.node_link_graph(json.load(open("graphify-out/graph.json")))
```
Node attributes include: `description`, `file`, `community`, and relationship data
accessible via in/out edges. Inspect an actual `graph.json` to confirm attribute names
before hardcoding — Graphify's schema may differ from what's documented here.

### MiniCPM5 thinking blocks
This model emits `<think>...</think>` chain-of-thought before its answer. Always strip
these before parsing JSON:
```python
import re
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

### Ollama API
Use the `/api/chat` endpoint, not `/api/generate`. Payload:
```python
{"model": "roger-local", "messages": [{"role": "user", "content": prompt}], "stream": False}
```
Response JSON path to content: `response["message"]["content"]`

### Question hash (cache key)
Hash the node's attributes + its 1-hop subgraph serialized to a stable string.
Use SHA-256. If the node or its immediate neighbors change, the hash changes and new
questions are generated. If the code is the same, the cache is hit.

### Guard skip mechanism
`ROGER_SKIP=1` environment variable → log the skip to history.db with reason, exit 0.
`git commit --no-verify` bypasses the hook at the git level — Roger can't control that,
but the skip won't be logged either since the hook doesn't run.

### Distractor selection (Tier 0 templates)
When building MCQ distractors for template questions, sample from nodes in the same
Leiden community as the target node. This keeps distractors plausible but wrong.
If the community has fewer than 3 other nodes, fall back to random sampling from the
full graph. Shuffle all four options before displaying.

---

## Code Style

- Type hints on all function signatures
- Dataclasses for all data models (Question, QuizResult, QuizAnswer, Config subclasses)
- No global mutable state — pass config and graph as arguments
- All Ollama/file I/O errors should raise descriptive custom exceptions, not generic ones
- Custom exceptions live in `roger/exceptions.py`:
  - `OllamaNotRunningError`
  - `GraphNotFoundError`
  - `ModelNotRegisteredError`
  - `CacheError`
- Use `pathlib.Path` not `os.path` for file operations
- All database connections opened and closed per-function (no persistent connection)

---

## Testing Approach

- Use pytest
- Mock Ollama calls in tests — do not require Ollama to be running to run tests
- Mock graphify output — include a small synthetic `graph.json` fixture in `tests/`
- Test Tier 0 template generation without any external dependencies
- Test the thinking-block stripping function with various edge cases
- Test hash stability — same input must always produce same hash

---

## What to Print on Error

When Ollama is not running:
```
✗ Roger: Ollama is not running.
  Start it with: ollama serve
  First-time setup: roger init
```

When graph.json is missing:
```
✗ Roger: No knowledge graph found at graphify-out/graph.json
  Build it with: roger init
  Or update it with: roger update
```

When model is not registered:
```
✗ Roger: Model 'roger-local' not found in Ollama.
  Register it with: roger init
  Or manually: ollama create roger-local -f .roger/Modelfile
```

---

## What NOT to Build

- No REST API or web server of any kind
- No authentication or user accounts
- No telemetry or usage reporting
- No network calls except to local Ollama (localhost:11434)
- No auto-update mechanism
- No GUI
- No VSCode extension (Phase 3+ if ever)
- Do not vendor or bundle Ollama or graphify — they are external dependencies the user
  installs separately
