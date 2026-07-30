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

**Local-first LLM; one sanctioned cloud exception (amended 2026-07-30 by Rohit).**
The default is fully local via Ollama; if Ollama is not running, raise a clear error
with setup instructions. The ONLY permitted cloud integration is the user's own
opt-in Azure AI Foundry resource, reachable through two API shapes (amended
2026-07-30 by Rohit): `provider = "azure-anthropic"` (Claude deployments,
Anthropic Messages API, roger/llm/azure.py, key from AZURE_ANTHROPIC_API_KEY)
and `provider = "azure-foundry"` (any other deployed model — gpt-4o-mini, Phi,
Mistral, Llama — via OpenAI-style chat-completions, roger/llm/foundry.py, key
from AZURE_FOUNDRY_API_KEY with AZURE_ANTHROPIC_API_KEY as fallback). Keys from
env vars only, never from config files; local remains the default; the README's
privacy note must stay accurate. The foundry payload carries NO sampling params
and NO token caps — model families disagree on which they accept (the Sonnet 5
temperature-400 lesson). Do not add any other cloud provider (no direct OpenAI/
Anthropic/Google APIs), no fallback between backends, no auto-switching.

**Graphify is the only graph/parsing layer.** Do not add tree-sitter, AST parsing, or
any other code parsing. Graphify (pip: `graphifyy`) handles all of that via its own
Tree-sitter + NetworkX pipeline. Roger reads `graphify-out/graph.json`.

**MiniCPM5-1B is the default LLM.** Model: `hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0`.
Users may point `[model].local` in `.roger/config.toml` at any locally pulled Ollama
model; `roger init` then verifies it exists (never `ollama create` over a user's tag).
This does not soften the no-cloud rule — Ollama on localhost is the only LLM backend.
Registered via the Modelfile at `local/Modelfile` as `roger-local`. The Modelfile
content is also embedded in `roger/llm/local.py` (MODELFILE_CONTENT, kept in sync by a
test) because wheel installs don't ship `local/Modelfile`; `roger init` writes it to
`.roger/Modelfile` when no checkout copy exists.

**Three tiers, not two.** Tier 0 (graph templates, no LLM) handles simple questions.
Tier 1 (local Ollama) handles medium and hard. There is no Tier 2.

**MCQ only for now.** Multiple choice with 4 options. No free-text answers in Phase 1.

**Typer + Rich for CLI.** Use Typer for command definitions. Use Rich for all terminal
output: panels, color, progress. Do not use Click directly or plain print() for UI.

**SQLite only.** No PostgreSQL, no Redis, no other database. SQLite files:
`.roger/cache.db` (shared question cache) and `.roger/vectors.db`
(machine-local semantic index, auto-gitignored). history.db was removed
2026-07-29 — Roger stores no quiz history.

**The Roger app is the one GUI (amended 2026-07-29 by Rohit).** `roger app`
runs a Streamlit app bound to 127.0.0.1 on a foreground process the user
starts and stops (Ctrl-C) — it never daemonizes, never binds beyond
localhost, and Streamlit's telemetry/first-run email prompt are suppressed
per-process via env (never by writing ~/.streamlit). Streamlit ships as the
optional `[app]` extra with a one-keypress on-demand install. The earlier
static-HTML web views and `roger record` were removed with it. Quiz history
was removed entirely (2026-07-29): no history.db, no session recording.

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
`ROGER_SKIP=1` environment variable → note the skip in .roger/activity.log
(visible via `roger log`), exit 0.
`git commit --no-verify` bypasses the hook at the git level — Roger can't control that,
but the skip won't be logged either since the hook doesn't run.

### Distractor selection (Tier 0 templates)
When building MCQ distractors for template questions, sample from nodes in the same
Leiden community as the target node. This keeps distractors plausible but wrong.
If the community has fewer than 3 other nodes, fall back to random sampling from the
full graph. Shuffle all four options before displaying.

---

## The Simplicity Doctrine (added 2026-07-28 by Rohit)

If a feature requires the developer to remember a command or read a doc before
it delivers value, it isn't finished. Roger does X itself only when X is local,
idempotent, confined to `.roger/` and `graphify-out/`, involves no download,
and either finishes in seconds or shows progress; anything else is a
one-keypress confirm. Never auto-download models, never start daemons, never
pass --force on the user's behalf without an explicit interactive confirmation,
and always scrub GRAPHIFY_FORCE from child environments.

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
  - `CloudBackendError` (Azure Foundry backend misconfigured/failed)
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

When model is not registered (non-interactive callers only; interactive
quiz/ask offer a one-keypress download instead — the default model installs
lazily on first LLM use, never during `roger init`; amended 2026-07-28 by
Rohit to keep first-run setup fast):
```
✗ Roger: Model 'roger-local' not found in Ollama.
  Install it by running: roger   (one keypress, ~1.15 GB one-time download)
  Or manually: ollama create roger-local -f .roger/Modelfile
```

---

## What NOT to Build

- No REST API or web server — except the localhost-only, user-started
  foreground `roger app` (see the amended GUI decision above)
- No authentication or user accounts
- No telemetry or usage reporting
- No network calls except to local Ollama (localhost:11434)
- No auto-update mechanism
- No GUI other than `roger app`
- No VSCode extension (Phase 3+ if ever)
- Do not vendor or bundle Ollama or graphify — they are external dependencies the user
  installs separately
