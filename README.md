# Roger

> Roger is a speed regulator for AI-assisted development. It uses a knowledge graph of
> your codebase and a fully local LLM to quiz you on the code before you commit — keeping
> your understanding in sync with your output. No cloud, no API keys, no tokens. Just you
> and your code.

<!-- demo GIF of the terminal quiz goes here -->

As AI coding agents write more of your code, it gets easy to ship things you don't
actually understand. Roger intercepts that: it builds a knowledge graph of your repo,
generates multiple-choice questions about *your actual code* with a local 1B model, and
asks them — on demand, or as a pre-commit gate.

Everything runs on your machine. The only network calls Roger makes are to Ollama on
`localhost:11434`.

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| [Ollama](https://ollama.ai) | Must be installed and running: `ollama serve` |
| ~1.2 GB disk | For the local model (MiniCPM5-1B, Q8_0), downloaded once |

The knowledge-graph engine ([graphify](https://pypi.org/project/graphifyy/)) is a normal
pip dependency and installs automatically.

## Install

```bash
pip install git+https://github.com/RohitSh26/Roger.git
```

Or for hacking on Roger itself:

```bash
git clone https://github.com/RohitSh26/Roger.git
cd Roger
pip install -e ".[dev]"
```

## Quick start

From the root of the repository you want to be quizzed on:

```bash
roger init            # build the knowledge graph + register the local model
roger quiz            # quiz yourself: 5 questions about this repo
roger guard install   # quiz on staged changes before every commit
```

`roger init` does all the setup: runs graphify over your code (local AST, no API key),
registers the `roger-local` model in Ollama (the Modelfile ships inside Roger and is
written to `.roger/Modelfile`; the ~1.2 GB model weights download the first time ever),
and creates `.roger/` with a default config and databases. If anything is missing it
tells you exactly what to run.

## Commands

| Command | What it does |
|---|---|
| `roger init` | One-time setup for a repo: graph build, model registration, config |
| `roger quiz` | Whole-repo quiz using the settings in `.roger/config.toml` |
| `roger guard` | Run the quiz on currently staged files (what the hook runs) |
| `roger guard install` | Write the pre-commit hook to `.git/hooks/pre-commit` |
| `roger guard uninstall` | Remove the hook (only if Roger installed it) |

Planned (not yet built): `roger ask`, `roger chat`, `roger report`, `roger update`,
`roger status`, and quiz scoping flags (`--module`, `--since`, `--difficulty`, `--count`).

## The guard workflow

After `roger guard install`, every `git commit` quizzes you on the staged files first:

- Files the graph doesn't know about (docs, configs) pass through silently.
- Score at or above `pass_threshold` → the commit proceeds.
- Below → the commit is blocked (or warned, if you set `block_on_fail = false`).

When you legitimately need to skip:

```bash
ROGER_SKIP=1 git commit -m "wip"   # skips the quiz; the skip is logged to history
git commit --no-verify             # git-native bypass; invisible to Roger
```

Skips are never punished — they're just recorded, so you can see your own pattern.

## Configuration

`roger init` writes `.roger/config.toml` with these defaults:

```toml
[model]
local = "roger-local"        # Ollama model name

[ollama]
url = "http://localhost:11434"
num_ctx = 8192

[quiz]
default_difficulty = "medium"   # simple | medium | hard
questions_per_session = 5
pass_threshold = 3              # correct answers needed to pass

[guard]
enabled = true
difficulty = "medium"
block_on_fail = true            # false = warn but never block commits

[graph]
path = "graphify-out/graph.json"
god_node_weight = true          # bias questions toward high-connectivity code
```

## How it works

```
your repo ──graphify──▶ graphify-out/graph.json     (nodes, call edges, communities)
                              │
                     roger picks nodes (god nodes first)
                              │
              ┌── difficulty: simple ──▶ Tier 0: template questions (no LLM, instant)
              └── medium / hard ───────▶ Tier 1: local Ollama (MiniCPM5-1B)
                              │
                    .roger/cache.db  (questions keyed by SHA-256 of the code's
                              │       graph neighborhood — unchanged code = cache hit)
                        terminal quiz
                              │
                    .roger/history.db  (sessions, answers, skips)
```

- **Tier 0 (simple):** structural questions built straight from the graph — "which of
  these calls `X`?", "in which file is `Y` defined?" — with distractors drawn from the
  same code community so they're plausible.
- **Tier 1 (medium/hard):** the local model receives the node plus its 1-hop
  neighborhood and writes comprehension questions about behavior and design.
- **Caching:** questions are keyed by a hash of the node and its neighborhood. Unchanged
  code never regenerates; changed code automatically gets fresh questions. You can commit
  `.roger/cache.db` so your team shares one question pool.

## Troubleshooting

**`Ollama is not running`** — start it with `ollama serve` (leave it running).

**`Model 'roger-local' not found in Ollama`** — run `roger init`, or manually:
`ollama create roger-local -f .roger/Modelfile` (init writes that file; the Modelfile
ships embedded inside Roger, so no checkout is needed).

**`No knowledge graph found`** — run `roger init` in the repo root. After large
refactors, re-run it (or `graphify ./ --code-only`) to rebuild the graph.

**Questions feel slow to generate** — first quiz on a set of nodes calls the local model
(a few seconds per node); repeat quizzes hit the cache and are instant.

**Odd or trivial questions** — a fully local 1B model has limits; question quality
tuning is active work. Tier 0 (`default_difficulty = "simple"`) is deterministic if you
want purely structural questions.

## Development

```bash
git clone https://github.com/RohitSh26/Roger.git && cd Roger
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # no Ollama or graphify needed — everything external is mocked
ruff check roger/ tests/
```

Project layout and the full technical spec live in [ROGER_SPEC.md](ROGER_SPEC.md).

## Status

Phase 1 (MVP) is complete and field-tested: init, quiz, guard hook, both question
tiers, caching, and history. Phases 2–3 (dashboard, ask/chat mode, quiz scoping flags,
incremental graph updates) are on the roadmap in the spec.

## License

[MIT](LICENSE)
