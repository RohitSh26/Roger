# Roger

> Roger is a speed regulator for AI-assisted development. It uses a knowledge graph of
> your codebase and a fully local LLM to quiz you on the code before you commit — keeping
> your understanding in sync with your output. No cloud, no API keys, no tokens. Just you
> and your code.

<!-- demo GIF of the terminal quiz goes here -->

AI writes more and more of your team's code. Roger makes sure the *team* still
understands it. It reads your repository, then does three jobs:

- **Quizzes you** on your own code and your team's written decisions — short,
  fair, multiple-choice, built to teach rather than test memory.
- **Answers questions** about the codebase, with the source and documents it
  used cited underneath.
- **Briefs AI coding agents** so they stop burning tokens reading whole files.

Everything runs on your computer by default. Nothing is uploaded anywhere.

---

## New to the terminal? Read this box first

Everything below happens in the **terminal** (the app called *Terminal* on a Mac).
A few conventions so nothing surprises you:

- Lines shown in code blocks are commands. Type them (or paste them) and press
  **Return**. Don't type the `$` if you see one — it just represents the prompt.
- Words starting with a dash, like `--web` or `-n`, are **options** (also called
  flags). They tweak how a command behaves. We explain every one we use.
- "Run this from your repository" means: use `cd` to go into your project's
  folder first, e.g. `cd ~/work/payments-service`.

That's all you need.

---

## Setting up (once per computer, about 5 minutes)

**1. You need Python 3.10 or newer.** Check with:

```bash
python3 --version
```

**2. You need Ollama** — a free Mac/Windows/Linux app that runs AI models
privately on your own machine. Download it from [ollama.ai](https://ollama.ai),
open it once, and leave it running. (If your company uses the Azure setup
instead, you can skip Ollama entirely — see "Where the AI runs" below.)

**3. Install Roger:**

```bash
pip install git+https://github.com/RohitSh26/Roger.git
```

`pip` is Python's installer; this tells it to fetch Roger straight from GitHub.

---

## Using Roger (this is most of the manual)

Go to your project and type one word:

```bash
roger
```

**The first time** in a repository, Roger asks a single question:

```
First time here — set up Roger for 'payments-service'? [Y/n]:
```

Press **Return** to accept. Roger builds its map of your code (usually under
a minute) and starts your first quiz. The quiz's AI model is a separate
one-time download (~1.15 GB per computer) — Roger asks before fetching it,
right at the moment a quiz or question first needs it:

```
Roger's AI model isn't installed yet (~1.15 GB, one time). Download now? [Y/n]
```

Nothing downloads behind your back, and nothing downloads that you don't
use: if you only ever run `roger context` (the AI-agent briefings), the
model is never fetched at all.

**Every time after that**, `roger` simply starts a five-question quiz:

```
╭─ Question 2 of 5 | process_payment (src/payments/processor.py) ─────╮
│ Why does process_payment validate the card before opening a         │
│ gateway session, rather than after?                                 │
│                                                                     │
│    1  def process_payment(order, card):        ← the real code,     │
│    2      if not validate_card(card):            shown so you can   │
│    3          return Declined("invalid card")    reason, not recall │
│                                                                     │
│   A) To fail fast before paying for a gateway round-trip            │
│   B) Because the gateway rejects unvalidated cards                  │
│   C) To keep the audit log ordered                                  │
│   D) To retry validation on gateway errors                          │
╰──────────────────────────────────────────────────────────────────────╯
```

Press **A**, **B**, **C**, or **D** — just the letter, no Return needed. You
get the answer and a short explanation immediately, right or wrong. Questions
are drawn from your actual code, your team's own documents (design decisions,
contracts), and your system's architecture. Your score is shown to you at the
end of the session and kept nowhere: Roger stores no quiz history, so there is
nothing for a manager to dashboard — by design, and forever.

Two options when you want them:

```bash
roger -n 10        # a longer session: -n means "number of questions"
roger app          # quiz and ask in your browser — the Roger app
```

**The Roger app** opens in your browser, quizzes you with instant grading
(click an answer, see the explanation right away), and has an Ask tab where
you can chat with your codebase. It runs entirely on your machine: the page
is served from your own computer (127.0.0.1), talks only to your local
Ollama, and stops the moment you press Ctrl-C in the terminal. The first
time, Roger offers to install the app's engine (Streamlit, ~250 MB of
Python packages) — one keypress, visible progress, and saying no costs
nothing: the terminal quiz keeps working as always.

---

## Asking questions

```bash
roger ask "why do we retry payments exactly twice?"
```

Roger finds the relevant code and documents, has its AI model read them, and
gives you an answer **with the sources listed underneath** so you can verify
it. Prefer a chat in the browser? The Roger app has an Ask tab:

```bash
roger app
```

If Roger can't find anything relevant, it says so plainly — it does not guess.

### Smarter search (optional, one keypress)

Out of the box, Roger finds code by matching the words in your question
against names and files. That works well when you know what things are
called. But sometimes you don't — you ask *"how do we slow down repeated
requests?"* and the code calls it `throttle`.

The first time you run `roger` on a computer, it offers to fix this:

```
Enable smarter search? Finds code by meaning, not just keywords
(~270 MB one-time download) [Y/n]
```

Press **Enter** to say yes. Roger downloads one small extra model into
Ollama (the same app it already uses) and quietly builds a meaning-based
index of your code in the background. From then on, questions find the
right code even when you don't know its name — in `roger ask`, in agent
context packs, everywhere.

Things worth knowing, in plain terms:

- **Saying no is fine.** Press `n` and Roger never asks again on this
  computer. Keyword search keeps working exactly as before. If you change
  your mind later, `roger doctor` offers again.
- **Nothing leaves your machine.** The index lives in `.roger/vectors.db`
  on your disk, is built by your local Ollama, and Roger automatically
  git-ignores it so it can't sneak into a commit.
- **It can never slow you down.** If the meaning lookup isn't ready within
  half a second, Roger silently uses keyword search for that question.
  There is no error to handle and no waiting.
- **It stays current by itself.** The index refreshes in the same
  background pass that keeps the code map fresh.
- **You can always see it working — and failing.** `roger update` builds
  the index right on your screen with a live count, and if anything goes
  wrong it prints the actual error (never a vague "off"). `roger doctor`
  goes further: if it finds the index unhealthy while you're at the
  keyboard, it rebuilds it right there with the same progress display —
  no "check back later". Ctrl-C is always safe; interrupted builds
  resume from where they stopped.

---

## The commit guard (optional, recommended)

A "pre-commit hook" is a small check that git runs automatically right before
each commit is saved. Roger's guard quizzes you briefly on the files you're
about to commit — the point where understanding matters most. Set it up once:

```bash
roger guard install
```

From then on, `git commit` first shows you a few questions about your staged
changes. Pass (3 of 5 by default) and the commit proceeds. Genuinely busy?
There's an honest skip:

```bash
ROGER_SKIP=1 git commit -m "wip"
```

That skips the quiz and notes the skip in your local activity log (see it
with `roger log`) — no shame, just a record for you. Remove the guard
anytime with `roger guard uninstall`.

---

## Staying current — you mostly don't have to think about it

Roger's map of your code refreshes itself in the background while you take
quizzes. If it ever needs a hand (for example after deleting a lot of code),
it tells you in one line, and one command finishes the job — including asking
you before anything drastic:

```bash
roger update
```

---

## Where the AI runs (one-time choice per repository)

Roger's question-writing and answering need an AI model. You have two options,
and switching between them is one command — no files to edit:

**Option 1 — your own computer (the default).** Private, free, works offline.
Roger sets this up automatically. Want a smarter local model and have ~5 GB
of memory to spare? Pull one and point Roger at it:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
roger use ollama --model qwen2.5:7b-instruct-q4_K_M
```

**Option 2 — your company's Azure (for enterprises).** If your organization
provides Claude through Azure AI Foundry, ask your platform team for two
values — an *endpoint* (a URL) and a *deployment name* — then:

```bash
roger use azure --endpoint https://YOUR-RESOURCE.services.ai.azure.com/anthropic --deployment YOUR-DEPLOYMENT-NAME
export AZURE_ANTHROPIC_API_KEY="your-key-here"
```

About that second line: `export` sets an *environment variable* — a value that
lives only in your terminal session, never in a file. That's deliberate: files
get committed to git; your key must not be. To avoid retyping it, add that
line to the file `~/.zshrc` (your terminal's startup file).

**Know before enabling Azure for a team:** with this option, the code and doc
excerpts Roger builds questions from are sent to *your company's own Azure*
— not to Roger, not to anyone else. The default option sends nothing anywhere.

---

## For teams

- Commit the `.roger/cache.db` file to your repository and teammates share the
  question pool — questions stay valid until the code they cover changes.
- Commit `.roger/config.toml` too and your repo's settings (model choice,
  question count) arrive with `git clone`. The Azure key is never in a file,
  so this is safe.
- Quiz results are never stored — there is nothing personal to leak.

---

## For AI coding agents (OpenCode, Copilot, Claude Code, Cursor…)

Agents waste enormous token budgets grepping and reading whole files. Roger
fixes that with one command, run once per repository:

```bash
roger agent install
```

This writes a short instruction into the files agents already read
(`AGENTS.md`, `CLAUDE.md`): *before exploring, run `roger context`*. From then
on your agents call:

```bash
roger context "how is authentication checked?" --budget 2000
```

…and receive a compact, cited briefing — the relevant functions in full, the
design decisions behind them, who-calls-what — capped at roughly the token
budget you set (`--budget 2000` ≈ 2,000 tokens). Two things worth knowing:

- **It uses no AI at all.** Roger looks things up; *your agent's own model*
  does the thinking — on whatever subscription and model you already picked
  in the agent. Roger needs no key and adds no cost.
- **You can see exactly what your agent saw** by running the same command
  yourself. No black box.

This works with **GitHub Copilot in VS Code** (via
`.github/copilot-instructions.md`, written for you), **OpenCode**, **Claude
Code**, Codex, and Aider — step-by-step instructions for each tool, including
what you'll click and see, are in
[docs/USING-WITH-AGENTS.md](docs/USING-WITH-AGENTS.md). No server to run,
nothing to configure in the agent. Undo anytime with `roger agent uninstall`.

---

## Every command, in one place

| Command | What it does |
|---|---|
| `roger` | The main event: sets up on first run, then quizzes you |
| `roger -n 10` | Quiz with a custom number of questions |
| `roger app` | Quiz and ask in your browser — all local |
| `roger ask "…"` | Answer a question about the codebase, with sources |
| `roger explain "…"` | Everything the graph knows about one symbol (no AI used) |
| `roger path "…" "…"` | How two symbols connect (no AI used) |
| `roger context "…"` | A cited briefing for AI agents (no AI used) |
| `roger context --interfaces "…"` | Contracts only — signatures and relationships, no bodies |
| `roger agent install` | Teach agents in this repo to use Roger |
| `roger log` | See what was recently asked of Roger — including by your agents |
| `roger guard install` | Quiz on staged changes before every commit |
| `roger update` | Refresh Roger's map of your code by hand |
| `roger doctor` | Check this environment and print fixes for anything wrong |
| `roger use ollama` / `roger use azure …` | Choose where the AI runs |
| `roger init` | Set up manually (bare `roger` does this for you) |

---

## When something goes wrong

First move, always:

```bash
roger doctor
```

It checks your environment end to end — Python, the code map, the AI backend,
agent files — and prints the exact fix for anything wrong. The common cases:

**"Ollama is not running"** — open the Ollama app (or run `ollama serve` in
another terminal window and leave it open).

**"Model 'roger-local' not found"** — run `roger init` once; it registers the
model for you.

**"No knowledge graph found"** — run `roger` in the repository; first-run
setup builds it.

**"command not found: roger" inside a dev container or Codespace** — the
container is a separate computer; install Roger inside it too. One-line fix
and the automatic version are in
[docs/USING-WITH-AGENTS.md](docs/USING-WITH-AGENTS.md#dev-containers-and-github-codespaces).

**"unknown model provider"** — there's a typo in `.roger/config.toml`. The
error lists the valid values; or just run `roger use ollama` to reset it.

**Questions feel slow the first time** — first-time questions are written by
the AI model (a few seconds each); repeats are instant because Roger caches
questions until the code they cover changes.

---

## Updating Roger itself

When a new version ships, run:

```bash
pip install --force-reinstall --no-deps git+https://github.com/RohitSh26/Roger.git
```

(The extra options force pip to fetch the newest code; plain `pip install`
would wrongly say you're up to date.) Check what you have:

```bash
pip show roger-cli | grep Version
```

---

## Our promises

1. **Never waste your time.** One word to use it. First question in seconds.
   Honest skips.
2. **Never lie to you.** Most question types are built so the correct answer
   is copied from your own code or docs — it *cannot* be wrong. AI-written
   questions pass seven validation checks before you ever see them.
3. **Never surveil you.** Scores live on your machine, period. By default,
   nothing about your code leaves your computer.

---

## For contributors

```bash
git clone https://github.com/RohitSh26/Roger.git && cd Roger
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # no Ollama needed — everything external is mocked
ruff check roger/ tests/
```

Full technical details live in [ROGER_SPEC.md](ROGER_SPEC.md) and the design
rules in [CLAUDE.md](CLAUDE.md). License: [MIT](LICENSE).
