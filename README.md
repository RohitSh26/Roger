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

Press **Return** to accept. Roger then builds its map of your code (usually
under a minute), downloads its small AI model if needed (~1.2 GB, one time
per computer), and starts your first quiz. It tells you beforehand if your
repository is unusually large or a download is coming — no surprises.

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
contracts), and your system's architecture. Your scores stay on your machine,
in your repository's `.roger` folder. They are yours — Roger has no dashboard
for managers and never will.

Two options when you want them:

```bash
roger -n 10        # a longer session: -n means "number of questions"
roger quiz --web   # take the quiz in your browser instead — nicer for
                   # reading code, with syntax colors and diagrams
```

The browser quiz ends by showing a short code like `BCADB` (your answers).
To save that session into your history, copy the command it shows you:

```bash
roger record BCADB
```

---

## Asking questions

```bash
roger ask "why do we retry payments exactly twice?"
```

Roger finds the relevant code and documents, has its AI model read them, and
gives you an answer **with the sources listed underneath** so you can verify
it. Add `--web` to get the answer as a nicely formatted page in your browser:

```bash
roger ask "how does search ranking work?" --web
```

If Roger can't find anything relevant, it says so plainly — it does not guess.

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

That skips the quiz and notes the skip in your own local history — no shame,
just a record for you. Remove the guard anytime with `roger guard uninstall`.

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
- Quiz history (`.roger/history.db`) is personal. Add it to `.gitignore`.

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
| `roger quiz --web` | The quiz, in your browser |
| `roger record BCADB` | Save a finished browser quiz to your history |
| `roger ask "…"` | Answer a question about the codebase, with sources |
| `roger ask "…" --web` | The same answer, formatted in your browser |
| `roger context "…"` | A cited briefing for AI agents (no AI used) |
| `roger agent install` | Teach agents in this repo to use Roger |
| `roger guard install` | Quiz on staged changes before every commit |
| `roger update` | Refresh Roger's map of your code by hand |
| `roger use ollama` / `roger use azure …` | Choose where the AI runs |
| `roger init` | Set up manually (bare `roger` does this for you) |

---

## When something goes wrong

Every Roger error tells you the fix, but here are the common ones:

**"Ollama is not running"** — open the Ollama app (or run `ollama serve` in
another terminal window and leave it open).

**"Model 'roger-local' not found"** — run `roger init` once; it registers the
model for you.

**"No knowledge graph found"** — run `roger` in the repository; first-run
setup builds it.

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
