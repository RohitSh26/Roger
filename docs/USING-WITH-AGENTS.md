# Using Roger with your AI coding tool

This guide walks through connecting Roger to the three most common setups:
**GitHub Copilot in VS Code**, **OpenCode**, and **Claude Code**. No prior
terminal expertise assumed.

## What "connecting" means here (30 seconds of background)

AI coding tools explore your repository by searching (grep) and reading whole
files — which is slow and burns a lot of your subscription's tokens. Roger
offers them a better move: one command, `roger context "<question>"`, that
returns a compact, cited briefing — the relevant functions in full, the design
decisions behind them, and who-calls-what.

There is **no plugin, no extension, and no server** to set up. Modern AI
coding tools all do two things: they read an *instructions file* in your
repository, and they can run terminal commands. So the entire connection is:
put one instruction in the file your tool reads. Roger does that for you:

```bash
roger agent install
```

Run once, inside your repository. It writes the same short instruction into
three files (creating them if needed):

| File | Who reads it |
|---|---|
| `AGENTS.md` | OpenCode, Codex, Copilot CLI, Aider |
| `.github/copilot-instructions.md` | GitHub Copilot in VS Code |
| `CLAUDE.md` (only if it already exists) | Claude Code |

Nothing else in those files is touched, and `roger agent uninstall` removes
exactly what was added. Commit these files so the whole team's agents benefit.

**One prerequisite for every tool below:** the `roger` command must work in a
fresh terminal in your project. Test it:

```bash
roger context "test" --budget 200
```

If you see "command not found", Roger is installed inside a Python
environment your terminal isn't using — the simplest fix is to install it
with [pipx](https://pipx.pypa.io) (`pipx install` makes a command available
everywhere), or ask whoever set up Roger on your team.

---

## GitHub Copilot in VS Code

**You need:** VS Code with the GitHub Copilot extension, a Copilot
subscription, and Copilot Chat's **agent mode** (the mode where Copilot can
edit files and run commands, not just chat).

**Steps:**

1. Open your project in VS Code.
2. Open the built-in terminal (menu: *Terminal → New Terminal*) and run:
   ```bash
   roger agent install
   ```
3. Open Copilot Chat (the chat icon in the sidebar) and switch the mode
   picker at the bottom of the chat panel to **Agent**.
4. Ask something real, e.g. *"Why does checkout retry payments twice?"*

**What you'll see:** Copilot reads `.github/copilot-instructions.md`, decides
to run `roger context "why does checkout retry payments"`, and asks your
permission to run the command (agent mode always asks before running terminal
commands — click **Continue**/**Allow**). Roger prints the briefing into the
terminal, Copilot reads it, and answers you — using whichever model you've
selected in Copilot's model picker, on your existing subscription. If VS Code
offers to remember the approval for this command, accepting it makes future
runs seamless.

**Tip:** since Copilot uses VS Code's integrated terminal, run the
prerequisite test above *in that same terminal* — that's the environment that
matters.

---

## OpenCode

**You need:** OpenCode installed and signed in (whatever provider/model your
team uses with it).

**Steps:**

1. In your project folder, run:
   ```bash
   roger agent install
   ```
2. Start OpenCode as usual:
   ```bash
   opencode
   ```
3. Ask a question about the codebase.

**What you'll see:** OpenCode reads `AGENTS.md` automatically at session
start. When your question needs code context, it runs `roger context …` as a
shell command (you'll see it in the tool-call log, like any grep it would
have run) and reasons over the briefing with the model you selected in
OpenCode. Switching OpenCode's model changes nothing about Roger — the
briefing is plain text; any model can read it.

---

## Claude Code

**You need:** the `claude` CLI installed and signed in.

**Steps:**

1. In your project folder, run:
   ```bash
   roger agent install
   ```
   (If your repo has a `CLAUDE.md`, the instruction lands there; Claude Code
   also reads `AGENTS.md`.)
2. Start `claude` in the project and ask about the codebase.

Claude Code will call `roger context …` through its shell tool exactly as it
would call grep — you'll see the command in its activity log.

---

## Dev containers and GitHub Codespaces

A dev container is, for all practical purposes, **a separate computer**. Tools
installed on your Mac — including Roger — don't exist inside it, which is why
the agent's terminal says `roger: command not found` even though `roger` works
in your regular terminal.

**First move in any broken environment:**

```bash
roger doctor
```

It checks the environment (Python, setuptools, the code map, the model
backend, agent files) and prints the exact fix for anything wrong. If `roger`
itself isn't installed yet, continue below.

**Immediate fix** — install Roger inside the container (open the container's
terminal in VS Code and run):

```bash
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/RohitSh26/Roger.git
```

The first line matters in containers: Python 3.12 removed a module called
`distutils` that older package builds still expect — without this line you
can hit `No module named 'distutils'` mid-install. Modern `setuptools`
restores it.

**Durable fix** — make it automatic for every rebuild and every teammate. Add
one line to your `.devcontainer/devcontainer.json`:

```json
"postCreateCommand": "pip install --upgrade pip setuptools wheel && pip install git+https://github.com/RohitSh26/Roger.git"
```

(`postCreateCommand` is a standard dev-container setting: commands that run
once when the container is first built. If you already have one, chain with
`&&`.)

**Two container-specific notes:**

- **The agent path needs no AI model at all.** `roger context` is pure lookup,
  so inside a container you don't need Ollama — just Roger and the code map.
  If your container mounts your working folder (the default for local dev
  containers), the map (`graphify-out/`) is already there. In a fresh clone
  (Codespaces), build it once with `graphify ./ --code-only` — graphify
  installs alongside Roger, and this needs no model either. Add it to
  `postCreateCommand` to automate.
- **Human features (`roger ask`, quizzes) need a model the container can
  reach.** Either your company's Azure setup (works anywhere), or point the
  container at Ollama running on your host machine:
  `roger use ollama` and set `url = "http://host.docker.internal:11434"`
  under `[ollama]` in `.roger/config.toml` — that special hostname is how
  containers reach the machine they run on.

## How to tell it's working (and worth it)

- **See what your agent saw:** run the identical command yourself —
  `roger context "your question"` — and read the briefing. If it's good,
  your agent's answers are grounded in it.
- **Watch the tool calls:** every one of these tools shows the commands it
  runs. `roger context` appearing instead of a chain of greps and file reads
  is the integration working.
- **The savings:** a briefing is capped at the token budget in the
  instruction (default ~2,000). The grep-and-read path for the same question
  routinely reads ten times that. Same subscription, smaller bills, faster
  answers.

## Troubleshooting

**Is the agent actually using Roger? Don't ask it — check the log:**

```bash
roger log
```

Every `roger context` and `roger ask` call is recorded locally (what was
asked, tokens served, and whether the caller was a human or a program), in
`.roger/activity.log`. If your agent claims it used Roger, the log is the
truth.

**The agent never runs `roger context`.** Agents follow instructions files
but aren't forced to — smaller local models especially. Check the instruction is present (open `AGENTS.md` /
`.github/copilot-instructions.md` — you should see a section marked
`<!-- roger:start -->`). Restart the agent session — instructions files are
read at session start. You can also just tell the agent once: *"use roger
context to look things up"* — it will remember for the session.

**"command not found: roger" in the agent's terminal.** The agent's shell
can't see Roger. Fix with pipx (see prerequisite above) or ensure the
environment where Roger is installed is active in that terminal.

**The briefing misses recent code.** Roger's map refreshes in the background,
but after a huge change you can force it: `roger update`.
