# Roger

> Roger is a speed regulator for AI-assisted development. It uses a knowledge graph of
> your codebase and a fully local LLM to quiz you on the code before you commit — keeping
> your understanding in sync with your output. No cloud, no API keys, no tokens. Just you
> and your code.

<!-- TODO: demo GIF of the terminal quiz goes here, first thing after the pitch -->

## Install

```bash
pip install roger-cli
```

External tools (install separately):

- [Ollama](https://ollama.ai) — must be installed and running (`ollama serve`)
- Model: `hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0` — pulled
  automatically on first `roger init`

## Quick start

```bash
roger init            # build the knowledge graph + register the local model
roger quiz            # quiz yourself on this repo
roger guard install   # set up the pre-commit hook
```

## Skipping the guard

```bash
ROGER_SKIP=1 git commit -m "wip"   # skip quiz, logged to history
git commit --no-verify             # bypass entire hook (git native)
```

## Team question pool

`.roger/cache.db` can be committed to the repo so teammates share the generated
question pool — questions are keyed by a hash of the code itself, so the cache stays
valid until the code changes.

## License

MIT
