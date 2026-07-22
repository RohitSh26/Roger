"""Tier 1: Ollama API client for the local MiniCPM5-1B model.

MiniCPM5 emits <think>...</think> chain-of-thought before its answer;
strip_thinking() must run before any JSON parsing.
"""

from __future__ import annotations

import json
import re

import requests

from roger.exceptions import ModelNotRegisteredError, OllamaNotRunningError

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "roger-local"

# Embedded copy of local/Modelfile (kept in sync by a test). Wheel installs
# don't ship the repo-root file, so `roger init` writes this to
# .roger/Modelfile before registering the model.
MODELFILE_CONTENT = '''FROM hf.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF:Q8_0

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
'''

OLLAMA_NOT_RUNNING_MSG = (
    "✗ Roger: Ollama is not running.\n"
    "  Start it with: ollama serve\n"
    "  First-time setup: roger init"
)

MODEL_NOT_REGISTERED_MSG = (
    "✗ Roger: Model '{model}' not found in Ollama.\n"
    "  Register it with: roger init\n"
    "  Or manually: ollama create roger-local -f .roger/Modelfile"
)

CUSTOM_MODEL_NOT_FOUND_MSG = (
    "✗ Roger: Model '{model}' not found in Ollama.\n"
    "  Pull it with: ollama pull {model}\n"
    '  Or set local = "roger-local" under [model] in .roger/config.toml'
)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def is_ollama_running(base_url: str = OLLAMA_BASE) -> bool:
    try:
        return requests.get(base_url, timeout=2).ok
    except requests.RequestException:
        return False


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by MiniCPM5 thinking mode."""
    return _THINK_RE.sub("", text).strip()


def call_local(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE,
    timeout: int = 60,
    num_ctx: int | None = None,
) -> dict:
    """Call Ollama, strip thinking blocks, parse JSON response.

    num_ctx overrides the model's context window per-request, so the
    config value applies to custom models that have no Roger Modelfile.

    Raises OllamaNotRunningError if Ollama is not available.
    Raises ModelNotRegisteredError if the model is not registered.
    Raises ValueError if the response is not valid JSON after stripping.
    """
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # MiniCPM5-Thinking otherwise spends its whole num_predict
        # budget in the thinking phase and truncates the JSON answer.
        "think": False,
    }
    if num_ctx is not None:
        payload["options"] = {"num_ctx": num_ctx}
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaNotRunningError(OLLAMA_NOT_RUNNING_MSG) from exc

    if resp.status_code == 404:
        template = (
            MODEL_NOT_REGISTERED_MSG if model == DEFAULT_MODEL else CUSTOM_MODEL_NOT_FOUND_MSG
        )
        raise ModelNotRegisteredError(template.format(model=model))
    if resp.status_code >= 400:
        # Surface Ollama's own error message (e.g. context-size exceeded)
        # instead of a bare HTTP status.
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise ValueError(
            f"✗ Roger: Ollama request failed (HTTP {resp.status_code}): {str(detail)[:300]}"
        )

    content = resp.json()["message"]["content"]
    content = strip_thinking(content)
    return _parse_json_lenient(content)


def _parse_json_lenient(content: str) -> dict:
    """Parse model output as JSON, salvaging what a small model mangles.

    A 1B model wraps JSON in prose/fences or truncates mid-array when it
    hits num_predict. Strip to the outermost braces first; if the tail is
    truncated, cut back to the last complete object and re-close the
    structure — parse_questions validates whatever survives.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        snippet = content[start:]
        brace_positions = [i for i, ch in enumerate(snippet) if ch == "}"]
        for pos in reversed(brace_positions[-20:]):
            base = snippet[: pos + 1]
            for closer in ("", "]}", "}", "}]}"):
                try:
                    return json.loads(base + closer)
                except json.JSONDecodeError:
                    continue

    raise ValueError(
        f"Local model returned invalid JSON after stripping thinking blocks: "
        f"{content[:200]!r}"
    )
