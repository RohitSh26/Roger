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

OLLAMA_NOT_RUNNING_MSG = (
    "✗ Roger: Ollama is not running.\n"
    "  Start it with: ollama serve\n"
    "  First-time setup: roger init"
)

MODEL_NOT_REGISTERED_MSG = (
    "✗ Roger: Model '{model}' not found in Ollama.\n"
    "  Register it with: roger init\n"
    "  Or manually: ollama create roger-local -f local/Modelfile"
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
) -> dict:
    """Call Ollama, strip thinking blocks, parse JSON response.

    Raises OllamaNotRunningError if Ollama is not available.
    Raises ModelNotRegisteredError if the model is not registered.
    Raises ValueError if the response is not valid JSON after stripping.
    """
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OllamaNotRunningError(OLLAMA_NOT_RUNNING_MSG) from exc

    if resp.status_code == 404:
        raise ModelNotRegisteredError(MODEL_NOT_REGISTERED_MSG.format(model=model))
    resp.raise_for_status()

    content = resp.json()["message"]["content"]
    content = strip_thinking(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Local model returned invalid JSON after stripping thinking blocks: "
            f"{content[:200]!r}"
        ) from exc
