"""Optional Azure AI Foundry (Anthropic) backend.

Opt-in per repo via `[model] provider = "azure-anthropic"` — the default
remains fully local Ollama. Privacy note: with this provider, generation
prompts (source excerpts, doc excerpts, module maps) leave the machine for
your Azure tenant. The API key is read ONLY from the AZURE_ANTHROPIC_API_KEY
environment variable, never from config files, which get committed.

Speaks the Anthropic Messages API that Foundry exposes for Claude models,
over plain HTTPS — no SDK dependency.
"""

from __future__ import annotations

import os

import requests

from roger.config import Config
from roger.exceptions import CloudBackendError
from roger.llm.local import _parse_json_lenient, strip_thinking

API_KEY_ENV = "AZURE_ANTHROPIC_API_KEY"
ANTHROPIC_VERSION = "2023-06-01"

SYSTEM_PROMPT = (
    "You are a code comprehension quiz generator embedded in the Roger "
    "developer tool. You receive structured context about code or "
    "documentation and output quiz questions as JSON. Output valid JSON "
    "only — no prose, no markdown fences, no preamble."
)


def ensure_ready(config: Config) -> None:
    """Raise CloudBackendError listing anything missing for the backend."""
    missing = []
    if not config.model.azure_endpoint:
        missing.append("  [model].azure_endpoint in .roger/config.toml")
    if not config.model.azure_deployment:
        missing.append("  [model].azure_deployment in .roger/config.toml")
    if not os.environ.get(API_KEY_ENV):
        missing.append(f"  {API_KEY_ENV} environment variable")
    if missing:
        raise CloudBackendError(
            "✗ Roger: Azure Foundry Anthropic backend is not configured. Missing:\n"
            + "\n".join(missing)
        )


def call_azure(prompt: str, config: Config, timeout: int = 60) -> dict:
    """Foundry Anthropic request parsed as JSON (quiz generation)."""
    return _parse_json_lenient(chat_azure(prompt, config, timeout=timeout))


def chat_azure(prompt: str, config: Config, timeout: int = 60) -> str:
    """Foundry Anthropic request returning the text answer (roger ask)."""
    ensure_ready(config)
    key = os.environ[API_KEY_ENV]
    url = config.model.azure_endpoint.rstrip("/")
    if not url.endswith("/v1/messages"):
        url += "/v1/messages"

    try:
        resp = requests.post(
            url,
            headers={
                # Foundry routes accept either header depending on setup;
                # sending both is harmless.
                "x-api-key": key,
                "api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            json={
                "model": config.model.azure_deployment,
                # No sampling params: Claude Sonnet 5 / Opus 4.7+ REJECT
                # temperature/top_p/top_k with a 400 ("temperature is
                # deprecated for this model"). Field-hit on a Sonnet 5
                # deployment. Steering happens in the prompt.
                # 4096: Sonnet 5 runs adaptive thinking by default, which
                # spends from max_tokens — 1024 risks truncated JSON.
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CloudBackendError(
            f"✗ Roger: Azure endpoint unreachable ({type(exc).__name__}).\n"
            "  Check [model].azure_endpoint in .roger/config.toml and your network."
        ) from exc

    if resp.status_code in (401, 403):
        raise CloudBackendError(
            f"✗ Roger: Azure rejected the API key (HTTP {resp.status_code}).\n"
            f"  Check the {API_KEY_ENV} environment variable."
        )
    if resp.status_code == 404:
        raise CloudBackendError(
            f"✗ Roger: deployment '{config.model.azure_deployment}' not found "
            f"(HTTP 404).\n  Check [model].azure_deployment and the endpoint path."
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        raise CloudBackendError(
            f"✗ Roger: Azure request failed (HTTP {resp.status_code}): {str(detail)[:300]}"
        )

    blocks = resp.json().get("content", [])
    text = "".join(
        block.get("text", "") for block in blocks if isinstance(block, dict)
    )
    return strip_thinking(text)
