"""Azure AI Foundry generic backend — any chat-completions model.

The second shape of Roger's single sanctioned cloud integration (amended
2026-07-30 by Rohit): the same Azure AI Foundry resource, but serving any
deployed model through the OpenAI-style chat-completions API — gpt-4o-mini,
Phi-4-mini, Mistral, Llama, whatever is cheap and good enough for quiz
generation and ask. Claude deployments keep the dedicated Anthropic
Messages backend (roger/llm/azure.py).

Key from the AZURE_FOUNDRY_API_KEY env var (falls back to
AZURE_ANTHROPIC_API_KEY — one Foundry resource usually has one key),
never from config files. No sampling parameters and no token caps in the
payload: model families disagree on which ones they accept (temperature
is rejected outright by some — field-learned), and defaults are sane.

Foundry resources expose two URL shapes depending on how the model is
deployed; Roger tries the model-inference route first and falls back to
the Azure OpenAI deployments route on 404, remembering what worked for
the rest of the process.
"""

from __future__ import annotations

import os

import requests

from roger.config import Config
from roger.exceptions import CloudBackendError
from roger.llm.azure import SYSTEM_PROMPT
from roger.llm.local import _parse_json_lenient, strip_thinking

API_KEY_ENV = "AZURE_FOUNDRY_API_KEY"
_FALLBACK_KEY_ENV = "AZURE_ANTHROPIC_API_KEY"
_INFERENCE_API_VERSION = "2024-05-01-preview"
_OPENAI_API_VERSION = "2024-10-21"

# endpoint -> url template index that worked (per process)
_working_route: dict[str, int] = {}


def _api_key() -> str:
    return os.environ.get(API_KEY_ENV) or os.environ.get(_FALLBACK_KEY_ENV) or ""


def ensure_ready(config: Config) -> None:
    """Raise CloudBackendError listing anything missing for the backend."""
    missing = []
    if not config.model.azure_endpoint:
        missing.append("  [model].azure_endpoint in .roger/config.toml")
    if not config.model.azure_deployment:
        missing.append("  [model].azure_deployment in .roger/config.toml")
    if not _api_key():
        missing.append(f"  {API_KEY_ENV} environment variable")
    if missing:
        raise CloudBackendError(
            "✗ Roger: Azure Foundry backend is not configured. Missing:\n"
            + "\n".join(missing)
        )


def _routes(config: Config) -> list[str]:
    base = config.model.azure_endpoint.rstrip("/")
    # Switching over from the Anthropic backend, people reuse the endpoint
    # they already have — which by convention ends in /anthropic. The
    # chat-completions routes hang off the resource root, so drop it.
    if base.endswith("/anthropic"):
        base = base[: -len("/anthropic")]
    deployment = config.model.azure_deployment
    return [
        f"{base}/models/chat/completions?api-version={_INFERENCE_API_VERSION}",
        f"{base}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={_OPENAI_API_VERSION}",
    ]


def call_foundry(prompt: str, config: Config, timeout: int = 60) -> dict:
    """Foundry request parsed as JSON (quiz generation)."""
    return _parse_json_lenient(chat_foundry(prompt, config, timeout=timeout))


def chat_foundry(prompt: str, config: Config, timeout: int = 60) -> str:
    """Foundry chat-completions request returning the text answer."""
    ensure_ready(config)
    key = _api_key()
    base = config.model.azure_endpoint
    routes = _routes(config)
    start = _working_route.get(base, 0)
    ordered = routes[start:] + routes[:start]

    payload = {
        # The inference route needs the model name; the deployments route
        # carries it in the URL and ignores this field.
        "model": config.model.azure_deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    last_status = None
    for offset, url in enumerate(ordered):
        try:
            resp = requests.post(
                url,
                headers={"api-key": key, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise CloudBackendError(
                f"✗ Roger: Azure Foundry endpoint unreachable ({type(exc).__name__}).\n"
                "  Check [model].azure_endpoint in .roger/config.toml and your network."
            ) from exc
        if resp.status_code == 404:
            last_status = 404
            continue  # wrong route shape for this resource — try the other
        if resp.status_code in (401, 403):
            raise CloudBackendError(
                f"✗ Roger: Azure Foundry rejected the API key (HTTP {resp.status_code}).\n"
                f"  Check the {API_KEY_ENV} environment variable."
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise CloudBackendError(
                f"✗ Roger: Azure Foundry request failed "
                f"(HTTP {resp.status_code}): {str(detail)[:300]}"
            )
        _working_route[base] = (start + offset) % len(routes)
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise CloudBackendError(
                "✗ Roger: Azure Foundry returned an unexpected response shape."
            ) from exc
        return strip_thinking(content)

    raise CloudBackendError(
        f"✗ Roger: deployment '{config.model.azure_deployment}' not found on "
        f"either Foundry route (HTTP {last_status}).\n"
        "  Check [model].azure_deployment and the endpoint URL."
    )
