"""Any OpenAI-compatible server running on this machine.

The second fully-local backend. Ollama stays the default because it keeps
the model resident between Roger's short-lived CLI invocations and manages
downloads behind one keypress — but people who already run their own
inference server should not have to keep Ollama installed just for Roger.

Deliberately named for the protocol rather than a vendor: llama.cpp's
`llama-server`, LM Studio, Jan, koboldcpp, llama-swap and ramalama all
speak OpenAI-style chat-completions, so this one client serves all of
them. Roger NEVER starts, stops, or installs that server — the user owns
that process (the Simplicity Doctrine forbids Roger daemonizing anything),
so the only job here is to talk to it and to fail with instructions when
it isn't there.

No API key: the server is on localhost and nothing leaves the machine.
No sampling parameters and no token caps — llama-server takes those at
launch (`-c`, `--temp`), and model families disagree about which ones they
accept over the wire (the Sonnet 5 temperature-400 lesson).
"""

from __future__ import annotations

import requests

from roger.config import Config
from roger.exceptions import LocalServerError
from roger.llm.azure import SYSTEM_PROMPT
from roger.llm.local import _parse_json_lenient, strip_thinking

# base URL -> model id the server reported (per process; a user restarting
# their own server restarts Roger's next command too).
_discovered_model: dict[str, str] = {}

_NOT_RUNNING = (
    "✗ Roger: no OpenAI-compatible server is answering at {url}\n"
    "  Start one, then re-run. With llama.cpp:\n"
    "    llama-server -hf <user>/<repo>-GGUF:Q4_K_M --port {port} -c 8192\n"
    "  Or point Roger at a server you already run:\n"
    "    roger use local-server --url http://127.0.0.1:1234   (LM Studio)\n"
    "  Or switch back to Ollama:  roger use ollama"
)


def _base(config: Config) -> str:
    return config.model.server_url.rstrip("/")


def _port(config: Config) -> str:
    tail = _base(config).rsplit(":", 1)[-1]
    return tail if tail.isdigit() else "8080"


def is_server_running(config: Config, timeout: float = 2.0) -> bool:
    """Is anything answering on the configured URL?

    /v1/models is the one endpoint every OpenAI-compatible server
    implements; llama.cpp also serves /health, but LM Studio does not.
    """
    try:
        resp = requests.get(f"{_base(config)}/v1/models", timeout=timeout)
    except requests.RequestException:
        return False
    return resp.status_code < 500


def ensure_ready(config: Config) -> None:
    if not is_server_running(config):
        raise LocalServerError(
            _NOT_RUNNING.format(url=_base(config), port=_port(config))
        )


def _model_id(config: Config, timeout: float = 3.0) -> str:
    """The model name to put in the request.

    llama-server ignores this field — it serves whatever was loaded at
    launch. LM Studio does NOT: it uses the name to pick the model and to
    just-in-time load it, answering 404 "No models loaded" when the name
    resolves to nothing. So a placeholder is not safe; ask the server what
    it actually has and send that. Cached per URL because it cannot change
    without the user restarting their own server.
    """
    if config.model.server_model:
        return config.model.server_model
    base = _base(config)
    if base in _discovered_model:
        return _discovered_model[base]
    try:
        data = requests.get(f"{base}/v1/models", timeout=timeout).json()
        name = str(data["data"][0]["id"])
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return "local-model"  # llama.cpp-style server that ignores the field
    _discovered_model[base] = name
    return name


def loaded_model(config: Config, timeout: float = 2.0) -> str:
    """A human-readable name for whatever the server has loaded.

    llama-server reports the model's full path as its id, which for a file
    served out of Ollama's blob store is a 64-char sha256 — useless in a
    banner. Fall back to something a person can read.
    """
    if config.model.server_model:
        return config.model.server_model
    try:
        data = requests.get(f"{_base(config)}/v1/models", timeout=timeout).json()
        name = str(data["data"][0]["id"]).rsplit("/", 1)[-1]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return "loaded model"
    if name.startswith("sha256") or len(name) > 40:
        return "loaded model"
    return name.removesuffix(".gguf")


def chat_local_server(
    prompt: str, config: Config, timeout: int = 300, system: str | None = None
) -> str:
    """One chat-completions request; returns the answer text.

    `system` is sent ONLY for quiz generation. Roger's system prompt says
    "output valid JSON only", which is right for questions and actively
    wrong for `roger ask` — a small model obeys it and wraps its prose in
    {"answer": ...}. Ollama's chat path sends no system message for the
    same reason; this mirrors it.

    The timeout is generous by default: a 14B model on CPU can spend
    minutes on a quiz batch, and a timeout mid-generation looks to the
    user exactly like a crash.
    """
    ensure_ready(config)
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    payload = {
        # Servers with one model loaded ignore this; llama-swap and
        # LM Studio use it to pick, so send it when the user set one.
        "model": _model_id(config),
        "messages": messages,
    }
    try:
        resp = requests.post(
            f"{_base(config)}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LocalServerError(
            _NOT_RUNNING.format(url=_base(config), port=_port(config))
        ) from exc
    if resp.status_code >= 400:
        detail = resp.text[:300]
        raise LocalServerError(
            f"✗ Roger: the local server at {_base(config)} refused the request "
            f"(HTTP {resp.status_code}): {detail}\n"
            "  If your server hosts several models, name the one to use:\n"
            "    roger use local-server --model <name>   "
            "(see the list at " + _base(config) + "/v1/models)"
        )
    try:
        content = resp.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as exc:
        raise LocalServerError(
            f"✗ Roger: the server at {_base(config)} answered in a shape Roger "
            "doesn't recognize — it must speak OpenAI-style chat-completions."
        ) from exc
    return strip_thinking(content)


def call_local_server(prompt: str, config: Config, timeout: int = 300) -> dict:
    """Chat-completions request parsed as JSON (quiz generation)."""
    return _parse_json_lenient(
        chat_local_server(prompt, config, timeout=timeout, system=SYSTEM_PROMPT)
    )
