"""Custom exceptions for Roger.

Every Ollama / file I/O failure surfaces as one of these, never a bare
generic exception, so the CLI can print actionable setup instructions.
"""


class RogerError(Exception):
    """Base class for all Roger errors."""


class OllamaNotRunningError(RogerError):
    """Ollama is not reachable at the configured URL."""


class GraphNotFoundError(RogerError):
    """graphify-out/graph.json does not exist or cannot be read."""


class ModelNotRegisteredError(RogerError):
    """The 'roger-local' model is not registered in Ollama."""


class CacheError(RogerError):
    """A SQLite storage operation (cache.db / history.db) failed."""
