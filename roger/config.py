"""Load .roger/config.toml with typed dataclass defaults."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROGER_DIR = Path(".roger")
CONFIG_PATH = ROGER_DIR / "config.toml"

DEFAULT_CONFIG_TOML = """\
[model]
local = "roger-local"

[ollama]
url = "http://localhost:11434"
num_ctx = 8192

[quiz]
default_difficulty = "medium"
questions_per_session = 5
pass_threshold = 3

[guard]
enabled = true
difficulty = "medium"
block_on_fail = true

[graph]
path = "graphify-out/graph.json"
god_node_weight = true
"""


@dataclass
class ModelConfig:
    local: str = "roger-local"


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    num_ctx: int = 8192


@dataclass
class QuizConfig:
    default_difficulty: str = "medium"
    questions_per_session: int = 5
    pass_threshold: int = 3


@dataclass
class GuardConfig:
    enabled: bool = True
    difficulty: str = "medium"
    block_on_fail: bool = True


@dataclass
class GraphConfig:
    path: str = "graphify-out/graph.json"
    god_node_weight: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    quiz: QuizConfig = field(default_factory=QuizConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


def _merge_section(section_cls: type, data: dict[str, Any]) -> Any:
    """Build a section dataclass from TOML data, ignoring unknown keys."""
    known = {f.name for f in fields(section_cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return section_cls(**kwargs)


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load from .roger/config.toml if it exists, else return defaults."""
    if not path.exists():
        return Config()

    with path.open("rb") as f:
        data = tomllib.load(f)

    kwargs: dict[str, Any] = {}
    for f_ in fields(Config):
        section = data.get(f_.name)
        section_cls = f_.default_factory  # each section default_factory is its class
        if isinstance(section, dict) and is_dataclass(section_cls):
            kwargs[f_.name] = _merge_section(section_cls, section)
    return Config(**kwargs)


def write_default_config(path: Path = CONFIG_PATH) -> None:
    """Write the default config.toml (used by `roger init`). Never overwrites."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
