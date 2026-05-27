"""Configuration loaders for shapes.yaml (and later, citations.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Shape = Literal["term-def", "cloze", "term-list", "term-steps"]


class ShapeConfig(BaseModel):
    """One entry in shapes.yaml — describes how the translator should treat an Anki note type."""

    model_config = ConfigDict(extra="forbid")

    shape: Shape
    fields: dict[str, str]
    cutoffs: dict[str, int] = Field(default_factory=dict)


class ConfigError(Exception):
    """Raised when configuration files are missing or invalid."""


def load_shapes(path: Path | str) -> dict[str, ShapeConfig]:
    """Load shapes.yaml and return a mapping of note-type-name → ShapeConfig.

    Raises ConfigError if the file is missing, unparseable, or fails schema validation.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"shapes config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"shapes config is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"shapes config must be a mapping at the top level, got {type(raw).__name__}")
    try:
        return {name: ShapeConfig(**cfg) for name, cfg in raw.items()}
    except Exception as e:
        raise ConfigError(f"shapes config failed schema validation: {e}") from e
