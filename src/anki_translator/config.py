"""Configuration loaders for shapes.yaml and citations.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Shape = Literal["term-def", "cloze", "term-list", "term-steps"]
SourceType = Literal["url", "doi", "pdf", "book", "manual"]


class ShapeConfig(BaseModel):
    """One entry in shapes.yaml — describes how the translator should treat an Anki note type."""

    model_config = ConfigDict(extra="forbid")

    shape: Shape
    fields: dict[str, str]
    cutoffs: dict[str, int] = Field(default_factory=dict)


class CitationConvention(BaseModel):
    """One entry in citations.yaml — describes how to populate Source/Position for a source type.

    source_template and position_template are Python str.format-style strings. Keys listed in
    source_required (and position_required) must be present in the metadata dict provided by
    the extractor. If a key in position_required is missing the position field is left empty,
    rather than raising — position is allowed to be empty.
    """

    model_config = ConfigDict(extra="forbid")

    description: str
    source_template: str
    position_template: str = ""
    source_required: list[str]
    position_required: list[str] = Field(default_factory=list)


class TaggerConfig(BaseModel):
    """Config for the LLM topic tagger. Loaded from tagger.yaml (optional).

    The seed vocabulary the tagger shows the LLM is filtered to avoid drifting
    scaffolding artifacts (agent names, test markers, date stamps, shape names)
    into topical tag output. See Great-Sarak/anki-translator#38 for background.
    """

    model_config = ConfigDict(extra="forbid")

    include_bare_leaves: bool = False
    """Whether to keep depth-1 tags (no `::`) in the seed vocabulary shown to the
    LLM. Default `false` — bare leaves are usually scaffolding artifacts. Set
    `true` if your collection uses flat single-segment topic tags."""

    extra_deny_patterns: list[str] = Field(default_factory=list)
    """Additional regex patterns to filter out of the seed vocabulary, on top of
    the built-in denylist (date stamps, `anki-skill-testrun-*`, fleet agent
    names, `spike`, `cloze`). Patterns are matched against the lowercased tag."""

    use_openclaw_agents: bool = True
    """Whether to query `openclaw config get agents.list` for the agent-name
    denylist. Disable in tests or environments where openclaw isn't installed."""


class ConfigError(Exception):
    """Raised when configuration files are missing or invalid."""


def _load_yaml_mapping(path: Path | str, kind: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{kind} config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{kind} config is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{kind} config must be a mapping at the top level, got {type(raw).__name__}"
        )
    return raw


def load_shapes(path: Path | str) -> dict[str, ShapeConfig]:
    """Load shapes.yaml and return a mapping of note-type-name → ShapeConfig.

    Raises ConfigError if the file is missing, unparseable, or fails schema validation.
    """
    raw = _load_yaml_mapping(path, "shapes")
    try:
        return {name: ShapeConfig(**cfg) for name, cfg in raw.items()}
    except Exception as e:
        raise ConfigError(f"shapes config failed schema validation: {e}") from e


def load_tagger_config(path: Path | str) -> TaggerConfig:
    """Load tagger.yaml. Missing file → return TaggerConfig() defaults.

    Unlike shapes / citations, the tagger config is optional: most users do not
    need to tune the seed-vocabulary filter, so a missing file is not an error.
    A present-but-invalid file IS an error — fail loud rather than silently.
    """
    p = Path(path)
    if not p.exists():
        return TaggerConfig()
    raw = _load_yaml_mapping(path, "tagger")
    try:
        return TaggerConfig(**raw)
    except Exception as e:
        raise ConfigError(f"tagger config failed schema validation: {e}") from e


def load_citations(path: Path | str) -> dict[str, CitationConvention]:
    """Load citations.yaml and return a mapping of source-type → CitationConvention.

    Validates that the keys are exactly the supported SourceType literals — extra keys, missing
    keys, or typos all fail validation rather than producing a partial config.
    """
    raw = _load_yaml_mapping(path, "citations")
    declared = set(raw.keys())
    expected: set[str] = {"url", "doi", "pdf", "book", "manual"}
    missing = expected - declared
    if missing:
        raise ConfigError(f"citations config missing required source types: {sorted(missing)}")
    extra = declared - expected
    if extra:
        raise ConfigError(f"citations config has unknown source types: {sorted(extra)}")
    try:
        return {name: CitationConvention(**cfg) for name, cfg in raw.items()}
    except Exception as e:
        raise ConfigError(f"citations config failed schema validation: {e}") from e
