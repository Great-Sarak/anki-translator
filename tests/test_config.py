"""Tests for config.load_shapes."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_translator.config import ConfigError, ShapeConfig, load_shapes

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_starter_shapes() -> None:
    """The shipped config/shapes.yaml loads cleanly and contains the AT-prefixed starter set."""
    shapes = load_shapes(REPO_ROOT / "config" / "shapes.yaml")
    assert set(shapes.keys()) == {"AT Basic", "AT Cloze", "AT List", "AT Steps"}
    assert shapes["AT Basic"].shape == "term-def"
    assert shapes["AT Basic"].fields["source"] == "Source"
    assert shapes["AT Basic"].cutoffs["back_max_chars"] == 200
    assert shapes["AT Cloze"].shape == "cloze"
    assert shapes["AT Cloze"].cutoffs["deletion_max_words"] == 5


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_shapes(tmp_path / "nope.yaml")


def test_load_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(":\n:bad")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_shapes(p)


def test_load_top_level_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- just a list\n- of strings\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_shapes(p)


def test_load_unknown_shape_value(tmp_path: Path) -> None:
    p = tmp_path / "bad_shape.yaml"
    p.write_text(
        '"Custom":\n'
        "  shape: not-a-real-shape\n"
        "  fields:\n"
        "    front: Front\n"
        "    back: Back\n"
    )
    with pytest.raises(ConfigError, match="schema validation"):
        load_shapes(p)


def test_load_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "no_fields.yaml"
    p.write_text('"Custom":\n  shape: term-def\n')
    with pytest.raises(ConfigError, match="schema validation"):
        load_shapes(p)


def test_load_rejects_extra_keys(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text(
        '"Custom":\n'
        "  shape: term-def\n"
        "  fields:\n"
        "    front: Front\n"
        "    back: Back\n"
        "  unknown_key: oops\n"
    )
    with pytest.raises(ConfigError, match="schema validation"):
        load_shapes(p)


def test_cutoffs_default_empty(tmp_path: Path) -> None:
    p = tmp_path / "no_cutoffs.yaml"
    p.write_text(
        '"Minimal":\n'
        "  shape: term-def\n"
        "  fields:\n"
        "    front: Front\n"
        "    back: Back\n"
    )
    shapes = load_shapes(p)
    assert shapes["Minimal"].cutoffs == {}
    assert isinstance(shapes["Minimal"], ShapeConfig)
