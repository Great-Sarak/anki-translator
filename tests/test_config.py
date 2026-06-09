"""Tests for config loaders (shapes + citations)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_translator.config import (
    CitationConvention,
    ConfigError,
    ShapeConfig,
    load_citations,
    load_shapes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_starter_shapes() -> None:
    """The shipped config/shapes.yaml loads cleanly and contains the AT-prefixed starter set."""
    shapes = load_shapes(REPO_ROOT / "config" / "shapes.yaml")
    assert set(shapes.keys()) == {"AT Basic", "AT Cloze", "AT List", "AT Steps", "AT Table"}
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


# ---- citations ----


def test_load_starter_citations() -> None:
    """The shipped config/citations.yaml loads cleanly and contains all five source types."""
    citations = load_citations(REPO_ROOT / "config" / "citations.yaml")
    assert set(citations.keys()) == {"url", "doi", "pdf", "book", "manual"}
    assert citations["url"].source_template == "{url}"
    assert citations["pdf"].position_template == "page {page}"
    assert citations["manual"].position_template == ""
    assert citations["book"].source_required == ["title", "edition"]
    assert isinstance(citations["url"], CitationConvention)


def test_citations_missing_source_type(tmp_path: Path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text(
        "url:\n"
        "  description: web\n"
        "  source_template: '{url}'\n"
        "  source_required: ['url']\n"
    )
    with pytest.raises(ConfigError, match="missing required source types"):
        load_citations(p)


def test_citations_unknown_source_type(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    body = (REPO_ROOT / "config" / "citations.yaml").read_text()
    body += (
        "\n"
        "podcast:\n"
        "  description: audio\n"
        "  source_template: '{episode}'\n"
        "  source_required: ['episode']\n"
    )
    p.write_text(body)
    with pytest.raises(ConfigError, match="unknown source types"):
        load_citations(p)


def test_citations_rejects_extra_keys(tmp_path: Path) -> None:
    body = (REPO_ROOT / "config" / "citations.yaml").read_text()
    body = body.replace(
        '  description: "Web page URL"',
        '  description: "Web page URL"\n  unknown_field: oops',
    )
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError, match="schema validation"):
        load_citations(p)


def test_citations_missing_required_template(tmp_path: Path) -> None:
    body = (REPO_ROOT / "config" / "citations.yaml").read_text()
    # Strip the source_template line from the url entry to trigger schema failure
    body = body.replace('  source_template: "{url}"\n', "")
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError, match="schema validation"):
        load_citations(p)


def test_load_citations_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_citations(tmp_path / "nope.yaml")
