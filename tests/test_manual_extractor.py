"""Tests for the manual/text extractor."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from anki_translator.extractors import ExtractionError
from anki_translator.extractors.manual import extract_file, extract_text


def test_extract_text_paragraph_split() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = extract_text(text, label="my notes")
    assert len(chunks) == 3
    assert chunks[0].text == "First paragraph."
    assert chunks[2].text == "Third paragraph."


def test_extract_text_label_becomes_source() -> None:
    chunks = extract_text("Some content here.", label="chat 2026-05-27")
    assert chunks[0].source == "chat 2026-05-27"
    assert chunks[0].source_type == "manual"
    assert chunks[0].metadata["label"] == "chat 2026-05-27"


def test_extract_text_position_always_empty() -> None:
    chunks = extract_text("Some content here.", label="notes")
    for c in chunks:
        assert c.position == ""


def test_extract_text_default_label_today() -> None:
    chunks = extract_text("Content goes here.")
    expected = f"manual {date.today().isoformat()}"
    assert chunks[0].source == expected


def test_extract_text_empty_text_raises() -> None:
    with pytest.raises(ExtractionError, match="empty"):
        extract_text("", label="notes")


def test_extract_text_whitespace_only_text_raises() -> None:
    with pytest.raises(ExtractionError, match="empty"):
        extract_text("   \n\n  ", label="notes")


def test_extract_text_empty_label_raises() -> None:
    with pytest.raises(ExtractionError, match="non-empty"):
        extract_text("Content goes here.", label="   ")


def test_extract_file_txt(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("First.\n\nSecond.\n")
    chunks = extract_file(p)
    assert len(chunks) == 2
    assert chunks[0].source == "notes"  # default label = file stem


def test_extract_file_txt_with_explicit_label(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("Content.\n")
    chunks = extract_file(p, label="custom label")
    assert chunks[0].source == "custom label"


def test_extract_file_md_splits_on_headings(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text(
        "Intro paragraph.\n\n"
        "## First section\n\n"
        "Body of first section.\n\n"
        "## Second section\n\n"
        "Body of second section.\n"
    )
    chunks = extract_file(p)
    # 1 preamble + 2 sections = 3 chunks
    assert len(chunks) == 3
    assert "First section" in chunks[1].text
    assert "Second section" in chunks[2].text


def test_extract_file_md_strips_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text(
        "---\n"
        "title: My Notes\n"
        "date: 2026-05-27\n"
        "---\n"
        "\n"
        "Actual content here.\n"
    )
    chunks = extract_file(p)
    assert len(chunks) == 1
    assert "title: My Notes" not in chunks[0].text
    assert "Actual content" in chunks[0].text


def test_extract_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="not found"):
        extract_file(tmp_path / "nope.txt")


# ---- markdown heading slug → Position (#47) ----


def test_extract_file_md_position_uses_heading_slug(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text(
        "## First section\n\n"
        "Body of first section.\n\n"
        "## Second section\n\n"
        "Body of second section.\n"
    )
    chunks = extract_file(p)
    assert chunks[0].position == "#first-section"
    assert chunks[1].position == "#second-section"
    assert chunks[0].metadata["anchor"] == "first-section"


def test_extract_file_md_preamble_has_empty_position(tmp_path: Path) -> None:
    """Content before the first heading has no slug to attribute to — empty Position."""
    p = tmp_path / "notes.md"
    p.write_text(
        "Intro paragraph with no heading.\n\n"
        "## First section\n\n"
        "Body.\n"
    )
    chunks = extract_file(p)
    assert chunks[0].text.startswith("Intro paragraph")
    assert chunks[0].position == ""
    assert chunks[1].position == "#first-section"


def test_extract_file_md_h3_picks_h3_not_parent_h2(tmp_path: Path) -> None:
    """Acceptance: chunk under ### picks the closest ### heading, not parent ##."""
    p = tmp_path / "notes.md"
    p.write_text(
        "## Top level\n\n"
        "Top body.\n\n"
        "### Nested subsection\n\n"
        "Nested body.\n"
    )
    chunks = extract_file(p)
    assert chunks[0].position == "#top-level"
    assert chunks[1].position == "#nested-subsection"


def test_extract_file_md_slug_handles_punctuation_and_case(tmp_path: Path) -> None:
    """Mixed-case + punctuation collapse to lowercase-hyphen runs, edges trimmed."""
    p = tmp_path / "notes.md"
    p.write_text(
        "### 1.4 Software introspection (free, zero hardware)\n\n"
        "Body.\n"
    )
    chunks = extract_file(p)
    assert chunks[0].position == "#1-4-software-introspection-free-zero-hardware"


def test_extract_file_txt_still_has_empty_position(tmp_path: Path) -> None:
    """Regression guard: .txt and inline text continue to produce empty Position."""
    p = tmp_path / "notes.txt"
    p.write_text("First paragraph.\n\nSecond paragraph.\n")
    chunks = extract_file(p)
    for c in chunks:
        assert c.position == ""
