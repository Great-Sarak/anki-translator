"""Tests for citation.cite()."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_translator.citation import CitationError, cite
from anki_translator.config import load_citations

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def conventions() -> dict:
    return load_citations(REPO_ROOT / "config" / "citations.yaml")


# ---- url ----


def test_url_with_anchor(conventions: dict) -> None:
    source, position = cite("url", {"url": "https://example.com/article", "anchor": "#install"}, conventions)
    assert source == "https://example.com/article"
    assert position == "#install"


def test_url_without_anchor(conventions: dict) -> None:
    source, position = cite("url", {"url": "https://example.com/article"}, conventions)
    assert source == "https://example.com/article"
    assert position == ""


def test_url_missing_required(conventions: dict) -> None:
    with pytest.raises(CitationError, match="missing required metadata"):
        cite("url", {"anchor": "#install"}, conventions)


# ---- doi ----


def test_doi_with_section(conventions: dict) -> None:
    source, position = cite("doi", {"doi": "10.1234/abc", "section": "3.2"}, conventions)
    assert source == "10.1234/abc"
    assert position == "3.2"


def test_doi_without_section(conventions: dict) -> None:
    source, position = cite("doi", {"doi": "10.1234/abc"}, conventions)
    assert source == "10.1234/abc"
    assert position == ""


# ---- pdf ----


def test_pdf_with_page(conventions: dict) -> None:
    source, position = cite("pdf", {"filename": "kleppmann.pdf", "page": 47}, conventions)
    assert source == "kleppmann.pdf"
    assert position == "page 47"


def test_pdf_missing_page_returns_empty_position(conventions: dict) -> None:
    """PDF page is in position_required; missing page → empty position, source still produced."""
    source, position = cite("pdf", {"filename": "kleppmann.pdf"}, conventions)
    assert source == "kleppmann.pdf"
    assert position == ""


def test_pdf_missing_filename_raises(conventions: dict) -> None:
    with pytest.raises(CitationError, match="missing required metadata"):
        cite("pdf", {"page": 47}, conventions)


# ---- book ----


def test_book_with_chapter_page(conventions: dict) -> None:
    source, position = cite(
        "book",
        {"title": "DDIA", "edition": "1st ed.", "chapter": 1, "page": 12},
        conventions,
    )
    assert source == "DDIA, 1st ed."
    assert position == "ch. 1, p. 12"


def test_book_missing_chapter_returns_empty_position(conventions: dict) -> None:
    """Both chapter and page are required for position; missing chapter → empty (no 'ch. , p. 12')."""
    source, position = cite(
        "book",
        {"title": "DDIA", "edition": "1st ed.", "page": 12},
        conventions,
    )
    assert source == "DDIA, 1st ed."
    assert position == ""


def test_book_missing_edition_raises(conventions: dict) -> None:
    with pytest.raises(CitationError, match="missing required metadata"):
        cite("book", {"title": "DDIA", "chapter": 1, "page": 12}, conventions)


# ---- manual ----


def test_manual_with_label(conventions: dict) -> None:
    source, position = cite("manual", {"label": "chat 2026-05-27"}, conventions)
    assert source == "chat 2026-05-27"
    assert position == ""  # always empty for manual


def test_manual_position_ignored_even_if_provided(conventions: dict) -> None:
    """Manual's position_template is empty; extra metadata is harmless."""
    source, position = cite("manual", {"label": "notes", "page": 99}, conventions)
    assert source == "notes"
    assert position == ""


def test_manual_missing_label_raises(conventions: dict) -> None:
    with pytest.raises(CitationError, match="missing required metadata"):
        cite("manual", {}, conventions)


# ---- general ----


def test_unknown_source_type_raises(conventions: dict) -> None:
    with pytest.raises(CitationError, match="unknown source type"):
        cite("podcast", {"episode": "42"}, conventions)


def test_whitespace_only_source_raises(conventions: dict) -> None:
    """Defense in depth: even if all required keys are 'present', whitespace doesn't count."""
    with pytest.raises(CitationError, match="missing required metadata"):
        cite("url", {"url": "   "}, conventions)
