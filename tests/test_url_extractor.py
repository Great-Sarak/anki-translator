"""Tests for the URL extractor.

No network calls — we feed canned HTML through extract() directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_translator.extractors import ExtractionError
from anki_translator.extractors.url import extract

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def article_html() -> str:
    return (FIXTURES / "article.html").read_text()


def test_extract_produces_paragraph_chunks(article_html: str) -> None:
    url = "https://example.com/cells"
    chunks = extract(article_html, url)
    # The article has 4 paragraph-sized chunks under main content
    assert len(chunks) >= 3
    # All chunks share the URL as source and 'url' as source_type
    for c in chunks:
        assert c.source == url
        assert c.source_type == "url"
        assert c.metadata["url"] == url


def test_extract_attaches_anchor_from_preceding_heading(article_html: str) -> None:
    url = "https://example.com/cells"
    chunks = extract(article_html, url)
    # Paragraphs under h2#organelles should have position '#organelles'
    organelle_chunks = [c for c in chunks if "mitochondria" in c.text.lower() or "nucleus" in c.text.lower()]
    assert organelle_chunks, "expected paragraphs under the Organelles heading"
    for c in organelle_chunks:
        assert c.position == "#organelles"


def test_extract_attaches_correct_anchor_to_membrane_paragraph(article_html: str) -> None:
    url = "https://example.com/cells"
    chunks = extract(article_html, url)
    membrane_chunks = [c for c in chunks if "selectively permeable" in c.text]
    assert len(membrane_chunks) == 1
    assert membrane_chunks[0].position == "#membrane"


def test_extract_strips_nav_and_footer(article_html: str) -> None:
    """trafilatura should drop <nav> and <footer>."""
    url = "https://example.com/cells"
    chunks = extract(article_html, url)
    all_text = " ".join(c.text for c in chunks)
    assert "Site Nav" not in all_text
    assert "Site Footer Boilerplate" not in all_text


def test_extract_raises_on_empty_html() -> None:
    with pytest.raises(ExtractionError):
        extract("", "https://example.com")


def test_extract_raises_on_unparseable_html() -> None:
    """Pure non-content HTML (a single button) should produce no chunks and raise."""
    minimal = "<html><body><button>Click</button></body></html>"
    with pytest.raises(ExtractionError):
        extract(minimal, "https://example.com/x")


def test_chunks_text_is_non_empty(article_html: str) -> None:
    url = "https://example.com/cells"
    chunks = extract(article_html, url)
    for c in chunks:
        assert c.text.strip(), "extractor produced an empty chunk"
