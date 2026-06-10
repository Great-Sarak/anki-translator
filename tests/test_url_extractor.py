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


# ---- structural pre-filter heuristics (#70, S6) ----

from anki_translator.classifier import PREFILTER_METADATA_KEY  # noqa: E402
from anki_translator.extractors.url import _boilerplate_kind  # noqa: E402


def test_boilerplate_kind_maps_known_sections() -> None:
    assert _boilerplate_kind("references") == "references"
    assert _boilerplate_kind("notes") == "references"
    assert _boilerplate_kind("bibliography") == "references"
    assert _boilerplate_kind("cited by") == "cited-by"
    assert _boilerplate_kind("author information") == "author-info"
    assert _boilerplate_kind("acknowledgements") == "author-info"
    assert _boilerplate_kind("external links") == "further-reading"
    assert _boilerplate_kind("see also") == "further-reading"


def test_boilerplate_kind_keeps_content_sections() -> None:
    """Conservative: an unrecognized (content) section keeps its paragraphs."""
    assert _boilerplate_kind("function") is None
    assert _boilerplate_kind("structure and evolution") is None
    assert _boilerplate_kind("") is None


def test_extract_flags_boilerplate_sections_not_article_body() -> None:
    """End-to-end: paragraphs under References / Cited by / Author information are
    pre-filtered by the enclosing heading; the article-body paragraph is kept."""
    html = """
    <html><body>
      <h2 id="function">Function</h2>
      <p>The mitochondrion generates most of the cell's supply of ATP.</p>
      <h2 id="references">References</h2>
      <p>Alberts B, Johnson A (2002). Molecular Biology of the Cell.</p>
      <h2 id="cited-by">Cited by</h2>
      <p>This article has been cited by 1,284 other works.</p>
      <h2 id="author-information">Author information</h2>
      <p>Affiliations: Department of Cell Biology, Harvard Medical School.</p>
    </body></html>
    """
    chunks = extract(html, "https://en.wikipedia.org/wiki/Mitochondrion")
    flags = [(c.position, c.metadata.get(PREFILTER_METADATA_KEY)) for c in chunks]
    assert flags == [
        ("#function", None),
        ("#references", "references"),
        ("#cited-by", "cited-by"),
        ("#author-information", "author-info"),
    ]


def test_extract_section_title_tracks_heading_text_not_id() -> None:
    """Flagging is driven by heading TEXT, so a varied/missing id still works."""
    html = """
    <html><body>
      <h2>Overview</h2>
      <p>A real overview paragraph with substantive content to learn.</p>
      <h2>External links</h2>
      <p>A list of outbound links that survived extraction.</p>
    </body></html>
    """
    chunks = extract(html, "https://example.com/article")
    flags = [c.metadata.get(PREFILTER_METADATA_KEY) for c in chunks]
    assert flags == [None, "further-reading"]
