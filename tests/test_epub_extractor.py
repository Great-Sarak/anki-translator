"""Tests for the EPUB extractor.

Synthesizes .epub archives in tmp_path with zipfile — no binary fixture lives in the
repo (same convention as test_pdf_extractor.py). Covers both real-world organizations
deliberately: a book where structural matter is its own spine items (EPUB 2-style,
with a toc.ncx), and a journal article where the whole body is one spine item and the
structural matter lives inside it as sections (EPUB 3, nav document in the spine).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from anki_translator import cli
from anki_translator.citation import cite
from anki_translator.classifier import (
    CardCandidate,
    PREFILTER_METADATA_KEY,
    classify_chunks,
    split_prefiltered,
)
from anki_translator.config import load_citations, load_shapes
from anki_translator.extractors import ExtractionError
from anki_translator.extractors.epub import MIN_CHUNK_CHARS, extract
from anki_translator.queue import overflow_bucket

REPO_ROOT = Path(__file__).resolve().parent.parent

_CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="{opf_path}" media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""


def _opf(
    title: str | None,
    items: list[tuple[str, str, str, str]],
    spine: list[str],
    *,
    ncx: bool = False,
) -> str:
    """Render a minimal OPF. items: (id, href, media_type, properties). spine: idrefs."""
    meta = f"<dc:title>{title}</dc:title>" if title else ""
    manifest = "".join(
        f'<item id="{i}" href="{href}" media-type="{mt}"'
        + (f' properties="{props}"' if props else "")
        + "/>"
        for i, href, mt, props in items
    )
    itemrefs = "".join(f'<itemref idref="{idref}"/>' for idref in spine)
    spine_attrs = ' toc="ncx"' if ncx else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">'
        f"<metadata>{meta}</metadata>"
        f"<manifest>{manifest}</manifest>"
        f"<spine{spine_attrs}>{itemrefs}</spine>"
        "</package>"
    )


def _xhtml(body: str, *, title: str = "doc") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<head><title>{title}</title></head><body>{body}</body></html>"
    )


def _write_epub(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        for name, content in files.items():
            z.writestr(name, content)
    return path


def _pad(sentence: str) -> str:
    """Ensure fixture sentences clear MIN_CHUNK_CHARS."""
    assert len(sentence) >= MIN_CHUNK_CHARS, f"fixture sentence too short: {sentence!r}"
    return sentence


@pytest.fixture
def book_epub(tmp_path: Path) -> Path:
    """Book-shaped EPUB 2: structural matter as its own spine items, plus a toc.ncx.

    titlepage/dedication/endnotes flag via epub:type on their <section>; colophon has
    no epub:type and flags via the filename-stem heuristic instead. chapter-2 has no
    section id — its anchor comes from the heading id, as in url.py.
    """
    items = [
        ("titlepage", "titlepage.xhtml", "application/xhtml+xml", ""),
        ("dedication", "dedication.xhtml", "application/xhtml+xml", ""),
        ("ch1", "chapter-1.xhtml", "application/xhtml+xml", ""),
        ("ch2", "chapter-2.xhtml", "application/xhtml+xml", ""),
        ("endnotes", "endnotes.xhtml", "application/xhtml+xml", ""),
        ("colophon", "colophon.xhtml", "application/xhtml+xml", ""),
        ("ncx", "toc.ncx", "application/x-dtbncx+xml", ""),
    ]
    spine = ["titlepage", "dedication", "ch1", "ch2", "endnotes", "colophon"]
    return _write_epub(tmp_path / "book.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf("Test Book Title", items, spine, ncx=True),
        "titlepage.xhtml": _xhtml(
            '<section epub:type="titlepage"><h1>Test Book Title</h1>'
            f"<p>{_pad('By A. Test Author, translated for the extractor test suite.')}</p>"
            "</section>"
        ),
        "dedication.xhtml": _xhtml(
            '<section epub:type="dedication">'
            f"<p>{_pad('To everyone who reads test fixtures with genuine care.')}</p>"
            "</section>"
        ),
        "chapter-1.xhtml": _xhtml(
            '<section id="chapter-1" epub:type="chapter"><h2>Laying Plans</h2>'
            f"<p>{_pad('The art of war is of vital importance to the State.')}</p>"
            f"<p>{_pad('It is a matter of life and death, a road either to safety or to ruin.')}</p>"
            "</section>"
        ),
        "chapter-2.xhtml": _xhtml(
            '<section epub:type="chapter"><h2 id="ch2">Waging War</h2>'
            f"<p>{_pad('When you engage in actual fighting, victory must be quick.')}</p>"
            "</section>"
        ),
        "endnotes.xhtml": _xhtml(
            '<section epub:type="endnotes"><h2>Notes</h2>'
            f"<p>{_pad('Note one: published at Paris in 1782, per the fixture record.')}</p>"
            "</section>"
        ),
        "colophon.xhtml": _xhtml(
            "<section>"
            f"<p>{_pad('This colophon paragraph describes the typeface used in the fixture.')}</p>"
            "</section>"
        ),
        "toc.ncx": (
            '<?xml version="1.0"?>'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
            "<navMap><navPoint id=\"n1\"><navLabel><text>Chapter 1</text></navLabel>"
            '<content src="chapter-1.xhtml"/></navPoint></navMap></ncx>'
        ),
    })


@pytest.fixture
def article_epub(tmp_path: Path) -> Path:
    """Article-shaped EPUB 3: one body spine item; structural matter is sections inside it.

    Mimics the SAGE journal epubs: body prose is <div role="paragraph">, references are
    <div role="listitem">, the nav document sits in the spine (linear="no") with
    properties="nav", and a figure-wrapper item carries only a <figcaption>. The
    acknowledgments section has no epub:type — it flags via heading text, exercising
    url.py's _boilerplate_kind seam.
    """
    items = [
        ("index", "xhtml/index.xhtml", "application/xhtml+xml", ""),
        ("nav", "xhtml/nav.xhtml", "application/xhtml+xml", "nav"),
        ("fig1", "xhtml/fig1.xhtml", "application/xhtml+xml", ""),
    ]
    return _write_epub(tmp_path / "article.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="EPUB/package.opf"),
        "EPUB/package.opf": _opf("Test Article Title", items, ["index", "nav", "fig1"]),
        "EPUB/xhtml/index.xhtml": _xhtml(
            "<h1>Test Article Title</h1>"
            '<section id="abstract" epub:type="abstract"><h2>Abstract</h2>'
            f'<div role="paragraph">{_pad("This abstract summarizes the fixture article findings.")}</div>'
            "</section>"
            '<section id="sec-1"><h2>Introduction</h2>'
            f'<div role="paragraph">{_pad("Whole slide imaging is central to modern pathology workflows.")}</div>'
            f'<div role="paragraph">{_pad("Color reproducibility across scanners remains poorly characterized.")}</div>'
            "</section>"
            '<section id="bibliography" epub:type="bibliography"><h2>References</h2>'
            f'<div role="listitem">{_pad("Adams E. T. (2013). Basic approaches in anatomic pathology.")}</div>'
            f'<div role="listitem">{_pad("Bolon B. (2015). Another citation entry for the fixture list.")}</div>'
            "</section>"
            '<section id="ack"><h2>Acknowledgments</h2>'
            f'<div role="paragraph">{_pad("The authors thank the fixture maintainers for their patience.")}</div>'
            "</section>"
        ),
        "EPUB/xhtml/nav.xhtml": _xhtml(
            '<nav epub:type="toc"><h1>Sections</h1><ol>'
            '<li><a href="index.xhtml#abstract">Abstract</a></li>'
            '<li><a href="index.xhtml#sec-1">Introduction</a></li>'
            "</ol></nav>"
        ),
        "EPUB/xhtml/fig1.xhtml": _xhtml(
            '<figure epub:type="figure"><img src="../images/fig1.jpg"/>'
            f"<figcaption>{_pad('Figure 1. Decision tree for the fixture grading criteria.')}</figcaption>"
            "</figure>"
        ),
    })


def _find(chunks, prefix: str):
    """Return the single chunk whose text starts with prefix."""
    matches = [c for c in chunks if c.text.startswith(prefix)]
    assert len(matches) == 1, f"expected 1 chunk starting {prefix!r}, got {len(matches)}"
    return matches[0]


# ---- book-shaped organization ----


def test_book_spine_order_and_titles(book_epub: Path) -> None:
    chunks = extract(book_epub)
    assert [c.source for c in chunks] == ["Test Book Title"] * len(chunks)
    assert all(c.source_type == "epub" for c in chunks)
    spine_sequence = [c.metadata["spine_item"] for c in chunks]
    assert spine_sequence == sorted(spine_sequence, key=["titlepage.xhtml", "dedication.xhtml", "chapter-1.xhtml", "chapter-2.xhtml", "endnotes.xhtml", "colophon.xhtml"].index)


def test_book_structural_items_flagged(book_epub: Path) -> None:
    chunks = extract(book_epub)
    kind_by_item: dict[str, set] = {}
    for c in chunks:
        kind_by_item.setdefault(str(c.metadata["spine_item"]), set()).add(
            c.metadata.get(PREFILTER_METADATA_KEY)
        )
    assert kind_by_item["titlepage.xhtml"] == {"title-page"}
    assert kind_by_item["dedication.xhtml"] == {"dedication"}
    assert kind_by_item["endnotes.xhtml"] == {"endnotes"}
    # colophon.xhtml carries no epub:type — flagged by filename stem alone
    assert kind_by_item["colophon.xhtml"] == {"colophon"}
    # body chapters are not flagged
    assert kind_by_item["chapter-1.xhtml"] == {None}
    assert kind_by_item["chapter-2.xhtml"] == {None}


def test_book_positions(book_epub: Path) -> None:
    chunks = extract(book_epub)
    # anchor from the enclosing section id
    assert _find(chunks, "The art of war").position == "chapter-1.xhtml#chapter-1"
    # anchor from the heading id when the section has none
    assert _find(chunks, "When you engage").position == "chapter-2.xhtml#ch2"
    for c in chunks:
        assert c.metadata["title"] == "Test Book Title"
        assert c.metadata["filename"] == "book.epub"
        assert "spine_item" in c.metadata and "anchor" in c.metadata


def test_toc_ncx_never_extracted(book_epub: Path) -> None:
    """The EPUB 2 NCX is not XHTML and must never become a chunk."""
    chunks = extract(book_epub)
    assert all("toc.ncx" not in str(c.metadata["spine_item"]) for c in chunks)


# ---- article-shaped organization ----


def test_article_sections_flagged(article_epub: Path) -> None:
    chunks = extract(article_epub)
    assert _find(chunks, "This abstract").metadata[PREFILTER_METADATA_KEY] == "abstract"
    assert _find(chunks, "Adams E. T.").metadata[PREFILTER_METADATA_KEY] == "bibliography"
    assert _find(chunks, "Bolon B.").metadata[PREFILTER_METADATA_KEY] == "bibliography"
    # acknowledgments has no epub:type — caught by the heading text seam
    assert _find(chunks, "The authors thank").metadata[PREFILTER_METADATA_KEY] == "author-info"
    # body sections are not flagged
    assert PREFILTER_METADATA_KEY not in _find(chunks, "Whole slide imaging").metadata
    assert PREFILTER_METADATA_KEY not in _find(chunks, "Color reproducibility").metadata


def test_article_position_uses_section_anchor(article_epub: Path) -> None:
    chunks = extract(article_epub)
    assert _find(chunks, "Whole slide imaging").position == "EPUB/xhtml/index.xhtml#sec-1"


def test_nav_item_produces_no_chunks(article_epub: Path) -> None:
    """The nav document is link lists only; even so, nothing unflagged may leak from it."""
    chunks = extract(article_epub)
    nav_chunks = [c for c in chunks if c.metadata["spine_item"] == "EPUB/xhtml/nav.xhtml"]
    assert all(PREFILTER_METADATA_KEY in c.metadata for c in nav_chunks)


def test_figure_caption_flagged(article_epub: Path) -> None:
    chunks = extract(article_epub)
    captions = [c for c in chunks if c.metadata["spine_item"] == "EPUB/xhtml/fig1.xhtml"]
    assert len(captions) == 1
    assert captions[0].metadata[PREFILTER_METADATA_KEY] == "figure-caption"


# ---- S3 wiring: zero classifier dispatch ----


def test_prefiltered_chunks_route_to_trimmed_without_classifier(book_epub: Path) -> None:
    """Negative check: structural chunks reach trimmed AND the classifier never sees them."""
    chunks = extract(book_epub)
    to_classify, prefiltered = split_prefiltered(chunks)

    flagged = [c for c in chunks if PREFILTER_METADATA_KEY in c.metadata]
    assert flagged, "fixture must produce structural chunks"
    assert len(prefiltered) == len(flagged)
    for ov in prefiltered:
        assert ov.reason.startswith("extractor: pre-filtered")
        assert overflow_bucket(ov.reason, ov.bucket) == "trimmed"

    dispatched: list[str] = []

    def spy_llm(prompt: str) -> str:
        dispatched.append(prompt)
        return '{"choice": "overflow", "reason": "stub", "bucket": "qa"}'

    shapes = load_shapes(REPO_ROOT / "config" / "shapes.yaml")
    results = classify_chunks(to_classify, shapes, llm=spy_llm, max_workers=1)
    assert len(results) == len(to_classify)
    # every dispatched prompt is for a body chunk; no flagged chunk text was classified
    for chunk in flagged:
        assert all(chunk.text not in prompt for prompt in dispatched)
    assert len(dispatched) == len(to_classify)


def test_classifier_stub_never_invoked_for_article_front_back_matter(article_epub: Path) -> None:
    """Same guarantee for the single-spine-item organization.

    #71 review, finding D: the previous version asserted
    ``calls == len(to_classify)``, which is trivially true for ANY input —
    classify_chunks calls its llm once per element, so leaking flagged chunks
    into to_classify kept the test green. This version observes the forbidden
    boundary directly: it records every dispatched prompt and asserts no
    flagged chunk's text appears in any of them, exactly like the book test.
    """
    chunks = extract(article_epub)
    to_classify, prefiltered = split_prefiltered(chunks)
    assert prefiltered, "article fixture must produce structural chunks"
    assert all(overflow_bucket(ov.reason, ov.bucket) == "trimmed" for ov in prefiltered)

    dispatched: list[str] = []

    def spy_llm(prompt: str) -> str:
        dispatched.append(prompt)
        return '{"choice": "overflow", "reason": "stub", "bucket": "qa"}'

    shapes = load_shapes(REPO_ROOT / "config" / "shapes.yaml")
    classify_chunks(to_classify, shapes, llm=spy_llm, max_workers=1)
    # Identity, not count: no flagged chunk text was dispatched to the LLM,
    # and every dispatched chunk is an unflagged body paragraph.
    flagged = [c for c in chunks if PREFILTER_METADATA_KEY in c.metadata]
    assert flagged, "article fixture must produce structural chunks"
    for chunk in flagged:
        assert all(chunk.text not in prompt for prompt in dispatched)
    assert {c.text for c in to_classify}.isdisjoint(c.text for c in flagged)
    assert len(dispatched) == len(to_classify)


def test_body_chunks_produce_candidates(book_epub: Path) -> None:
    """The happy path alone is not enough, but body chapters must still be classifiable."""
    chunks = extract(book_epub)
    to_classify, _ = split_prefiltered(chunks)
    assert to_classify, "body chapters must survive the pre-filter"
    stub = lambda _prompt: (  # noqa: E731
        '{"choice": "AT Basic", "fields": {"Front": "q", "Back": "a"}}'
    )
    shapes = load_shapes(REPO_ROOT / "config" / "shapes.yaml")
    results = classify_chunks(to_classify, shapes, llm=stub, max_workers=1)
    assert all(isinstance(r, CardCandidate) for r in results)


# ---- citation integration ----


def test_citations_yaml_epub_entry_loads() -> None:
    conventions = load_citations(REPO_ROOT / "config" / "citations.yaml")
    assert "epub" in conventions
    assert conventions["epub"].source_required == ["title"]


def test_cite_epub_chunk(book_epub: Path) -> None:
    conventions = load_citations(REPO_ROOT / "config" / "citations.yaml")
    chunk = extract(book_epub)[0]
    source, position = cite("epub", chunk.metadata, conventions)
    assert source == "Test Book Title"
    assert position == chunk.position


# ---- CLI dispatch ----


def test_cli_dispatches_epub_suffix(book_epub: Path) -> None:
    import argparse

    args = argparse.Namespace(source=str(book_epub), text=None, label=None)
    chunks = cli._dispatch_extractor(args)
    assert chunks and all(c.source_type == "epub" for c in chunks)


# ---- error handling ----


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="EPUB not found"):
        extract(tmp_path / "nope.epub")


def test_non_zip_raises(tmp_path: Path) -> None:
    p = tmp_path / "fake.epub"
    p.write_bytes(b"this is not a zip archive at all")
    with pytest.raises(ExtractionError, match="not a zip archive"):
        extract(p)


def test_zip_without_container_raises(tmp_path: Path) -> None:
    p = tmp_path / "bare.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("index.html", "<p>hello</p>")
    with pytest.raises(ExtractionError, match="container.xml missing"):
        extract(p)


def test_no_paragraphs_raises(tmp_path: Path) -> None:
    """An epub whose only spine item is a nav document yields nothing."""
    p = _write_epub(tmp_path / "navonly.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Nav Only",
            [("nav", "nav.xhtml", "application/xhtml+xml", "nav")],
            ["nav"],
        ),
        "nav.xhtml": _xhtml('<nav epub:type="toc"><ol><li><a href="x">One</a></li></ol></nav>'),
    })
    with pytest.raises(ExtractionError, match="no paragraphs extracted"):
        extract(p)


def test_title_falls_back_to_filename(tmp_path: Path) -> None:
    p = _write_epub(tmp_path / "untitled.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            None,
            [("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ch"],
        ),
        "chapter.xhtml": _xhtml(
            f"<p>{_pad('A body paragraph in a title-less epub fixture document.')}</p>"
        ),
    })
    chunks = extract(p)
    assert chunks[0].source == "untitled.epub"
    assert chunks[0].metadata["title"] == "untitled.epub"
    assert chunks[0].metadata["filename"] == "untitled.epub"


def test_sub_floor_unflagged_fragments_dropped(tmp_path: Path) -> None:
    """Unflagged paragraphs under MIN_CHUNK_CHARS are dropped silently (pdf.py parity),
    but sub-floor STRUCTURAL matter (here a short caption) is preserved and flagged —
    the floor must not delete known structural content before S3 can route it."""
    p = _write_epub(tmp_path / "tiny.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Tiny",
            [("fig", "fig1.xhtml", "application/xhtml+xml", ""),
             ("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["fig", "ch"],
        ),
        "fig1.xhtml": _xhtml("<figure><figcaption>Figure 1.</figcaption></figure>"),
        "chapter.xhtml": _xhtml(
            "<p>Short.</p>"
            f"<p>{_pad('A real body paragraph that clearly survives the size floor.')}</p>"
        ),
    })
    chunks = extract(p)
    assert len("Figure 1.") < MIN_CHUNK_CHARS and len("Short.") < MIN_CHUNK_CHARS
    # sub-floor caption: preserved, flagged, routed to trimmed via S3
    caption = _find(chunks, "Figure 1.")
    assert caption.metadata[PREFILTER_METADATA_KEY] == "figure-caption"
    _, prefiltered = split_prefiltered(chunks)
    assert any(ov.reason.startswith("extractor: pre-filtered") for ov in prefiltered)
    # sub-floor UNFLAGGED body fragment: still dropped silently
    assert all(not c.text.startswith("Short.") for c in chunks)
    assert _find(chunks, "A real body paragraph").metadata.get(PREFILTER_METADATA_KEY) is None
# ---- #71 review regressions ----


def test_heading_state_scoped_to_its_section(tmp_path: Path) -> None:
    """Finding 1: a heading inside a structural section must not flag a heading-less
    body sibling that follows it. The References heading flags the citation inside
    its own section; the trailing body paragraph stays classifiable."""
    p = _write_epub(tmp_path / "scope.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Scoped",
            [("body", "body.xhtml", "application/xhtml+xml", "")],
            ["body"],
        ),
        "body.xhtml": _xhtml(
            '<section epub:type="bibliography"><h2>References</h2>'
            f"<p>{_pad('Smith J. (2020). A citation entry inside the references section.')}</p>"
            "</section>"
            "<section>"
            f"<p>{_pad('A substantive body paragraph that follows the references and has no heading.')}</p>"
            "</section>"
        ),
    })
    chunks = extract(p)
    citation = _find(chunks, "Smith J.")
    assert citation.metadata[PREFILTER_METADATA_KEY] in ("bibliography", "references")
    body = _find(chunks, "A substantive body paragraph")
    assert PREFILTER_METADATA_KEY not in body.metadata
    # and the body paragraph still reaches the classifier path
    to_classify, prefiltered = split_prefiltered(chunks)
    assert [c.text for c in to_classify] == [body.text]
    assert all(
        overflow_bucket(ov.reason, ov.bucket) == "trimmed" for ov in prefiltered
    )


def test_structural_headings_recognized_without_epub_type(tmp_path: Path) -> None:
    """Finding 2(a): index.xhtml carrying <h1>Index</h1> but no epub:type — the bare
    'index' stem is (correctly) not a filename signal, so the heading must catch it."""
    p = _write_epub(tmp_path / "bookindex.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Book Index",
            [("idx", "index.xhtml", "application/xhtml+xml", ""),
             ("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["idx", "ch"],
        ),
        "index.xhtml": _xhtml(
            "<section><h1>Index</h1>"
            f"<p>{_pad('Alpha, 12, 45. Beta, 33. Gamma, 78. A book-index entry paragraph.')}</p>"
            "</section>"
        ),
        "chapter.xhtml": _xhtml(
            f"<p>{_pad('Ordinary body content living in the chapter spine item.')}</p>"
        ),
    })
    chunks = extract(p)
    assert _find(chunks, "Alpha, 12, 45").metadata[PREFILTER_METADATA_KEY] == "index"
    assert PREFILTER_METADATA_KEY not in _find(chunks, "Ordinary body content").metadata


def test_generic_backmatter_class_with_structural_heading(tmp_path: Path) -> None:
    """Finding 2(b): epub:type='backmatter' stays a non-signal by itself, but a
    Colophon heading inside it must still flag the colophon paragraph."""
    p = _write_epub(tmp_path / "backmatter.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Backmatter",
            [("bm", "backmatter.xhtml", "application/xhtml+xml", ""),
             ("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["bm", "ch"],
        ),
        "backmatter.xhtml": _xhtml(
            '<section epub:type="backmatter"><h2>Colophon</h2>'
            f"<p>{_pad('This book was set in a fixture typeface by the test suite press.')}</p>"
            "</section>"
        ),
        "chapter.xhtml": _xhtml(
            f"<p>{_pad('Ordinary body content living in the chapter spine item.')}</p>"
        ),
    })
    chunks = extract(p)
    assert _find(chunks, "This book was set").metadata[PREFILTER_METADATA_KEY] == "colophon"
    assert PREFILTER_METADATA_KEY not in _find(chunks, "Ordinary body content").metadata


def test_short_structural_content_routed_through_s3(tmp_path: Path) -> None:
    """Finding 3: a dedication shorter than MIN_CHUNK_CHARS must survive as a flagged
    chunk and reach trimmed through S3 with zero classifier dispatch — not vanish."""
    p = _write_epub(tmp_path / "shortded.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Short Dedication",
            [("ded", "dedication.xhtml", "application/xhtml+xml", ""),
             ("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ded", "ch"],
        ),
        "dedication.xhtml": _xhtml(
            '<section epub:type="dedication"><p>For Alice.</p></section>'
        ),
        "chapter.xhtml": _xhtml(
            f"<p>{_pad('Ordinary body content living in the chapter spine item.')}</p>"
        ),
    })
    assert len("For Alice.") < MIN_CHUNK_CHARS  # guards the fixture's premise
    chunks = extract(p)
    dedication = _find(chunks, "For Alice.")
    assert dedication.metadata[PREFILTER_METADATA_KEY] == "dedication"
    to_classify, prefiltered = split_prefiltered(chunks)
    assert len(prefiltered) == 1
    assert overflow_bucket(prefiltered[0].reason, prefiltered[0].bucket) == "trimmed"
    # zero dispatch for the structural chunk; the body paragraph is all the classifier sees
    assert [c.text for c in to_classify] == ["Ordinary body content living in the chapter spine item."]


# ---- #71 review round 2 regressions ----


def test_dpub_role_token_list_still_prefilters(tmp_path: Path) -> None:
    """Finding A: role is a space-separated token list. A recognized doc-* role
    must prefilter even when accompanied by additional roles — the parser used
    to do one exact lookup of the whole attribute value, so
    role="doc-bibliography region" fell through to the classifier."""
    p = _write_epub(tmp_path / "roles.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Role Tokens",
            [("body", "body.xhtml", "application/xhtml+xml", "")],
            ["body"],
        ),
        "body.xhtml": _xhtml(
            '<section role="doc-bibliography region"><h2>Works</h2>'
            f"<p>{_pad('Smith J. (2021). A citation inside a multi-role bibliography section.')}</p>"
            "</section>"
            "<section>"
            f"<p>{_pad('A substantive body paragraph outside the bibliography section.')}</p>"
            "</section>"
        ),
    })
    chunks = extract(p)
    citation = _find(chunks, "Smith J.")
    assert citation.metadata[PREFILTER_METADATA_KEY] == "bibliography"
    assert PREFILTER_METADATA_KEY not in _find(chunks, "A substantive body paragraph").metadata


def test_unknown_spine_idref_fails_loudly(tmp_path: Path) -> None:
    """Finding B: a spine itemref whose idref resolves to no manifest item must
    abort extraction naming the bad idref, not be silently dropped."""
    items = [
        ("one", "one.xhtml", "application/xhtml+xml", ""),
        ("two", "two.xhtml", "application/xhtml+xml", ""),
    ]
    p = _write_epub(tmp_path / "ghost.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf("Ghost", items, ["one", "ghost", "two"]),
        "one.xhtml": _xhtml(
            f"<p>{_pad('First body paragraph of a book with a broken spine reference.')}</p>"
        ),
        "two.xhtml": _xhtml(
            f"<p>{_pad('Second body paragraph of a book with a broken spine reference.')}</p>"
        ),
    })
    with pytest.raises(ExtractionError, match="unknown manifest id 'ghost'"):
        extract(p)


def test_missing_spine_member_fails_loudly(tmp_path: Path) -> None:
    """Finding B: a declared XHTML spine member absent from the ZIP must abort
    extraction naming the href, not be silently skipped (which produced a
    partial book with no warning)."""
    items = [
        ("one", "one.xhtml", "application/xhtml+xml", ""),
        ("gone", "missing.xhtml", "application/xhtml+xml", ""),
        ("two", "two.xhtml", "application/xhtml+xml", ""),
    ]
    p = _write_epub(tmp_path / "partial.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf("Partial", items, ["one", "gone", "two"]),
        "one.xhtml": _xhtml(
            f"<p>{_pad('First body paragraph of a book missing a spine member.')}</p>"
        ),
        # missing.xhtml deliberately absent from the archive
        "two.xhtml": _xhtml(
            f"<p>{_pad('Second body paragraph of a book missing a spine member.')}</p>"
        ),
    })
    with pytest.raises(ExtractionError, match="missing.xhtml.*absent from the archive"):
        extract(p)


def _write_epub_deflated(path: Path, files: dict[str, str]) -> Path:
    """Like _write_epub but DEFLATE-compresses members — needed to exercise the
    compression-ratio guard (the default stored fixtures all sit at ratio 1)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        for name, content in files.items():
            z.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
    return path


def test_large_member_ratio_over_10x_rejected(tmp_path: Path) -> None:
    """Finding C: a spine member >5 MB uncompressed may compress at most 10x.
    This ~6.6 MB member deflates ~1000:1 — a zip-bomb shape — and must be
    rejected before decompression, with the error naming the member."""
    body = f"<p>{_pad('A body paragraph of a hostile archive member.')}</p>" * 110_000
    p = _write_epub_deflated(tmp_path / "bomb.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Bomb",
            [("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ch"],
        ),
        "chapter.xhtml": _xhtml(body),
    })
    with zipfile.ZipFile(p) as zf:
        info = zf.getinfo("chapter.xhtml")
        assert info.file_size > 5 * 1024 * 1024  # premise: large-member rule applies
        assert info.file_size / info.compress_size > 10
    with pytest.raises(ExtractionError, match="chapter.xhtml.*decompression budget"):
        extract(p)


def test_small_member_ratio_over_20x_rejected(tmp_path: Path) -> None:
    """Finding C: a member at or under 5 MB uncompressed may compress at most
    20x. A ~150 KB run of one repeated character deflates ~1000:1 and trips the
    small-member limit."""
    body = f"<p>{'a' * 150_000}</p>"
    p = _write_epub_deflated(tmp_path / "smallbomb.epub", {
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Small Bomb",
            [("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ch"],
        ),
        "chapter.xhtml": _xhtml(body),
    })
    with zipfile.ZipFile(p) as zf:
        info = zf.getinfo("chapter.xhtml")
        assert info.file_size <= 5 * 1024 * 1024  # premise: small-member rule applies
        assert info.file_size / info.compress_size > 20
    with pytest.raises(ExtractionError, match="chapter.xhtml.*decompression budget"):
        extract(p)


def test_bomb_guard_covers_container_and_opf_reads(tmp_path: Path) -> None:
    """Finding C: the guard is on all three read sites, not just the spine loop.
    A bomb-shaped container.xml must be rejected before the OPF is even located."""
    padded_container = _CONTAINER.format(opf_path="content.opf").replace(
        "</container>", "<!--" + " " * 200_000 + "--></container>"
    )
    p = _write_epub_deflated(tmp_path / "cbomb.epub", {
        "META-INF/container.xml": padded_container,
        "content.opf": _opf(
            "Container Bomb",
            [("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ch"],
        ),
        "chapter.xhtml": _xhtml(
            f"<p>{_pad('A body paragraph that must never be reached.')}</p>"
        ),
    })
    with zipfile.ZipFile(p) as zf:
        info = zf.getinfo("META-INF/container.xml")
        assert info.file_size / info.compress_size > 20
    with pytest.raises(ExtractionError, match="META-INF/container.xml.*decompression budget"):
        extract(p)


def test_stored_large_member_passes_guard(tmp_path: Path) -> None:
    """Finding C calibration: the budget gates compression RATIO, not size.
    A >5 MB member stored uncompressed (ratio 1) is a big book, not a bomb,
    and must extract normally."""
    paragraph = _pad("A real body paragraph of a large but honestly stored book.")
    body = f"<p>{paragraph}</p>" * 90_000  # ~6.7 MB uncompressed
    p = _write_epub(tmp_path / "bigstored.epub", {  # stored, not deflated
        "META-INF/container.xml": _CONTAINER.format(opf_path="content.opf"),
        "content.opf": _opf(
            "Big Stored",
            [("ch", "chapter.xhtml", "application/xhtml+xml", "")],
            ["ch"],
        ),
        "chapter.xhtml": _xhtml(body),
    })
    with zipfile.ZipFile(p) as zf:
        info = zf.getinfo("chapter.xhtml")
        assert info.file_size > 5 * 1024 * 1024
        assert info.compress_size == info.file_size  # stored: ratio 1
    chunks = extract(p)
    assert chunks and all(c.text == paragraph for c in chunks)
