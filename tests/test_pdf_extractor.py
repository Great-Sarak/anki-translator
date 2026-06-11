"""Tests for the PDF extractor.

Uses pymupdf to synthesize a small test PDF in a tmp_path fixture so no binary needs to
live in the repo.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from anki_translator.extractors import ExtractionError
from anki_translator.extractors.pdf import MIN_CHUNK_CHARS, extract, extract_bytes


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Two-page PDF: page 1 has two paragraph-shaped blocks, page 2 has one."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()  # new empty doc

    page1 = doc.new_page()
    page1.insert_textbox(
        pymupdf.Rect(72, 72, 540, 200),
        "The mitochondria is the powerhouse of the cell, "
        "producing ATP through oxidative phosphorylation in eukaryotic organisms.",
        fontsize=11,
    )
    page1.insert_textbox(
        pymupdf.Rect(72, 220, 540, 340),
        "The nucleus houses the cell's genetic material and coordinates "
        "activities including growth, metabolism, and reproduction.",
        fontsize=11,
    )

    page2 = doc.new_page()
    page2.insert_textbox(
        pymupdf.Rect(72, 72, 540, 200),
        "The cell membrane is a selectively permeable barrier "
        "that controls what enters and leaves the cell.",
        fontsize=11,
    )

    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def short_blocks_pdf(tmp_path: Path) -> Path:
    """PDF with only short text blocks — should produce no chunks (header/footer noise)."""
    path = tmp_path / "short.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 100), "Header", fontsize=11)
    page.insert_textbox(pymupdf.Rect(72, 700, 540, 750), "Page 1", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_extract_produces_per_block_chunks(sample_pdf: Path) -> None:
    chunks = extract(sample_pdf)
    # Two paragraphs on page 1, one on page 2 = 3 chunks
    assert len(chunks) == 3


def test_chunks_have_filename_as_source(sample_pdf: Path) -> None:
    chunks = extract(sample_pdf)
    for c in chunks:
        assert c.source == "sample.pdf"
        assert c.source_type == "pdf"
        assert c.metadata["filename"] == "sample.pdf"


def test_chunks_have_correct_page_position(sample_pdf: Path) -> None:
    chunks = extract(sample_pdf)
    page1_chunks = [c for c in chunks if c.position == "page 1"]
    page2_chunks = [c for c in chunks if c.position == "page 2"]
    assert len(page1_chunks) == 2
    assert len(page2_chunks) == 1
    assert page2_chunks[0].metadata["page"] == 2


def test_extract_skips_short_blocks(short_blocks_pdf: Path) -> None:
    """Headers/footers/page numbers are noise; should raise rather than emit junk chunks."""
    with pytest.raises(ExtractionError, match="image-only|no text blocks"):
        extract(short_blocks_pdf)


def test_extract_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="not found"):
        extract(tmp_path / "nope.pdf")


def test_extract_raises_on_non_pdf(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_text("This is not a PDF at all")
    with pytest.raises(ExtractionError, match="could not open"):
        extract(fake)


def test_extract_raises_on_image_only_pdf(tmp_path: Path) -> None:
    """A PDF with no text content should fail with a clear OCR hint."""
    path = tmp_path / "image_only.pdf"
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    doc.save(path)
    doc.close()
    with pytest.raises(ExtractionError, match="image-only"):
        extract(path)


def test_chunk_text_meets_min_length(sample_pdf: Path) -> None:
    """Sanity: every emitted chunk text is at least MIN_CHUNK_CHARS long."""
    chunks = extract(sample_pdf)
    for c in chunks:
        assert len(c.text) >= MIN_CHUNK_CHARS


# ---- extract_bytes ----


@pytest.fixture
def sample_pdf_bytes(sample_pdf: Path) -> bytes:
    return sample_pdf.read_bytes()


def test_extract_bytes_produces_same_chunks_as_extract(sample_pdf: Path, sample_pdf_bytes: bytes) -> None:
    from_file = extract(sample_pdf)
    from_bytes = extract_bytes(sample_pdf_bytes, "sample.pdf")
    assert len(from_file) == len(from_bytes)
    for a, b in zip(from_file, from_bytes):
        assert a.text == b.text
        assert a.position == b.position
        assert a.source_type == b.source_type


def test_extract_bytes_uses_source_label(sample_pdf_bytes: bytes) -> None:
    label = "https://example.com/paper.pdf"
    chunks = extract_bytes(sample_pdf_bytes, label)
    for c in chunks:
        assert c.source == label
        assert c.metadata["filename"] == label


def test_extract_bytes_raises_on_non_pdf() -> None:
    with pytest.raises(ExtractionError, match="could not open"):
        extract_bytes(b"this is not pdf data", "fake.pdf")


def test_extract_bytes_raises_on_empty_pdf() -> None:
    """An in-memory PDF with no text should raise the image-only error."""
    doc = pymupdf.open()
    doc.new_page()
    buf = doc.tobytes()
    doc.close()
    with pytest.raises(ExtractionError, match="image-only"):
        extract_bytes(buf, "blank.pdf")


# ---- structural pre-filter heuristics (#68, S4) ----

from anki_translator.classifier import PREFILTER_METADATA_KEY  # noqa: E402
from anki_translator.extractors.pdf import _prefilter_kind  # noqa: E402


def test_prefilter_flags_references_heading_and_sets_section_state() -> None:
    kind, in_refs = _prefilter_kind("References. Boal JG (2006) Anim Cogn 9:171-180.", 5, False)
    assert kind == "bibliography"
    assert in_refs is True


def test_prefilter_flags_citation_blocks_within_references_run() -> None:
    kind, in_refs = _prefilter_kind("Smith J, Jones K (2019). On cables. J. Cables 4:1-9.", 6, True)
    assert kind == "bibliography" and in_refs is True


def test_prefilter_non_citation_block_ends_references_run_conservatively() -> None:
    """A non-citation block after References must NOT be flagged — an appendix or
    later section must not be swept up. The run ends instead."""
    kind, in_refs = _prefilter_kind(
        "Appendix A presents the full derivation of the recognition model used above.", 7, True
    )
    assert kind is None and in_refs is False


def test_prefilter_flags_acknowledgments_heading() -> None:
    kind, _ = _prefilter_kind("Acknowledgements. We thank the marine station staff.", 4, False)
    assert kind == "acknowledgments"
    # British spelling too.
    kind_uk, _ = _prefilter_kind("Acknowledgments: funded by grant 123.", 4, False)
    assert kind_uk == "acknowledgments"


def test_prefilter_flags_title_page_affiliation_block() -> None:
    """A front-matter (page 1) block with two institutional markers is an
    author/affiliation block."""
    kind, _ = _prefilter_kind(
        "Jane Doe, Department of Marine Biology, University of Naples Federico II, Italy.", 1, False
    )
    assert kind == "author-affiliations"


def test_prefilter_is_conservative_about_affiliations() -> None:
    """One marker, or page>1, is not enough — real prose mentioning a single
    institution must not be trimmed."""
    # single marker on page 1 → not flagged
    one_marker, _ = _prefilter_kind(
        "The university hospital reported a 30 percent reduction in readmissions.", 1, False
    )
    assert one_marker is None
    # two markers but not front matter → not flagged (body content)
    body, _ = _prefilter_kind(
        "The department and the university jointly funded the laboratory expansion.", 4, False
    )
    assert body is None


def test_prefilter_does_not_flag_ordinary_body_prose() -> None:
    kind, in_refs = _prefilter_kind(
        "Octopus vulgaris individuals can recognise and remember other octopuses.", 1, False
    )
    assert kind is None and in_refs is False


def test_extract_flags_references_and_affiliations_not_body(tmp_path: Path) -> None:
    """End-to-end: a paper-shaped PDF pre-filters the author block and the
    references entry, leaving the body fact for the classifier."""
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    top = 72.0
    for text in [
        "Recognition in cephalopods. Jane Doe, Department of Marine Biology, "
        "University of Naples Federico II, Naples, Italy.",
        "Octopuses can distinguish familiar neighbours from unfamiliar strangers "
        "in their natural reef habitat.",
        "References. Boal JG (2006) Social recognition in cephalopods. Anim Cogn 9:171-180.",
    ]:
        page.insert_textbox(pymupdf.Rect(72, top, 540, top + 90), text, fontsize=11, fontname="helv")
        top += 120.0
    doc.save(path)
    doc.close()

    chunks = extract(path)
    flags = [c.metadata.get(PREFILTER_METADATA_KEY) for c in chunks]
    assert flags == ["author-affiliations", None, "bibliography"]


def test_extract_arms_references_run_from_standalone_short_heading(tmp_path: Path) -> None:
    """A standalone 'REFERENCES' heading is ~10 chars — shorter than
    MIN_CHUNK_CHARS. It must still arm the references run so the citations below
    are pre-filtered as bibliography (zero LLM cost), instead of the whole list
    falling through to the classifier (#68 follow-up). A short page-number
    artifact between citations must NOT end the run."""
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    top = 72.0
    for text in [
        "Octopuses can distinguish familiar neighbours from unfamiliar strangers "
        "in their natural reef habitat, a sign of individual recognition.",   # body fact
        "REFERENCES",                                                          # standalone short heading
        "Boal JG (2006) Social recognition in cephalopods. Anim Cogn 9:171-180.",
        "1124",                                                                # short layout artifact
        "Tricarico E, Borrelli L (2011) I know my neighbour. PLoS ONE 6:e18710.",
    ]:
        page.insert_textbox(pymupdf.Rect(72, top, 540, top + 70), text, fontsize=11, fontname="helv")
        top += 100.0
    doc.save(path)
    doc.close()

    chunks = extract(path)
    # The short heading and the page-number artifact are below MIN_CHUNK_CHARS,
    # so neither is emitted as a chunk.
    assert all(c.text not in ("REFERENCES", "1124") for c in chunks)
    # Both citations are flagged bibliography — proving the short heading armed
    # the run and the interstitial artifact did not cut it short.
    bibliography = [c for c in chunks if c.metadata.get(PREFILTER_METADATA_KEY) == "bibliography"]
    assert len(bibliography) == 2
    # The body fact is left for the classifier.
    body = [c for c in chunks if c.metadata.get(PREFILTER_METADATA_KEY) is None]
    assert len(body) == 1 and "recognition" in body[0].text
