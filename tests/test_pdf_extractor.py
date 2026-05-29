"""Tests for the PDF extractor.

Uses pymupdf to synthesize a small test PDF in a tmp_path fixture so no binary needs to
live in the repo.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from anki_translator.extractors import ExtractionError
from anki_translator.extractors.pdf import MIN_CHUNK_CHARS, extract


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
