"""PDF extractor: PDF file → list[Chunk] with per-page Position.

Uses pymupdf (fitz) for text extraction. Chunks by text block (pymupdf groups text into
paragraph-shaped blocks based on layout), keeping a per-chunk page number for Position.
Encrypted PDFs and image-only PDFs raise ExtractionError with a clear message — we don't
attempt OCR.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..chunk import Chunk
from . import ExtractionError

# Heuristic: text blocks shorter than this are usually page numbers, headers, or footers —
# strip them before turning into chunks. 30 chars catches "Page 42", "Chapter 3", etc.
MIN_CHUNK_CHARS = 30


def extract(path: Path | str) -> list[Chunk]:
    """Extract chunks from a PDF file.

    Returns one Chunk per text block on each page, with the filename as Source and
    "page N" as Position. Blocks shorter than MIN_CHUNK_CHARS are skipped — those are
    usually running headers, footers, or page numbers.
    """
    p = Path(path)
    if not p.exists():
        raise ExtractionError(f"PDF not found: {p}")

    try:
        doc = pymupdf.open(p)
    except Exception as e:
        raise ExtractionError(f"could not open PDF {p}: {e}") from e

    if doc.is_encrypted:
        doc.close()
        raise ExtractionError(f"PDF is encrypted, cannot extract: {p}")

    filename = p.name
    chunks: list[Chunk] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            # blocks is a list of (x0, y0, x1, y1, text, block_no, block_type)
            for block in blocks:
                # block_type == 1 is image; we want text only (type 0)
                if len(block) < 7 or block[6] != 0:
                    continue
                text = block[4].strip()
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(
                    Chunk(
                        text=text,
                        source=filename,
                        position=f"page {page_num}",
                        source_type="pdf",
                        metadata={"filename": filename, "page": page_num},
                    )
                )
    finally:
        doc.close()

    if not chunks:
        raise ExtractionError(
            f"no text blocks extracted from {p} — PDF may be image-only (scanned without OCR)"
        )
    return chunks
