"""PDF extractor: PDF file → list[Chunk] with per-page Position.

Uses pymupdf (fitz) for text extraction. Chunks by text block (pymupdf groups text into
paragraph-shaped blocks based on layout), keeping a per-chunk page number for Position.
Encrypted PDFs and image-only PDFs raise ExtractionError with a clear message — we don't
attempt OCR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ..chunk import Chunk
from ..classifier import PREFILTER_METADATA_KEY
from . import ExtractionError

# Heuristic: text blocks shorter than this are usually page numbers, headers, or footers —
# strip them before turning into chunks. 30 chars catches "Page 42", "Chapter 3", etc.
MIN_CHUNK_CHARS = 30

# --- Structural pre-filter heuristics (#68, S4) ----------------------------------
#
# Flag references/bibliography section bodies and the title-page author/affiliation
# block as structural chaff (PREFILTER_METADATA_KEY), so they bypass the LLM and
# route straight to trimmed (#67). Conservative bias: a chaff chunk slipping
# through to the classifier (false negative) is far cheaper than real content
# silently trimmed (false positive). Heuristics only fire on high-confidence
# markers, and the references run flags only consecutive citation-shaped blocks —
# the first non-citation block ends it.

_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|notes and references)\b",
    re.IGNORECASE,
)
_ACK_HEADING_RE = re.compile(r"^\s*acknowledge?ments\b", re.IGNORECASE)
# Within a references section, a block that carries a citation fingerprint:
# a (YYYY) year, a vol:page range, "et al.", or "pp. N".
_CITATION_RE = re.compile(
    r"\(\d{4}\)|\b\d{1,4}:\d{1,4}[–-]\d{1,4}\b|\bet al\.?|\bpp?\.\s*\d",
    re.IGNORECASE,
)
# Affiliation markers in the academic "<unit> of <place>" form, plus email and an
# explicit "affiliation" label. Two or more distinct markers in a single
# front-matter (page 1) block is a strong author/affiliation signal. The phrase
# form is deliberately narrow: it matches "Department of Marine Biology,
# University of Naples" but NOT ordinary prose like "the university hospital
# reported…", which stacks bare institutional words without the "of" form.
_AFFILIATION_MARKERS = (
    "department of",
    "university of",
    "institute of",
    "school of",
    "faculty of",
    "laboratory of",
    "college of",
    "division of",
    "affiliation",
    "@",
)


def _count_affiliation_markers(text: str) -> int:
    lowered = text.lower()
    return sum(1 for marker in _AFFILIATION_MARKERS if marker in lowered)


def _normalize_repeat(text: str) -> str:
    """Normalize a block for cross-page repetition detection: drop digits (page
    numbers, dates, volume/issue) and collapse whitespace, so a running header or
    footer matches itself across pages despite the changing page number."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", text)).strip().lower()[:80]


def _detect_running_blocks(doc: pymupdf.Document) -> set[str]:
    """Normalized text of short blocks that recur on >=2 pages — running headers
    and footers (journal masthead, the article title repeated per page, the page
    footer). These are boilerplate chaff, and — critically — when multi-column
    reading order interleaves them into a references section, they must NOT end
    the references run. Long body prose is unique per page and never matches."""
    if doc.page_count < 2:
        return set()
    pages: dict[str, set[int]] = {}
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            if len(block) < 7 or block[6] != 0:
                continue
            text = block[4].strip()
            if not text or len(text) > 150:
                continue
            pages.setdefault(_normalize_repeat(text), set()).add(page_num)
    return {norm for norm, pgs in pages.items() if norm and len(pgs) >= 2}


def _make_chunk(text: str, source_label: str, page_num: int, kind: str | None) -> Chunk:
    metadata: dict[str, object] = {"filename": source_label, "page": page_num}
    if kind:
        metadata[PREFILTER_METADATA_KEY] = kind
    return Chunk(
        text=text, source=source_label, position=f"page {page_num}",
        source_type="pdf", metadata=metadata,
    )


def _prefilter_kind(text: str, page_num: int, in_references: bool) -> tuple[str | None, bool]:
    """Classify a block as structural chaff. Returns (kind, in_references).

    kind is the prefilter tag ('bibliography' / 'acknowledgments' /
    'author-affiliations') or None. The returned in_references threads the
    references-section state to the next block.
    """
    head = text.lstrip()
    if _REFERENCES_HEADING_RE.match(head):
        return "bibliography", True
    if _ACK_HEADING_RE.match(head):
        return "acknowledgments", in_references
    if in_references:
        if _CITATION_RE.search(text):
            return "bibliography", True
        # First non-citation block ends the references run (conservative — an
        # appendix or later section must not be swept up).
        return None, False
    if page_num == 1 and _count_affiliation_markers(text) >= 2:
        return "author-affiliations", in_references
    return None, in_references


def _extract_doc(doc: pymupdf.Document, source_label: str) -> list[Chunk]:
    """Walk an open pymupdf Document and return Chunks. Caller owns doc lifetime.

    Structural-chaff blocks (references, acknowledgments, title-page author/
    affiliation, running headers/footers) are flagged via PREFILTER_METADATA_KEY
    rather than dropped, so the pipeline routes them to trimmed without an LLM
    call (#67/#68).
    """
    chunks: list[Chunk] = []
    running = _detect_running_blocks(doc)
    in_references = False
    ack_page: int | None = None  # page an acknowledgments run started on (page-bounded)
    for page_num, page in enumerate(doc, start=1):
        # blocks is a list of (x0, y0, x1, y1, text, block_no, block_type)
        for block in page.get_text("blocks"):
            # block_type == 1 is image; we want text only (type 0)
            if len(block) < 7 or block[6] != 0:
                continue
            text = block[4].strip()

            # Running header/footer: boilerplate chaff. Flag it (when long enough
            # to emit), but keep it TRANSPARENT to section state — multi-column
            # reading order interleaves it into the references stream, and it must
            # not end an active references run (the #70-era two-column gap).
            if _normalize_repeat(text) in running:
                if len(text) >= MIN_CHUNK_CHARS:
                    chunks.append(_make_chunk(text, source_label, page_num, "running-header"))
                continue

            # Section headings arm their run BEFORE the length skip — a standalone
            # "REFERENCES"/"ACKNOWLEDGMENTS" heading is ~10-15 chars and would
            # otherwise be dropped below before _prefilter_kind ever saw it, so
            # the section body fell through to the LLM (#68 follow-up). A heading
            # inline with content (a long block) still flows through _prefilter_kind.
            head = text.lstrip()
            if _REFERENCES_HEADING_RE.match(head):
                in_references, ack_page = True, None
            elif _ACK_HEADING_RE.match(head):
                ack_page, in_references = page_num, False

            # Short layout noise (page numbers, column-split artifacts, a
            # standalone heading) is never emitted, and is skipped WITHOUT ending
            # an active run so a stray artifact can't cut a section short.
            if len(text) < MIN_CHUNK_CHARS:
                continue

            kind, in_references = _prefilter_kind(text, page_num, in_references)
            # Acknowledgments body — prose with no citation shape. Flag blocks on
            # the heading's page; the run is page-bounded so it can't swallow a
            # later section that lacks its own heading.
            if ack_page is not None and page_num != ack_page:
                ack_page = None
            if kind is None and ack_page == page_num and not in_references:
                kind = "acknowledgments"

            chunks.append(_make_chunk(text, source_label, page_num, kind))
    return chunks


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

    try:
        chunks = _extract_doc(doc, p.name)
    finally:
        doc.close()

    if not chunks:
        raise ExtractionError(
            f"no text blocks extracted from {p} — PDF may be image-only (scanned without OCR)"
        )
    return chunks


def extract_bytes(data: bytes, source_label: str) -> list[Chunk]:
    """Extract chunks from raw PDF bytes (e.g. fetched from a URL).

    source_label is used as the Source field (typically the URL or a filename).
    """
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ExtractionError(f"could not open PDF from bytes [{source_label}]: {e}") from e

    if doc.is_encrypted:
        doc.close()
        raise ExtractionError(f"PDF is encrypted, cannot extract: {source_label}")

    try:
        chunks = _extract_doc(doc, source_label)
    finally:
        doc.close()

    if not chunks:
        raise ExtractionError(
            f"no text blocks extracted from {source_label} — PDF may be image-only (scanned without OCR)"
        )
    return chunks
