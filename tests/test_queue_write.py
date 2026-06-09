"""Tests for the queue-writer side of queue.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from anki_translator.chunk import Chunk
from anki_translator.classifier import CardCandidate, Overflow
from anki_translator.queue import (
    TaggedCandidate,
    make_slug,
    write_queue,
)


def _chunk(text: str = "Mitochondria is the powerhouse of the cell.") -> Chunk:
    return Chunk(
        text=text,
        source="https://example.com/cells",
        position="#organelles",
        source_type="url",
        metadata={"url": "https://example.com/cells", "anchor": "organelles"},
    )


def _candidate(note_type: str = "AT Basic", shape: str = "term-def", fields: dict | None = None) -> CardCandidate:
    return CardCandidate(
        note_type=note_type,
        shape=shape,
        fields=fields or {"Front": "mitochondria", "Back": "powerhouse of the cell"},
        chunk=_chunk(),
    )


# ---- make_slug ----


def test_make_slug_url() -> None:
    assert make_slug("https://example.com/cells", "url") == "example-com-cells"


def test_make_slug_url_with_path() -> None:
    assert make_slug("https://example.com/biology/cells", "url") == "example-com-biology-cells"


def test_make_slug_pdf() -> None:
    assert make_slug("kleppmann.pdf", "pdf") == "kleppmann"


def test_make_slug_manual_label() -> None:
    assert make_slug("chat 2026-05-27", "manual") == "chat-2026-05-27"


def test_make_slug_truncates() -> None:
    long_source = "x" * 200
    slug = make_slug(long_source, "manual")
    assert len(slug) <= 80


def test_make_slug_empty_fallback() -> None:
    assert make_slug("!!!", "manual") == "untitled"


def test_make_slug_collapses_runs_of_hyphens() -> None:
    assert make_slug("foo---bar___baz", "manual") == "foo-bar-baz"


# ---- write_queue ----


def test_write_queue_creates_both_files(tmp_path: Path) -> None:
    queue_path, qa_path = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
        overflow=[],
        deck="Reading",
        slug="example-com-cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    assert queue_path.exists()
    assert qa_path.exists()
    assert queue_path.name == "2026-05-27-example-com-cells.md"
    assert qa_path.name == "2026-05-27-example-com-cells.md"


def test_queue_file_block_structure(tmp_path: Path) -> None:
    queue_path, _ = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — term-def" in body
    assert "**Front:** mitochondria" in body
    assert "**Back:** powerhouse of the cell" in body
    assert "**Source:** https://example.com/cells" in body
    assert "**Position:** #organelles" in body
    assert "**Deck:** Reading" in body
    assert "**Model:** AT Basic" in body
    assert "**Tags:** biology, biology::organelles" in body
    assert body.rstrip().endswith("---")


def test_queue_file_separates_multiple_cards(tmp_path: Path) -> None:
    candidates = [
        TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
        TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
        TaggedCandidate(_candidate(fields={"Front": "E", "Back": "F"}), tags=["t3"]),
    ]
    queue_path, _ = write_queue(
        tagged=candidates,
        overflow=[],
        deck="Reading",
        slug="multi",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — term-def" in body
    assert "## Card 2 — term-def" in body
    assert "## Card 3 — term-def" in body
    # Three blocks → three `---` separators
    assert body.count("\n---\n") == 3


def test_queue_file_cloze_uses_text_field(tmp_path: Path) -> None:
    cloze_candidate = _candidate(
        note_type="AT Cloze",
        shape="cloze",
        fields={"Text": "The {{c1::mitochondria}} is the powerhouse of the cell."},
    )
    queue_path, _ = write_queue(
        tagged=[TaggedCandidate(cloze_candidate, tags=["biology"])],
        overflow=[],
        deck="Reading",
        slug="cloze-test",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — cloze" in body
    assert "**Text:** The {{c1::mitochondria}}" in body
    # No Front/Back for cloze
    assert "**Front:**" not in body
    assert "**Back:**" not in body


def test_qa_file_renders_overflow(tmp_path: Path) -> None:
    overflow = [
        Overflow(chunk=_chunk("Some long discursive paragraph that does not fit a card."), reason="no_shape_fit"),
        Overflow(chunk=_chunk("Another overflow chunk with context."), reason="exceeds_budget: Back is 250 chars"),
    ]
    _, qa_path = write_queue(
        tagged=[],
        overflow=overflow,
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = qa_path.read_text()
    assert "# Q&A — cells" in body
    assert "## 1. From https://example.com/cells" in body
    assert "#organelles" in body
    assert "Reason: no_shape_fit" in body
    assert "Reason: exceeds_budget" in body
    assert "Some long discursive paragraph" in body


def test_qa_file_empty_overflow_still_written(tmp_path: Path) -> None:
    _, qa_path = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = qa_path.read_text()
    assert "No overflow chunks" in body


def test_queue_file_empty_candidates_still_written(tmp_path: Path) -> None:
    queue_path, _ = write_queue(
        tagged=[],
        overflow=[Overflow(chunk=_chunk(), reason="no_shape_fit")],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "empty" in body.lower()


def test_write_queue_creates_missing_directories(tmp_path: Path) -> None:
    queue_dir = tmp_path / "deeply" / "nested" / "queue"
    qa_dir = tmp_path / "other" / "qa"
    write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=queue_dir,
        qa_dir=qa_dir,
        ingestion_date=date(2026, 5, 27),
    )
    assert queue_dir.exists()
    assert qa_dir.exists()


def test_queue_file_no_tags_renders_cleanly(tmp_path: Path) -> None:
    queue_path, _ = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=[])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "**Tags:** " in body  # empty but present


def test_queue_file_multiline_field_inlined(tmp_path: Path) -> None:
    """Field values with embedded newlines must be inlined onto one markdown line."""
    multiline = _candidate(fields={"Front": "x", "Back": "line one\nline two\nline three"})
    queue_path, _ = write_queue(
        tagged=[TaggedCandidate(multiline, tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "**Back:** line one line two line three" in body


def test_queue_file_renders_term_table_row_as_separate_block(tmp_path: Path) -> None:
    """For #52: each row of a term-table chunk is a CardCandidate after the cli
    flattens MultiCardCandidate. Each lands as its own `## Card N — term-table`
    block, preserving per-row delete semantics during review. The natural
    ordering of classify_chunks keeps same-source rows adjacent in the queue
    file — that's the 'grouped' behavior, without breaking the parser's
    one-block-per-card split."""
    rows = [
        _candidate(
            note_type="AT Table",
            shape="term-table",
            fields={
                "Key": "DP 1.2",
                "Attr1Name": "Link rate", "Attr1Value": "HBR2",
                "Attr2Name": "Total bandwidth", "Attr2Value": "21.6 Gbps",
                "Attr3Name": "Top resolution", "Attr3Value": "4K@60",
            },
        ),
        _candidate(
            note_type="AT Table",
            shape="term-table",
            fields={
                "Key": "DP 1.4",
                "Attr1Name": "Link rate", "Attr1Value": "HBR3",
                "Attr2Name": "Total bandwidth", "Attr2Value": "25.92 Gbps",
                "Attr3Name": "Top resolution", "Attr3Value": "8K@60",
            },
        ),
    ]
    queue_path, _qa_path = write_queue(
        tagged=[TaggedCandidate(c, tags=["video::displayport"]) for c in rows],
        overflow=[],
        deck="Myrzka::Cables",
        slug="dp",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
    )
    body = queue_path.read_text()
    # Two blocks, one per row, in order.
    assert body.count("## Card") == 2
    assert body.index("DP 1.2") < body.index("DP 1.4")
    # Per-row block contains the row's full attribute set so the reviewer can
    # judge it without re-reading the source.
    assert "**Key:** DP 1.2" in body
    assert "**Attr1Name:** Link rate" in body
    assert "**Attr1Value:** HBR2" in body
    assert "**Attr3Value:** 4K@60" in body
    assert "**Model:** AT Table" in body
