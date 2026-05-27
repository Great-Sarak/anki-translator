"""Tests for the queue parser + commit pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anki_translator.chunk import Chunk
from anki_translator.classifier import CardCandidate, Overflow
from anki_translator.queue import (
    ParsedBlock,
    QueueParseError,
    TaggedCandidate,
    commit_queue,
    parse_queue,
    write_queue,
)


def _candidate(fields: dict[str, str] | None = None, note_type: str = "AT Basic", shape: str = "term-def") -> CardCandidate:
    chunk = Chunk(
        text="The mitochondria is the powerhouse of the cell.",
        source="https://example.com/cells",
        position="#organelles",
        source_type="url",
        metadata={"url": "https://example.com/cells", "anchor": "organelles"},
    )
    return CardCandidate(
        note_type=note_type,
        shape=shape,
        fields=fields or {"Front": "mitochondria", "Back": "powerhouse of the cell"},
        chunk=chunk,
    )


def _write_sample_queue(tmp_path: Path, candidates: list[TaggedCandidate]) -> Path:
    queue_path, _ = write_queue(
        tagged=candidates,
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
    )
    return queue_path


def _fake_mgr(created_then_updated: bool = True) -> MagicMock:
    """Mock AnkiManager whose upsert_note returns a result with created=True/False alternating."""
    mgr = MagicMock()
    call_count = [0]
    def upsert(deck, model, fields, *, tags=None, stable_guid=None, dry_run=False):
        call_count[0] += 1
        result = MagicMock()
        result.note_id = 1000 + call_count[0]
        result.stable_guid = f"anki-manager::{call_count[0]:04x}"
        result.created = call_count[0] == 1 if created_then_updated else True
        result.dry_run = dry_run
        return result
    mgr.upsert_note.side_effect = upsert
    return mgr


# ---- round-trip ----


def test_parse_queue_round_trips_simple_block(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(
        tmp_path,
        [TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
    )
    blocks = parse_queue(queue_path)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.shape == "term-def"
    assert b.fields["Front"] == "mitochondria"
    assert b.fields["Back"] == "powerhouse of the cell"
    assert b.fields["Source"] == "https://example.com/cells"
    assert b.fields["Position"] == "#organelles"
    assert b.deck == "Reading"
    assert b.model == "AT Basic"
    assert b.tags == ["biology", "biology::organelles"]


def test_parse_queue_round_trips_multiple_blocks(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(
        tmp_path,
        [
            TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
            TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
            TaggedCandidate(_candidate(fields={"Front": "E", "Back": "F"}), tags=[]),
        ],
    )
    blocks = parse_queue(queue_path)
    assert len(blocks) == 3
    assert blocks[0].tags == ["t1"]
    assert blocks[2].tags == []


def test_parse_queue_handles_cloze_text_field(tmp_path: Path) -> None:
    cloze = _candidate(
        note_type="AT Cloze",
        shape="cloze",
        fields={"Text": "The {{c1::mitochondria}} is the powerhouse."},
    )
    queue_path = _write_sample_queue(tmp_path, [TaggedCandidate(cloze, tags=["biology"])])
    blocks = parse_queue(queue_path)
    assert blocks[0].shape == "cloze"
    assert "{{c1::mitochondria}}" in blocks[0].fields["Text"]


def test_parse_queue_empty_sentinel_file_yields_no_blocks(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(tmp_path, [])
    blocks = parse_queue(queue_path)
    assert blocks == []


def test_parse_queue_user_deleted_blocks_simulation(tmp_path: Path) -> None:
    """Simulate user editing the file to delete the middle block."""
    queue_path = _write_sample_queue(
        tmp_path,
        [
            TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
            TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
            TaggedCandidate(_candidate(fields={"Front": "E", "Back": "F"}), tags=["t3"]),
        ],
    )
    body = queue_path.read_text()
    # Remove the middle block entirely
    blocks_raw = body.split("\n---\n")
    edited = blocks_raw[0] + "\n---\n" + "\n---\n".join(blocks_raw[2:])
    queue_path.write_text(edited)
    blocks = parse_queue(queue_path)
    assert len(blocks) == 2
    assert blocks[0].fields["Front"] == "A"
    assert blocks[1].fields["Front"] == "E"


# ---- parse error cases ----


def test_parse_queue_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(QueueParseError, match="not found"):
        parse_queue(tmp_path / "nope.md")


def test_parse_queue_missing_card_header_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.md"
    p.write_text("**Front:** something\n**Back:** else\n**Deck:** D\n**Model:** M\n**Tags:** \n---\n")
    with pytest.raises(QueueParseError, match="missing.*header"):
        parse_queue(p)


def test_parse_queue_missing_deck_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_deck.md"
    p.write_text("## Card 1 — term-def\n**Front:** a\n**Back:** b\n**Model:** M\n**Tags:** \n---\n")
    with pytest.raises(QueueParseError, match="missing .*Deck"):
        parse_queue(p)


def test_parse_queue_missing_model_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_model.md"
    p.write_text("## Card 1 — term-def\n**Front:** a\n**Back:** b\n**Deck:** D\n**Tags:** \n---\n")
    with pytest.raises(QueueParseError, match="missing .*Model"):
        parse_queue(p)


def test_parse_queue_no_content_fields_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_fields.md"
    p.write_text("## Card 1 — term-def\n**Deck:** D\n**Model:** M\n**Tags:** \n---\n")
    with pytest.raises(QueueParseError, match="no content fields"):
        parse_queue(p)


# ---- commit ----


def test_commit_calls_upsert_per_block(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(
        tmp_path,
        [
            TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
            TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
        ],
    )
    mgr = _fake_mgr()
    result = commit_queue(queue_path, mgr)
    assert mgr.upsert_note.call_count == 2
    assert len(result.created) + len(result.updated) == 2
    assert result.failed == []


def test_commit_archives_file_after_success(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(tmp_path, [TaggedCandidate(_candidate(), tags=["t"])])
    mgr = _fake_mgr()
    result = commit_queue(queue_path, mgr)
    assert not queue_path.exists()
    expected_archive = queue_path.parent / "committed" / queue_path.name
    assert expected_archive.exists()
    assert result.archived_to == expected_archive


def test_commit_does_not_archive_on_failure(tmp_path: Path) -> None:
    """If any block fails, leave the file in place for retry (upsert is idempotent)."""
    queue_path = _write_sample_queue(
        tmp_path,
        [
            TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
            TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
        ],
    )
    mgr = MagicMock()
    mgr.upsert_note.side_effect = [
        MagicMock(stable_guid="anki-manager::ok", created=True),
        RuntimeError("anki connect timeout"),
    ]
    result = commit_queue(queue_path, mgr)
    assert queue_path.exists()  # not archived
    assert result.archived_to is None
    assert len(result.created) == 1
    assert len(result.failed) == 1
    assert result.failed[0][0] == 2
    assert "RuntimeError" in result.failed[0][1]


def test_commit_dry_run_does_not_archive(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(tmp_path, [TaggedCandidate(_candidate(), tags=["t"])])
    mgr = _fake_mgr()
    result = commit_queue(queue_path, mgr, dry_run=True)
    assert queue_path.exists()
    assert result.archived_to is None
    # mgr.upsert_note was called with dry_run=True
    call = mgr.upsert_note.call_args_list[0]
    assert call.kwargs["dry_run"] is True


def test_commit_passes_tags_correctly(tmp_path: Path) -> None:
    queue_path = _write_sample_queue(
        tmp_path,
        [TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
    )
    mgr = _fake_mgr()
    commit_queue(queue_path, mgr)
    call = mgr.upsert_note.call_args_list[0]
    assert call.kwargs["tags"] == ["biology", "biology::organelles"]


def test_commit_empty_tags_passes_none(tmp_path: Path) -> None:
    """Tags=[] should become tags=None when calling mgr to match its signature default."""
    queue_path = _write_sample_queue(tmp_path, [TaggedCandidate(_candidate(), tags=[])])
    mgr = _fake_mgr()
    commit_queue(queue_path, mgr)
    call = mgr.upsert_note.call_args_list[0]
    assert call.kwargs["tags"] is None
