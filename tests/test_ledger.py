"""Tests for the chunk-dedup ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from anki_translator.ledger import Ledger, chunk_key


def test_chunk_key_stable() -> None:
    """Same inputs produce same hash."""
    a = chunk_key("source-a", "hello world")
    b = chunk_key("source-a", "hello world")
    assert a == b


def test_chunk_key_collision_resistant_to_boundary_swap() -> None:
    """('ab', 'c') and ('a', 'bc') must hash differently — the separator byte enforces this."""
    assert chunk_key("ab", "c") != chunk_key("a", "bc")


def test_chunk_key_content_change_changes_hash() -> None:
    """Slight content changes produce different hashes — meaningful revisions re-process."""
    a = chunk_key("article.html", "The quick brown fox.")
    b = chunk_key("article.html", "The quick brown fox jumps.")
    assert a != b


def test_seen_false_on_fresh_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.seen("source", "chunk") is False
    assert ledger.lookup("source", "chunk") is None
    assert len(ledger) == 0


def test_record_then_seen(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    entry = ledger.record(
        source="https://example.com/article",
        chunk="The mitochondria is the powerhouse of the cell.",
        stable_guid="anki-manager::abc123",
        deck="Reading",
    )
    assert ledger.seen("https://example.com/article", "The mitochondria is the powerhouse of the cell.")
    looked_up = ledger.lookup("https://example.com/article", "The mitochondria is the powerhouse of the cell.")
    assert looked_up == entry
    assert entry.stable_guid == "anki-manager::abc123"
    assert entry.deck == "Reading"
    assert len(ledger) == 1


def test_persistence_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger1 = Ledger(path)
    ledger1.record("source-1", "chunk text", "anki-manager::xyz", "Deck1")
    ledger1.record("source-2", "other chunk", "anki-manager::pdq", "Deck2")

    # Reload from disk
    ledger2 = Ledger(path)
    assert len(ledger2) == 2
    assert ledger2.seen("source-1", "chunk text")
    assert ledger2.seen("source-2", "other chunk")
    e = ledger2.lookup("source-1", "chunk text")
    assert e is not None
    assert e.stable_guid == "anki-manager::xyz"


def test_record_creates_parent_dir(tmp_path: Path) -> None:
    """ledger/ may not exist on a fresh checkout; record() must create it."""
    nested = tmp_path / "ledger" / "ledger.jsonl"
    assert not nested.parent.exists()
    ledger = Ledger(nested)
    ledger.record("s", "c", "anki-manager::guid", "Deck")
    assert nested.exists()


def test_record_re_recording_overwrites_in_index(tmp_path: Path) -> None:
    """Re-recording the same (source, chunk) updates the in-memory entry (e.g., new GUID after re-add)."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("s", "c", "anki-manager::first", "Deck")
    ledger.record("s", "c", "anki-manager::second", "Deck")
    e = ledger.lookup("s", "c")
    assert e is not None
    assert e.stable_guid == "anki-manager::second"
    # JSONL still has both lines on disk (append-only); the latest wins in the in-memory index
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_corrupt_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text("not json at all\n")
    with pytest.raises(ValueError, match="corrupt"):
        Ledger(path)


def test_blank_lines_ignored(tmp_path: Path) -> None:
    """Blank lines (e.g., from manual edits) should be skipped, not crash the loader."""
    path = tmp_path / "ledger.jsonl"
    ledger1 = Ledger(path)
    ledger1.record("s", "c", "anki-manager::guid", "Deck")
    # Inject a blank line
    with path.open("a") as f:
        f.write("\n\n")
    ledger2 = Ledger(path)
    assert len(ledger2) == 1
