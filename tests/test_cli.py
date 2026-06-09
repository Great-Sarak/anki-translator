"""Tests for the CLI dispatch layer.

We exercise the entry point with mocked LLM, mocked AnkiManager, and synthesized inputs.
The CLI is thin glue — these tests are integration-shaped, validating that arguments
reach the right modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anki_translator import cli


REPO_ROOT = Path(__file__).resolve().parent.parent


def _classify_to_basic(chunk, shapes, llm=None):
    """Stub classifier that turns every chunk into an AT Basic card."""
    from anki_translator.classifier import CardCandidate
    return CardCandidate(
        note_type="AT Basic",
        shape="term-def",
        fields={"Front": chunk.text[:30], "Back": chunk.text[:60]},
        chunk=chunk,
    )


def _generate_tags_stub(candidate, existing_tags, batch_tag=None, llm=None):
    return ["biology"] + ([batch_tag] if batch_tag else [])


# ---- bootstrap ----


def test_bootstrap_invokes_module_with_mocked_mgr(monkeypatch, capsys) -> None:
    fake_mgr = MagicMock()
    fake_mgr.list_models.return_value = {}
    fake_result = MagicMock()
    fake_result.created = ["AT Basic", "AT Cloze"]
    fake_result.already_present = []
    monkeypatch.setattr("anki_translator.bootstrap.bootstrap", lambda mgr, dry_run=False: fake_result)
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)
    rc = cli.main(["bootstrap"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["created"] == ["AT Basic", "AT Cloze"]


# ---- ingest (manual / --text path) ----


def test_ingest_text_path(monkeypatch, tmp_path: Path, capsys) -> None:
    """End-to-end through ingest with --text input, stubbed classifier+tagger."""
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _generate_tags_stub)
    # Avoid hitting Anki for existing tags
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main([
        "ingest",
        "--text", "First paragraph here.\n\nSecond paragraph here.",
        "--label", "test-notes",
        "--deck", "Reading",
        "--tag", "test-batch",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["candidates"] == 2
    assert parsed["overflow_qa"] == 0
    assert parsed["overflow_trimmed"] == 0
    assert Path(parsed["trimmed_file"]).exists()
    qpath = Path(parsed["queue_file"])
    assert qpath.exists()
    body = qpath.read_text()
    assert "**Front:**" in body
    assert "**Tags:** biology, test-batch" in body


def test_ingest_md_file_path(monkeypatch, tmp_path: Path, capsys) -> None:
    """File-based dispatch: .md → manual extractor."""
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _generate_tags_stub)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    md_path = tmp_path / "notes.md"
    md_path.write_text("Intro paragraph.\n\n## A section\n\nBody.\n")
    rc = cli.main([
        "ingest", str(md_path),
        "--deck", "Reading",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["candidates"] >= 1


def test_ingest_rejects_both_source_and_text(monkeypatch, tmp_path: Path, capsys) -> None:
    rc = cli.main([
        "ingest", "https://example.com",
        "--text", "inline",
        "--deck", "X",
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 2
    assert "exactly one" in capsys.readouterr().err


def test_ingest_rejects_neither_source_nor_text(monkeypatch, tmp_path: Path, capsys) -> None:
    rc = cli.main([
        "ingest",
        "--deck", "X",
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 2


def test_ingest_handles_extraction_error(monkeypatch, tmp_path: Path, capsys) -> None:
    rc = cli.main([
        "ingest", str(tmp_path / "nonexistent.pdf"),
        "--deck", "X",
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 2
    assert "extraction failed" in capsys.readouterr().err


def test_ingest_unknown_source_type_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    """A file with an unsupported extension (e.g. .docx) should produce a clear error."""
    fake = tmp_path / "thing.docx"
    fake.write_text("anything")
    rc = cli.main([
        "ingest", str(fake),
        "--deck", "X",
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 2
    assert "cannot determine source type" in capsys.readouterr().err


# ---- commit ----


def test_commit_invokes_commit_queue(monkeypatch, tmp_path: Path, capsys) -> None:
    fake_mgr = MagicMock()
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    fake_result = MagicMock()
    fake_result.created = ["anki-manager::a", "anki-manager::b"]
    fake_result.updated = []
    fake_result.failed = []
    fake_result.archived_to = tmp_path / "queue" / "committed" / "x.md"
    monkeypatch.setattr("anki_translator.cli.commit_queue", lambda *a, **kw: fake_result)

    rc = cli.main(["commit", str(tmp_path / "queue" / "x.md")])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["created"] == ["anki-manager::a", "anki-manager::b"]


def test_commit_nonzero_exit_on_failures(monkeypatch, tmp_path: Path, capsys) -> None:
    fake_mgr = MagicMock()
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    fake_result = MagicMock()
    fake_result.created = []
    fake_result.updated = []
    fake_result.failed = [(2, "RuntimeError: timeout")]
    fake_result.archived_to = None
    monkeypatch.setattr("anki_translator.cli.commit_queue", lambda *a, **kw: fake_result)

    rc = cli.main(["commit", str(tmp_path / "x.md")])
    assert rc == 1


def test_commit_dry_run_passes_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    fake_mgr = MagicMock()
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)
    captured: dict = {}
    def fake_commit(path, mgr, *, dry_run=False, archive_dir=None):
        captured["dry_run"] = dry_run
        result = MagicMock()
        result.created = []
        result.updated = []
        result.failed = []
        result.archived_to = None
        return result
    monkeypatch.setattr("anki_translator.cli.commit_queue", fake_commit)
    cli.main(["commit", str(tmp_path / "x.md"), "--dry-run"])
    assert captured["dry_run"] is True
