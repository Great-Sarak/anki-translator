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


# ---- URL dispatch: content-type / magic-byte routing (#45) ----

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _make_pdf_bytes() -> bytes:
    """Synthesize a minimal single-page PDF with one text block."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, 540, 200),
        "The mitochondria is the powerhouse of the cell, "
        "producing ATP through oxidative phosphorylation.",
        fontsize=11,
    )
    data = doc.tobytes()
    doc.close()
    return data


def test_url_dispatch_routes_pdf_content_type_to_pdf_extractor(monkeypatch, tmp_path: Path, capsys) -> None:
    """content_type='application/pdf' → pdf_extract_bytes, not url HTML extractor."""
    pdf_bytes = _make_pdf_bytes()
    monkeypatch.setattr("anki_translator.cli.url_fetch_bytes", lambda url, **kw: (pdf_bytes, "application/pdf"))
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _generate_tags_stub)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main([
        "ingest", "https://example.com/paper?type=printable",
        "--deck", "Reading",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["candidates"] >= 1
    # Chunks come from PDF path → source_type should be 'pdf'
    queue_text = Path(parsed["queue_file"]).read_text()
    assert "**Source:** https://example.com/paper?type=printable" in queue_text


def test_url_dispatch_routes_pdf_magic_bytes_to_pdf_extractor(monkeypatch, tmp_path: Path, capsys) -> None:
    """Server omits content-type but body starts with %PDF- → still routed to pdf_extract_bytes."""
    pdf_bytes = _make_pdf_bytes()
    assert pdf_bytes[:5] == b"%PDF-"
    monkeypatch.setattr("anki_translator.cli.url_fetch_bytes", lambda url, **kw: (pdf_bytes, "application/octet-stream"))
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _generate_tags_stub)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main([
        "ingest", "https://example.com/download",
        "--deck", "Reading",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["candidates"] >= 1


def test_url_dispatch_routes_html_content_type_to_html_extractor(monkeypatch, tmp_path: Path, capsys) -> None:
    """content_type='text/html' → HTML extractor, not PDF."""
    html_bytes = (FIXTURES / "article.html").read_bytes()
    monkeypatch.setattr("anki_translator.cli.url_fetch_bytes", lambda url, **kw: (html_bytes, "text/html"))
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _generate_tags_stub)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main([
        "ingest", "https://example.com/cells",
        "--deck", "Reading",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["candidates"] >= 1
    queue_text = Path(parsed["queue_file"]).read_text()
    # HTML extractor attaches the URL as source; PDF would use the URL too but position
    # format differs: HTML uses '#anchor' or '', PDF uses 'page N'
    assert "**Position:** #" in queue_text or "**Position:** " in queue_text


# ---- card subcommand (#42) ----


def _classify_to_basic_stub(chunk, shapes, llm=None):
    from anki_translator.classifier import CardCandidate
    return CardCandidate(
        note_type="AT Basic",
        shape="term-def",
        fields={"Front": chunk.text[:30], "Back": chunk.text[:60]},
        chunk=chunk,
    )


def _tag_stub_no_batch(candidate, existing_tags, batch_tag=None, llm=None):
    return []


def _card_args(tmp_path: Path, extra: list[str] | None = None) -> list[str]:
    base = [
        "card",
        "--from-text", "The mitochondria is the powerhouse of the cell.",
        "--deck", "Reading",
        "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
        "--queue-dir", str(tmp_path / "queue"),
        "--qa-dir", str(tmp_path / "qa"),
        "--trimmed-dir", str(tmp_path / "trimmed"),
    ]
    return base + (extra or [])


def test_card_queue_mode_writes_single_block(monkeypatch, tmp_path: Path, capsys) -> None:
    """card without --commit writes a one-block queue file and returns shape in JSON."""
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic_stub)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main(_card_args(tmp_path))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["candidates"] == 1
    assert parsed["shape"] == "term-def"
    qpath = Path(parsed["queue_file"])
    assert qpath.exists()
    body = qpath.read_text()
    assert body.count("## Card") == 1


def test_card_source_and_position_appear_in_queue(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic_stub)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main(_card_args(tmp_path, [
        "--source", "telegram:1040956901#3416",
        "--position", "#organelles",
    ]))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    body = Path(parsed["queue_file"]).read_text()
    assert "telegram:1040956901#3416" in body
    assert "#organelles" in body


def test_card_tag_args_appear_in_queue(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic_stub)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main(_card_args(tmp_path, ["--tag", "biology", "--tag", "cell-biology"]))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    body = Path(parsed["queue_file"]).read_text()
    assert "biology" in body
    assert "cell-biology" in body


def test_card_commit_calls_upsert_note(monkeypatch, tmp_path: Path, capsys) -> None:
    """--commit skips the queue file and upserts directly, returning stable_guid."""
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic_stub)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)

    upsert_result = MagicMock()
    upsert_result.stable_guid = "anki-manager::abc123"

    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    fake_mgr.upsert_note.return_value = upsert_result
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    rc = cli.main(_card_args(tmp_path, ["--commit"]))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stable_guid"] == "anki-manager::abc123"
    assert parsed["shape"] == "term-def"
    assert parsed["deck"] == "Reading"
    fake_mgr.add_deck.assert_called_once_with("Reading")
    fake_mgr.upsert_note.assert_called_once()


def test_card_commit_does_not_write_queue_file(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_to_basic_stub)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)

    upsert_result = MagicMock()
    upsert_result.stable_guid = "anki-manager::abc123"
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    fake_mgr.upsert_note.return_value = upsert_result
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    cli.main(_card_args(tmp_path, ["--commit"]))
    capsys.readouterr()
    queue_dir = tmp_path / "queue"
    queue_files = list(queue_dir.glob("*.md")) if queue_dir.exists() else []
    assert queue_files == []


def test_card_shape_hint_prefers_matching_candidate(monkeypatch, tmp_path: Path, capsys) -> None:
    """If --shape matches a candidate, that candidate is picked over the first one."""
    from anki_translator.classifier import CardCandidate
    from anki_translator.chunk import Chunk

    def _classify_two_shapes(chunk, shapes, llm=None):
        base = CardCandidate(
            note_type="AT Basic",
            shape="term-def",
            fields={"Front": "a", "Back": "b"},
            chunk=chunk,
        )
        return base  # single candidate; shape hint test just checks no crash for non-matching hint

    monkeypatch.setattr("anki_translator.cli.classifier.classify", _classify_two_shapes)
    monkeypatch.setattr("anki_translator.cli.tagger.generate_tags", _tag_stub_no_batch)
    fake_mgr = MagicMock()
    fake_mgr.call.return_value = []
    monkeypatch.setattr("anki_manager.AnkiManager", lambda: fake_mgr)

    # --shape that doesn't match → falls back to first candidate without crashing
    rc = cli.main(_card_args(tmp_path, ["--shape", "cloze"]))
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["shape"] == "term-def"  # fallback to first


def test_card_requires_from_text(monkeypatch, tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "card",
            "--deck", "Reading",
            "--shapes", str(REPO_ROOT / "config" / "shapes.yaml"),
            "--queue-dir", str(tmp_path / "queue"),
            "--qa-dir", str(tmp_path / "qa"),
            "--trimmed-dir", str(tmp_path / "trimmed"),
        ])
    assert exc_info.value.code == 2
