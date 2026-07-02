"""anki-translator CLI — ingest, commit, card, bootstrap.

Thin glue layer over the library modules. Mirrors anki-manager's CLI style (argparse,
JSON output for machine-readable commands, plain text for human-readable ones).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from . import bootstrap as bootstrap_mod
from . import classifier, tagger
from .config import load_citations, load_shapes, load_tagger_config
from .extractors import ExtractionError
from .extractors.manual import extract_file as manual_extract_file
from .extractors.manual import extract_text as manual_extract_text
from .extractors.pdf import extract as pdf_extract
from .extractors.pdf import extract_bytes as pdf_extract_bytes
from .extractors.url import extract as url_extract_html
from .extractors.url import extract_url
from .extractors.url import fetch_bytes as url_fetch_bytes
from .queue import (
    TaggedCandidate,
    commit_queue,
    make_slug,
    write_queue,
)


def _state_default(name: str) -> str:
    """Default output dir for an overflow lane (``queue``/``qa``/``trimmed``).

    Rooted at ``$ANKI_TRANSLATOR_STATE_DIR`` when set — the agent-independent
    install (anki-manager #32) points this at ``/var/lib/anki-translator/`` so the
    tool writes system state, not into its package tree. Unset → a cwd-relative
    dir, preserving the in-tree dev behavior. ``--queue-dir`` etc. still override.
    """
    base = os.environ.get("ANKI_TRANSLATOR_STATE_DIR")
    return str(Path(base) / name) if base else name


# Repo root that ships the config/ dir, resolved from this file's location rather
# than the cwd: <root>/src/anki_translator/cli.py -> parents[2] == <root>.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _config_default(name: str) -> str:
    """Default path for a bundled config file (``shapes.yaml`` / ``tagger.yaml``).

    Resolves to ``$ANKI_TRANSLATOR_CONFIG_DIR/<name>`` when set, else the copy that
    ships alongside the install (``<package-root>/config/``). Anchoring on the
    package location instead of the cwd means the tool works from any directory —
    not just the repo root — which is what the /opt install needs (#88; the v0.1
    smoke hit ``ConfigError: shapes config not found`` from another cwd). The
    ``--shapes`` / ``--tagger-config`` flags still override.
    """
    base = os.environ.get("ANKI_TRANSLATOR_CONFIG_DIR")
    root = Path(base) if base else _PACKAGE_ROOT / "config"
    return str(root / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anki-translator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_boot = sub.add_parser("bootstrap", help="Create the AT-prefixed starter note types in Anki if missing")
    p_boot.add_argument("--dry-run", action="store_true")

    p_ing = sub.add_parser("ingest", help="Extract a source → write the queue + qa files")
    p_ing.add_argument("source", nargs="?", help="URL, .pdf path, or .txt/.md path (exactly one of source or --text must be given)")
    p_ing.add_argument("--text", help="Inline text payload (alternative to a source path/URL)")
    p_ing.add_argument("--deck", required=True, help="Anki deck name (e.g. 'Reading')")
    p_ing.add_argument("--tag", help="Batch tag applied to every produced note (e.g. 'book-club-2026')")
    p_ing.add_argument("--label", help="Source label override (default: file stem, today's date for --text)")
    p_ing.add_argument("--shapes", default=_config_default("shapes.yaml"))
    p_ing.add_argument("--tagger-config", default=_config_default("tagger.yaml"),
                       help="Optional tagger config (filter rules for seed vocabulary). Missing file → defaults.")
    p_ing.add_argument("--queue-dir", default=_state_default("queue"))
    p_ing.add_argument("--qa-dir", default=_state_default("qa"))
    p_ing.add_argument("--trimmed-dir", default=_state_default("trimmed"),
                       help="Directory for trimmed-chaff overflow (TOC, citations, section headers). "
                            "Substantive overflow still lands in --qa-dir.")

    p_card = sub.add_parser(
        "card",
        help="Shape one paragraph into a card via the LLM, then commit or write a queue file",
    )
    p_card.add_argument("--from-text", required=True, metavar="TEXT",
                        help="The prose paragraph to turn into a card")
    p_card.add_argument("--deck", required=True, help="Anki deck name")
    p_card.add_argument("--source", default=None,
                        help="Source citation label (e.g. 'telegram:1040956901#3416'). "
                             "Defaults to today's date.")
    p_card.add_argument("--position", default=None,
                        help="Freeform position string (e.g. '#section-id', 'page 3'). "
                             "Defaults to empty.")
    p_card.add_argument("--shape", default=None,
                        help="Shape hint: if the classifier produces a candidate matching "
                             "this shape, prefer it. E.g. 'term-def', 'cloze', 'list-recall'.")
    p_card.add_argument("--commit", action="store_true",
                        help="Commit directly to Anki instead of writing a queue file. "
                             "Default: write queue file for review.")
    p_card.add_argument("--tag", action="append", default=[],
                        help="Tag to apply to the note (repeatable)")
    p_card.add_argument("--shapes", default=_config_default("shapes.yaml"))
    p_card.add_argument("--tagger-config", default=_config_default("tagger.yaml"))
    p_card.add_argument("--queue-dir", default=_state_default("queue"))
    p_card.add_argument("--qa-dir", default=_state_default("qa"))
    p_card.add_argument("--trimmed-dir", default=_state_default("trimmed"))

    p_com = sub.add_parser("commit", help="Parse a reviewed queue file and create notes via anki-manager")
    p_com.add_argument("queue_file", help="Path to a queue/<date>-<slug>.md file")
    p_com.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "bootstrap":
        return _cmd_bootstrap(args)
    if args.cmd == "ingest":
        return _cmd_ingest(args)
    if args.cmd == "card":
        return _cmd_card(args)
    if args.cmd == "commit":
        return _cmd_commit(args)
    return 1


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    from anki_manager import AnkiManager  # imported lazily so bootstrap is the only path that needs it

    mgr = AnkiManager()
    result = bootstrap_mod.bootstrap(mgr, dry_run=args.dry_run)
    print(json.dumps({"created": result.created, "already_present": result.already_present}, indent=2))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    if bool(args.source) == bool(args.text):
        print("error: provide exactly one of <source> or --text", file=sys.stderr)
        return 2

    # 1. Extract
    try:
        chunks = _dispatch_extractor(args)
    except ExtractionError as e:
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 2

    if not chunks:
        print("error: extractor produced no chunks", file=sys.stderr)
        return 2

    # 2. Classify each chunk (fan out — see classifier.resolve_concurrency)
    shapes = load_shapes(args.shapes)
    # Extractor pre-filter seam (#67): structural-chaff chunks skip the LLM
    # entirely and route straight to trimmed; only the rest are classified.
    to_classify, prefiltered = classifier.split_prefiltered(chunks)
    candidates: list[classifier.CardCandidate] = []
    overflow: list[classifier.Overflow] = list(prefiltered)
    for result in classifier.classify_chunks(to_classify, shapes):
        if isinstance(result, classifier.CardCandidate):
            candidates.append(result)
        elif isinstance(result, classifier.MultiCardCandidate):
            candidates.extend(result.rows)
        else:
            overflow.append(result)

    # 3. Tag each candidate (fan out — same width as classification)
    try:
        from anki_manager import AnkiManager
        existing_tags = list(AnkiManager().call("getTags") or [])
    except Exception:
        existing_tags = []  # tagger handles empty vocabulary gracefully

    tagger_cfg = load_tagger_config(args.tagger_config)
    tag_lists = tagger.tag_candidates(
        candidates, existing_tags, batch_tag=args.tag, tagger_config=tagger_cfg
    )
    tagged = [TaggedCandidate(c, tags) for c, tags in zip(candidates, tag_lists)]

    # 4. Write queue + qa + trimmed (overflow is split via overflow_bucket)
    first_chunk = chunks[0]
    slug = make_slug(first_chunk.source, first_chunk.source_type)
    queue_path, qa_path, trimmed_path = write_queue(
        tagged=tagged,
        overflow=overflow,
        deck=args.deck,
        slug=slug,
        queue_dir=args.queue_dir,
        qa_dir=args.qa_dir,
        trimmed_dir=args.trimmed_dir,
    )
    from .queue import overflow_bucket
    qa_count = sum(1 for ov in overflow if overflow_bucket(ov.reason, ov.bucket) == "qa")
    trimmed_count = len(overflow) - qa_count
    print(json.dumps({
        "queue_file": str(queue_path),
        "qa_file": str(qa_path),
        "trimmed_file": str(trimmed_path),
        "candidates": len(tagged),
        "overflow_qa": qa_count,
        "overflow_trimmed": trimmed_count,
    }, indent=2))
    return 0


def _dispatch_extractor(args: argparse.Namespace) -> list:
    """Pick the right extractor based on the source form."""
    if args.text:
        return manual_extract_text(args.text, label=args.label)
    src = args.source
    if src.startswith(("http://", "https://")):
        body, content_type = url_fetch_bytes(src)
        if content_type == "application/pdf" or body[:5] == b"%PDF-":
            return pdf_extract_bytes(body, src)
        return url_extract_html(body.decode("utf-8", errors="replace"), src)
    p = Path(src)
    if p.suffix.lower() == ".pdf":
        return pdf_extract(p)
    if p.suffix.lower() in {".txt", ".md"}:
        return manual_extract_file(p, label=args.label)
    raise ExtractionError(f"cannot determine source type for {src!r}")


def _cmd_card(args: argparse.Namespace) -> int:
    # 1. Extract — single text chunk
    try:
        chunks = manual_extract_text(args.from_text, label=args.source)
    except ExtractionError as e:
        print(f"error: extraction failed: {e}", file=sys.stderr)
        return 2

    # Patch position if provided
    if args.position:
        chunks = [dataclasses.replace(c, position=args.position) for c in chunks]

    # 2. Classify
    shapes = load_shapes(args.shapes)
    # Extractor pre-filter seam (#67): structural-chaff chunks skip the LLM
    # entirely and route straight to trimmed; only the rest are classified.
    to_classify, prefiltered = classifier.split_prefiltered(chunks)
    candidates: list[classifier.CardCandidate] = []
    overflow: list[classifier.Overflow] = list(prefiltered)
    for result in classifier.classify_chunks(to_classify, shapes):
        if isinstance(result, classifier.CardCandidate):
            candidates.append(result)
        elif isinstance(result, classifier.MultiCardCandidate):
            candidates.extend(result.rows)
        else:
            overflow.append(result)

    if not candidates:
        print("error: classifier produced no card candidates", file=sys.stderr)
        return 2

    # Shape hint: prefer first candidate matching the hint, fall back to first overall
    candidate = candidates[0]
    if args.shape:
        match = next((c for c in candidates if c.shape == args.shape), None)
        if match:
            candidate = match

    # 3. Tag
    try:
        from anki_manager import AnkiManager
        existing_tags = list(AnkiManager().call("getTags") or [])
    except Exception:
        existing_tags = []

    tagger_cfg = load_tagger_config(args.tagger_config)
    tag_lists = tagger.tag_candidates(
        [candidate], existing_tags, batch_tag=None, tagger_config=tagger_cfg
    )
    # Merge tagger output with user-supplied --tag values (deduplicated, order preserved)
    tags: list[str] = list(dict.fromkeys(tag_lists[0] + args.tag))

    if args.commit:
        # 4a. Direct commit to Anki
        from anki_manager import AnkiManager
        mgr = AnkiManager()
        mgr.add_deck(args.deck)
        result = mgr.upsert_note(
            args.deck,
            candidate.note_type,
            candidate.fields,
            tags=tags or None,
        )
        print(json.dumps({
            "stable_guid": result.stable_guid,
            "shape": candidate.shape,
            "deck": args.deck,
        }, indent=2))
        return 0

    # 4b. Write single-block queue file for review
    tagged = [TaggedCandidate(candidate, tags)]
    from .queue import overflow_bucket
    slug = "card-" + make_slug(chunks[0].source, chunks[0].source_type)
    queue_path, qa_path, trimmed_path = write_queue(
        tagged=tagged,
        overflow=overflow,
        deck=args.deck,
        slug=slug,
        queue_dir=args.queue_dir,
        qa_dir=args.qa_dir,
        trimmed_dir=args.trimmed_dir,
    )
    qa_count = sum(1 for ov in overflow if overflow_bucket(ov.reason, ov.bucket) == "qa")
    print(json.dumps({
        "queue_file": str(queue_path),
        "qa_file": str(qa_path),
        "trimmed_file": str(trimmed_path),
        "candidates": 1,
        "shape": candidate.shape,
        "overflow_qa": qa_count,
        "overflow_trimmed": len(overflow) - qa_count,
    }, indent=2))
    return 0


def _cmd_commit(args: argparse.Namespace) -> int:
    from anki_manager import AnkiManager

    mgr = AnkiManager()
    try:
        result = commit_queue(args.queue_file, mgr, dry_run=args.dry_run)
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({
        "created": result.created,
        "updated": result.updated,
        "failed": result.failed,
        "archived_to": str(result.archived_to) if result.archived_to else None,
        "already_committed": result.already_committed,
    }, indent=2))
    return 0 if not result.failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
