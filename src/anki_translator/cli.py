"""anki-translator CLI — ingest, commit, bootstrap.

Thin glue layer over the library modules. Mirrors anki-manager's CLI style (argparse,
JSON output for machine-readable commands, plain text for human-readable ones).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bootstrap as bootstrap_mod
from . import classifier, tagger
from .config import load_citations, load_shapes, load_tagger_config
from .extractors import ExtractionError
from .extractors.manual import extract_file as manual_extract_file
from .extractors.manual import extract_text as manual_extract_text
from .extractors.pdf import extract as pdf_extract
from .extractors.url import extract_url
from .queue import (
    TaggedCandidate,
    commit_queue,
    make_slug,
    write_queue,
)


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
    p_ing.add_argument("--shapes", default="config/shapes.yaml")
    p_ing.add_argument("--tagger-config", default="config/tagger.yaml",
                       help="Optional tagger config (filter rules for seed vocabulary). Missing file → defaults.")
    p_ing.add_argument("--queue-dir", default="queue")
    p_ing.add_argument("--qa-dir", default="qa")

    p_com = sub.add_parser("commit", help="Parse a reviewed queue file and create notes via anki-manager")
    p_com.add_argument("queue_file", help="Path to a queue/<date>-<slug>.md file")
    p_com.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "bootstrap":
        return _cmd_bootstrap(args)
    if args.cmd == "ingest":
        return _cmd_ingest(args)
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
    candidates: list[classifier.CardCandidate] = []
    overflow: list[classifier.Overflow] = []
    for result in classifier.classify_chunks(chunks, shapes):
        if isinstance(result, classifier.CardCandidate):
            candidates.append(result)
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

    # 4. Write queue + qa
    first_chunk = chunks[0]
    slug = make_slug(first_chunk.source, first_chunk.source_type)
    queue_path, qa_path = write_queue(
        tagged=tagged,
        overflow=overflow,
        deck=args.deck,
        slug=slug,
        queue_dir=args.queue_dir,
        qa_dir=args.qa_dir,
    )
    print(json.dumps({
        "queue_file": str(queue_path),
        "qa_file": str(qa_path),
        "candidates": len(tagged),
        "overflow": len(overflow),
    }, indent=2))
    return 0


def _dispatch_extractor(args: argparse.Namespace) -> list:
    """Pick the right extractor based on the source form."""
    if args.text:
        return manual_extract_text(args.text, label=args.label)
    src = args.source
    if src.startswith(("http://", "https://")):
        return extract_url(src)
    p = Path(src)
    if p.suffix.lower() == ".pdf":
        return pdf_extract(p)
    if p.suffix.lower() in {".txt", ".md"}:
        return manual_extract_file(p, label=args.label)
    raise ExtractionError(f"cannot determine source type for {src!r}")


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
    }, indent=2))
    return 0 if not result.failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
