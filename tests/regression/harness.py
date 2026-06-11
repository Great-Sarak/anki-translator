"""S0 — baseline overflow-routing regression harness (#59 Layer 0, issue #64).

The objective baseline every later layer (S1-S6) asserts against:
*correctness held, token count dropped.* It snapshots, for each of three
representative corpora (one per extractor), how every extracted chunk routes —
``card`` vs ``qa`` vs ``trimmed`` — plus the qa/trimmed split counts and the
LLM-facing prompt cost. The snapshot is committed as ``golden.json``.

Determinism: the LLM is stubbed with a record/replay fixture (``responses.json``,
keyed by ``sha1(chunk.text)``), so the harness needs **zero network**. The
captured responses represent today's live classifier behavior; refresh them
deliberately when the corpora change.

How later layers assert against this:

* **Split unchanged-or-improved** — a layer must not move a chunk that is real
  content (``card``/``qa``) into ``trimmed``. Compare ``counts`` and the
  per-chunk ``routing`` against the golden; ``qa``/``card`` totals may only hold
  or rise, never fall, on the clean corpora.
* **Tokens <= baseline** — ``prompt_token_chars`` is the summed cost of every
  prompt actually dispatched to ``classify()`` (a character proxy; chunks routed
  without an LLM call, e.g. S3 extractor pre-filtering, contribute zero). A
  token-saving layer drives this down; no layer may drive it up on the same
  corpora.

The token metric is a character proxy, not a tokenizer count — stated here so the
number is interpreted as a stable relative measure, not an absolute token bill.

Usage::

    python tests/regression/harness.py            # print snapshot, exit 0
    python tests/regression/harness.py --update    # regenerate golden.json
    python tests/regression/harness.py --check      # diff vs golden, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from anki_translator.classifier import (
    CardCandidate,
    MultiCardCandidate,
    Overflow,
    build_prompt,
    classify,
    prefilter_overflow,
)
from anki_translator.chunk import Chunk
from anki_translator.config import load_shapes
from anki_translator.extractors import manual, pdf, url
from anki_translator.queue import overflow_bucket

HERE = Path(__file__).parent
CORPORA_DIR = HERE / "corpora"
ORIGINALS_DIR = CORPORA_DIR / "originals"
SHAPES_PATH = HERE.parent.parent / "config" / "shapes.yaml"

SCHEMA_VERSION = 1

# Originals mode (issue: real-corpora baseline) — when the full-size source files
# are present in the gitignored ``corpora/originals/`` folder, the harness runs
# against *them* instead of the committed synthetic stubs, and reads/writes the
# ``*.originals.json`` fixture variant so the portable (CI) golden is never
# clobbered. Each corpus falls back independently: present original wins, else
# the stub. See README "Originals mode".
RESPONSES_PATH = HERE / "responses.json"
GOLDEN_PATH = HERE / "golden.json"
RESPONSES_ORIGINALS_PATH = HERE / "responses.originals.json"
GOLDEN_ORIGINALS_PATH = HERE / "golden.originals.json"


def _src(name: str) -> Path:
    """The original in ``corpora/originals/`` if it exists, else the committed stub."""
    original = ORIGINALS_DIR / name
    return original if original.exists() else CORPORA_DIR / name


def using_originals() -> bool:
    """True when at least one real original is present — selects the fixture variant."""
    return ORIGINALS_DIR.is_dir() and any(
        (ORIGINALS_DIR / n).exists()
        for n in ("cable_identification.md", "octopus.pdf", "plos.pdf", "mitochondrion.html")
    )


def golden_path() -> Path:
    return GOLDEN_ORIGINALS_PATH if using_originals() else GOLDEN_PATH


def _extract_cable() -> list[Chunk]:
    return manual.extract_file(_src("cable_identification.md"))


def _extract_mitochondrion() -> list[Chunk]:
    html = _src("mitochondrion.html").read_text(encoding="utf-8")
    return url.extract(html, "https://en.wikipedia.org/wiki/Mitochondrion")


def _extract_octopus() -> list[Chunk]:
    return pdf.extract(_src("octopus.pdf"))


def _extract_plos() -> list[Chunk]:
    # PLOS ONE printable URL serves application/pdf, so it routes through the PDF
    # extractor exactly as cli.py does for a fetched PDF body. Only present in
    # originals mode (no committed stub).
    return pdf.extract(ORIGINALS_DIR / "plos.pdf")


def _corpora() -> tuple[tuple[str, str, object], ...]:
    """The active corpus list — adds PLOS as a 4th item when its original is present."""
    corpora = [
        ("cable-identification", "manual", _extract_cable),
        ("mitochondrion", "url", _extract_mitochondrion),
        ("octopus", "pdf", _extract_octopus),
    ]
    if (ORIGINALS_DIR / "plos.pdf").exists():
        corpora.append(("plos", "pdf", _extract_plos))
    return tuple(corpora)


def _chunk_key(chunk: Chunk) -> str:
    return hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()


def _load_responses() -> dict[str, str]:
    """Committed stub responses, with the gitignored originals fixture merged on top.

    Both are keyed by ``sha1(chunk.text)``, so a chunk only ever hits its own
    entry; the merge lets a single run mix stub-backed and original-backed corpora
    without collision.
    """
    raw = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    merged = {k: v for k, v in raw.items() if not k.startswith("_")}
    if RESPONSES_ORIGINALS_PATH.exists():
        extra = json.loads(RESPONSES_ORIGINALS_PATH.read_text(encoding="utf-8"))
        merged.update({k: v for k, v in extra.items() if not k.startswith("_")})
    return merged


def _prefilter(chunk: Chunk) -> Overflow | None:
    """Structural pre-filter seam (S3, #67).

    Delegates to the real pipeline logic (``classifier.prefilter_overflow``): a
    chunk an extractor flagged as structural chaff becomes an Overflow that
    bypasses the LLM entirely (zero token cost, ``dispatched=False``). When no
    extractor flags anything — as on the clean S0 corpora — this returns None and
    every chunk is dispatched, so the baseline is unchanged.
    """
    return prefilter_overflow(chunk)


def _route_chunk(
    chunk: Chunk,
    shapes: dict,
    responses: dict[str, str],
) -> tuple[dict, int]:
    """Route one chunk, returning (record, prompt_chars_dispatched).

    prompt_chars_dispatched is 0 for chunks pre-filtered without an LLM call.
    """
    base = {"position": chunk.position}

    prefiltered = _prefilter(chunk)
    if prefiltered is not None:
        bucket = overflow_bucket(prefiltered.reason)
        return {**base, "routing": bucket, "bucket": bucket,
                "reason": prefiltered.reason, "note_type": None, "dispatched": False}, 0

    key = _chunk_key(chunk)
    if key not in responses:
        raise KeyError(
            f"no captured response for chunk {key} (pos={chunk.position!r}); "
            f"corpus extraction drifted — refresh responses.json. text={chunk.text[:80]!r}"
        )
    prompt_chars = len(build_prompt(chunk, shapes))
    result = classify(chunk, shapes, llm=lambda _prompt, _r=responses[key]: _r)

    if isinstance(result, CardCandidate):
        return {**base, "routing": "card", "bucket": None, "reason": None,
                "note_type": result.note_type, "dispatched": True}, prompt_chars
    if isinstance(result, MultiCardCandidate):
        note_type = result.rows[0].note_type if result.rows else None
        return {**base, "routing": "card", "bucket": None, "reason": None,
                "note_type": note_type, "cards": len(result.rows), "dispatched": True}, prompt_chars
    # Overflow
    bucket = overflow_bucket(result.reason)
    return {**base, "routing": bucket, "bucket": bucket, "reason": result.reason,
            "note_type": None, "dispatched": True}, prompt_chars


def build_snapshot() -> dict:
    """Run all corpora through extract -> replay-classify -> route; return the snapshot."""
    shapes = load_shapes(SHAPES_PATH)
    responses = _load_responses()
    corpora_out = []
    for name, extractor, extract in _corpora():
        chunks = extract()
        records = []
        prompt_chars = 0
        for chunk in chunks:
            record, chars = _route_chunk(chunk, shapes, responses)
            records.append(record)
            prompt_chars += chars
        counts = {"card": 0, "qa": 0, "trimmed": 0}
        for record in records:
            counts[record["routing"]] += 1
        corpora_out.append({
            "name": name,
            "extractor": extractor,
            "chunk_count": len(chunks),
            "dispatched_count": sum(1 for r in records if r["dispatched"]),
            "prompt_token_chars": prompt_chars,
            "counts": counts,
            "chunks": records,
        })
    return {"schema_version": SCHEMA_VERSION, "corpora": corpora_out}


def load_golden() -> dict:
    return json.loads(golden_path().read_text(encoding="utf-8"))


def write_golden(snapshot: dict) -> None:
    golden_path().write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def record_responses() -> int:
    """Capture live-classifier responses for the active corpora into the gitignored
    ``responses.originals.json``.

    Calls the real ``classifier._default_llm`` (``openclaw infer model run``) once
    per dispatched chunk — the only step that touches the network. Pre-filtered
    chunks (structural chaff, ``dispatched=False``) are skipped, exactly as in the
    snapshot. Existing entries are reused so re-runs only fill gaps.
    """
    from anki_translator.classifier import _default_llm

    shapes = load_shapes(SHAPES_PATH)
    captured: dict[str, str] = {}
    if RESPONSES_ORIGINALS_PATH.exists():
        captured = {
            k: v
            for k, v in json.loads(RESPONSES_ORIGINALS_PATH.read_text(encoding="utf-8")).items()
            if not k.startswith("_")
        }
    committed = {
        k: v
        for k, v in json.loads(RESPONSES_PATH.read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    new_calls = 0
    for name, _extractor, extract in _corpora():
        for chunk in extract():
            if _prefilter(chunk) is not None:
                continue
            key = _chunk_key(chunk)
            if key in captured or key in committed:
                continue
            captured[key] = _default_llm(build_prompt(chunk, shapes))
            new_calls += 1
            print(f"  [{name}] recorded {key[:12]} ({new_calls} live calls)", file=sys.stderr)
    payload = {
        "_comment": "Live-classifier responses for the real originals (gitignored "
        "source). Keyed by sha1(chunk.text). Regenerate with harness.py --record.",
        **captured,
    }
    RESPONSES_ORIGINALS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESPONSES_ORIGINALS_PATH} (+{new_calls} new)")
    return 0


def diff_message(current: dict, golden: dict) -> str:
    """Human-readable diff between a fresh snapshot and the golden."""
    cur = json.dumps(current, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
    gold = json.dumps(golden, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
    import difflib

    delta = list(difflib.unified_diff(gold, cur, fromfile="golden.json", tofile="current", lineterm=""))
    return "\n".join(delta) or "(snapshots are equal)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S0 overflow-routing regression harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true", help="regenerate the active golden")
    group.add_argument("--check", action="store_true", help="diff vs golden; exit 1 on drift")
    group.add_argument(
        "--record",
        action="store_true",
        help="capture live-classifier responses for the originals (network; writes "
        "responses.originals.json)",
    )
    args = parser.parse_args(argv)

    if args.record:
        return record_responses()

    snapshot = build_snapshot()
    if args.update:
        write_golden(snapshot)
        print(f"wrote {golden_path()}")
        return 0
    if args.check:
        golden = load_golden()
        if snapshot != golden:
            print(diff_message(snapshot, golden), file=sys.stderr)
            return 1
        print("golden matches")
        return 0
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
