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
)
from anki_translator.chunk import Chunk
from anki_translator.config import load_shapes
from anki_translator.extractors import manual, pdf, url
from anki_translator.queue import overflow_bucket

HERE = Path(__file__).parent
CORPORA_DIR = HERE / "corpora"
RESPONSES_PATH = HERE / "responses.json"
GOLDEN_PATH = HERE / "golden.json"
SHAPES_PATH = HERE.parent.parent / "config" / "shapes.yaml"

SCHEMA_VERSION = 1


def _extract_cable() -> list[Chunk]:
    return manual.extract_file(CORPORA_DIR / "cable_identification.md")


def _extract_mitochondrion() -> list[Chunk]:
    html = (CORPORA_DIR / "mitochondrion.html").read_text(encoding="utf-8")
    return url.extract(html, "https://en.wikipedia.org/wiki/Mitochondrion")


def _extract_octopus() -> list[Chunk]:
    return pdf.extract(CORPORA_DIR / "octopus.pdf")


CORPORA = (
    ("cable-identification", "manual", _extract_cable),
    ("mitochondrion", "url", _extract_mitochondrion),
    ("octopus", "pdf", _extract_octopus),
)


def _chunk_key(chunk: Chunk) -> str:
    return hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()


def _load_responses() -> dict[str, str]:
    raw = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _prefilter(chunk: Chunk) -> Overflow | None:
    """Structural pre-filter seam (S3, #67).

    Returns an :class:`Overflow` for chunks an extractor flagged as structural
    chaff, which bypass the LLM entirely (zero token cost). In S0 nothing is
    pre-filtered — every chunk is dispatched to ``classify()``. Later layers
    plug their logic in here so the token metric reflects the saving.
    """
    return None


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
    for name, extractor, extract in CORPORA:
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
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def write_golden(snapshot: dict) -> None:
    GOLDEN_PATH.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
    group.add_argument("--update", action="store_true", help="regenerate golden.json")
    group.add_argument("--check", action="store_true", help="diff vs golden; exit 1 on drift")
    args = parser.parse_args(argv)

    snapshot = build_snapshot()
    if args.update:
        write_golden(snapshot)
        print(f"wrote {GOLDEN_PATH}")
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
