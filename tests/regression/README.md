# Overflow-routing regression harness (S0, #64)

The objective baseline the #59 redesign asserts against: **correctness held,
token count dropped.** Every later layer (S1–S6) must show, against this
harness, that the overflow split held-or-improved and the LLM-facing token cost
did not rise.

## What it measures

For each of three representative corpora — one per extractor —
`harness.build_snapshot()` runs `extract → classify → route` and records:

| corpus | extractor | fixture |
| --- | --- | --- |
| `cable-identification` | `manual` | `corpora/cable_identification.md` |
| `mitochondrion` | `url` | `corpora/mitochondrion.html` |
| `octopus` | `pdf` | `corpora/octopus.pdf` (built by `corpora/make_octopus_pdf.py`) |

Per chunk it snapshots the routing (`card` / `qa` / `trimmed`), and per corpus
the qa/trimmed/card split counts plus `prompt_token_chars` — the summed length
of every prompt actually dispatched to `classify()`.

These fixtures stand in for the live corpora the redesign was driven by (the
real Octopus PDF, the Wikipedia Mitochondrion article, the cable-identification
`.md`). The repo commits those corpora's *routed outputs* under `qa/` and
`trimmed/`, not their source inputs; the fixtures here reproduce the same
extractor shapes and routing behavior deterministically.

## Determinism

The LLM is stubbed with a record/replay fixture, `responses.json`, keyed by
`sha1(chunk.text)`. **Zero network.** A chunk with no captured response is a
hard error — that means corpus extraction drifted and the fixture must be
refreshed. The token figure is a **character proxy**, not a tokenizer count: a
stable relative measure, not an absolute token bill.

## How S1–S6 assert against it

* **Split unchanged-or-improved** — no layer may move real content (`card`/`qa`)
  into `trimmed`. Compare per-chunk `routing` and the `counts` against the
  golden; `card`/`qa` totals on the clean corpora may hold or rise, never fall.
* **Tokens ≤ baseline, with one reviewed exception** — `prompt_token_chars` may
  only drop *through routing*. Chunks routed without an LLM call (S3 extractor
  pre-filtering sets `dispatched=False`) contribute zero, so the saving is
  visible in this number, and no layer may claw it back by dispatching more.

  A deliberate edit to `prompts/classify.txt` is the one thing that may raise it,
  because the prompt is summed into every dispatched chunk. Such a rise is
  allowed only when all of the following hold, and the regenerating commit must
  say so:

  1. The golden diff is confined to `prompt_token_chars` — no `routing`,
     `counts`, `chunk_count`, `dispatched_count` or drift movement.
  2. The rise equals *(prompt delta) × (dispatched chunks)* exactly, so it is
     accounted for rather than merely observed.
  3. The prompt change buys a stated correctness improvement, and the per-chunk
     delta is proportionate to it.
  4. It lands as its own labelled commit, never folded into a behaviour change.

  Note the stub corpora are clean by construction, so a prompt edit that teaches
  the classifier to *trim* some new kind of chaff shows its cost here and none of
  its benefit — the benefit appears only on real corpora containing that chaff.
  That asymmetry is a reason to check the arithmetic, not a reason to wave the
  rise through.

## Running

```sh
python tests/regression/harness.py            # print the live snapshot
python tests/regression/harness.py --check     # diff vs golden, exit 1 on drift
python tests/regression/harness.py --update    # regenerate golden.json (deliberate)
pytest tests/test_overflow_regression.py        # CI assertion
```

`--update` is intentional: regenerate the golden only when a routing change is
expected and reviewed (e.g. landing S1–S6), never to paper over an unexplained
diff.

## Originals mode — the real-corpora baseline

The committed stubs are small, deterministic stand-ins, but the redesign was
driven by the *real* sources. Drop the full-size originals into the gitignored
`corpora/originals/` folder and the harness runs against **them** instead, using
a separate fixture variant so the portable golden is never touched:

| corpus | source file in `corpora/originals/` |
| --- | --- |
| `cable-identification` | `cable_identification.md` (the real ~36 KB note) |
| `octopus` | `octopus.pdf` (the real 13-page Current Biology PDF) |
| `plos` *(4th item)* | `plos.pdf` — PLOS ONE `pone.0018710`, `?type=printable` (served as `application/pdf`, so it routes through the PDF extractor exactly as `cli.py` does for a fetched PDF body) |

Each corpus falls back **independently**: a present original wins, else the
committed stub. `mitochondrion` has no original, so it always uses its stub.
PLOS is added only when `originals/plos.pdf` is present.

When any original is present the harness reads/writes the `*.originals.json`
variant (`golden.originals.json`, `responses.originals.json`) — both gitignored,
alongside the copyrighted source. CI (no originals) is unaffected: it runs the
3 stubs against the committed `golden.json`.

```sh
# 1. populate corpora/originals/ with the real files (gitignored)
# 2. capture live-classifier responses for them (network; haiku via the gateway)
python tests/regression/harness.py --record    # writes responses.originals.json
# 3. lock the real baseline
python tests/regression/harness.py --update     # writes golden.originals.json
python tests/regression/harness.py --check      # verifies the real snapshot
```

`--record` calls the live classifier once per *dispatched* chunk (pre-filtered
chaff is skipped), keyed by `sha1(chunk.text)` like the replay fixture. Re-runs
only fill gaps.

Note: the `test_clean_corpora_have_zero_routing_drift` invariant is a property of
the hand-built clean stubs and is **skipped** in originals mode — real corpora
legitimately exercise the S2 backstop (the live model self-tags chunks `trimmed`
whose reasons aren't a recognised chaff pattern; #66 records that as telemetry,
not failure). The real routing is locked by the `golden.originals.json` match.
