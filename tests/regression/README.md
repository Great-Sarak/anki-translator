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
* **Tokens ≤ baseline** — `prompt_token_chars` may only drop. Chunks routed
  without an LLM call (S3 extractor pre-filtering sets `dispatched=False`)
  contribute zero, so the saving is visible in this number.

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
