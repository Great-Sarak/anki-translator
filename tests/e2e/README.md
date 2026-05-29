# End-to-end smoke test

This directory holds the v0.1 exit-criterion test: a real URL ingestion → human review → live commit against the user's Anki collection, with verification that re-running on the same source is idempotent.

It is **not** part of the unit test suite and is **not** runnable in CI without manual setup. It exists as a runbook for the user to execute on their machine.

## Prerequisites

- The `kryshanti-anki` systemd unit is running (`systemctl --user status kryshanti-anki` or `anki-manager status`).
- AnkiConnect is reachable from the host shell.
- The invoking user is in the `kryshanti-anki-users` group.
- `openclaw` is on `PATH` and a model auth profile is configured for the default `anthropic/claude-haiku-4-5` (or whatever `ANKI_TRANSLATOR_MODEL` overrides to). Verify with `openclaw infer model run --json --prompt "ping" --model anthropic/claude-haiku-4-5`.
- Classifier and tagger fan out chunk calls in parallel. The default scales with the host CPU count — `(os.cpu_count() - 1) // 2`, floored at 1. On a 20-core host that's 9 workers; on a 2-core box it backs off to 1. Tune with `ANKI_TRANSLATOR_CONCURRENCY=<n>` if your gateway/account throttles aggressively or if you want sequential debug runs (`ANKI_TRANSLATOR_CONCURRENCY=1`). Background: each `openclaw infer model run` invocation has fixed ~7s of CLI bootstrap overhead per process at ~100% core time, but invocations parallelize cleanly up to the CPU-count-derived ceiling (see `spikes/001-gateway-direct-llm/`).
- `Myrzka::Testing` is allowed in `/var/lib/kryshanti-anki/allowlist.toml` (or the Myrzka section has the `<new>` capability flag — it does by default).
- A fresh checkout: `pip install -e ../anki-manager_main && pip install -e .`

## Procedure

### Step 1 — Bootstrap starter note types

```sh
anki-translator bootstrap
```

Expected: JSON output listing AT Basic, AT Cloze, AT List, AT Steps as either `created` or `already_present`. A second run should show all four as `already_present`.

### Step 2 — Ingest a real article

Pick a short, stable article URL (Wikipedia is reliable). Suggestion:

```sh
anki-translator ingest \
  https://en.wikipedia.org/wiki/Mitochondrion \
  --deck "Myrzka::Testing" \
  --tag "e2e-smoke-$(date +%Y-%m-%d)"
```

Expected: JSON output reporting the queue file path, qa file path, and counts of candidates + overflow.

### Step 3 — Inspect the queue file

Open `queue/<date>-en-wikipedia-org-wiki-mitochondrion.md`. Verify:

- One `## Card N — <shape>` block per candidate
- Each block has Front/Back (or Text for cloze), Source, Position, Deck, Model, Tags
- Source field contains the full URL
- Position field contains an anchor like `#Structure` for headings within the page (or empty if a paragraph isn't under a `<h2 id=...>`)
- Tags include both the LLM-generated topic tags and the batch tag

### Step 4 — Human review

Edit the queue file:

- **Delete** one card block entirely (verify presence-equals-approved by removing one)
- **Edit** one field on another block (e.g., shorten a Back value)

Note the count of remaining blocks. You'll need it for verification in step 6.

### Step 5 — Commit

```sh
anki-translator commit queue/<date>-en-wikipedia-org-wiki-mitochondrion.md
```

Expected:

- `created` list contains stable_guids for all remaining blocks
- `failed` is empty
- `archived_to` points to `queue/committed/<same-name>.md`
- The original `queue/<name>.md` no longer exists

### Step 6 — Verify in Anki

```sh
# Card count
anki-manager call findNotes 'query=tag:e2e-smoke-<date>'
```

Or open Anki Desktop and browse `tag:e2e-smoke-<date>` in the deck `Myrzka::Testing`.

Verify:

- Number of notes equals the number of blocks remaining in the queue file after your step-4 edits
- The edited field's value reflects your edit
- Source and Position fields are populated correctly
- The batch tag is on every note

### Step 7 — Retry idempotence

```sh
anki-translator commit queue/committed/<date>-en-wikipedia-org-wiki-mitochondrion.md
```

Expected: should fail at the `parse_queue` step or commit cleanly with zero new notes created (existing stable_guids → upsert returns `created=False` for each → goes to the `updated` list, not `created`). No duplicates appear in Anki.

If you want to test the chunk-level dedup, re-run step 2 (same URL). The ledger should suppress chunks that already produced cards, so the new queue file should be smaller or empty.

## Exit criterion

This test passes when:

1. Steps 1-6 all behave as expected
2. The Anki note count matches the queue file's surviving block count exactly
3. Edited fields show your edits
4. Retry (step 7) produces zero duplicates

When all four hold, anki-translator v0.1 is shippable.

## Capturing failures

If anything diverges from expected:

- Don't close issue #17
- File a follow-up issue describing what you observed and which step failed
- Link the follow-up from #17

## Future automation

The natural automation path is a pytest marker (`@pytest.mark.e2e`) that runs the same procedure with assertions, gated on `openclaw infer model run` succeeding for the target model and `anki-manager status` returning `ready=true`. Deferred until the v0.1 manual run has stabilized.
