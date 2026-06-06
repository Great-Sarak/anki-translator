# End-to-end smoke test

This directory holds the v0.1 exit-criterion test: a real PDF ingestion → human review → live commit against the user's Anki collection, with verification that re-running on the same source is idempotent.

It is **not** part of the unit test suite and is **not** runnable in CI without manual setup. It exists as a runbook for the user to execute on their machine.

## Prerequisites

- The `kryshanti-anki` systemd unit is running (`systemctl status kryshanti-anki` or `anki-manager status`).
- The unit is loaded with the **`sorotassu` profile** (real collection), not `_anki_skill_testrun`. Confirm:

  ```sh
  curl -s -X POST http://127.0.0.1:8765 \
    -d '{"action":"getActiveProfile","version":6}'
  # Expect: {"result":"sorotassu","error":null}
  ```

  If the active profile is wrong, set `KRYSHANTI_ANKI_DEFAULT_PROFILE=sorotassu` in `/var/lib/kryshanti-anki/anki.env` and `sudo systemctl restart kryshanti-anki`. To bootstrap the profile from a `.colpkg` for the first time, see `ops/container/bootstrap-profile.sh` in `anki-manager`.
- AnkiConnect is reachable from the host shell.
- The invoking user is in the `kryshanti-anki-users` group.
- `openclaw` is on `PATH` and a model auth profile is configured for the default `anthropic/claude-haiku-4-5` (or whatever `ANKI_TRANSLATOR_MODEL` overrides to). Verify with `openclaw infer model run --json --prompt "ping" --model anthropic/claude-haiku-4-5`.
- Classifier and tagger fan out chunk calls in parallel. The default scales with the host CPU count — `(os.cpu_count() - 1) // 2`, floored at 1. On a 20-core host that's 9 workers; on a 2-core box it backs off to 1. Tune with `ANKI_TRANSLATOR_CONCURRENCY=<n>` if your gateway/account throttles aggressively or if you want sequential debug runs (`ANKI_TRANSLATOR_CONCURRENCY=1`). Background: each `openclaw infer model run` invocation has fixed ~7s of CLI bootstrap overhead per process at ~100% core time, but invocations parallelize cleanly up to the CPU-count-derived ceiling (see `spikes/001-gateway-direct-llm/`).
- `Myrzka::Octopus` is allowed in `/var/lib/kryshanti-anki/allowlist.toml` (or the Myrzka section has the `<new>` capability flag — it does by default; the deck will be created on first ingest).
- A fresh checkout: `pip install -e ../anki-manager_main && pip install -e .`

## Procedure

### Step 1 — Bootstrap starter note types

```sh
anki-translator bootstrap
```

Expected: JSON output listing AT Basic, AT Cloze, AT List, AT Steps as either `created` or `already_present`. A second run should show all four as `already_present`.

### Step 2 — Ingest a real article

The target is a *Current Biology* (Cell) PDF on octopuses — real scholarly prose, not a Wikipedia article. Exercises the PDF extractor on representative source material.

```sh
anki-translator ingest \
  https://www.cell.com/action/showPdf?pii=S0960-9822%2823%2901221-6 \
  --deck "Myrzka::Octopus" \
  --tag "octopus-smoke-$(date +%Y-%m-%d)"
```

Expected: JSON output reporting the queue file path, qa file path, and counts of candidates + overflow.

If the PDF URL ever 404s (Cell sometimes rotates direct-download URLs), swap in any short scholarly PDF you can vouch for and update the `--tag` accordingly.

### Step 3 — Inspect the queue file

Open the queue file under `queue/<date>-<slug>.md`. Verify:

- One `## Card N — <shape>` block per candidate
- Each block has Front/Back (or Text for cloze), Source, Position, Deck, Model, Tags
- Source field contains the full URL
- Position field references the section heading or page anchor the chunk came from (PDF extractor uses page numbers)
- Tags include both the LLM-generated topic tags (e.g. `biology`, `cephalopod`) and the `octopus-smoke-<date>` batch tag

### Step 4 — Human review

Edit the queue file:

- **Delete** one card block entirely (verify presence-equals-approved by removing one)
- **Edit** one field on another block (e.g., shorten a Back value)

Note the count of remaining blocks. You'll need it for verification in step 6.

### Step 5 — Commit

```sh
anki-translator commit queue/<date>-<slug>.md
```

Expected:

- `created` list contains stable_guids for all remaining blocks
- `failed` is empty
- `archived_to` points to `queue/committed/<same-name>.md`
- The original `queue/<name>.md` no longer exists

### Step 6 — Verify in Anki

```sh
# Card count
anki-manager call findNotes 'query=tag:octopus-smoke-<date>'
```

Or open Anki Desktop and browse `tag:octopus-smoke-<date>` in the deck `Myrzka::Octopus`.

Verify:

- Number of notes equals the number of blocks remaining in the queue file after your step-4 edits
- The edited field's value reflects your edit
- Source and Position fields are populated correctly
- The batch tag is on every note

### Step 7 — Retry idempotence

```sh
anki-translator commit queue/committed/<date>-<slug>.md
```

Expected: should fail at the `parse_queue` step or commit cleanly with zero new notes created (existing stable_guids → upsert returns `created=False` for each → goes to the `updated` list, not `created`). No duplicates appear in Anki.

If you want to test the chunk-level dedup, re-run step 2 (same URL). The ledger should suppress chunks that already produced cards, so the new queue file should be smaller or empty.

### Step 8 — Sync to AnkiWeb (and your other devices)

```sh
anki-manager sync
```

Expected: sync completes, your AnkiWeb account receives the new notes. They'll propagate to any other Anki client (mobile, web) on next pull.

## Exit criterion

This test passes when:

1. Steps 1–6 all behave as expected
2. The Anki note count matches the queue file's surviving block count exactly
3. Edited fields show your edits
4. Retry (step 7) produces zero duplicates
5. Sync (step 8) propagates to AnkiWeb without error

When all five hold, anki-translator v0.1 is shippable.

## Cleanup (optional)

The smoke run leaves real cards in your real collection. Options:

- **Keep them**: `Myrzka::Octopus` becomes a permanent topical deck. Reasonable if the cards are good.
- **Suspend them**: in Anki Desktop, browse `tag:octopus-smoke-<date>`, select all, Ctrl-J to toggle suspend. Cards remain in the deck but don't enter review rotation.
- **Delete the deck**: `anki-manager call deleteDecks 'decks=["Myrzka::Octopus"]' cardsToo=true`. Removes the entire smoke-run output. Use if the cards aren't useful or if you want a fresh run.

Whatever you choose, sync (step 8) so the cleanup propagates.

## Capturing failures

If anything diverges from expected:

- Don't close issue #17
- File a follow-up issue describing what you observed and which step failed
- Link the follow-up from #17

## Future automation

The natural automation path is a pytest marker (`@pytest.mark.e2e`) that runs the same procedure with assertions, gated on `openclaw infer model run` succeeding for the target model and `anki-manager status` returning `ready=true`. Deferred until the v0.1 manual run has stabilized.
