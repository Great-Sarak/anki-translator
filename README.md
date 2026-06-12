# anki-translator

Document → Anki flashcards translator with a human-reviewed markdown queue. Layer 3 of the [Myrzka anki-skill stack](https://github.com/Great-Sarak/myrzka):

- **Layer 1** — [`anki-rpc`](https://github.com/Great-Sarak/anki-rpc): typed Python client for AnkiConnect.
- **Layer 2** — [`anki-manager`](https://github.com/Great-Sarak/anki-manager): lifecycle + domain ops for the headless Anki container.
- **Layer 3** — `anki-translator` (this repo): document extraction, semantic shape classification, citation, tag generation, and the human-review queue. Calls anki-manager to persist notes.

## What it does

Given a source (URL, PDF, Markdown/text file, or pasted text):

1. Extracts content into chunks with citation metadata (Source + Position).
2. Classifies each chunk: does it fit one of the configured note-type shapes (term/definition, cloze, list, steps, …) within field budgets? Fit → a card candidate. No fit → **overflow**, routed to one of two markdown lanes (`qa` or `trimmed`, see below).
3. Writes the card candidates to a `queue` markdown file, and the overflow to the `qa`/`trimmed` files — one set per ingestion.
4. On `commit`, parses the reviewed queue and creates notes via anki-manager.

Card-worthiness is mechanical: "does this content fit a shape within the field budget?" Nothing substantive is discarded — content that can't be carded is preserved as reference, and chaff is set aside for audit rather than deleted.

## Output lanes

Each ingestion writes three markdown files (same `<date>-<slug>.md` name in three sibling dirs). All three are always written; empty lanes get a placeholder so downstream tooling stays consistent.

| Lane | Holds | Lifecycle |
|------|-------|-----------|
| **`queue/`** | Card candidates, as structured blocks separated by `---`. | **Review → commit.** Presence-equals-approved: delete blocks you don't want; `commit` creates the rest as Anki notes, then archives the file to `queue/committed/` to prevent double-commits. |
| **`qa/`** | **Substantive overflow** — reference material worth keeping that didn't fit a card shape within field budget (too long, multiple distinct facts, or a mechanical `exceeds_budget` / `no_shape_fit` / `invalid_response` / `llm_error`). | **Kept permanently** as standalone reference. Unknown phrasings default here — better to over-keep than silently discard. |
| **`trimmed/`** | **Chaff** — non-substantive structural material: citations, bibliography, TOC, section headers, author/affiliations, acknowledgments, boilerplate, fragments. | **Disposable audit.** Exists for debug visibility — a reviewer spot-check that nothing substantive was tossed. Carries a routing-drift telemetry footer. |

Overflow files are **not** future-batch input — each ingest is one-shot. The `overflow_qa` / `overflow_trimmed` counts in the helper output are lane sizes, not a backlog to re-run. `qa` is reference; `trimmed` is audit.

Routing precedence (`qa` vs `trimmed`) lives in `classify_overflow_bucket` (`src/anki_translator/queue.py`): mechanical classifier prefixes → `qa`; extractor-flagged structural chaff → `trimmed`; otherwise the LLM's self-tagged bucket, with a conservative strong-chaff backstop that only ever downgrades `qa → trimmed`, never the reverse.

## Status

v0.1 shipped. Installs as a system tool alongside `anki-manager`; smoke-tested end-to-end (ingest → commit → sync). See [`docs/design.md`](docs/design.md) for the full architecture, decisions log, and open items.

Track progress on the [Anki Skill Stack — Full Coverage](https://github.com/orgs/Great-Sarak/projects/5) project board.

## Install

Editable install alongside the sibling `anki-manager` checkout:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ../anki-manager_main   # sibling repo (Layer 2)
.venv/bin/pip install -e .
```

State (queue / qa / trimmed) is written under `$ANKI_TRANSLATOR_STATE_DIR` when set (the system install points it at `/var/lib/anki-translator/`); unset, it falls back to cwd-relative dirs for in-tree development. Bundled config (`shapes.yaml`, `tagger.yaml`) resolves from `$ANKI_TRANSLATOR_CONFIG_DIR` or the copy shipped beside the package.

## CLI

Console script `anki-translator`:

```sh
anki-translator bootstrap                                  # create the AT-prefixed starter note types in Anki if missing
anki-translator ingest <url|path|--text> \
    --deck "Myrzka::Reading" \
    --tag book-club-2026                                   # extract → classify → write queue + qa + trimmed files
anki-translator card --from-text "One paragraph." \
    --deck "Myrzka::Reading"                               # shape a single paragraph into a card (review by default; --commit to write straight to Anki)
anki-translator commit /var/lib/anki-translator/queue/<date>-<slug>.md   # parse reviewed queue, call anki-manager
```

For the no-review-gate path (ingest → commit → sync in one shot, with a batch tag for rollback), the skill ships `skills/anki-translator/scripts/ingest_commit_sync.py`. See [`skills/anki-translator/SKILL.md`](skills/anki-translator/SKILL.md).

## License

MIT. Same as anki-rpc and anki-manager.
