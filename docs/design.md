# Anki-Translator — Design

**Status**: design, pre-implementation
**Date**: 2026-05-27
**Paired with**: anki-manager (existing)

## Purpose

Convert source documents (URLs, PDFs, books, journal articles, manual notes) into:

1. Anki flashcards, via a human-reviewed queue
2. A companion Q&A markdown file for content that doesn't fit flashcard shapes

Anki-Translator owns all semantic decisions about what is card-worthy and how to shape it. Anki-Manager remains dumb storage.

## Architecture

Two-layer split.

- **Anki-Translator** (new repo): extraction, shape classification, citation, tag generation, queue management. All LLM-driven semantic work happens here.
- **Anki-Manager** (existing): given note type + fields + deck + tags, calls Anki-Connect to create/update notes. No semantic intelligence; no card-type inference; no shape decisions.

Translator imports Manager as a Python library. This keeps the dumb-storage boundary honest and lets Manager remain independently useful for the "I want to add one card by hand" path.

Rejected alternatives:
- *Manager owns shape inference*: bloats Manager with translator-specific heuristics that won't generalize to other producers (scripts, manual entry).
- *Translator with its own AnkiConnect client*: duplicates Manager. Not actually "independence" — just shifted dependency.

## Pipeline

1. **Extract** — pull text + structure from source (URL/PDF/book/manual).
2. **Classify** — for each chunk, decide whether it fits a configured shape within budget. Yes → card candidate. No → Q&A prose.
3. **Cite** — produce `Source` + `Position` for each chunk per source-type conventions.
4. **Shape** — assemble card candidate as a note (fields per shape).
5. **Tag** — apply user-specified batch tag + LLM-generated topic tags (seeded from existing deck tags).
6. **Queue** — write candidates to a markdown queue file; write overflow prose to Q&A file.
7. **Review** — human edits queue file, deletes unwanted blocks (presence = approved).
8. **Commit** — translator parses surviving blocks, calls Manager to create notes, archives the queue file.

## Shape-as-Specification

Card-worthiness is **mechanical**, not vibe-based: does the content fit a configured shape within field budgets?

- Fits → card candidate in the queue
- Doesn't fit → Q&A markdown overflow

This converts a fuzzy LLM judgment ("is this card-worthy?") into a hard test.

### Starting cutoffs (tunable, hard not soft)

| Field | Cutoff |
|-------|--------|
| Back-field | ≤ 200 chars |
| List | ≤ 7 items |
| Steps | ≤ 5 |
| Cloze deletion | ≤ K words per deletion (TBD) |

Loosening later is easy; tightening retroactively means rewriting cards.

### Shapes are not hardcoded

Translator pulls available note types from Anki at runtime:

- `modelNames` → which note types exist
- `modelFieldNames` → fields per note type

A separate `shapes.yaml` config maps **note-type-name → semantic shape + field roles + content cutoffs**. Example:

```yaml
"Basic":
  shape: term-def
  fields:
    front: Front
    back: Back
    source: Source
    position: Position
  cutoffs:
    back_max_chars: 200

"Cloze":
  shape: cloze
  fields:
    text: Text
    source: Source
    position: Position
  cutoffs:
    deletion_max_words: 5
```

Why explicit mapping (not auto-discovery from field names): same field names mean different things in different note types. "Front/Back" could be term-def, Q/A, or cloze-adjacent. One-time user markup beats forever-fragile heuristics.

### Bootstrap

If a fresh deck has no matching note types, translator creates a starter set via Anki-Connect `createModel` (including the `Source` and `Position` fields — see Source Citation below). Gives a fresh install something to work with on day one.

### Shape inventory (starting set, not exhaustive)

- `term-def` — Term → Definition (most encyclopedic content)
- `cloze` — sentence with one or more deletions
- `term-list` — Term → bounded list
- `term-steps` — Term → ordered steps
- `term-image` — Term → single image

Translator's known shapes are a *subset* of deck note types. Custom note types the user invents stay manual unless added to `shapes.yaml`.

Comparison/contrast cards (e.g., "Mitosis vs Meiosis") not modeled as a primary shape; rendered as two term-def notes plus an optional cloze. Revisit if usage shows this is wrong.

## Source Citation

Every note carries two fields: **Source** + **Position**. Both are designed in from the start; backfilling Source into existing notes later is painful.

### Source — canonical, stable, human-readable

| Source type | Source value |
|-------------|--------------|
| URL | full URL (`https://...`) |
| DOI | bare DOI (`10.1234/foo`) — not the doi.org URL |
| PDF | filename (translator ledger keeps a hash for re-find if renamed) |
| Book | title + edition (`DDIA, 1st ed.`) |
| Manual (chat/notes/transcript) | readable label (`chat 2026-05-27`, `Sorotassu notes 2026-05-27`) |

**Source MUST NOT be empty.** Cards without verifiable origin become useless within months.

### Position — freeform per-source-type

| Source type | Position value | Empty when |
|-------------|----------------|------------|
| URL | `#anchor` | no anchor exists |
| DOI | section string (`§3.2`) | unknown |
| PDF | `page N` (or `page N, ¶M` if needed) | unknown |
| Book | `ch. N, p. N` | unknown |
| Manual | — | always empty |

Position holds heterogeneous content across source types. Don't enforce cross-type structure; Position only needs to make sense *within* a Source. Document conventions per type in translator config alongside `shapes.yaml`.

Multi-level locations (volume + chapter + page; chapter + paragraph) collapse into Position as freeform string (`"vol. 2, ch. 3, p. 47"`). Resist adding a third structured field — simpler beats more granular for a flashcard tool.

## Dedup

Two layers, answering different questions.

### Card-level: "does this card already exist in Anki?"

Handled entirely by Anki-Connect's `canAddNotes`, which uses Anki's per-note-type duplicate config (default: first-field match). Manager calls it before `addNote` and surfaces conflicts. No translator-side logic needed.

### Chunk-level: "did I already process this source chunk?"

Translator-side ledger keyed by `Source + content-hash(chunk)`. Skips chunks already ingested. Stored in the translator repo, not in Anki.

Tolerates slight content changes: a meaningfully-revised paragraph will hash differently and re-process. Better to occasionally re-process than to silently skip a real revision.

## Notes vs. Cards

Translator emits **notes**, not cards. Anki note types generate one or more cards per note (recognition + production, etc.). Keeping the translator at the note level keeps it out of card-template territory.

## Deck Assignment

User-specified per ingestion: `anki-translator ingest <source> --deck "Reading"`.

Auto-routing by topic/classifier deferred until card volume justifies it. Easy to add later; hard to undo if it's wrong from day one.

## Tags

### Used

- **User-specified batch tag** (`--tag book-club-2026`): applied to every note from one ingestion. Enables "undo this batch later" via `tag:book-club-2026`.
- **Translator-generated topic tags** (`topic`, `topic::subtopic`): LLM generates these per note. Depth capped at 2 levels — Anki's tag UI degrades beyond that. Generation is **seeded from the deck's existing tag list** so the LLM prefers reusing `biology::organelles` rather than inventing `cell-biology::organelles` for the same concept. Without seeding, drift across ingestions is guaranteed.

### Not used

- Source name (`source::Kleppmann_DDIA`) — already in the Source field.
- Source type (`pdf`, `url`) — derivable from the Source field.
- Ingestion date — encoded in Anki's note metadata (creation time).

Tags shouldn't duplicate fields.

## Review Queue

Markdown file. **Presence-equals-approved**: deleting a block rejects it; no per-card approve flag to toggle.

### Ingestion outputs two files per source

- `queue/<date>-<slug>.md` — flashcard candidates, structured blocks separated by `---`. Reviewed and committed.
- `qa/<date>-<slug>.md` — prose overflow. **Standalone reference material** — separate lifecycle from the queue, kept permanently.

### Queue file format

```markdown
## Card 1 — term-def
**Front:** Mitochondria
**Back:** Powerhouse of the cell
**Source:** Kleppmann_DDIA
**Position:** ch. 1, p. 12
**Deck:** Reading
**Tags:** biology, biology::organelles
---
## Card 2 — cloze
**Text:** The {{c1::mitochondria}} is the powerhouse of the cell.
**Source:** Kleppmann_DDIA
**Position:** ch. 1, p. 12
**Deck:** Reading
**Tags:** biology, biology::organelles
---
```

### Workflow

```
$ anki-translator ingest https://example.com/article --deck "Reading" --tag book-club-2026
  → wrote queue/2026-05-27-example-article.md  (8 candidates)
  → wrote qa/2026-05-27-example-article.md     (3 prose chunks)

# User opens queue file, edits prose, deletes blocks they don't want

$ anki-translator commit queue/2026-05-27-example-article.md
  → 6 notes added via anki-manager
  → 2 skipped (card-level dedup via canAddNotes)
  → moved to queue/committed/2026-05-27-example-article.md
```

Committed files are archived (not deleted) to prevent double-commits and to preserve ingestion history.

Why markdown not JSON: editing prose during review is the main activity; JSON is hostile for that. Markdown opens in any editor (Obsidian works), is git-trackable, and the parser is ~30 lines (split on `---`, parse key-value).

## Repo Structure (proposed)

```
anki-translator/
  src/anki_translator/
    extractors/          # per source type (url, pdf, book, manual)
      __init__.py
      url.py
      pdf.py
      book.py
      manual.py
    classifier.py        # shape-fit decisions
    citation.py          # Source + Position formatting per source type
    queue.py             # markdown queue read/write/commit
    ledger.py            # chunk-level dedup (Source + content-hash)
    tagger.py            # topic-tag generation with existing-tag seeding
    config.py            # load shapes.yaml + citation conventions
    cli.py               # ingest, commit
  config/
    shapes.yaml          # note-type-name → semantic shape + field roles
    citations.yaml       # per-source-type Source/Position conventions
  docs/
    design.md            # this file
  queue/                 # active candidates
    committed/           # archived after commit
  qa/                    # prose overflow files, kept permanently
  ledger/                # chunk-hash ingestion log
  pyproject.toml
```

Depends on `anki-manager` as a library import.

## Open Items (deferred, not blocking start)

- **Workflow trigger**: v1 is CLI. Telegram/file-watcher triggers deferred until CLI shakes out.
- **Extraction tooling**: pypdf vs. pymupdf for PDFs, trafilatura vs. readability for URLs — pick per extractor when implementing. Choice of library affects what structure (page numbers, headings) is recoverable, which feeds Position.
- **Failure handling for Anki-Connect unavailable**: queue files are durable on disk; commit can retry. Document the recovery path.
- **Auto-add mode** for trusted sources: skip the queue, commit directly. Add only after enough hand-review to calibrate the classifier.
- **Cloze deletion budget**: pick `K` words per deletion before implementing the cloze shape.
- **Comparison/contrast shape**: may earn its own shape if two-term-def + cloze pattern turns out to be awkward in practice.

## Decisions Log (key tradeoffs)

| Decision | Choice | Rejected alternative |
|----------|--------|----------------------|
| Card-type inference location | Translator | Manager (couples Manager to translator-specific heuristics) |
| Shape vocabulary | Pulled from Anki + explicit `shapes.yaml` mapping | Auto-discovery from field names (fragile) |
| Source citation structure | 2 fields (Source + Position), freeform Position | 1 combined field (hard to query); 3+ structured fields (over-engineered) |
| Card-level dedup | Anki-Connect `canAddNotes` | Translator-side card-content hash (duplicates Anki's logic) |
| Chunk-level dedup | Translator ledger | None (re-ingestion produces dup candidates) |
| Emission unit | Notes | Cards (forces translator into template territory) |
| Deck routing | User-specified per ingestion | Auto-routing by topic (premature) |
| Tag strategy | Batch tag + topic/subtopic (seeded), no field-duplicating tags | Tag every facet (noise) |
| Review model | Markdown queue, presence-equals-approved | Per-card approve flag (toggling overhead); auto-add (uncalibrated) |
| Queue format | Markdown blocks | JSON (hostile for prose editing); Anki review-deck (rejection cleanup awkward) |
| Overflow handling | Separate `qa/` markdown file, standalone reference | Inline in queue (mixes lifecycles) |
