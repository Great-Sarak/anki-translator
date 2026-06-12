---
name: anki-translator
description: "Create Anki flashcards from URLs, PDFs, Markdown/text files, or pasted text through anki-translator, then commit and sync them."
metadata: {"openclaw":{"emoji":"🃏","requires":{"bins":["anki-translator","anki-manager","openclaw"]}}}
---

# anki-translator

Use when the user asks to make flashcards from a document, URL, PDF, note file, or pasted text.

This is Layer 3 of the Anki stack:
- `anki-translator` extracts, classifies, tags, writes a queue, and commits generated cards.
- `anki-manager` owns the container lifecycle, Anki writes, deck allowlist, and sync.

## Default Workflow

Run the helper. It intentionally has no human approval gate: generated cards are committed and synced immediately, with one batch tag for rollback by created date/tag.

```sh
python3 skills/anki-translator/scripts/ingest_commit_sync.py \
  --deck "Myrzka::Reading" \
  --tag "batch-source-slug" \
  "https://example.com/article"
```

For pasted text:

```sh
python3 skills/anki-translator/scripts/ingest_commit_sync.py \
  --deck "Myrzka::Reading" \
  --tag "batch-note-slug" \
  --label "source label" \
  --text "Text to turn into cards."
```

If `--tag` is omitted, the helper generates `anki-translator-YYYYMMDD-HHMMSS`.

Other helper options:

- `--label` — source label for `--text` or manual files (becomes the note `Source`).
- `--concurrency` — sets `ANKI_TRANSLATOR_CONCURRENCY` (default `1`; see Prerequisites).
- `--model` — override `ANKI_TRANSLATOR_MODEL` for the classifier/tagger.
- `--skip-bootstrap` — skip the `anki-translator bootstrap` step when the `AT` note types already exist (the helper runs bootstrap first by default, which is idempotent).

## Output lanes

Every ingest writes three markdown files under `$ANKI_TRANSLATOR_STATE_DIR` (`/var/lib/anki-translator/`) — the same `<date>-<slug>.md` name in three sibling dirs. The helper's `ingest` block reports their sizes as `candidates`, `overflow_qa`, and `overflow_trimmed`.

- **`queue/`** — card candidates. The helper auto-commits them; on `commit` each becomes an Anki note and the file is archived to `queue/committed/`. (In the manual, no-helper path this is the human-review file: delete a block to reject that card.)
- **`qa/`** — **substantive overflow**: reference material worth keeping that did not fit a card shape within field budget (too long, multiple distinct facts, or a mechanical budget/parse failure). **Kept permanently** as standalone reference.
- **`trimmed/`** — **chaff**: non-substantive structural material (citations, bibliography, TOC, section headers, author/affiliations, boilerplate, fragments). **Disposable audit** — it exists so a reviewer can spot-check that nothing substantive was tossed; it carries a routing-drift telemetry footer.

The `overflow_qa` / `overflow_trimmed` counts are **lane sizes, not a backlog** — overflow files are *not* future-batch input. Each ingest is one-shot: `qa` is reference, `trimmed` is audit. A large overflow count next to a small `candidates` count is expected (a dense PDF yields few card-shaped facts), not a sign that cards were dropped.

## Reading the result

The helper prints one JSON object to stdout and exits `0` on success. On any step failure it prints `{"ok": false, "error", "command", "returncode", "stdout", "stderr"}` to **stderr** and exits non-zero.

- Check **`ok`** first.
- On success, the **`commit`** block is the source of truth for what landed: `created` / `updated` / `failed` (note GUIDs). `failed` should be empty — a non-empty `failed` with `ok: true` means some notes were rejected at the Anki layer (e.g. duplicate, deck-not-allowed) while the run otherwise completed; surface it rather than reporting a clean pass.
- `ingest.candidates` is the card count; `overflow_qa` / `overflow_trimmed` are lane sizes (see Output lanes).
- On `ok: false`, **stop and report** `error` + `stderr` — do not retry blindly. A deck-allowlist rejection or a failed service start will simply repeat.

## Idempotency & rollback

- **Re-running the same source is safe.** A chunk-level ledger (keyed by `Source` + content-hash) skips chunks already ingested, so a repeat run yields few or zero new candidates instead of duplicates. Anki's card-level `canAddNotes` is a second backstop.
- **Undo a batch by its tag.** Every note from one run carries the `--tag` value, so searching `tag:<batch-tag>` in Anki selects exactly that batch for bulk delete.

## Prerequisites

- The calling identity can run `anki-manager start` for `kryshanti-anki.service` (normally via `kryshanti-anki-users` + polkit).
- The calling identity has deck write permission in `/var/lib/kryshanti-anki/allowlist.toml`.
- `openclaw infer model run --json` works for the classifier/tagger.
- On shared gateways, keep `ANKI_TRANSLATOR_CONCURRENCY=1`. The helper sets this default unless the environment or `--concurrency` overrides it.

## Profile Selection

Until per-user profile switching lands in `anki-manager`, writes go to the currently active Anki profile, so no slug reaches this skill today.

When profile routing does land, keep two identities separate:

- **Authorization identity** — the *calling* identity (the Unix user running this skill, e.g. `teva`). This is what polkit and `allowlist.toml` check for `anki-manager start` and deck writes. In a subagent dispatch this is correctly the agent, not the human.
- **Destination identity** — whose Anki profile the cards belong to (the **target user slug**, e.g. `sorotassu`). In a subagent call the human is never the direct requester, so this must arrive as an **explicit parameter threaded as data** through the caller/workflow — never inferred from the calling identity.

A routing-capable build must **error if it is given no target slug** rather than silently falling back to the active profile; silent fallback is how cross-user contamination sneaks in once more than one profile exists. Do not guess another user's profile.

## Useful Direct Commands

```sh
anki-translator ingest <source> --deck "Deck" --tag "batch-tag"
anki-translator commit /var/lib/anki-translator/queue/<file>.md
anki-manager sync
```

The `commit` step archives the queue file to `queue/committed/`; the `qa` and `trimmed` lanes are left in place (see **Output lanes**).
