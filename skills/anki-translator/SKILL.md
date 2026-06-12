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

The queue, qa, and trimmed lanes remain available in `/var/lib/anki-translator/` for audit even though v1 auto-commits.
