# anki-translator

Document → Anki flashcards translator with a human-reviewed markdown queue. Layer 3 of the [Myrzka anki-skill stack](https://github.com/Great-Sarak/myrzka):

- **Layer 1** — [`anki-rpc`](https://github.com/Great-Sarak/anki-rpc): typed Python client for AnkiConnect.
- **Layer 2** — [`anki-manager`](https://github.com/Great-Sarak/anki-manager): lifecycle + domain ops for the headless Anki container.
- **Layer 3** — `anki-translator` (this repo): document extraction, semantic shape classification, citation, tag generation, and the human-review queue. Calls anki-manager to persist notes.

## What it does

Given a source (URL, PDF, or pasted text):

1. Extracts content into chunks with citation metadata (Source + Position).
2. Classifies each chunk as a fit for one of the configured note-type shapes (term/definition, cloze, list, steps, …) within field budgets — or routes it to a Q&A markdown overflow file.
3. Writes the card candidates to a markdown queue file for human review.
4. On `commit`, parses the queue and creates notes via anki-manager.

Card-worthiness is mechanical: "does this content fit a shape within the field budget?" Overflow is preserved as standalone reference material, not discarded.

## Status

Pre-implementation. See [`docs/design.md`](docs/design.md) (coming in [#3](https://github.com/Great-Sarak/anki-translator/issues/3)) for the full architecture, decisions log, and open items.

Track progress on the [Anki Skill Stack — Full Coverage](https://github.com/orgs/Great-Sarak/projects/5) project board.

## Install

Not yet shipped. Once scaffolded:

```sh
python3 -m venv .venv
.venv/bin/pip install -e ../anki-manager_main   # sibling repo
.venv/bin/pip install -e .
```

## CLI

Not yet implemented. Planned commands:

```sh
anki-translator bootstrap                                  # create starter note types in Anki if missing
anki-translator ingest <url|path|--text> \
    --deck "Reading" \
    --tag book-club-2026                                   # extract → classify → write queue + qa files
anki-translator commit queue/<date>-<slug>.md              # parse reviewed queue, call anki-manager
```

See [#16](https://github.com/Great-Sarak/anki-translator/issues/16) for the CLI implementation issue.

## License

MIT. Same as anki-rpc and anki-manager.
