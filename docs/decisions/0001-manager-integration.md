# ADR-0001: Integration with anki-manager

**Status**: accepted
**Date**: 2026-05-27
**Resolves**: [#2](https://github.com/Great-Sarak/anki-translator/issues/2)

## Decision

anki-translator imports `AnkiManager` as a Python library and dispatches per-candidate (in-process) during queue commit. We do not shell out to the `anki-manager` CLI.

## Reasoning

- **Public surface already exists.** `AnkiManager` and its supporting types (`AddResult`, `UpsertResult`, the public error classes, `compute_guid`, `file_lock`) are re-exported via `__all__` in `anki_manager/__init__.py`. The class is effectively public; the README just hasn't documented it as such yet.
- **CLI is a thin wrapper.** `anki_manager/cli.py` is dispatch-only — it calls `AnkiManager` methods directly. Verification (live model schema check, deck allowlist enforcement, stable-GUID derivation, lifecycle readiness gate) lives in the manager class, so library callers get the same guarantees as CLI callers.
- **Subprocess-per-note is bad for our workload.** Queue commits process N notes at a time. Per-note `subprocess.run` adds 100ms+ of process spawn overhead each and forces every field value through shell argument escaping — fragile for cloze cards or any content containing quotes, newlines, or backticks. In-process calls sidestep both problems.

## Cross-repo dependency

[Great-Sarak/anki-manager#1](https://github.com/Great-Sarak/anki-manager/issues/1) formalizes the public Python API contract (docs + verification-parity audit) and adds a batch CLI mode. The audit is not strictly blocking — we can call `mgr.upsert_note(...)` today and trust the thin-wrapper assumption — but it should land before v0.1 ships so we are not depending on undocumented behavior.

## Implications

- `pyproject.toml` declares `anki-manager` as a dependency (sibling install from local checkout until published).
- Queue commit (#15) iterates parsed candidates and calls `mgr.upsert_note(...)` per candidate.
- Bootstrap (#8) calls `mgr` for model queries and creation.
- All anki-related error handling re-raises from the documented anki-manager error types (`AnkiManagerError`, `InvalidNoteError`, `DeckNotAllowedError`, etc.).
