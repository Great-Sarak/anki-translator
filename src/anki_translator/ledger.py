"""Chunk-level dedup ledger.

Tracks which (source, chunk) pairs the translator has already ingested. Used to skip
re-processing of unchanged content when an extraction runs twice on the same source. A
slight content change produces a different hash, so meaningfully-revised content is
re-processed rather than silently skipped.

Storage is an append-only JSON Lines file under the translator's ledger/ directory.
Each line records the (source, chunk) hash, the stable GUID of the Anki note that was
created from it (from anki-manager), the deck, and an ISO-8601 timestamp.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

HASH_LENGTH = 16  # truncated hex chars from SHA-256 — 2^64 keyspace is sufficient


def chunk_key(source: str, chunk: str) -> str:
    """Stable hash of (source, chunk) used to key the ledger."""
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\0")  # separator prevents collisions like (\"ab\", \"c\") vs (\"a\", \"bc\")
    h.update(chunk.encode("utf-8"))
    return h.hexdigest()[:HASH_LENGTH]


@dataclass(frozen=True)
class LedgerEntry:
    key: str
    source: str
    stable_guid: str
    deck: str
    ingested_at: str  # ISO-8601 UTC


class Ledger:
    """In-memory index over an append-only JSONL on disk."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._index: dict[str, LedgerEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = LedgerEntry(**data)
                except (json.JSONDecodeError, TypeError) as e:
                    raise ValueError(f"ledger {self.path} line {line_num} is corrupt: {e}") from e
                self._index[entry.key] = entry

    def seen(self, source: str, chunk: str) -> bool:
        """True if this (source, chunk) pair has been ingested before."""
        return chunk_key(source, chunk) in self._index

    def lookup(self, source: str, chunk: str) -> LedgerEntry | None:
        """Return the LedgerEntry for this (source, chunk), or None if not present."""
        return self._index.get(chunk_key(source, chunk))

    def record(self, source: str, chunk: str, stable_guid: str, deck: str) -> LedgerEntry:
        """Record an ingestion. Idempotent — re-recording the same (source, chunk) updates the entry."""
        key = chunk_key(source, chunk)
        entry = LedgerEntry(
            key=key,
            source=source,
            stable_guid=stable_guid,
            deck=deck,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._index[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def __len__(self) -> int:
        return len(self._index)
