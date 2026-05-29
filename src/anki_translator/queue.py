"""Queue file producer + (later) consumer.

The producer (write_queue) takes the output of the classifier+tagger pipeline and writes
two markdown files per ingestion:

- queue/<date>-<slug>.md — flashcard candidates as structured blocks, separated by `---`.
  Presence-equals-approved: user deletes blocks they don't want; the rest are committed.
- qa/<date>-<slug>.md — Overflow chunks as standalone reference material. Different
  lifecycle from queue — qa files are kept permanently, queue files are committed/archived.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from .classifier import CardCandidate, Overflow

MAX_SLUG_LEN = 80


@dataclass(frozen=True)
class TaggedCandidate:
    """A CardCandidate paired with its generated tags. Ready for queue serialization."""
    candidate: CardCandidate
    tags: list[str]


def make_slug(source: str, source_type: str) -> str:
    """Produce a filesystem-safe slug from a Source value, based on source type.

    URL: hostname + path joined with '-'.
    PDF: filename minus extension.
    Other: source as-is, lowercased and ASCII-sanitized.
    """
    if source_type == "url":
        p = urlparse(source)
        raw = (p.netloc + p.path).strip("/").replace("/", "-")
    elif source_type == "pdf":
        raw = source.rsplit(".", 1)[0]
    else:
        raw = source

    safe = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    safe = re.sub(r"-+", "-", safe)  # collapse runs of hyphens
    return safe[:MAX_SLUG_LEN] or "untitled"


def write_queue(
    tagged: list[TaggedCandidate],
    overflow: list[Overflow],
    *,
    deck: str,
    slug: str,
    queue_dir: Path | str,
    qa_dir: Path | str,
    ingestion_date: date | None = None,
) -> tuple[Path, Path]:
    """Write the queue file and the qa-overflow file for one ingestion.

    Returns (queue_path, qa_path). Both files are always written, even if their respective
    lists are empty — keeps downstream tooling (commit, archives, audits) consistent.
    """
    d = ingestion_date or date.today()
    filename = f"{d.isoformat()}-{slug}.md"

    queue_path = Path(queue_dir) / filename
    qa_path = Path(qa_dir) / filename
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)

    queue_path.write_text(_render_queue(tagged, deck=deck), encoding="utf-8")
    qa_path.write_text(_render_qa(overflow, slug=slug, ingestion_date=d), encoding="utf-8")

    return queue_path, qa_path


def _render_queue(tagged: list[TaggedCandidate], *, deck: str) -> str:
    if not tagged:
        return (
            "# Queue (empty)\n\n"
            "No card candidates were produced by this ingestion. All extracted content "
            "either routed to overflow or failed extraction.\n"
        )

    blocks: list[str] = []
    for i, tc in enumerate(tagged, start=1):
        blocks.append(_render_block(i, tc, deck=deck))
    return "\n".join(blocks)


def _render_block(index: int, tc: TaggedCandidate, *, deck: str) -> str:
    c = tc.candidate
    lines: list[str] = [f"## Card {index} — {c.shape}"]

    field_order = ("Front", "Back", "Text")  # canonical order for the visible content fields
    seen_fields: set[str] = set()
    for fname in field_order:
        if fname in c.fields:
            lines.append(f"**{fname}:** {_inline(c.fields[fname])}")
            seen_fields.add(fname)
    for fname, fvalue in c.fields.items():
        if fname not in seen_fields and fname not in ("Source", "Position"):
            lines.append(f"**{fname}:** {_inline(fvalue)}")

    # Source + Position last, since they're metadata not the card body
    if "Source" in c.fields:
        lines.append(f"**Source:** {_inline(c.fields['Source'])}")
    else:
        lines.append(f"**Source:** {_inline(c.chunk.source)}")
    if "Position" in c.fields:
        lines.append(f"**Position:** {_inline(c.fields['Position'])}")
    else:
        lines.append(f"**Position:** {_inline(c.chunk.position)}")

    lines.append(f"**Deck:** {deck}")
    lines.append(f"**Model:** {c.note_type}")
    lines.append(f"**Tags:** {', '.join(tc.tags)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _inline(value: str) -> str:
    """Inline a field value into a single markdown line. Preserves cloze braces; collapses newlines."""
    return value.replace("\r\n", "\n").replace("\n", " ").strip()


def _render_qa(overflow: list[Overflow], *, slug: str, ingestion_date: date) -> str:
    """Render the Q&A markdown — one ## section per overflow chunk with its citation header."""
    header = f"# Q&A — {slug} ({ingestion_date.isoformat()})\n\n"
    if not overflow:
        return header + "No overflow chunks from this ingestion.\n"

    sections: list[str] = [header]
    for i, ov in enumerate(overflow, start=1):
        c = ov.chunk
        sections.append(
            f"## {i}. From {c.source}"
            + (f" ({c.position})" if c.position else "")
            + f"\n\n_Reason: {ov.reason}_\n\n{c.text.strip()}\n"
        )
    return "\n".join(sections)
