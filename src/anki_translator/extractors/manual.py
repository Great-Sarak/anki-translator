"""Manual/text extractor: raw text or .txt/.md file → list[Chunk] with a user-supplied label.

For pasted notes, transcripts, hand-written content — anything without a stable URL or
filename to cite. The user supplies a readable label (e.g. "chat 2026-05-27") that becomes
the Source value. Position is always empty for manual sources by design.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..chunk import Chunk
from . import ExtractionError

# Strip an optional YAML frontmatter block at the start of markdown files.
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
# Markdown headings (## or ###) split the body into chunks.
MD_HEADING_SPLIT_RE = re.compile(r"^#{2,3}\s+.*$", re.MULTILINE)


def _default_label() -> str:
    return f"manual {date.today().isoformat()}"


def extract_text(text: str, label: str | None = None) -> list[Chunk]:
    """Split a raw text string into paragraph chunks.

    label defaults to "manual YYYY-MM-DD" if not provided. Must end up non-empty after
    stripping — Source MUST NOT be empty per the citation contract.
    """
    label = (label or _default_label()).strip()
    if not label:
        raise ExtractionError("manual extractor: label must be non-empty (Source cannot be empty)")
    if not text or not text.strip():
        raise ExtractionError("manual extractor: text content is empty")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        raise ExtractionError("manual extractor: no paragraphs found after splitting")

    return [
        Chunk(
            text=p,
            source=label,
            position="",
            source_type="manual",
            metadata={"label": label},
        )
        for p in paragraphs
    ]


def extract_file(path: Path | str, label: str | None = None) -> list[Chunk]:
    """Read a .txt or .md file and split into chunks.

    For .md: strip YAML frontmatter and split on H2/H3 headings. For .txt: paragraph-split.
    label defaults to the file stem if not provided.
    """
    p = Path(path)
    if not p.exists():
        raise ExtractionError(f"manual extractor: file not found: {p}")

    body = p.read_text(encoding="utf-8")
    effective_label = label or p.stem

    if p.suffix.lower() == ".md":
        body = FRONTMATTER_RE.sub("", body, count=1)
        # Split on H2/H3 headings. The heading line itself goes with the section following it.
        sections = MD_HEADING_SPLIT_RE.split(body)
        # Re-stitch: pair each heading with its content if present
        heads = MD_HEADING_SPLIT_RE.findall(body)
        parts: list[str] = []
        # Preamble (before first heading) — keep if non-empty
        if sections and sections[0].strip():
            parts.append(sections[0].strip())
        for head, section in zip(heads, sections[1:]):
            chunk_text = f"{head.strip()}\n\n{section.strip()}".strip()
            if chunk_text:
                parts.append(chunk_text)
        if not parts:
            raise ExtractionError(f"manual extractor: no content after frontmatter/heading split in {p}")
        return [
            Chunk(
                text=part,
                source=effective_label,
                position="",
                source_type="manual",
                metadata={"label": effective_label},
            )
            for part in parts
        ]

    # Plain text / .txt fallthrough
    return extract_text(body, label=effective_label)
