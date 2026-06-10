"""Manual/text extractor: raw text or .txt/.md file → list[Chunk] with a user-supplied label.

For pasted notes, transcripts, hand-written content — anything without a stable URL or
filename to cite. The user supplies a readable label (e.g. "chat 2026-05-27") that becomes
the Source value. Position is empty for raw text and .txt sources; for .md sources, each
chunk's Position is `#<heading-slug>` of the H2/H3 heading the chunk lives under (parity
with the URL extractor's anchor positions).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..chunk import Chunk
from ..classifier import PREFILTER_METADATA_KEY
from . import ExtractionError

# --- Structural pre-filter heuristics (#69, S5) ----------------------------------
#
# Flag heading-only chunks, prose-less front-matter, and link-list / table-of-
# contents bodies as structural chaff (PREFILTER_METADATA_KEY) so they bypass the
# LLM and route straight to trimmed (#67). Conservative bias: a chunk only flags
# on an unambiguous structural shape — any real prose line defeats every rule.
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
_H1_LINE_RE = re.compile(r"^#\s")
# A "**Label:** value" front-matter metadata line (Scope:, Date:, Author:, ...).
_LABEL_LINE_RE = re.compile(r"^\*\*[^*]+:\*\*")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)|https?://\S+")


def _prefilter_kind(text: str) -> str | None:
    """Classify a markdown chunk as structural chaff, or None to keep it.

    - 'heading': a chunk that is only heading line(s) with no body.
    - 'front-matter': a preamble led by an H1 whose body is solely
      "**Label:** value" metadata lines (no plain prose).
    - 'table-of-contents' / 'link-list': a body whose every line is a list item
      and every item contains a link, with no other prose.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    heading_lines = [ln for ln in lines if _HEADING_LINE_RE.match(ln)]
    body_lines = [ln for ln in lines if not _HEADING_LINE_RE.match(ln)]

    if heading_lines and not body_lines:
        return "heading"

    if body_lines and _H1_LINE_RE.match(lines[0]) and all(_LABEL_LINE_RE.match(ln) for ln in body_lines):
        return "front-matter"

    list_items = [m.group(1) for ln in body_lines if (m := _LIST_ITEM_RE.match(ln))]
    if body_lines and list_items and len(list_items) == len(body_lines) and all(
        _LINK_RE.search(item) for item in list_items
    ):
        heading_text = heading_lines[0].lower() if heading_lines else ""
        return "table-of-contents" if "contents" in heading_text else "link-list"

    return None


def _with_prefilter(metadata: dict[str, object], text: str) -> dict[str, object]:
    """Return metadata with a prefilter tag added if the chunk is structural chaff."""
    kind = _prefilter_kind(text)
    if kind:
        return {**metadata, PREFILTER_METADATA_KEY: kind}
    return metadata

# Strip an optional YAML frontmatter block at the start of markdown files.
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
# Markdown headings (## or ###) split the body into chunks.
MD_HEADING_SPLIT_RE = re.compile(r"^#{2,3}\s+.*$", re.MULTILINE)
# Capture the visible text of a heading line, stripping the leading `#`s and any
# trailing `#`s used by the closed-form heading style.
_HEADING_TEXT_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_SLUG_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def _slugify_heading(line: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to hyphens, strip trim hyphens.

    Pragmatic ASCII slugifier — the resulting Position is human-grep bait for
    locating a card's origin in the source `.md`, not an HTML id. Headings that
    contain only non-alphanumeric characters (rare) produce an empty slug, which
    the caller treats the same as "no preceding heading".
    """
    match = _HEADING_TEXT_RE.match(line.strip())
    if not match:
        return ""
    text = match.group(1)
    return _SLUG_SEPARATOR_RE.sub("-", text.lower()).strip("-")


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
            metadata=_with_prefilter({"label": label}, p),
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
        # (text, slug) pairs — slug is empty for the preamble (content before the
        # first heading), per #47 acceptance.
        parts: list[tuple[str, str]] = []
        if sections and sections[0].strip():
            parts.append((sections[0].strip(), ""))
        for head, section in zip(heads, sections[1:]):
            chunk_text = f"{head.strip()}\n\n{section.strip()}".strip()
            if chunk_text:
                parts.append((chunk_text, _slugify_heading(head)))
        if not parts:
            raise ExtractionError(f"manual extractor: no content after frontmatter/heading split in {p}")
        return [
            Chunk(
                text=part_text,
                source=effective_label,
                position=f"#{slug}" if slug else "",
                source_type="manual",
                metadata=_with_prefilter({"label": effective_label, "anchor": slug}, part_text),
            )
            for part_text, slug in parts
        ]

    # Plain text / .txt fallthrough
    return extract_text(body, label=effective_label)
