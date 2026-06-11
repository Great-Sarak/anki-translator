"""Queue file producer + consumer.

The producer (write_queue) takes the output of the classifier+tagger pipeline and writes
two markdown files per ingestion:

- queue/<date>-<slug>.md — flashcard candidates as structured blocks, separated by `---`.
  Presence-equals-approved: user deletes blocks they don't want; the rest are committed.
- qa/<date>-<slug>.md — Overflow chunks as standalone reference material. Different
  lifecycle from queue — qa files are kept permanently, queue files are committed/archived.

The consumer (parse_queue + commit_queue) reads a reviewed queue file, calls anki-manager
to create/update notes for surviving blocks, and archives the queue file to prevent
double-commits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .classifier import CardCandidate, Overflow

if TYPE_CHECKING:
    from anki_manager import AnkiManager

MAX_SLUG_LEN = 80
CARD_HEADER_RE = re.compile(r"^##\s+Card\s+\d+\s+—\s+(.+?)\s*$")
FIELD_LINE_RE = re.compile(r"^\*\*([^:]+):\*\*\s*(.*)$")
META_FIELDS = {"Deck", "Model", "Tags"}
CITATION_FIELDS = {"Source", "Position"}


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


_SUBSTANTIVE_REASON_PREFIXES = (
    "exceeds_budget:",
    "invalid_response:",
    "llm_error:",
    "no_shape_fit",
)

_SUBSTANTIVE_SIGNALS = (
    "multiple distinct facts",
    "too complex",
    "too long",
    "cannot fit",
    "can't fit",
    "couldn't fit",
    "cannot be reduced",
    "exceed",
)

_CHAFF_SIGNALS = (
    # citation / bibliography
    "bibliographic",
    "bibliography",  # "Passage is a bibliography entry/citation" appeared in Octopus PDF
    "citation header",
    "citation metadata",
    "citation/header",
    "is a citation",
    "is a journal citation",
    "citation/reference",
    "passage contains only a citation",
    "journal citation",
    "citation with multiple",
    "publication metadata",
    # document scaffolding
    "table of contents",
    "section header",
    "section heading",
    "document header",
    "document overview",
    "page header",
    "structural metadata",
    "header block",  # "passage is a header block with affiliations and metadata"
    "list of topic headings",
    "minireview title",
    # author / affiliations / acknowledgments
    "author names",
    "author names and affiliations",
    "list of authors",
    "affiliations and citations",
    "affiliations and metadata",
    "acknowledgments",
    "acknowledgements",
    "funding information",
    # transitional / boilerplate / introductory
    "transitional text",
    "introductory/transitional",
    "introductory material",
    "previews future sections",
    "upcoming content structure",
    "boilerplate",
    "disclaimer",
    # reference link lists
    "reference list",
    "reference links",
    "list of links",
    "list of reference links",
    "collection of reference links",
    # fragments / incomplete passages — extractor caught a stub of text where the
    # body is elsewhere; not card material on its own
    "is a fragment",
    "passage is fragmentary",
    "passage is incomplete",
    "bare taxonomic label",
    # general "no content"
    "no substantive content",
    "no factual content",
    "no learnable fact",
    "no discrete fact",
    "metadata rather than",
)


def overflow_bucket(reason: str, bucket: str | None = None) -> str:
    """Route an Overflow to either 'qa' (substantive) or 'trimmed' (chaff).

    Substantive overflow is content the classifier wanted to make a card from but
    couldn't quite fit — worth keeping next to the queue as reference material.
    Trimmed chaff is content with no factual value (citation metadata, TOC entries,
    section headers, author lists, transitional prose) — useful for spot-checking
    that nothing important was dropped, but disposable once verified.

    Precedence (#65) — classifier-prefix > LLM bucket > pattern-match:

    1. Classifier-mechanical prefixes (`exceeds_budget:`, `invalid_response:`,
       `llm_error:`, `no_shape_fit`) are authoritative `qa`. They fire on real
       content that didn't fit a budget or on a dispatch fault — mechanical facts,
       not content judgments — so they win even if the LLM mislabels the bucket.
    2. The LLM's self-tagged `bucket` ('qa'|'trimmed') is the primary content
       signal when present and valid. `classify()` has already coerced any
       missing/out-of-vocabulary value to None.
    3. Otherwise fall back to reason pattern-matching (pre-#65 behavior), so a
       run whose model didn't emit a bucket routes exactly as it did before.

    Default = qa. Unknown phrasings are kept rather than discarded — easier to
    extend `_CHAFF_SIGNALS` as new patterns appear than to recover content lost
    to overreaching trim rules.
    """
    if reason.startswith(_SUBSTANTIVE_REASON_PREFIXES):
        return "qa"
    if bucket in ("qa", "trimmed"):
        return bucket
    lowered = reason.lower()
    # Chaff check runs first. The LLM frequently rationalizes why a
    # bibliographic / citation chunk overflowed by saying it "exceeds field
    # capacity" or "requires multiple metadata fields" — the generic substantive
    # signal "exceed" then ate the wrong bucket on a live ingest (Cell octopus
    # PDF, 2026-06-08). A reason that names a chaff content type belongs in
    # trimmed regardless of how the LLM phrased the budget violation.
    if any(s in lowered for s in _CHAFF_SIGNALS):
        return "trimmed"
    # No chaff signal — fall back to substantive intent ("multiple distinct
    # facts", "too complex", "cannot fit"). Default to qa for unknown
    # phrasings; better to over-keep than to silently discard new patterns.
    if any(s in lowered for s in _SUBSTANTIVE_SIGNALS):
        return "qa"
    return "qa"


def write_queue(
    tagged: list[TaggedCandidate],
    overflow: list[Overflow],
    *,
    deck: str,
    slug: str,
    queue_dir: Path | str,
    qa_dir: Path | str,
    trimmed_dir: Path | str,
    ingestion_date: date | None = None,
) -> tuple[Path, Path, Path]:
    """Write the queue, qa, and trimmed files for one ingestion.

    Returns (queue_path, qa_path, trimmed_path). All three files are always
    written, even if their respective lists are empty — keeps downstream tooling
    (commit, archives, audits, reviewer spot-checks) consistent. Empty buckets
    render to a sentinel "no chunks" placeholder.

    Overflows are routed via :func:`overflow_bucket`: substantive overflow
    (content that wanted to be a card) lands in `qa_dir`, chaff (citation
    metadata, TOC entries, section headers, etc.) lands in `trimmed_dir`.
    """
    d = ingestion_date or date.today()
    filename = f"{d.isoformat()}-{slug}.md"

    queue_path = Path(queue_dir) / filename
    qa_path = Path(qa_dir) / filename
    trimmed_path = Path(trimmed_dir) / filename
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed_path.parent.mkdir(parents=True, exist_ok=True)

    qa_overflows = [ov for ov in overflow if overflow_bucket(ov.reason, ov.bucket) == "qa"]
    trimmed_overflows = [ov for ov in overflow if overflow_bucket(ov.reason, ov.bucket) == "trimmed"]

    queue_path.write_text(_render_queue(tagged, deck=deck), encoding="utf-8")
    qa_path.write_text(
        _render_overflow_file(qa_overflows, title="Q&A", slug=slug, ingestion_date=d),
        encoding="utf-8",
    )
    trimmed_path.write_text(
        _render_overflow_file(trimmed_overflows, title="Trimmed", slug=slug, ingestion_date=d),
        encoding="utf-8",
    )

    return queue_path, qa_path, trimmed_path


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


def _render_overflow_file(
    overflow: list[Overflow], *, title: str, slug: str, ingestion_date: date
) -> str:
    """Render an overflow markdown file — one ## section per overflow chunk with its
    citation header. Used for both `qa` (substantive) and `trimmed` (chaff) files;
    the only difference is the heading."""
    header = f"# {title} — {slug} ({ingestion_date.isoformat()})\n\n"
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


# ---- consumer side ----


class QueueParseError(Exception):
    """Raised when a queue file is malformed beyond what we can sensibly recover from."""


@dataclass(frozen=True)
class ParsedBlock:
    """One block parsed from a queue file. Round-trips with what write_queue serialized."""
    shape: str
    fields: dict[str, str]   # everything except Deck/Model/Tags
    deck: str
    model: str
    tags: list[str]


@dataclass
class CommitResult:
    created: list[str] = field(default_factory=list)        # stable_guids of new notes
    updated: list[str] = field(default_factory=list)        # stable_guids of existing notes whose fields changed
    failed: list[tuple[int, str]] = field(default_factory=list)  # (block_index, error message)
    archived_to: Path | None = None


def parse_queue(path: Path | str) -> list[ParsedBlock]:
    """Parse a reviewed queue file into ParsedBlocks.

    Splits on `\\n---\\n`. Blocks that are entirely whitespace are skipped. A block that is
    present but missing its `## Card N — <shape>` header raises QueueParseError — that's
    an editor mistake worth surfacing loudly rather than silently dropping.
    """
    p = Path(path)
    if not p.exists():
        raise QueueParseError(f"queue file not found: {p}")
    body = p.read_text(encoding="utf-8")
    raw_blocks = body.split("\n---\n")
    parsed: list[ParsedBlock] = []
    for i, raw in enumerate(raw_blocks):
        raw = raw.strip()
        if not raw:
            continue
        # The "empty queue" sentinel file (written by write_queue when there are no candidates)
        # starts with "# Queue (empty)" — skip it cleanly.
        if raw.startswith("# Queue"):
            continue
        try:
            parsed.append(_parse_one_block(raw))
        except QueueParseError as e:
            raise QueueParseError(f"block {i + 1} in {p}: {e}") from e
    return parsed


def _parse_one_block(raw: str) -> ParsedBlock:
    lines = raw.splitlines()
    header = lines[0] if lines else ""
    m = CARD_HEADER_RE.match(header)
    if not m:
        raise QueueParseError(f"missing '## Card N — <shape>' header (got: {header!r})")
    shape = m.group(1)

    fields: dict[str, str] = {}
    deck: str | None = None
    model: str | None = None
    tags: list[str] = []

    for line in lines[1:]:
        line = line.rstrip()
        if not line:
            continue
        fm = FIELD_LINE_RE.match(line)
        if not fm:
            continue  # tolerate stray lines (user notes between fields, etc.)
        name, value = fm.group(1).strip(), fm.group(2).strip()
        if name == "Deck":
            deck = value
        elif name == "Model":
            model = value
        elif name == "Tags":
            tags = [t.strip() for t in value.split(",") if t.strip()]
        else:
            fields[name] = value

    if deck is None:
        raise QueueParseError("missing **Deck:** line")
    if model is None:
        raise QueueParseError("missing **Model:** line")
    if not fields:
        raise QueueParseError("block has no content fields (Front/Back/Text/...)")

    return ParsedBlock(shape=shape, fields=fields, deck=deck, model=model, tags=tags)


def commit_queue(
    path: Path | str,
    mgr: "AnkiManager",
    *,
    dry_run: bool = False,
    archive_dir: Path | str | None = None,
) -> CommitResult:
    """Parse a queue file, call mgr.upsert_note for each block, then archive.

    dry_run=True: validate + call mgr in dry-run mode but do not archive the file.
    archive_dir defaults to <queue_dir>/committed/.

    On success the file is moved to archive_dir to prevent double-commits. On failure
    (any block fails) the file stays in place so the user can fix and retry; the dups are
    safe to retry because upsert_note is idempotent by stable_guid.
    """
    p = Path(path)
    blocks = parse_queue(p)
    result = CommitResult()

    # Ensure each referenced deck exists before the per-block upserts. mgr.add_deck
    # is idempotent and respects the allowlist — if the agent has the <new>
    # capability or a matching pattern (explicit or wildcard), the deck is
    # created here. Surfaced per-block so the user sees a clear "deck X cannot
    # be created" rather than an opaque "deck was not found" from AnkiConnect
    # at addNote time. Skipped in dry_run since deck creation is a mutation.
    deck_errors: dict[str, str] = {}
    if not dry_run:
        for deck in dict.fromkeys(b.deck for b in blocks):  # unique, preserves order
            try:
                mgr.add_deck(deck)
            except Exception as e:  # noqa: BLE001 — propagate per-block below
                deck_errors[deck] = f"{type(e).__name__}: {e}"

    for i, block in enumerate(blocks, start=1):
        if block.deck in deck_errors:
            result.failed.append(
                (i, f"deck setup failed for {block.deck!r}: {deck_errors[block.deck]}")
            )
            continue
        try:
            # Citation fields are part of the note fields; mgr's live schema validation
            # will catch any field-name mismatch with the user's note type.
            upsert = mgr.upsert_note(
                deck=block.deck,
                model=block.model,
                fields=block.fields,
                tags=block.tags or None,
                dry_run=dry_run,
            )
            if upsert.created:
                result.created.append(upsert.stable_guid)
            else:
                result.updated.append(upsert.stable_guid)
        except Exception as e:  # noqa: BLE001 — surface per-block, keep going
            result.failed.append((i, f"{type(e).__name__}: {e}"))

    if not dry_run and not result.failed:
        archive = Path(archive_dir) if archive_dir else (p.parent / "committed")
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / p.name
        p.rename(target)
        result.archived_to = target

    return result
