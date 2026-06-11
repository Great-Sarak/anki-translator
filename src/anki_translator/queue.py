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

# Extractor pre-filter reason prefix (#67). A chunk an extractor flagged as
# structural chaff never reaches the LLM; its reason is "extractor: pre-filtered
# <kind>". This is definitionally trimmed — bypasses both LLM bucket and
# pattern-match.
_EXTRACTOR_REASON_PREFIX = "extractor:"

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


# High-precision structural-chaff markers (#66). A strict subset of
# _CHAFF_SIGNALS: phrasings that are essentially *never* real subject-matter
# content. Because they're high-precision, they're safe to use as a conservative
# backstop that can override an LLM `qa` mislabel — unlike the fuzzier full list,
# which we no longer trust to override an explicit content judgment.
_STRONG_CHAFF_SIGNALS = (
    "bibliograph",
    "table of contents",
    "section header",
    "section heading",
    "document header",
    "page header",
    "header block",
    "author names",
    "affiliations",
    "acknowledgment",
    "acknowledgement",
    "reference list",
    "reference links",
    "list of links",
    "is a citation",
    "journal citation",
    "citation metadata",
    "citation header",
    "publication metadata",
)


@dataclass(frozen=True)
class BucketDecision:
    """The routing decision for one Overflow, with drift telemetry (#66).

    bucket:       final routing — 'qa' or 'trimmed'.
    llm_bucket:   the LLM's self-tag ('qa'|'trimmed'|None).
    downgraded:   True iff a strong chaff signal forced an LLM 'qa' to 'trimmed'.
    disagreement: 'qa_with_chaff' (LLM said qa, strong chaff overrode it),
                  'trimmed_without_chaff' (LLM said trimmed, no pattern chaff
                  signal — recorded but NOT acted on), or None.
    """
    bucket: str
    llm_bucket: str | None
    downgraded: bool
    disagreement: str | None


def _pattern_bucket(reason: str) -> str:
    """Pre-#65 reason pattern-matching. The fallback when no LLM bucket is present.

    A reason that names a chaff content type belongs in trimmed regardless of how
    the LLM phrased the budget violation — the generic 'exceed' signal otherwise
    ate the wrong bucket on a live ingest (Cell octopus PDF, 2026-06-08). Default
    to qa for unknown phrasings; better to over-keep than to silently discard.
    """
    lowered = reason.lower()
    if any(s in lowered for s in _CHAFF_SIGNALS):
        return "trimmed"
    if any(s in lowered for s in _SUBSTANTIVE_SIGNALS):
        return "qa"
    return "qa"


def classify_overflow_bucket(reason: str, bucket: str | None = None) -> BucketDecision:
    """Route an Overflow to 'qa' (substantive) or 'trimmed' (chaff), with telemetry.

    Substantive overflow is content the classifier wanted to card but couldn't
    quite fit — kept next to the queue as reference. Trimmed chaff has no factual
    value (citations, TOC, section headers, author lists) — disposable once a
    reviewer confirms nothing important was dropped.

    Precedence — classifier-prefix > LLM bucket (+ conservative chaff backstop) >
    pattern-match:

    1. Classifier-mechanical prefixes (`exceeds_budget:`, `invalid_response:`,
       `llm_error:`, `no_shape_fit`) are authoritative `qa` — mechanical facts,
       not content judgments.
    2. The LLM's self-tagged bucket (#65) is the primary content signal, but with
       `_CHAFF_SIGNALS` demoted to a safety net (#66): if the LLM says `qa` yet a
       *strong* chaff signal fires, downgrade to `trimmed` and record the
       disagreement. The backstop is conservative — it only ever downgrades
       qa→trimmed on a high-precision signal, never silently upgrades
       trimmed→qa. An LLM `trimmed` with no pattern chaff signal is recorded as a
       disagreement for drift-watching but left as `trimmed`.
    3. With no LLM bucket, fall back to reason pattern-matching (pre-#65), so a
       run whose model didn't emit a bucket routes exactly as it did before.
    """
    if reason.startswith(_EXTRACTOR_REASON_PREFIX):
        # Extractor-flagged structural chaff: definitionally trimmed, no LLM
        # involved. Bypasses both the LLM bucket and pattern-match (#67).
        return BucketDecision("trimmed", None, False, None)

    if reason.startswith(_SUBSTANTIVE_REASON_PREFIXES):
        return BucketDecision("qa", bucket if bucket in ("qa", "trimmed") else None, False, None)

    if bucket in ("qa", "trimmed"):
        lowered = reason.lower()
        strong_chaff = any(s in lowered for s in _STRONG_CHAFF_SIGNALS)
        if bucket == "qa" and strong_chaff:
            return BucketDecision("trimmed", "qa", True, "qa_with_chaff")
        if bucket == "trimmed" and not any(s in lowered for s in _CHAFF_SIGNALS):
            # Disagreement: the LLM trimmed something pattern-match wouldn't have.
            # Recorded for drift, but trusted — we never upgrade trimmed→qa.
            return BucketDecision("trimmed", "trimmed", False, "trimmed_without_chaff")
        return BucketDecision(bucket, bucket, False, None)

    return BucketDecision(_pattern_bucket(reason), None, False, None)


def overflow_bucket(reason: str, bucket: str | None = None) -> str:
    """Final routing string for an Overflow. Thin wrapper over
    :func:`classify_overflow_bucket` for callers that only need the bucket."""
    return classify_overflow_bucket(reason, bucket).bucket


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

    decisions = [(ov, classify_overflow_bucket(ov.reason, ov.bucket)) for ov in overflow]
    qa_overflows = [(ov, dec) for ov, dec in decisions if dec.bucket == "qa"]
    trimmed_overflows = [(ov, dec) for ov, dec in decisions if dec.bucket == "trimmed"]
    drift = {
        "qa_with_chaff": sum(1 for _, dec in decisions if dec.disagreement == "qa_with_chaff"),
        "trimmed_without_chaff": sum(1 for _, dec in decisions if dec.disagreement == "trimmed_without_chaff"),
    }

    queue_path.write_text(_render_queue(tagged, deck=deck), encoding="utf-8")
    qa_path.write_text(
        _render_overflow_file(qa_overflows, title="Q&A", slug=slug, ingestion_date=d),
        encoding="utf-8",
    )
    trimmed_path.write_text(
        _render_overflow_file(trimmed_overflows, title="Trimmed", slug=slug, ingestion_date=d, drift=drift),
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


def _render_drift_footer(drift: dict[str, int]) -> str:
    """Render the routing-drift telemetry footer for the trimmed file (#66).

    A durable, greppable per-ingest record of LLM-bucket vs pattern disagreement,
    so model drift is watchable over time:
      - qa_with_chaff: chunks the LLM tagged `qa` that a strong chaff signal
        overrode to `trimmed` (the conservative backstop fired).
      - trimmed_without_chaff: chunks the LLM tagged `trimmed` that no pattern
        chaff signal would have caught (recorded, not acted on).
    """
    return (
        "\n---\n\n"
        f"_Routing drift (LLM bucket vs pattern): "
        f"{drift['qa_with_chaff']} qa→trimmed override(s) on strong chaff signal; "
        f"{drift['trimmed_without_chaff']} llm-trimmed chunk(s) with no pattern signal._\n"
    )


def _stage_label(reason: str, decision: BucketDecision) -> str:
    """Human-readable rejection stage for an overflow item.

    Recovers *which* stage routed a chunk out of the queue — otherwise only
    inferable by eye from the reason string. Three rejection points:
    structural pre-filter (extractor, pre-LLM, zero tokens), classifier-mechanical
    (budget/shape/parse faults), and the LLM's content judgment — plus the S2
    backstop when it downgrades an LLM `qa` to `trimmed`.
    """
    if reason.startswith(_EXTRACTOR_REASON_PREFIX):
        return "extractor (structural pre-filter, no LLM call)"
    if reason.startswith(_SUBSTANTIVE_REASON_PREFIXES):
        return "classifier (mechanical: budget/shape/parse)"
    if decision.downgraded:
        return "LLM → backstop downgrade (chaff signal)"
    return "LLM (content judgment)"


def _render_overflow_file(
    overflow: list[tuple[Overflow, BucketDecision]], *, title: str, slug: str,
    ingestion_date: date, drift: dict[str, int] | None = None,
) -> str:
    """Render an overflow markdown file — one ## section per overflow chunk with its
    citation header. Used for both `qa` (substantive) and `trimmed` (chaff) files;
    the only difference is the heading. Each item carries a `_Stage:_` line naming
    the rejection point (#66 backstop activity included). The trimmed file also
    carries a drift telemetry footer (#66) when `drift` is supplied."""
    header = f"# {title} — {slug} ({ingestion_date.isoformat()})\n\n"
    footer = _render_drift_footer(drift) if drift is not None else ""
    if not overflow:
        return header + "No overflow chunks from this ingestion.\n" + footer

    sections: list[str] = [header]
    for i, (ov, dec) in enumerate(overflow, start=1):
        c = ov.chunk
        sections.append(
            f"## {i}. From {c.source}"
            + (f" ({c.position})" if c.position else "")
            + f"\n\n_Stage: {_stage_label(ov.reason, dec)}_  \n"
            + f"_Reason: {ov.reason}_\n\n{c.text.strip()}\n"
        )
    return "\n".join(sections) + footer


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
