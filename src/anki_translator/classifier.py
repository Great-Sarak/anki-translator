"""LLM-driven shape classifier with hard cutoff enforcement.

For each Chunk, decide whether it fits one of the configured shapes within the shape's
budget. Yes → CardCandidate. No → Overflow, which the queue writer routes to the qa/
markdown bucket instead of the queue file.

The LLM call returns structured JSON; we then validate the proposed fields against the
shape's cutoffs, because the LLM may hallucinate compliance with the budgets it was
shown. Validation failure routes the chunk to overflow with a specific reason — the
classifier never trusts the LLM's self-report on budget compliance.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Union

from .chunk import Chunk
from .config import ShapeConfig

DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
PROMPT_PATH = Path(__file__).parent / "prompts" / "classify.txt"


def _default_concurrency() -> int:
    """Scale fan-out width to the host's CPU count.

    Each openclaw CLI dispatch is CPU-heavy during Node bootstrap (Spike 001
    saw ~100% core time per process), so the practical ceiling tracks core
    count. Heuristic: (cpu_count - 1) // 2, floored at 1. On a 20-core host
    that yields 9 workers, just below the empirically-measured saturation
    knee at 10. On a 2-core host it backs off to 1.

    Falls back to 4 if os.cpu_count() returns None (rare; some sandboxes).
    """
    cores = os.cpu_count()
    if cores is None:
        return 4
    return max(1, (cores - 1) // 2)


def resolve_concurrency() -> int:
    """Effective fan-out width: env override if set+positive, else CPU-scaled default."""
    raw = os.environ.get("ANKI_TRANSLATOR_CONCURRENCY")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _default_concurrency()

LLMCall = Callable[[str], str]
"""Type for the LLM dispatch function. Takes a prompt, returns the raw text response."""


@dataclass(frozen=True)
class CardCandidate:
    """A passage that fits a shape — ready for the review queue."""
    note_type: str
    shape: str
    fields: dict[str, str]
    chunk: Chunk


@dataclass(frozen=True)
class MultiCardCandidate:
    """A passage that produces multiple CardCandidates from one classification call.

    Used by the term-table shape: a reference table with M value columns and N rows
    yields N rows × M attribute templates = N×M cards from a single chunk. Callers
    that consume Classification must flatten MultiCardCandidate into its `rows` list
    of CardCandidates before downstream tagging / queue writing.
    """
    rows: tuple[CardCandidate, ...]


@dataclass(frozen=True)
class Overflow:
    """A passage that does not fit any shape's budget — routed to the qa/ markdown bucket."""
    chunk: Chunk
    reason: str  # one of: 'no_shape_fit', 'exceeds_budget', 'llm_error', 'invalid_response'
    bucket: str | None = None  # LLM self-tagged overflow bucket: 'qa' | 'trimmed' | None


Classification = Union[CardCandidate, MultiCardCandidate, Overflow]

PREFILTER_METADATA_KEY = "prefilter"
"""Chunk.metadata key an extractor sets to mark a chunk as structural chaff.

The value is the chaff *kind* (e.g. 'bibliography', 'table-of-contents',
'author-affiliations'). Flagged chunks are converted to Overflow *before*
classify() — they never reach the LLM, which is the whole token saving. The
extractor-agnostic seam (#67); the concrete filter rules live in the extractors
themselves (#68/#69/#70)."""


def prefilter_overflow(chunk: Chunk) -> Overflow | None:
    """Return a pre-classified Overflow if an extractor flagged this chunk as
    structural chaff, else None.

    Design (a) from #67: extractors keep emitting a single output type (Chunk)
    and flag chaff via ``metadata[PREFILTER_METADATA_KEY]``; the pipeline
    converts the flag here. The reason carries the ``extractor:`` prefix so
    :func:`anki_translator.queue.overflow_bucket` routes it straight to trimmed,
    bypassing both the LLM bucket and pattern-match — structural chaff is
    definitionally trimmed.
    """
    kind = chunk.metadata.get(PREFILTER_METADATA_KEY)
    if kind:
        return Overflow(chunk=chunk, reason=f"extractor: pre-filtered {kind}")
    return None


def split_prefiltered(chunks: Iterable[Chunk]) -> tuple[list[Chunk], list[Overflow]]:
    """Partition chunks into (to_classify, pre_filtered_overflows).

    Pre-filtered chunks must never reach classify() — the token saving is the
    point. Non-flagged chunks pass through untouched, so behavior is identical to
    today when no extractor flags anything.
    """
    to_classify: list[Chunk] = []
    prefiltered: list[Overflow] = []
    for chunk in chunks:
        overflow = prefilter_overflow(chunk)
        if overflow is not None:
            prefiltered.append(overflow)
        else:
            to_classify.append(chunk)
    return to_classify, prefiltered


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _shapes_block(shapes: dict[str, ShapeConfig]) -> str:
    """Render the shapes list as it appears in the prompt.

    Field names are emitted in their canonical (model-defined) casing only.
    Earlier renderings showed `role=FieldName` which led some models to emit
    JSON keys in the lowercase role form ("back") instead of the field name
    ("Back"); the resulting notes were rejected by anki-manager's live schema
    validation.
    """
    lines: list[str] = []
    for name, cfg in shapes.items():
        cutoffs_str = ", ".join(f"{k}={v}" for k, v in cfg.cutoffs.items()) or "(no budget)"
        if cfg.shape == "term-table":
            # term-table flattens attribute slots into many flat Anki fields
            # (Attr1Name, Attr1Value, ...). Listing them all confuses the LLM
            # about the response shape — it should be returning `rows`+`attrs`,
            # not filling flat fields. Surface the conceptual model instead.
            attr_slots = sum(1 for role in cfg.fields if role.startswith("attr") and role.endswith("_value"))
            lines.append(
                f"- \"{name}\" (shape={cfg.shape}): one lookup key per row + up to {attr_slots} attribute pairs "
                f"PER ROW (row count is unbounded — emit one entry in `rows` per source-table row); "
                f"budget: {cutoffs_str}"
            )
            continue
        visible_fields = [
            field_name
            for role, field_name in cfg.fields.items()
            if role not in {"source", "position"}
        ]
        lines.append(
            f"- \"{name}\" (shape={cfg.shape}): fields {', '.join(visible_fields)}; budget: {cutoffs_str}"
        )
    return "\n".join(lines)


def build_prompt(chunk: Chunk, shapes: dict[str, ShapeConfig]) -> str:
    """Render the classifier prompt for one chunk + the available shapes."""
    return _load_prompt_template().format(
        shapes_block=_shapes_block(shapes),
        chunk_text=chunk.text,
    )


def _default_llm(prompt: str, model: str | None = None) -> str:
    """Default LLM dispatcher — delegates to openclaw infer model run via the gateway."""
    import json
    import re
    import subprocess

    resolved_model = model or os.environ.get("ANKI_TRANSLATOR_MODEL", DEFAULT_MODEL)
    result = subprocess.run(
        ["openclaw", "infer", "model", "run", "--json", "--prompt", prompt, "--model", resolved_model],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    text = data["outputs"][0]["text"]
    # Strip markdown code fences that some models add despite prompt instructions.
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text).rstrip().removesuffix("```").rstrip()
    return text


def classify(
    chunk: Chunk,
    shapes: dict[str, ShapeConfig],
    llm: LLMCall | None = None,
) -> Classification:
    """Classify one chunk into a CardCandidate or Overflow.

    The LLM is the default classifier. Pass a custom `llm` callable in tests to avoid network.
    """
    if not shapes:
        return Overflow(chunk=chunk, reason="no_shape_fit")

    llm_fn = llm or _default_llm
    prompt = build_prompt(chunk, shapes)

    try:
        raw = llm_fn(prompt)
    except Exception as e:  # noqa: BLE001 — network/API failures should not crash the pipeline
        return Overflow(chunk=chunk, reason=f"llm_error: {type(e).__name__}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return Overflow(chunk=chunk, reason="invalid_response: not JSON")

    if isinstance(parsed, list):
        return Overflow(
            chunk=chunk,
            reason="invalid_response: got array (expected single object)",
        )
    if not isinstance(parsed, dict) or "choice" not in parsed:
        return Overflow(chunk=chunk, reason="invalid_response: missing 'choice'")

    choice = parsed["choice"]
    if choice == "overflow":
        # The LLM self-tags qa vs trimmed (#65). Tolerate missing/invalid — an
        # absent or out-of-vocabulary bucket falls back to reason pattern-match
        # in queue.overflow_bucket, preserving pre-#65 routing.
        bucket = parsed.get("bucket")
        if bucket not in ("qa", "trimmed"):
            bucket = None
        return Overflow(
            chunk=chunk,
            reason=str(parsed.get("reason", "no_shape_fit")),
            bucket=bucket,
        )

    if choice not in shapes:
        return Overflow(chunk=chunk, reason=f"invalid_response: unknown note type {choice!r}")

    shape_cfg = shapes[choice]

    # term-table is the only shape that returns >1 card per chunk. It uses a
    # different LLM response shape — `rows` instead of `fields` — and gets
    # expanded here into a MultiCardCandidate.
    if shape_cfg.shape == "term-table":
        return _classify_term_table(chunk, choice, shape_cfg, parsed)

    fields = parsed.get("fields")
    if not isinstance(fields, dict):
        return Overflow(chunk=chunk, reason="invalid_response: missing or non-dict 'fields'")

    # Canonicalize LLM-returned field names to the model's actual casing.
    # Some models occasionally emit lowercase ("back") instead of the canonical
    # ("Back") despite the prompt — anki-manager's live schema check then
    # rejects the note. Match case-insensitively against the shape's known
    # field names and rewrite to canonical casing here, so downstream cutoff
    # checks and the note payload both see the right keys.
    canonical_by_lower = {f.lower(): f for f in shape_cfg.fields.values()}
    canonical_fields: dict[str, str] = {}
    for key, value in fields.items():
        if not isinstance(key, str):
            return Overflow(chunk=chunk, reason="invalid_response: non-string field key")
        canonical_fields[canonical_by_lower.get(key.lower(), key)] = str(value)
    fields = canonical_fields

    # Normalize malformed cloze deletion markers. Anki requires {{cN::text}}
    # but the LLM occasionally emits the first deletion as `{cN::text}}` (one
    # opening brace) or `{{cN::text}` (one closing brace) — Anki then renders
    # the literal text instead of producing a cloze card. Fix in place so the
    # cutoff check below counts deletion-words against the corrected payload.
    if shape_cfg.shape == "cloze":
        text_field = shape_cfg.fields.get("text")
        if text_field and text_field in fields:
            fields[text_field] = _normalize_cloze_braces(fields[text_field])

    budget_violation = _check_cutoffs(fields, shape_cfg)
    if budget_violation:
        return Overflow(chunk=chunk, reason=f"exceeds_budget: {budget_violation}")

    return CardCandidate(
        note_type=choice,
        shape=shape_cfg.shape,
        fields=fields,
        chunk=chunk,
    )


def _classify_term_table(
    chunk: Chunk,
    note_type: str,
    shape_cfg: ShapeConfig,
    parsed: dict,
) -> Classification:
    """Expand a term-table LLM response into one CardCandidate per row.

    Expected response shape:
        {"choice": "AT Table",
         "rows": [
            {"key": "DP 1.2",
             "attrs": [{"name": "Link rate", "value": "HBR2"}, ...]},
            ...
         ]}

    Each row becomes one CardCandidate populating Key + AttrN/AttrNValue slots.
    Anki's conditional templates emit one card per non-empty AttrNName slot, so a
    row with M attributes yields M cards at note creation. Unused slots stay empty.

    Cutoffs enforced here:
        attrs_per_row_max  — reject any row with more attributes than the per-row cap
                             (NOT a cap on the row count; row count is unbounded).
        attr_max_chars     — reject if any attribute name or value exceeds the limit.
    """
    rows = parsed.get("rows")
    if not isinstance(rows, list) or not rows:
        return Overflow(chunk=chunk, reason="invalid_response: missing or empty 'rows'")

    attrs_per_row_max = shape_cfg.cutoffs.get("attrs_per_row_max", 4)
    attr_max_chars = shape_cfg.cutoffs.get("attr_max_chars", 80)
    candidates: list[CardCandidate] = []

    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            return Overflow(chunk=chunk, reason=f"invalid_response: row {row_idx} is not an object")
        key = row.get("key")
        attrs = row.get("attrs")
        if not isinstance(key, str) or not key.strip():
            return Overflow(chunk=chunk, reason=f"invalid_response: row {row_idx} missing 'key'")
        if not isinstance(attrs, list) or not attrs:
            return Overflow(chunk=chunk, reason=f"invalid_response: row {row_idx} missing 'attrs'")
        if len(attrs) > attrs_per_row_max:
            return Overflow(
                chunk=chunk,
                reason=f"exceeds_budget: row {row_idx} has {len(attrs)} attrs, per-row limit {attrs_per_row_max}",
            )

        fields: dict[str, str] = {shape_cfg.fields["key"]: key.strip()}
        for attr_idx, attr in enumerate(attrs, start=1):
            if not isinstance(attr, dict):
                return Overflow(chunk=chunk, reason=f"invalid_response: row {row_idx} attr {attr_idx} is not an object")
            name = attr.get("name")
            value = attr.get("value")
            if not isinstance(name, str) or not isinstance(value, str):
                return Overflow(
                    chunk=chunk,
                    reason=f"invalid_response: row {row_idx} attr {attr_idx} missing 'name'/'value'",
                )
            if len(name) > attr_max_chars:
                return Overflow(
                    chunk=chunk,
                    reason=f"exceeds_budget: row {row_idx} attr {attr_idx} name is {len(name)} chars, limit {attr_max_chars}",
                )
            if len(value) > attr_max_chars:
                return Overflow(
                    chunk=chunk,
                    reason=f"exceeds_budget: row {row_idx} attr {attr_idx} value is {len(value)} chars, limit {attr_max_chars}",
                )
            name_role = f"attr{attr_idx}_name"
            value_role = f"attr{attr_idx}_value"
            if name_role not in shape_cfg.fields or value_role not in shape_cfg.fields:
                return Overflow(
                    chunk=chunk,
                    reason=f"invalid_response: shape has no slot for attr {attr_idx}",
                )
            fields[shape_cfg.fields[name_role]] = name
            fields[shape_cfg.fields[value_role]] = value

        # Backfill unused attribute slots with empty strings so anki-manager's
        # validate-all-fields-present check passes. Conditional templates
        # ({{#AttrNName}}...{{/AttrNName}}) suppress empty cards at render time.
        for unused_idx in range(len(attrs) + 1, attrs_per_row_max + 1):
            name_role = f"attr{unused_idx}_name"
            value_role = f"attr{unused_idx}_value"
            if name_role in shape_cfg.fields:
                fields.setdefault(shape_cfg.fields[name_role], "")
            if value_role in shape_cfg.fields:
                fields.setdefault(shape_cfg.fields[value_role], "")

        candidates.append(CardCandidate(
            note_type=note_type,
            shape=shape_cfg.shape,
            fields=fields,
            chunk=chunk,
        ))

    return MultiCardCandidate(rows=tuple(candidates))


def classify_chunks(
    chunks: Iterable[Chunk],
    shapes: dict[str, ShapeConfig],
    llm: LLMCall | None = None,
    max_workers: int | None = None,
) -> list[Classification]:
    """Classify many chunks in parallel, preserving input order.

    The default subprocess dispatcher has high per-call overhead but parallelizes well —
    see spikes/001-gateway-direct-llm. With a single-thread `llm` (or `max_workers=1`),
    behavior is identical to calling `classify()` in a loop.
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return []
    workers = max_workers if max_workers is not None else resolve_concurrency()
    if workers <= 1:
        return [classify(chunk, shapes, llm=llm) for chunk in chunks_list]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: classify(c, shapes, llm=llm), chunks_list))


_LIST_ITEM_NUMBER_RE = re.compile(r"(?:^|\s)\d+\.\s")
_LIST_ITEM_BULLET_RE = re.compile(r"(?:^|\s)[•\-\*]\s")


def _count_list_items(text: str) -> int:
    """Count list items in a Back-field value, handling both multi-line and inline layouts.

    The classifier output may come back from the LLM in two shapes:
      - Multi-line, one item per line: `"1. A\n2. B\n3. C"`. The original
        implementation used `splitlines()` which catches this case.
      - Single logical line with inline delimiters: `"1. A 2. B 3. C"` or
        `"- A - B - C"`. `splitlines()` returns one line and the cutoff
        previously passed silently regardless of bullet count.

    When only a single non-empty line is present, count numbered (`\\d+\\. `)
    or bullet (`• `, `- `, `* `) delimiter markers. Returns the larger of the
    two counts in case both forms appear, floored at 1 so plain prose still
    registers as a single item.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return len(lines)
    numbered = len(_LIST_ITEM_NUMBER_RE.findall(text))
    bulleted = len(_LIST_ITEM_BULLET_RE.findall(text))
    return max(1, numbered, bulleted)


def _check_cutoffs(fields: dict[str, object], shape_cfg: ShapeConfig) -> str | None:
    """Return a violation reason if any cutoff is exceeded, else None.

    LLM may hallucinate compliance — verify against the actual rendered field lengths.
    """
    role_for_field = {field_name: role for role, field_name in shape_cfg.fields.items()}

    for cutoff_name, limit in shape_cfg.cutoffs.items():
        if cutoff_name == "back_max_chars":
            back_field = shape_cfg.fields.get("back")
            if back_field and back_field in fields:
                value = str(fields[back_field])
                if len(value) > limit:
                    return f"{back_field} is {len(value)} chars, limit {limit}"
        elif cutoff_name == "list_max_items":
            back_field = shape_cfg.fields.get("back")
            if back_field and back_field in fields:
                count = _count_list_items(str(fields[back_field]))
                if count > limit:
                    return f"{back_field} has {count} items, limit {limit}"
        elif cutoff_name == "steps_max":
            back_field = shape_cfg.fields.get("back")
            if back_field and back_field in fields:
                count = _count_list_items(str(fields[back_field]))
                if count > limit:
                    return f"{back_field} has {count} steps, limit {limit}"
        elif cutoff_name == "deletion_max_words":
            text_field = shape_cfg.fields.get("text")
            if text_field and text_field in fields:
                value = str(fields[text_field])
                violation = _check_cloze_deletion_words(value, limit)
                if violation:
                    return violation
    return None


_CLOZE_SINGLE_OPEN_RE = re.compile(r"(?<!\{)\{(c\d+::[^{}]*?)\}\}")
_CLOZE_SINGLE_CLOSE_RE = re.compile(r"\{\{(c\d+::[^{}]*?)\}(?!\})")
_CLOZE_SINGLE_BOTH_RE = re.compile(r"(?<!\{)\{(c\d+::[^{}]*?)\}(?!\})")


def _normalize_cloze_braces(text: str) -> str:
    """Repair malformed cloze deletion markers to canonical `{{cN::...}}`.

    Three observed shapes from Claude haiku-4-5, all on the first deletion:
      - `{cN::...}}`  (single open)
      - `{{cN::...}`  (single close)
      - `{cN::...}`   (single on both ends) — kicked in after we hardened the
        prompt's example; the model still copies whatever brace count it sees
        on the first deletion and self-corrects from `c2` on.

    Order matters: run single-both LAST so the asymmetric patterns above (which
    contain a single `{` or `}`) aren't snagged by it first.
    """
    text = _CLOZE_SINGLE_OPEN_RE.sub(r"{{\1}}", text)
    text = _CLOZE_SINGLE_CLOSE_RE.sub(r"{{\1}}", text)
    text = _CLOZE_SINGLE_BOTH_RE.sub(r"{{\1}}", text)
    return text


def _check_cloze_deletion_words(text: str, max_words: int) -> str | None:
    """For cloze: each {{cN::...}} deletion must contain ≤ max_words."""
    import re

    pattern = re.compile(r"\{\{c\d+::([^}]+)\}\}")
    for match in pattern.finditer(text):
        deletion = match.group(1).strip()
        word_count = len(deletion.split())
        if word_count > max_words:
            return f"cloze deletion {match.group(0)!r} has {word_count} words, limit {max_words}"
    return None
