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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Union

from .chunk import Chunk
from .config import ShapeConfig

DEFAULT_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_CONCURRENCY = 8
"""Default fan-out width for classify_chunks/tag_candidates.

The default subprocess dispatcher pays ~7s of openclaw CLI bootstrap per call,
but those bootstraps parallelize cleanly. Spike 001 measured a ~5x throughput
win at 8 workers vs sequential, with diminishing/negative returns past ~10.
"""
PROMPT_PATH = Path(__file__).parent / "prompts" / "classify.txt"


def resolve_concurrency() -> int:
    """Effective fan-out width: env override if set+positive, else DEFAULT_CONCURRENCY."""
    raw = os.environ.get("ANKI_TRANSLATOR_CONCURRENCY")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_CONCURRENCY

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
class Overflow:
    """A passage that does not fit any shape's budget — routed to the qa/ markdown bucket."""
    chunk: Chunk
    reason: str  # one of: 'no_shape_fit', 'exceeds_budget', 'llm_error', 'invalid_response'


Classification = Union[CardCandidate, Overflow]


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
        visible_fields = [
            field_name
            for role, field_name in cfg.fields.items()
            if role not in {"source", "position"}
        ]
        cutoffs_str = ", ".join(f"{k}={v}" for k, v in cfg.cutoffs.items()) or "(no budget)"
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

    if not isinstance(parsed, dict) or "choice" not in parsed:
        return Overflow(chunk=chunk, reason="invalid_response: missing 'choice'")

    choice = parsed["choice"]
    if choice == "overflow":
        return Overflow(chunk=chunk, reason=str(parsed.get("reason", "no_shape_fit")))

    if choice not in shapes:
        return Overflow(chunk=chunk, reason=f"invalid_response: unknown note type {choice!r}")

    shape_cfg = shapes[choice]
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

    budget_violation = _check_cutoffs(fields, shape_cfg)
    if budget_violation:
        return Overflow(chunk=chunk, reason=f"exceeds_budget: {budget_violation}")

    return CardCandidate(
        note_type=choice,
        shape=shape_cfg.shape,
        fields=fields,
        chunk=chunk,
    )


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
                items = [ln for ln in str(fields[back_field]).splitlines() if ln.strip()]
                if len(items) > limit:
                    return f"{back_field} has {len(items)} items, limit {limit}"
        elif cutoff_name == "steps_max":
            back_field = shape_cfg.fields.get("back")
            if back_field and back_field in fields:
                items = [ln for ln in str(fields[back_field]).splitlines() if ln.strip()]
                if len(items) > limit:
                    return f"{back_field} has {len(items)} steps, limit {limit}"
        elif cutoff_name == "deletion_max_words":
            text_field = shape_cfg.fields.get("text")
            if text_field and text_field in fields:
                value = str(fields[text_field])
                violation = _check_cloze_deletion_words(value, limit)
                if violation:
                    return violation
    return None


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
