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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

from .chunk import Chunk
from .config import ShapeConfig

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PROMPT_PATH = Path(__file__).parent / "prompts" / "classify.txt"

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
    """Render the shapes list as it appears in the prompt."""
    lines: list[str] = []
    for name, cfg in shapes.items():
        non_cite_fields = [
            f"{role}={field_name}"
            for role, field_name in cfg.fields.items()
            if role not in {"source", "position"}
        ]
        cutoffs_str = ", ".join(f"{k}={v}" for k, v in cfg.cutoffs.items()) or "(no budget)"
        lines.append(
            f"- \"{name}\" (shape={cfg.shape}): fields {', '.join(non_cite_fields)}; budget: {cutoffs_str}"
        )
    return "\n".join(lines)


def build_prompt(chunk: Chunk, shapes: dict[str, ShapeConfig]) -> str:
    """Render the classifier prompt for one chunk + the available shapes."""
    return _load_prompt_template().format(
        shapes_block=_shapes_block(shapes),
        chunk_text=chunk.text,
    )


def _default_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Default LLM dispatcher — uses the Anthropic SDK with the configured haiku model."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic returns a list of content blocks; for plain prompts we expect one text block.
    parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
    return "".join(parts)


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

    budget_violation = _check_cutoffs(fields, shape_cfg)
    if budget_violation:
        return Overflow(chunk=chunk, reason=f"exceeds_budget: {budget_violation}")

    return CardCandidate(
        note_type=choice,
        shape=shape_cfg.shape,
        fields={k: str(v) for k, v in fields.items()},
        chunk=chunk,
    )


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
