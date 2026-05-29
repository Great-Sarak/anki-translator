"""LLM-driven topic tag generation, seeded from the deck's existing tag vocabulary.

The point of seeding is drift prevention. Without showing the LLM what tags already exist,
it will happily invent "cell-biology::organelles" for one card and "biology::organelles"
for the next, splintering the vocabulary across ingestions.

Tags from this module do NOT include: source name (lives in Source field), source type
(derivable from Source), ingestion date (encoded in note creation time). Just topic tags.
The optional batch tag is appended unmodified — that's the user's lever for "find every
card from this ingestion later."
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from .classifier import CardCandidate, LLMCall, _default_llm, resolve_concurrency

PROMPT_PATH = Path(__file__).parent / "prompts" / "tag.txt"
MAX_DEPTH = 2  # cap at topic::subtopic — Anki's tag UI degrades beyond this
TAG_NORMALIZE_RE = re.compile(r"[^a-z0-9:\-]+")


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _existing_tags_block(existing_tags: list[str]) -> str:
    if not existing_tags:
        return "(no existing tags — this is a fresh deck)"
    # Sort + dedup; cap the list to avoid blowing up the prompt on a huge collection.
    seen = sorted(set(existing_tags))
    return "\n".join(f"- {t}" for t in seen[:200])


def _fields_block(fields: dict[str, str]) -> str:
    # Drop Source/Position from the prompt — they're not informative for topic classification
    relevant = {k: v for k, v in fields.items() if k not in {"Source", "Position"}}
    return "\n".join(f"{k}: {v}" for k, v in relevant.items())


def build_prompt(candidate: CardCandidate, existing_tags: list[str]) -> str:
    return _load_prompt_template().format(
        existing_tags_block=_existing_tags_block(existing_tags),
        note_type=candidate.note_type,
        fields_block=_fields_block(candidate.fields),
    )


def _normalize_tag(tag: str) -> str | None:
    """Lowercase, ASCII, hyphens-not-spaces. Cap depth at MAX_DEPTH. Return None if unusable."""
    tag = tag.strip().lower().replace(" ", "-")
    tag = TAG_NORMALIZE_RE.sub("", tag)
    if not tag:
        return None
    # Cap depth: split on ::, keep only first MAX_DEPTH segments
    parts = [p for p in tag.split("::") if p]
    if not parts:
        return None
    return "::".join(parts[:MAX_DEPTH])


def generate_tags(
    candidate: CardCandidate,
    existing_tags: list[str],
    batch_tag: str | None = None,
    llm: LLMCall | None = None,
) -> list[str]:
    """Generate normalized topic tags for a CardCandidate.

    Returns up to 3 generated tags (deduplicated, normalized, depth-capped), plus the
    batch_tag appended unchanged if provided. On any LLM failure returns just the
    batch_tag (or an empty list) — the pipeline doesn't crash on tagging hiccups.
    """
    llm_fn = llm or _default_llm
    prompt = build_prompt(candidate, existing_tags)

    generated: list[str] = []
    try:
        raw = llm_fn(prompt)
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("tags"), list):
            for raw_tag in parsed["tags"]:
                if not isinstance(raw_tag, str):
                    continue
                normalized = _normalize_tag(raw_tag)
                if normalized and normalized not in generated:
                    generated.append(normalized)
                if len(generated) >= 3:
                    break
    except Exception:  # noqa: BLE001 — tagging failures fall back to batch_tag only
        pass

    result: list[str] = list(generated)
    if batch_tag:
        bt = batch_tag.strip()
        if bt and bt not in result:
            result.append(bt)
    return result


def tag_candidates(
    candidates: Iterable[CardCandidate],
    existing_tags: list[str],
    batch_tag: str | None = None,
    llm: LLMCall | None = None,
    max_workers: int | None = None,
) -> list[list[str]]:
    """Generate tags for many candidates in parallel, preserving input order.

    `existing_tags` is captured once and shared as the seed across calls — workers do
    not see each other's tag suggestions. That's intentional: parallel tag generation
    on a single batch shouldn't drift the vocabulary mid-flight.
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return []
    workers = max_workers if max_workers is not None else resolve_concurrency()
    if workers <= 1:
        return [generate_tags(c, existing_tags, batch_tag=batch_tag, llm=llm) for c in candidates_list]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda c: generate_tags(c, existing_tags, batch_tag=batch_tag, llm=llm),
                candidates_list,
            )
        )
