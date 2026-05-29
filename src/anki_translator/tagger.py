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
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from .classifier import CardCandidate, LLMCall, _default_llm, resolve_concurrency
from .config import TaggerConfig

PROMPT_PATH = Path(__file__).parent / "prompts" / "tag.txt"
MAX_DEPTH = 2  # cap at topic::subtopic — Anki's tag UI degrades beyond this
TAG_NORMALIZE_RE = re.compile(r"[^a-z0-9:\-]+")

_SEED_DENY_EXACT: frozenset[str] = frozenset({"spike", "cloze"})
_SEED_DENY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d{4}-\d{2}-\d{2}"),     # date stamp anywhere (e.g. e2e-smoke-2026-05-29)
    re.compile(r"^anki-skill-testrun-"),  # any sibling-repo test marker
)


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _get_openclaw_agent_names() -> set[str]:
    """Read agent ids+names from `openclaw config get agents.list`.

    Returns the union of `id` and `name` for every agent, lowercased. Empty set
    on any failure (openclaw missing, command errors, malformed JSON, timeout).
    """
    try:
        result = subprocess.run(
            ["openclaw", "config", "get", "agents.list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return set()
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, list):
        return set()
    names: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for key in ("id", "name"):
            v = entry.get(key)
            if isinstance(v, str) and v.strip():
                names.add(v.strip().lower())
    return names


def _filter_seed_vocabulary(
    tags: Iterable[str],
    *,
    config: TaggerConfig,
    agent_names: Iterable[str] = (),
) -> list[str]:
    """Strip scaffolding artifacts from the seed vocabulary before it goes into
    the LLM prompt. See Great-Sarak/anki-translator#38.

    Two-layer filter:
      1. If `config.include_bare_leaves` is False (default), drop tags with no
         `::` separator. Catches most scaffolding artifacts for free.
      2. Hard denylist (always applied regardless of bare-leaf setting):
         agent ids/names from openclaw, the exact terms `spike` and `cloze`,
         dated stamps (`\\d{4}-\\d{2}-\\d{2}`), `anki-skill-testrun-*` markers,
         and any `extra_deny_patterns` from the tagger config.
    """
    deny_exact = set(_SEED_DENY_EXACT) | {n.lower() for n in agent_names}
    deny_patterns: list[re.Pattern[str]] = list(_SEED_DENY_PATTERNS)
    for raw in config.extra_deny_patterns:
        try:
            deny_patterns.append(re.compile(raw))
        except re.error:
            continue  # silently skip invalid user-supplied patterns
    kept: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        normalized = tag.strip().lower()
        if not normalized:
            continue
        if normalized in deny_exact:
            continue
        if any(p.search(normalized) for p in deny_patterns):
            continue
        if not config.include_bare_leaves and "::" not in normalized:
            continue
        kept.append(tag)
    return kept


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
    tagger_config: TaggerConfig | None = None,
    agent_names: Iterable[str] | None = None,
) -> list[list[str]]:
    """Generate tags for many candidates in parallel, preserving input order.

    `existing_tags` is filtered once via `_filter_seed_vocabulary` (controlled by
    `tagger_config`) and then shared as the seed across all `generate_tags`
    calls. Workers do not see each other's tag suggestions — intentional, so
    parallel tag generation on a single batch doesn't drift the vocabulary
    mid-flight.

    `agent_names` is the fleet agent-name denylist. When `None`, this function
    consults `openclaw config get agents.list` if `tagger_config.use_openclaw_agents`
    is True. Pass an explicit iterable (including an empty one) to skip the
    openclaw subprocess — useful in tests.
    """
    candidates_list = list(candidates)
    if not candidates_list:
        return []
    cfg = tagger_config or TaggerConfig()
    if agent_names is None:
        resolved_agent_names: Iterable[str] = (
            _get_openclaw_agent_names() if cfg.use_openclaw_agents else ()
        )
    else:
        resolved_agent_names = agent_names
    filtered_tags = _filter_seed_vocabulary(
        existing_tags, config=cfg, agent_names=resolved_agent_names
    )
    workers = max_workers if max_workers is not None else resolve_concurrency()
    if workers <= 1:
        return [generate_tags(c, filtered_tags, batch_tag=batch_tag, llm=llm) for c in candidates_list]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda c: generate_tags(c, filtered_tags, batch_tag=batch_tag, llm=llm),
                candidates_list,
            )
        )
