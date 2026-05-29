"""Tests for the LLM topic tagger.

Stub-LLM only; no network. Exercises the normalization, depth cap, dedup, and
fallback-to-batch-tag-only-on-LLM-failure behavior.
"""

from __future__ import annotations

import json
from typing import Callable

import pytest

from anki_translator.chunk import Chunk
from anki_translator.classifier import CardCandidate
from anki_translator.tagger import build_prompt, generate_tags, tag_candidates


@pytest.fixture
def candidate() -> CardCandidate:
    chunk = Chunk(
        text="The mitochondria is the powerhouse of the cell.",
        source="https://example.com/cells",
        position="#organelles",
        source_type="url",
        metadata={"url": "https://example.com/cells"},
    )
    return CardCandidate(
        note_type="AT Basic",
        shape="term-def",
        fields={"Front": "mitochondria", "Back": "powerhouse of the cell"},
        chunk=chunk,
    )


def _stub(response: str) -> Callable[[str], str]:
    return lambda _prompt: response


def test_generate_tags_happy_path(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["biology", "biology::organelles"]}))
    tags = generate_tags(candidate, existing_tags=["biology", "biology::organelles"], llm=stub)
    assert tags == ["biology", "biology::organelles"]


def test_generate_tags_caps_depth_at_two(candidate: CardCandidate) -> None:
    """An LLM that returns 3+ level tag must be capped to 2."""
    stub = _stub(json.dumps({"tags": ["cell-biology::organelles::mitochondria"]}))
    tags = generate_tags(candidate, existing_tags=[], llm=stub)
    assert tags == ["cell-biology::organelles"]


def test_generate_tags_normalizes_spaces_and_case(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["Cell Biology", "Distributed Systems::Consensus"]}))
    tags = generate_tags(candidate, existing_tags=[], llm=stub)
    assert tags == ["cell-biology", "distributed-systems::consensus"]


def test_generate_tags_dedupes(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["biology", "Biology", "BIOLOGY"]}))
    tags = generate_tags(candidate, existing_tags=[], llm=stub)
    assert tags == ["biology"]


def test_generate_tags_caps_at_three(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["a", "b", "c", "d", "e"]}))
    tags = generate_tags(candidate, existing_tags=[], llm=stub)
    assert len(tags) == 3
    assert tags == ["a", "b", "c"]


def test_generate_tags_appends_batch_tag(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["biology"]}))
    tags = generate_tags(candidate, existing_tags=[], batch_tag="book-club-2026", llm=stub)
    assert "biology" in tags
    assert "book-club-2026" in tags
    assert tags[-1] == "book-club-2026"


def test_generate_tags_batch_tag_not_duplicated(candidate: CardCandidate) -> None:
    """If LLM happens to include the batch tag, don't duplicate it."""
    stub = _stub(json.dumps({"tags": ["book-club-2026", "biology"]}))
    tags = generate_tags(candidate, existing_tags=[], batch_tag="book-club-2026", llm=stub)
    assert tags.count("book-club-2026") == 1


def test_generate_tags_llm_failure_falls_back_to_batch_only(candidate: CardCandidate) -> None:
    def raising_llm(_prompt: str) -> str:
        raise RuntimeError("API down")
    tags = generate_tags(candidate, existing_tags=[], batch_tag="book-club-2026", llm=raising_llm)
    assert tags == ["book-club-2026"]


def test_generate_tags_llm_failure_no_batch_returns_empty(candidate: CardCandidate) -> None:
    def raising_llm(_prompt: str) -> str:
        raise RuntimeError("API down")
    tags = generate_tags(candidate, existing_tags=[], llm=raising_llm)
    assert tags == []


def test_generate_tags_invalid_json_falls_back(candidate: CardCandidate) -> None:
    stub = _stub("Sorry I cannot")
    tags = generate_tags(candidate, existing_tags=[], batch_tag="batch", llm=stub)
    assert tags == ["batch"]


def test_generate_tags_missing_tags_key_falls_back(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"other": "data"}))
    tags = generate_tags(candidate, existing_tags=[], batch_tag="batch", llm=stub)
    assert tags == ["batch"]


def test_build_prompt_seeds_existing_tags(candidate: CardCandidate) -> None:
    prompt = build_prompt(candidate, existing_tags=["biology", "physics", "biology::organelles"])
    assert "biology" in prompt
    assert "biology::organelles" in prompt
    assert "physics" in prompt


def test_build_prompt_handles_empty_existing_tags(candidate: CardCandidate) -> None:
    prompt = build_prompt(candidate, existing_tags=[])
    assert "fresh deck" in prompt or "no existing tags" in prompt


def test_build_prompt_includes_card_fields(candidate: CardCandidate) -> None:
    prompt = build_prompt(candidate, existing_tags=[])
    assert "mitochondria" in prompt
    assert "powerhouse" in prompt


def test_build_prompt_excludes_source_position(candidate: CardCandidate) -> None:
    """Source and Position aren't informative for topic — should not appear in the prompt's fields block."""
    candidate_with_source = CardCandidate(
        note_type="AT Basic",
        shape="term-def",
        fields={
            "Front": "mito",
            "Back": "atp factory",
            "Source": "should-not-appear-as-tag-hint",
            "Position": "page 99",
        },
        chunk=candidate.chunk,
    )
    prompt = build_prompt(candidate_with_source, existing_tags=[])
    assert "should-not-appear-as-tag-hint" not in prompt


# ---- batch / parallel ----


def test_tag_candidates_preserves_order_with_concurrency(candidate: CardCandidate) -> None:
    """Each candidate gets its own tag list, returned in input order."""
    candidates = []
    for i in range(20):
        c = CardCandidate(
            note_type="AT Basic",
            shape="term-def",
            fields={"Front": f"front-{i}", "Back": "b"},
            chunk=candidate.chunk,
        )
        candidates.append(c)

    def stub(prompt: str) -> str:
        # Reflect the per-candidate Front value back as the tag, so we can detect reordering.
        for line in prompt.splitlines():
            if line.startswith("Front:"):
                front = line.split(":", 1)[1].strip()
                return json.dumps({"tags": [f"echo::{front}"]})
        raise AssertionError("stub didn't find Front: in prompt")

    results = tag_candidates(candidates, existing_tags=[], llm=stub, max_workers=8)
    assert len(results) == 20
    for i, tags in enumerate(results):
        assert tags == [f"echo::front-{i}"], f"candidate {i} got out-of-order tags: {tags}"


def test_tag_candidates_empty_input() -> None:
    assert tag_candidates([], existing_tags=[], llm=_stub("")) == []


def test_tag_candidates_appends_batch_tag_to_each(candidate: CardCandidate) -> None:
    stub = _stub(json.dumps({"tags": ["biology"]}))
    results = tag_candidates([candidate, candidate], existing_tags=[], batch_tag="batch-x", llm=stub, max_workers=2)
    assert results == [["biology", "batch-x"], ["biology", "batch-x"]]
