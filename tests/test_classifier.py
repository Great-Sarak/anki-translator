"""Tests for the LLM-driven shape classifier.

Uses a stub LLM (a callable returning canned JSON) to avoid network calls and to exercise
each failure mode deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from anki_translator.chunk import Chunk
from anki_translator.classifier import CardCandidate, Overflow, build_prompt, classify
from anki_translator.config import load_shapes

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def shapes() -> dict:
    return load_shapes(REPO_ROOT / "config" / "shapes.yaml")


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        text="The mitochondria is the powerhouse of the cell.",
        source="https://example.com/cells",
        position="#organelles",
        source_type="url",
        metadata={"url": "https://example.com/cells", "anchor": "organelles"},
    )


def _stub(response: str) -> Callable[[str], str]:
    """Make an LLM stub that returns the given string verbatim."""
    return lambda _prompt: response


# ---- happy paths ----


def test_classify_basic_term_def(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "mitochondria", "Back": "powerhouse of the cell"},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)
    assert result.note_type == "AT Basic"
    assert result.shape == "term-def"
    assert result.fields == {"Front": "mitochondria", "Back": "powerhouse of the cell"}
    assert result.chunk is chunk


def test_classify_cloze(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "The {{c1::mitochondria}} is the powerhouse of the cell."},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)
    assert result.note_type == "AT Cloze"
    assert "{{c1::mitochondria}}" in result.fields["Text"]


# ---- overflow paths ----


def test_classify_overflow_decided_by_llm(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({"choice": "overflow", "reason": "too complex for one card"}))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "complex" in result.reason


def test_classify_overflow_when_no_shapes_available(chunk: Chunk) -> None:
    result = classify(chunk, {}, llm=_stub(""))
    assert isinstance(result, Overflow)
    assert result.reason == "no_shape_fit"


# ---- cutoff enforcement (defensive against LLM hallucinated compliance) ----


def test_classify_routes_to_overflow_when_back_exceeds_chars(shapes: dict, chunk: Chunk) -> None:
    """LLM picks AT Basic but produces a 250-char Back when limit is 200 → overflow."""
    long_back = "x" * 250
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "mito", "Back": long_back},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason
    assert "Back" in result.reason


def test_classify_routes_to_overflow_when_list_too_many_items(shapes: dict, chunk: Chunk) -> None:
    """AT List limit is 7 items; produce 10 → overflow."""
    long_list = "\n".join(f"- item {i}" for i in range(10))
    stub = _stub(json.dumps({
        "choice": "AT List",
        "fields": {"Front": "organelles", "Back": long_list},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason


def test_classify_routes_to_overflow_when_steps_too_many(shapes: dict, chunk: Chunk) -> None:
    """AT Steps limit is 5; produce 7 → overflow."""
    long_steps = "\n".join(f"{i}. step" for i in range(7))
    stub = _stub(json.dumps({
        "choice": "AT Steps",
        "fields": {"Front": "process", "Back": long_steps},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason


def test_classify_routes_to_overflow_when_cloze_deletion_too_long(shapes: dict, chunk: Chunk) -> None:
    """AT Cloze deletion_max_words=5; produce a 6-word deletion → overflow."""
    stub = _stub(json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "The {{c1::powerhouse of the cell making ATP}} is mitochondria."},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason


def test_classify_accepts_borderline_at_budget(shapes: dict, chunk: Chunk) -> None:
    """Exactly at the limit should NOT route to overflow."""
    exactly_200 = "x" * 200
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "mito", "Back": exactly_200},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)


# ---- invalid LLM responses ----


def test_classify_handles_non_json_response(shapes: dict, chunk: Chunk) -> None:
    stub = _stub("Sorry, I cannot do that.")
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "invalid_response" in result.reason
    assert "not JSON" in result.reason


def test_classify_handles_missing_choice(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({"fields": {"Front": "x"}}))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "missing 'choice'" in result.reason


def test_classify_handles_unknown_note_type(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({"choice": "Phantom Type", "fields": {}}))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "unknown note type" in result.reason


def test_classify_handles_missing_fields(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({"choice": "AT Basic"}))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "missing or non-dict 'fields'" in result.reason


def test_classify_catches_llm_exception(shapes: dict, chunk: Chunk) -> None:
    def raising_llm(_prompt: str) -> str:
        raise RuntimeError("API timeout")
    result = classify(chunk, shapes, llm=raising_llm)
    assert isinstance(result, Overflow)
    assert "llm_error" in result.reason
    assert "RuntimeError" in result.reason


# ---- prompt construction ----


def test_build_prompt_includes_chunk_text(shapes: dict, chunk: Chunk) -> None:
    prompt = build_prompt(chunk, shapes)
    assert chunk.text in prompt


def test_build_prompt_lists_all_shapes(shapes: dict, chunk: Chunk) -> None:
    prompt = build_prompt(chunk, shapes)
    for name in shapes:
        assert name in prompt


def test_build_prompt_includes_cutoffs(shapes: dict, chunk: Chunk) -> None:
    prompt = build_prompt(chunk, shapes)
    assert "back_max_chars=200" in prompt
    assert "deletion_max_words=5" in prompt
