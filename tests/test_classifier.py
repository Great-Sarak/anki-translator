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
from anki_translator.classifier import (
    CardCandidate,
    Overflow,
    _default_concurrency,
    build_prompt,
    classify,
    classify_chunks,
    resolve_concurrency,
)
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


def test_build_prompt_uses_canonical_field_names_only(shapes: dict, chunk: Chunk) -> None:
    """Prompt must surface the model's canonical field name (e.g. 'Back'), not the
    lowercase role ('back'), so the LLM doesn't mirror the wrong casing in its JSON."""
    prompt = build_prompt(chunk, shapes)
    # Field listings should not include the `role=Name` form (e.g., 'back=Back').
    assert "front=Front" not in prompt
    assert "back=Back" not in prompt
    assert "text=Text" not in prompt
    # But the canonical names themselves must still appear.
    assert "Front" in prompt
    assert "Back" in prompt
    assert "Text" in prompt


# ---- canonical field-name normalization ----


def test_classify_canonicalizes_lowercase_field_keys(shapes: dict, chunk: Chunk) -> None:
    """LLM occasionally emits lowercase JSON keys ('back') instead of 'Back'.
    The classifier should rewrite to canonical casing so anki-manager's schema
    check passes downstream."""
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"front": "mitochondria", "back": "powerhouse of the cell"},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)
    assert result.fields == {"Front": "mitochondria", "Back": "powerhouse of the cell"}


def test_classify_canonicalizes_mixed_casing(shapes: dict, chunk: Chunk) -> None:
    """Some keys canonical, some not — all should land canonical."""
    stub = _stub(json.dumps({
        "choice": "AT List",
        "fields": {"Front": "organelles", "back": "mitochondria\\nribosomes\\nnucleus"},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)
    assert "Front" in result.fields and "Back" in result.fields
    assert "back" not in result.fields and "front" not in result.fields


def test_classify_canonicalization_enforces_cutoff_on_lowercase_back(shapes: dict, chunk: Chunk) -> None:
    """Cutoff enforcement must run on the canonicalized field — a lowercase 'back'
    that exceeds the budget must still route to overflow, not slip through."""
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"front": "f", "back": "x" * 201},  # 201 chars vs 200 cutoff
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason


# ---- cloze brace normalization ----


def test_classify_normalizes_single_open_brace_on_cloze(shapes: dict, chunk: Chunk) -> None:
    """LLM bug: first cloze deletion comes out as `{c1::...}}` (single open).
    Classifier must repair to `{{c1::...}}` so Anki renders it as a cloze."""
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "Mitochondria is {c1::the powerhouse}} of the cell."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Text"] == "Mitochondria is {{c1::the powerhouse}} of the cell."


def test_classify_normalizes_single_close_brace_on_cloze(shapes: dict, chunk: Chunk) -> None:
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "Mitochondria is {{c1::the powerhouse} of the cell."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Text"] == "Mitochondria is {{c1::the powerhouse}} of the cell."


def test_classify_leaves_well_formed_cloze_alone(shapes: dict, chunk: Chunk) -> None:
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "{{c1::Mitochondria}} produces {{c2::ATP}}."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Text"] == "{{c1::Mitochondria}} produces {{c2::ATP}}."


def test_classify_normalizes_mixed_malformed_and_correct_clozes(shapes: dict, chunk: Chunk) -> None:
    """Observed in the smoke test: first deletion broken, later ones correct."""
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "Evidence of {c1::recombination in mtDNA}}. Enzymes are {{c2::present}}."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Text"] == "Evidence of {{c1::recombination in mtDNA}}. Enzymes are {{c2::present}}."


def test_classify_does_not_touch_braces_on_non_cloze_shapes(shapes: dict, chunk: Chunk) -> None:
    """Normalization is scoped to cloze shapes — a literal `{c1::...}` in an AT Basic
    Back field is not a cloze; pass through untouched."""
    raw = json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "syntax", "Back": "the {c1::pattern}} is literal here"},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Back"] == "the {c1::pattern}} is literal here"


def test_classify_cloze_cutoff_runs_on_normalized_text(shapes: dict, chunk: Chunk) -> None:
    """The deletion-words cutoff must see the normalized braces so it can count
    deletion words — otherwise a broken `{c1::...}}` slips past the budget check."""
    # 6 words > 5-word limit. Bracket is malformed; cutoff should still catch it.
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "Cells need {c1::a lot of energy from ATP}} daily."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason


def test_classify_unknown_field_name_passes_through_for_visibility(shapes: dict, chunk: Chunk) -> None:
    """An LLM that hallucinates a field name we don't know about should keep that
    key as-is — better the user sees the typo in queue review than us silently
    dropping content."""
    stub = _stub(json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "f", "Back": "b", "Notes": "extra"},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, CardCandidate)
    assert result.fields.get("Notes") == "extra"


# ---- batch / parallel ----


def test_classify_chunks_preserves_order_with_concurrency(shapes: dict) -> None:
    """Order of returned classifications must match input order, even when parallel."""
    chunks = [
        Chunk(text=f"chunk {i}", source="s", position="", source_type="manual", metadata={})
        for i in range(20)
    ]
    # Stub returns a Front field encoding the chunk index, so we can detect reordering.
    def stub(prompt: str) -> str:
        # Pull the chunk number out of the rendered prompt.
        for line in prompt.splitlines():
            if line.startswith("chunk "):
                idx = line.split()[1]
                return json.dumps({"choice": "AT Basic", "fields": {"Front": f"front-{idx}", "Back": "b"}})
        raise AssertionError("test stub didn't find chunk marker in prompt")

    results = classify_chunks(chunks, shapes, llm=stub, max_workers=8)
    assert len(results) == 20
    for i, result in enumerate(results):
        assert isinstance(result, CardCandidate), f"chunk {i} should classify"
        assert result.fields["Front"] == f"front-{i}", f"chunk {i} got out-of-order result"


def test_classify_chunks_empty_input(shapes: dict) -> None:
    assert classify_chunks([], shapes, llm=_stub("")) == []


def test_classify_chunks_max_workers_one_matches_sequential(shapes: dict, chunk: Chunk) -> None:
    stub = _stub(json.dumps({"choice": "AT Basic", "fields": {"Front": "f", "Back": "b"}}))
    sequential = [classify(chunk, shapes, llm=stub) for _ in range(3)]
    batched = classify_chunks([chunk] * 3, shapes, llm=stub, max_workers=1)
    assert sequential == batched


# ---- concurrency heuristic ----


def test_default_concurrency_uses_cpu_minus_one_over_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 20)
    assert _default_concurrency() == 9  # (20-1)//2
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 8)
    assert _default_concurrency() == 3  # (8-1)//2
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 2)
    assert _default_concurrency() == 1  # floor at 1
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 1)
    assert _default_concurrency() == 1  # single-core box still works


def test_default_concurrency_handles_unknown_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some sandboxes (containers without cpu_set) return None — fall back."""
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: None)
    assert _default_concurrency() == 4


def test_resolve_concurrency_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANKI_TRANSLATOR_CONCURRENCY", "3")
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 64)  # would yield 31
    assert resolve_concurrency() == 3


def test_resolve_concurrency_ignores_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANKI_TRANSLATOR_CONCURRENCY", "not-a-number")
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 20)
    assert resolve_concurrency() == 9  # falls through to CPU heuristic


def test_resolve_concurrency_ignores_zero_or_negative_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANKI_TRANSLATOR_CONCURRENCY", "0")
    monkeypatch.setattr("anki_translator.classifier.os.cpu_count", lambda: 20)
    assert resolve_concurrency() == 9
    monkeypatch.setenv("ANKI_TRANSLATOR_CONCURRENCY", "-1")
    assert resolve_concurrency() == 9
