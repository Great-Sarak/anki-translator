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


def test_classify_routes_to_overflow_when_inline_list_too_many_items(shapes: dict, chunk: Chunk) -> None:
    """AT List limit is 7; an LLM that returns an 8-item list inline on one line
    (no newlines) must still route to overflow. Pre-#39 fix this slipped through."""
    inline_list = "1. one 2. two 3. three 4. four 5. five 6. six 7. seven 8. eight"
    stub = _stub(json.dumps({
        "choice": "AT List",
        "fields": {"Front": "organelles", "Back": inline_list},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason
    assert "8 items" in result.reason


def test_classify_routes_to_overflow_when_inline_bulleted_list_too_many_items(shapes: dict, chunk: Chunk) -> None:
    """Same as above but with bullet markers (•/-/*) inline."""
    inline_list = "• one • two • three • four • five • six • seven • eight"
    stub = _stub(json.dumps({
        "choice": "AT List",
        "fields": {"Front": "things", "Back": inline_list},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason
    assert "8 items" in result.reason


def test_classify_routes_to_overflow_when_inline_steps_too_many(shapes: dict, chunk: Chunk) -> None:
    """AT Steps limit is 5; inline 6-step list must route to overflow."""
    inline_steps = "1. boot 2. extract 3. classify 4. tag 5. queue 6. commit"
    stub = _stub(json.dumps({
        "choice": "AT Steps",
        "fields": {"Front": "process", "Back": inline_steps},
    }))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "exceeds_budget" in result.reason
    assert "6 steps" in result.reason


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


def test_classify_handles_array_response(shapes: dict, chunk: Chunk) -> None:
    """Pin for #48: haiku-4-5 sometimes returns a JSON array of multiple cards
    for chunks holding multiple distinct facts (multi-table markdown sections,
    multi-bullet structures). The classifier must surface this as a specific
    overflow reason instead of the misleading "missing 'choice'" message."""
    stub = _stub(json.dumps([
        {"choice": "AT Basic", "fields": {"Front": "a", "Back": "b"}},
        {"choice": "AT Basic", "fields": {"Front": "c", "Back": "d"}},
    ]))
    result = classify(chunk, shapes, llm=stub)
    assert isinstance(result, Overflow)
    assert "array" in result.reason


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


def test_prompt_budget_matches_cutoff_enforcement(shapes: dict, chunk: Chunk) -> None:
    """The budget rules the LLM sees in the prompt must match what _check_cutoffs
    actually enforces. Symmetric closure of the PR #37 cloze-brace gap where the
    prompt and the downstream code diverged silently."""
    prompt = build_prompt(chunk, shapes)
    for name, cfg in shapes.items():
        for cutoff_name, limit in cfg.cutoffs.items():
            # Every cutoff the classifier checks must be visible in the prompt
            # so the LLM is asked to satisfy it. If a new cutoff is added in
            # config but the prompt block doesn't surface it, this test fires.
            assert f"{cutoff_name}={limit}" in prompt, (
                f"shape {name!r} cutoff {cutoff_name}={limit} not surfaced in prompt"
            )


def test_build_prompt_shows_canonical_cloze_braces_to_llm(shapes: dict, chunk: Chunk) -> None:
    """The rendered prompt must show `{{c1::word}}` (double braces) to the LLM —
    Python's .format() collapses `{{` → `{`, so the template needs four braces
    per side. If we regress to two, the LLM copies single-brace clozes."""
    prompt = build_prompt(chunk, shapes)
    assert "{{c1::word}}" in prompt
    assert "{c1::word}" not in prompt.replace("{{c1::word}}", "")


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


def test_classify_normalizes_single_brace_on_both_ends(shapes: dict, chunk: Chunk) -> None:
    """Observed after the prompt-escape fix: LLM emits `{c1::...}` with single
    braces on BOTH ends. Normalizer must repair to `{{c1::...}}`."""
    raw = json.dumps({
        "choice": "AT Cloze",
        "fields": {"Text": "Mitochondria divide by {c1::budding}."},
    })
    result = classify(chunk, shapes, llm=_stub(raw))
    assert isinstance(result, CardCandidate)
    assert result.fields["Text"] == "Mitochondria divide by {{c1::budding}}."


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


# ---- term-table multi-card classification (#52) ----


# Import here to avoid touching the import block at the top during the PR.
from anki_translator.classifier import MultiCardCandidate  # noqa: E402


DISPLAYPORT_TABLE_TEXT = (
    "#### Version / bandwidth\n"
    "\n"
    "| Version | Link rate | Total bandwidth | Top resolution (with DSC) |\n"
    "|---|---|---|---|\n"
    "| DP 1.2 | HBR2 | 21.6 Gbps | 4K@60 |\n"
    "| DP 1.4 | HBR3 | 25.92 Gbps | 8K@60 |\n"
    "| DP 2.0 | UHBR 20 | 80 Gbps | 16K@60 |\n"
)


def _table_chunk() -> Chunk:
    return Chunk(
        text=DISPLAYPORT_TABLE_TEXT,
        source="2026-05-30-cable-identification-and-testing",
        position="#2-2-displayport",
        source_type="manual",
        metadata={},
    )


def test_classify_term_table_expands_rows_to_multi_card_candidate(shapes: dict) -> None:
    """Pinning test for #52: an AT Table response with N rows × M attrs yields
    a MultiCardCandidate wrapping N CardCandidates (one per row), each with its
    Key + AttrN/AttrNValue slots populated. Anki's conditional templates emit
    one card per non-empty AttrN slot at note creation; the classifier's job is
    just to produce the row-level CardCandidates."""
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": "DP 1.2", "attrs": [
                {"name": "Link rate", "value": "HBR2"},
                {"name": "Total bandwidth", "value": "21.6 Gbps"},
                {"name": "Top resolution (with DSC)", "value": "4K@60"},
            ]},
            {"key": "DP 1.4", "attrs": [
                {"name": "Link rate", "value": "HBR3"},
                {"name": "Total bandwidth", "value": "25.92 Gbps"},
                {"name": "Top resolution (with DSC)", "value": "8K@60"},
            ]},
            {"key": "DP 2.0", "attrs": [
                {"name": "Link rate", "value": "UHBR 20"},
                {"name": "Total bandwidth", "value": "80 Gbps"},
                {"name": "Top resolution (with DSC)", "value": "16K@60"},
            ]},
        ],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, MultiCardCandidate), result
    assert len(result.rows) == 3
    first = result.rows[0]
    assert first.note_type == "AT Table"
    assert first.shape == "term-table"
    assert first.fields["Key"] == "DP 1.2"
    assert first.fields["Attr1Name"] == "Link rate"
    assert first.fields["Attr1Value"] == "HBR2"
    assert first.fields["Attr2Name"] == "Total bandwidth"
    assert first.fields["Attr3Value"] == "4K@60"
    # Unused slots are present as empty strings — anki-manager rejects notes with
    # missing model fields, but Anki's conditional templates ({{#Attr4Name}}...{{/}})
    # suppress empty cards at render time, so this is the correct shape.
    assert first.fields["Attr4Name"] == ""
    assert first.fields["Attr4Value"] == ""


def test_classify_term_table_backfills_all_unused_attr_slots_with_empty_strings(shapes: dict) -> None:
    """Regression for the InvalidNoteError when a row has fewer than attrs_per_row_max
    attributes: every attr slot defined in the shape must be present in fields, with
    empty string for unused slots. Otherwise anki-manager rejects the note with
    'Model 'AT Table' requires fields not provided: [...]'."""
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": "Single-attr row", "attrs": [
                {"name": "Color", "value": "red"},
            ]},
            {"key": "Two-attr row", "attrs": [
                {"name": "Color", "value": "blue"},
                {"name": "Size", "value": "large"},
            ]},
        ],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, MultiCardCandidate), result
    # Every row, every attr slot present (populated or "")
    for row in result.rows:
        for i in range(1, 5):
            assert f"Attr{i}Name" in row.fields, f"missing Attr{i}Name in {row.fields}"
            assert f"Attr{i}Value" in row.fields, f"missing Attr{i}Value in {row.fields}"
    # Populated slots have content; unused slots are empty strings
    one_attr = result.rows[0]
    assert one_attr.fields["Attr1Name"] == "Color"
    assert one_attr.fields["Attr2Name"] == ""
    assert one_attr.fields["Attr3Name"] == ""
    assert one_attr.fields["Attr4Name"] == ""


def test_classify_term_table_overflows_when_row_exceeds_attrs_per_row_max(shapes: dict) -> None:
    """Five attrs in one row exceeds the attrs_per_row_max=4 cutoff → overflow.
    The cutoff is enforced post-LLM, the same way every other shape's cutoff is,
    because the LLM may self-report compliance without actually complying.

    The reason string says 'per-row limit' so it's clear the cap is per-row, not
    a cap on the total row count (caps on row count don't exist — a term-table
    response with 20 rows of 2 attrs each is fine)."""
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": "DP 1.2", "attrs": [
                {"name": "A", "value": "1"},
                {"name": "B", "value": "2"},
                {"name": "C", "value": "3"},
                {"name": "D", "value": "4"},
                {"name": "E", "value": "5"},  # one too many
            ]},
        ],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, Overflow), result
    assert "exceeds_budget" in result.reason
    assert "5 attrs" in result.reason
    assert "per-row" in result.reason


def test_classify_term_table_accepts_many_rows_with_few_attrs(shapes: dict) -> None:
    """attrs_per_row_max is per-row, not a cap on rows. A 20-row table with 2
    attrs per row is well within budget and produces 20 CardCandidates. Pins
    the per-row semantics against the LLM (or a future reader of the prompt)
    misreading it as a row-count cap."""
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": f"R{i}", "attrs": [
                {"name": "Pick", "value": f"P{i}"},
                {"name": "Cost", "value": f"$ {i*10}"},
            ]}
            for i in range(1, 21)  # 20 rows
        ],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, MultiCardCandidate), result
    assert len(result.rows) == 20


def test_classify_term_table_overflows_when_attr_value_too_long(shapes: dict) -> None:
    """An attribute value longer than attr_max_chars=80 → overflow, not silent
    truncation."""
    long_value = "x" * 81
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": "DP 1.2", "attrs": [
                {"name": "Link rate", "value": long_value},
            ]},
        ],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, Overflow), result
    assert "exceeds_budget" in result.reason


def test_classify_term_table_overflows_on_missing_rows(shapes: dict) -> None:
    stub = _stub(json.dumps({"choice": "AT Table", "fields": {"Key": "DP 1.2"}}))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, Overflow), result
    assert "invalid_response" in result.reason
    assert "rows" in result.reason


def test_classify_term_table_overflows_on_empty_rows(shapes: dict) -> None:
    stub = _stub(json.dumps({"choice": "AT Table", "rows": []}))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, Overflow), result
    assert "invalid_response" in result.reason


def test_classify_term_table_overflows_on_missing_key(shapes: dict) -> None:
    stub = _stub(json.dumps({
        "choice": "AT Table",
        "rows": [{"attrs": [{"name": "Link rate", "value": "HBR2"}]}],
    }))
    result = classify(_table_chunk(), shapes, llm=stub)
    assert isinstance(result, Overflow), result
    assert "missing 'key'" in result.reason


def test_build_prompt_includes_term_table_response_shape(shapes: dict, chunk: Chunk) -> None:
    """The prompt must teach the model the term-table response shape (`rows`+`attrs`)
    so the LLM doesn't try to flatten attribute columns into `fields`."""
    prompt = build_prompt(chunk, shapes)
    assert "term-table" in prompt
    # Response shape advertised:
    assert "rows" in prompt
    assert "attrs" in prompt
    # AT Table shape gets the conceptual rendering, not the flat field list:
    assert "attribute pairs" in prompt
    assert "Attr1Name" not in prompt  # confusing flat names suppressed for term-table


def test_build_prompt_clarifies_attrs_per_row_max_is_per_row(shapes: dict, chunk: Chunk) -> None:
    """First live ingest after #52 landed showed the LLM reading the old name
    `row_max_attrs=4` as a cap on row count (overflowing 7-row tables with the
    reason 'exceeds row_max_attrs=4'). The renamed cutoff + per-row explainer
    in the prompt must make it unambiguous that the cap is per-row, not
    per-response. Pin both pieces so a future prompt rewrite doesn't silently
    regress them."""
    prompt = build_prompt(chunk, shapes)
    assert "attrs_per_row_max" in prompt
    # The conceptual rendering of AT Table calls out the per-row meaning
    # explicitly so the cutoff name + line agree.
    assert "PER ROW" in prompt
    assert "row count is unbounded" in prompt
    # And the standalone rule reinforces it for the LLM that skims cutoffs.
    assert "PER-ROW cap" in prompt


def test_build_prompt_prefers_term_table_for_tables_in_multi_fact_passages(
    shapes: dict, chunk: Chunk
) -> None:
    """The merged-in #48 rule pushed the model toward 'pick one central fact +
    overflow' for multi-fact passages, which caused the SAS-family table (a
    clean term-table fit inside a chunk with other prose) to overflow as
    'multiple distinct facts'. The rule must now carve out: if any of those
    facts is a clean term-table fit, prefer term-table for that table."""
    prompt = build_prompt(chunk, shapes)
    # The old "pick one central fact + overflow" rule still exists but is now
    # gated on "AND none of them is a reference table that fits term-table".
    lowered = prompt.lower()
    assert "term-table fit" in lowered
    assert "prefer term-table" in lowered


def test_classify_chunks_flattens_multi_card_into_results(shapes: dict) -> None:
    """classify_chunks() returns a flat list in input order, with MultiCardCandidate
    preserved as a single entry (caller flattens). Today the cli does that flatten
    in its loop — this test pins the contract so a future refactor doesn't
    silently start spreading the rows here."""
    table_response = json.dumps({
        "choice": "AT Table",
        "rows": [
            {"key": "DP 1.2", "attrs": [{"name": "Link rate", "value": "HBR2"}]},
            {"key": "DP 1.4", "attrs": [{"name": "Link rate", "value": "HBR3"}]},
        ],
    })
    basic_response = json.dumps({
        "choice": "AT Basic",
        "fields": {"Front": "mito", "Back": "powerhouse"},
    })
    responses = iter([table_response, basic_response])
    llm = lambda _prompt: next(responses)

    chunks = [_table_chunk(), Chunk(text="x", source="s", position="", source_type="manual", metadata={})]
    results = classify_chunks(chunks, shapes, llm=llm, max_workers=1)
    assert len(results) == 2
    assert isinstance(results[0], MultiCardCandidate)
    assert len(results[0].rows) == 2
    assert isinstance(results[1], CardCandidate)
