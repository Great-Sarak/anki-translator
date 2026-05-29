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
from anki_translator.config import TaggerConfig
from anki_translator.tagger import (
    _filter_seed_vocabulary,
    _get_openclaw_agent_names,
    build_prompt,
    generate_tags,
    tag_candidates,
)


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


# ---- seed-vocabulary filter (#38) ----


def test_filter_drops_bare_leaves_by_default() -> None:
    """Default config: bare-leaf tags (no '::') are dropped from the seed."""
    cfg = TaggerConfig()
    kept = _filter_seed_vocabulary(
        ["biology", "biology::organelles", "cell-biology", "physics::quantum"],
        config=cfg,
    )
    assert kept == ["biology::organelles", "physics::quantum"]


def test_filter_keeps_bare_leaves_when_opted_in() -> None:
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(
        ["biology", "biology::organelles", "cell-biology"],
        config=cfg,
    )
    assert kept == ["biology", "biology::organelles", "cell-biology"]


def test_filter_drops_builtin_exact_denylist_even_with_bare_leaves() -> None:
    """`spike` and `cloze` are always dropped, even with include_bare_leaves=True."""
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(
        ["spike", "cloze", "biology", "Spike", "CLOZE"],
        config=cfg,
    )
    assert kept == ["biology"]  # cases collapsed via .lower()


def test_filter_drops_dated_batch_tags() -> None:
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(
        ["e2e-smoke-2026-05-28", "book-club-2026-01-15", "biology"],
        config=cfg,
    )
    assert kept == ["biology"]


def test_filter_drops_anki_skill_testrun_markers() -> None:
    """The unified test-marker prefix from the rename in sibling repos."""
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(
        ["anki-skill-testrun-rpc", "anki-skill-testrun-manager",
         "anki-skill-testrun-translator", "biology"],
        config=cfg,
    )
    assert kept == ["biology"]


def test_filter_drops_agent_names() -> None:
    """Agent ids/names from the openclaw roster (passed in explicitly here)."""
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(
        ["myrzka", "fheliza", "rukha", "Myrzka", "biology"],
        config=cfg,
        agent_names={"myrzka", "fheliza", "rukha", "tava"},
    )
    assert kept == ["biology"]


def test_filter_applies_extra_deny_patterns_from_config() -> None:
    cfg = TaggerConfig(include_bare_leaves=True, extra_deny_patterns=[r"^prov-", r"-draft$"])
    kept = _filter_seed_vocabulary(
        ["prov-something", "topic-draft", "biology::organelles"],
        config=cfg,
    )
    assert kept == ["biology::organelles"]


def test_filter_ignores_invalid_user_regex_patterns() -> None:
    """Bad regex in config should silently skip, not crash the pipeline."""
    cfg = TaggerConfig(include_bare_leaves=True, extra_deny_patterns=["[unclosed", "biology"])
    kept = _filter_seed_vocabulary(["biology", "physics"], config=cfg)
    # "biology" pattern still drops biology; "[unclosed" is silently skipped.
    assert "biology" not in kept
    assert "physics" in kept


def test_filter_handles_non_string_and_empty_tags() -> None:
    cfg = TaggerConfig(include_bare_leaves=True)
    kept = _filter_seed_vocabulary(["biology", "", None, 42, "  ", "physics"], config=cfg)
    assert kept == ["biology", "physics"]


def test_filter_real_smoke_run_vocab_keeps_only_topical(candidate: CardCandidate) -> None:
    """End-to-end pin: the exact 8 bare-leaves from the v0.1 smoke run get
    filtered, leaving only the hierarchical biology tags."""
    cfg = TaggerConfig()  # default: drop bare leaves
    smoke_vocab = [
        "cloze", "spike", "myrzka", "cli-smoke",
        "anki-rpc-test", "anki-manager-test",
        "e2e-smoke-2026-05-28", "cell-biology",
        "biology::organelles", "biology::cellular-respiration",
        "biochemistry::metabolism",
    ]
    kept = _filter_seed_vocabulary(
        smoke_vocab, config=cfg, agent_names={"myrzka", "fheliza"}
    )
    assert kept == [
        "biology::organelles",
        "biology::cellular-respiration",
        "biochemistry::metabolism",
    ]


def test_tag_candidates_filters_seed_before_calling_llm(candidate: CardCandidate) -> None:
    """The LLM's prompt must contain only the post-filter vocabulary."""
    captured: list[str] = []
    def capturing_llm(prompt: str) -> str:
        captured.append(prompt)
        return json.dumps({"tags": ["biology"]})

    tag_candidates(
        [candidate],
        existing_tags=["myrzka", "spike", "biology::organelles", "cell-biology"],
        llm=capturing_llm,
        tagger_config=TaggerConfig(),
        agent_names={"myrzka"},
    )
    prompt = captured[0]
    assert "biology::organelles" in prompt
    for scrap in ("myrzka", "spike", "cell-biology"):
        # The seed bullets are rendered as "- <tag>". Make sure no such bullet exists.
        assert f"- {scrap}\n" not in prompt and not prompt.rstrip().endswith(f"- {scrap}")


def test_tag_candidates_skips_openclaw_when_agent_names_provided(candidate: CardCandidate) -> None:
    """Passing agent_names=<iterable> should not invoke the openclaw subprocess.
    Verified by running with a patched _get_openclaw_agent_names that would raise."""
    import anki_translator.tagger as tagger_mod
    original = tagger_mod._get_openclaw_agent_names
    tagger_mod._get_openclaw_agent_names = lambda: (_ for _ in ()).throw(AssertionError("should not be called"))
    try:
        results = tag_candidates(
            [candidate],
            existing_tags=["biology::organelles"],
            llm=_stub(json.dumps({"tags": ["biology"]})),
            agent_names=(),  # explicit empty — skips lookup
        )
        assert results == [["biology"]]
    finally:
        tagger_mod._get_openclaw_agent_names = original


def test_tag_candidates_skips_openclaw_when_disabled_in_config(candidate: CardCandidate) -> None:
    import anki_translator.tagger as tagger_mod
    original = tagger_mod._get_openclaw_agent_names
    tagger_mod._get_openclaw_agent_names = lambda: (_ for _ in ()).throw(AssertionError("should not be called"))
    try:
        results = tag_candidates(
            [candidate],
            existing_tags=["biology::organelles"],
            llm=_stub(json.dumps({"tags": ["biology"]})),
            tagger_config=TaggerConfig(use_openclaw_agents=False),
        )
        assert results == [["biology"]]
    finally:
        tagger_mod._get_openclaw_agent_names = original


def test_get_openclaw_agent_names_parses_canonical_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the parser against the real `openclaw config get agents.list` shape."""
    import subprocess
    fake_output = json.dumps([
        {"id": "myrzka", "name": "Myrzka", "workspace": "/path"},
        {"id": "fheliza", "name": "Fheliza"},
        {"id": "no-name-only-id"},
    ])
    class FakeResult:
        stdout = fake_output
    def fake_run(*args, **kwargs):
        return FakeResult()
    monkeypatch.setattr(subprocess, "run", fake_run)
    names = _get_openclaw_agent_names()
    assert names == {"myrzka", "fheliza", "no-name-only-id"}


def test_get_openclaw_agent_names_returns_empty_on_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """openclaw missing or erroring shouldn't crash the tagger."""
    import subprocess
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("openclaw not installed")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _get_openclaw_agent_names() == set()


def test_get_openclaw_agent_names_returns_empty_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    class FakeResult:
        stdout = "not json"
    def fake_run(*args, **kwargs):
        return FakeResult()
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _get_openclaw_agent_names() == set()
