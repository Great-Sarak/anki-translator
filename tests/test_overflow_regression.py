"""S0 baseline regression test (#64).

Asserts the committed golden overflow-routing snapshot still holds, with zero
network (the harness replays captured LLM responses). This is the objective
baseline S1-S6 build on: a later layer runs ``harness.build_snapshot()`` and
proves the split held-or-improved and ``prompt_token_chars`` did not rise.
"""

from __future__ import annotations

import pytest

from tests.regression import harness


def test_snapshot_matches_golden() -> None:
    current = harness.build_snapshot()
    golden = harness.load_golden()
    assert current == golden, "\n" + harness.diff_message(current, golden)


def test_snapshot_is_deterministic() -> None:
    assert harness.build_snapshot() == harness.build_snapshot()


def test_both_overflow_buckets_are_exercised() -> None:
    """The fixtures are only a meaningful baseline if they route into both buckets."""
    snapshot = harness.build_snapshot()
    totals = {"card": 0, "qa": 0, "trimmed": 0}
    for corpus in snapshot["corpora"]:
        for bucket, n in corpus["counts"].items():
            totals[bucket] += n
    assert totals["qa"] > 0, "no chunk routed to qa — baseline cannot detect qa regressions"
    assert totals["trimmed"] > 0, "no chunk routed to trimmed — baseline cannot detect trimmed regressions"
    assert totals["card"] > 0, "no chunk produced a card candidate"


def test_token_metric_counts_only_dispatched_chunks() -> None:
    """prompt_token_chars must reflect chunks actually sent to classify().

    Guards the contract S3 relies on: a pre-filtered chunk (dispatched=False)
    contributes zero prompt cost, so token savings are measurable.
    """
    snapshot = harness.build_snapshot()
    for corpus in snapshot["corpora"]:
        dispatched = [c for c in corpus["chunks"] if c["dispatched"]]
        assert corpus["dispatched_count"] == len(dispatched)
        if dispatched:
            assert corpus["prompt_token_chars"] > 0
        # Every dispatched chunk routes somewhere; nothing is silently dropped.
        assert len(corpus["chunks"]) == corpus["chunk_count"]


def test_clean_corpora_have_zero_routing_drift() -> None:
    """#66 acceptance: the clean baseline corpora produce no LLM-vs-pattern
    disagreement. (Their captured responses carry no LLM bucket, so routing goes
    through pattern-match and the backstop never fires.)"""
    from anki_translator.classifier import Overflow, classify
    from anki_translator.config import load_shapes
    from anki_translator.queue import classify_overflow_bucket

    shapes = load_shapes(harness.SHAPES_PATH)
    responses = harness._load_responses()
    for name, _extractor, extract in harness.CORPORA:
        for chunk in extract():
            key = harness._chunk_key(chunk)
            result = classify(chunk, shapes, llm=lambda _p, r=responses[key]: r)
            if isinstance(result, Overflow):
                decision = classify_overflow_bucket(result.reason, result.bucket)
                assert decision.disagreement is None, f"{name}: drift on {result.reason!r}"


def test_prefiltered_chunk_is_trimmed_with_zero_token_cost() -> None:
    """#67 acceptance: a flagged chunk routes to trimmed via the `extractor:`
    reason, is NOT dispatched (dispatched=False), and incurs zero prompt cost —
    while the unflagged form of the same chunk IS dispatched and costs exactly
    its build_prompt length. The delta is the measurable token saving."""
    import dataclasses

    from anki_translator.classifier import build_prompt, prefilter_overflow
    from anki_translator.config import load_shapes

    shapes = load_shapes(harness.SHAPES_PATH)
    responses = harness._load_responses()
    # Take a real corpus chunk that dispatches (octopus[1] is the body fact;
    # octopus[0]/[2] are pre-filtered by the S4 PDF rules).
    chunk = harness._extract_octopus()[1]
    unflagged_record, unflagged_chars = harness._route_chunk(chunk, shapes, responses)
    assert unflagged_record["dispatched"] is True
    assert unflagged_chars == len(build_prompt(chunk, shapes)) > 0

    # Flag the same chunk; it must now bypass the LLM at zero cost.
    flagged = dataclasses.replace(chunk, metadata={**chunk.metadata, "prefilter": "author-affiliations"})
    assert prefilter_overflow(flagged) is not None
    flagged_record, flagged_chars = harness._route_chunk(flagged, shapes, responses)
    assert flagged_record["dispatched"] is False
    assert flagged_record["routing"] == "trimmed"
    assert flagged_record["reason"] == "extractor: pre-filtered author-affiliations"
    assert flagged_chars == 0
    # Token saving for this chunk equals its full prompt cost.
    assert unflagged_chars - flagged_chars == len(build_prompt(chunk, shapes))


def test_prefilter_does_not_change_clean_baseline() -> None:
    """With no extractor flagging anything, the S3 seam is inert — the snapshot
    still matches the committed golden."""
    assert harness.build_snapshot() == harness.load_golden()


def test_missing_response_is_a_hard_error() -> None:
    """Extraction drift (a chunk with no captured response) must fail loudly."""
    from anki_translator.chunk import Chunk
    from anki_translator.config import load_shapes

    shapes = load_shapes(harness.SHAPES_PATH)
    rogue = Chunk(text="a chunk with no captured response", source="x", position="", source_type="manual", metadata={})
    with pytest.raises(KeyError):
        harness._route_chunk(rogue, shapes, harness._load_responses())
