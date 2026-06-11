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


def test_missing_response_is_a_hard_error() -> None:
    """Extraction drift (a chunk with no captured response) must fail loudly."""
    from anki_translator.chunk import Chunk
    from anki_translator.config import load_shapes

    shapes = load_shapes(harness.SHAPES_PATH)
    rogue = Chunk(text="a chunk with no captured response", source="x", position="", source_type="manual", metadata={})
    with pytest.raises(KeyError):
        harness._route_chunk(rogue, shapes, harness._load_responses())
