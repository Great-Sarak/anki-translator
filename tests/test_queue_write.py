"""Tests for the queue-writer side of queue.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from anki_translator.chunk import Chunk
from anki_translator.classifier import CardCandidate, Overflow
from anki_translator.queue import (
    TaggedCandidate,
    make_slug,
    write_queue,
)


def _chunk(text: str = "Mitochondria is the powerhouse of the cell.") -> Chunk:
    return Chunk(
        text=text,
        source="https://example.com/cells",
        position="#organelles",
        source_type="url",
        metadata={"url": "https://example.com/cells", "anchor": "organelles"},
    )


def _candidate(note_type: str = "AT Basic", shape: str = "term-def", fields: dict | None = None) -> CardCandidate:
    return CardCandidate(
        note_type=note_type,
        shape=shape,
        fields=fields or {"Front": "mitochondria", "Back": "powerhouse of the cell"},
        chunk=_chunk(),
    )


# ---- make_slug ----


def test_make_slug_url() -> None:
    assert make_slug("https://example.com/cells", "url") == "example-com-cells"


def test_make_slug_url_with_path() -> None:
    assert make_slug("https://example.com/biology/cells", "url") == "example-com-biology-cells"


def test_make_slug_pdf() -> None:
    assert make_slug("kleppmann.pdf", "pdf") == "kleppmann"


def test_make_slug_manual_label() -> None:
    assert make_slug("chat 2026-05-27", "manual") == "chat-2026-05-27"


def test_make_slug_truncates() -> None:
    long_source = "x" * 200
    slug = make_slug(long_source, "manual")
    assert len(slug) <= 80


def test_make_slug_empty_fallback() -> None:
    assert make_slug("!!!", "manual") == "untitled"


def test_make_slug_collapses_runs_of_hyphens() -> None:
    assert make_slug("foo---bar___baz", "manual") == "foo-bar-baz"


# ---- write_queue ----


def test_write_queue_creates_both_files(tmp_path: Path) -> None:
    queue_path, qa_path, trimmed_path = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
        overflow=[],
        deck="Reading",
        slug="example-com-cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    assert queue_path.exists()
    assert qa_path.exists()
    assert trimmed_path.exists()
    assert queue_path.name == "2026-05-27-example-com-cells.md"
    assert qa_path.name == "2026-05-27-example-com-cells.md"
    assert trimmed_path.name == "2026-05-27-example-com-cells.md"


def test_queue_file_block_structure(tmp_path: Path) -> None:
    queue_path, _, _ = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["biology", "biology::organelles"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — term-def" in body
    assert "**Front:** mitochondria" in body
    assert "**Back:** powerhouse of the cell" in body
    assert "**Source:** https://example.com/cells" in body
    assert "**Position:** #organelles" in body
    assert "**Deck:** Reading" in body
    assert "**Model:** AT Basic" in body
    assert "**Tags:** biology, biology::organelles" in body
    assert body.rstrip().endswith("---")


def test_queue_file_separates_multiple_cards(tmp_path: Path) -> None:
    candidates = [
        TaggedCandidate(_candidate(fields={"Front": "A", "Back": "B"}), tags=["t1"]),
        TaggedCandidate(_candidate(fields={"Front": "C", "Back": "D"}), tags=["t2"]),
        TaggedCandidate(_candidate(fields={"Front": "E", "Back": "F"}), tags=["t3"]),
    ]
    queue_path, _, _ = write_queue(
        tagged=candidates,
        overflow=[],
        deck="Reading",
        slug="multi",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — term-def" in body
    assert "## Card 2 — term-def" in body
    assert "## Card 3 — term-def" in body
    # Three blocks → three `---` separators
    assert body.count("\n---\n") == 3


def test_queue_file_cloze_uses_text_field(tmp_path: Path) -> None:
    cloze_candidate = _candidate(
        note_type="AT Cloze",
        shape="cloze",
        fields={"Text": "The {{c1::mitochondria}} is the powerhouse of the cell."},
    )
    queue_path, _, _ = write_queue(
        tagged=[TaggedCandidate(cloze_candidate, tags=["biology"])],
        overflow=[],
        deck="Reading",
        slug="cloze-test",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "## Card 1 — cloze" in body
    assert "**Text:** The {{c1::mitochondria}}" in body
    # No Front/Back for cloze
    assert "**Front:**" not in body
    assert "**Back:**" not in body


def test_qa_file_renders_overflow(tmp_path: Path) -> None:
    overflow = [
        Overflow(chunk=_chunk("Some long discursive paragraph that does not fit a card."), reason="no_shape_fit"),
        Overflow(chunk=_chunk("Another overflow chunk with context."), reason="exceeds_budget: Back is 250 chars"),
    ]
    _, qa_path, _ = write_queue(
        tagged=[],
        overflow=overflow,
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = qa_path.read_text()
    assert "# Q&A — cells" in body
    assert "## 1. From https://example.com/cells" in body
    assert "#organelles" in body
    assert "Reason: no_shape_fit" in body
    assert "Reason: exceeds_budget" in body
    assert "Some long discursive paragraph" in body


def test_qa_file_empty_overflow_still_written(tmp_path: Path) -> None:
    _, qa_path, _ = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = qa_path.read_text()
    assert "No overflow chunks" in body


def test_queue_file_empty_candidates_still_written(tmp_path: Path) -> None:
    queue_path, _, _ = write_queue(
        tagged=[],
        overflow=[Overflow(chunk=_chunk(), reason="no_shape_fit")],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "empty" in body.lower()


def test_write_queue_creates_missing_directories(tmp_path: Path) -> None:
    queue_dir = tmp_path / "deeply" / "nested" / "queue"
    qa_dir = tmp_path / "other" / "qa"
    trimmed_dir = tmp_path / "yet" / "another" / "trimmed"
    write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=queue_dir,
        qa_dir=qa_dir,
        trimmed_dir=trimmed_dir,
        ingestion_date=date(2026, 5, 27),
    )
    assert queue_dir.exists()
    assert qa_dir.exists()
    assert trimmed_dir.exists()


def test_queue_file_no_tags_renders_cleanly(tmp_path: Path) -> None:
    queue_path, _, _ = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=[])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "**Tags:** " in body  # empty but present


def test_queue_file_multiline_field_inlined(tmp_path: Path) -> None:
    """Field values with embedded newlines must be inlined onto one markdown line."""
    multiline = _candidate(fields={"Front": "x", "Back": "line one\nline two\nline three"})
    queue_path, _, _ = write_queue(
        tagged=[TaggedCandidate(multiline, tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = queue_path.read_text()
    assert "**Back:** line one line two line three" in body


def test_queue_file_renders_term_table_row_as_separate_block(tmp_path: Path) -> None:
    """For #52: each row of a term-table chunk is a CardCandidate after the cli
    flattens MultiCardCandidate. Each lands as its own `## Card N — term-table`
    block, preserving per-row delete semantics during review. The natural
    ordering of classify_chunks keeps same-source rows adjacent in the queue
    file — that's the 'grouped' behavior, without breaking the parser's
    one-block-per-card split."""
    rows = [
        _candidate(
            note_type="AT Table",
            shape="term-table",
            fields={
                "Key": "DP 1.2",
                "Attr1Name": "Link rate", "Attr1Value": "HBR2",
                "Attr2Name": "Total bandwidth", "Attr2Value": "21.6 Gbps",
                "Attr3Name": "Top resolution", "Attr3Value": "4K@60",
            },
        ),
        _candidate(
            note_type="AT Table",
            shape="term-table",
            fields={
                "Key": "DP 1.4",
                "Attr1Name": "Link rate", "Attr1Value": "HBR3",
                "Attr2Name": "Total bandwidth", "Attr2Value": "25.92 Gbps",
                "Attr3Name": "Top resolution", "Attr3Value": "8K@60",
            },
        ),
    ]
    queue_path, _qa_path, _trimmed_path = write_queue(
        tagged=[TaggedCandidate(c, tags=["video::displayport"]) for c in rows],
        overflow=[],
        deck="Myrzka::Cables",
        slug="dp",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
    )
    body = queue_path.read_text()
    # Two blocks, one per row, in order.
    assert body.count("## Card") == 2
    assert body.index("DP 1.2") < body.index("DP 1.4")
    # Per-row block contains the row's full attribute set so the reviewer can
    # judge it without re-reading the source.
    assert "**Key:** DP 1.2" in body
    assert "**Attr1Name:** Link rate" in body
    assert "**Attr1Value:** HBR2" in body
    assert "**Attr3Value:** 4K@60" in body
    assert "**Model:** AT Table" in body


# ---- overflow_bucket + qa/trimmed split (#46) ----


from anki_translator.queue import overflow_bucket  # noqa: E402


def test_overflow_bucket_routes_classifier_emitted_reasons_to_qa() -> None:
    """Reasons emitted directly by the classifier (not the LLM) always indicate
    substantive overflow — they fired because real content didn't fit a budget
    or a parse failed. Never trimmed."""
    assert overflow_bucket("exceeds_budget: Back is 208 chars, limit 200") == "qa"
    assert overflow_bucket("exceeds_budget: row 1 attr 1 value is 114 chars, limit 80") == "qa"
    assert overflow_bucket("invalid_response: missing 'choice'") == "qa"
    assert overflow_bucket("invalid_response: got array (expected single object)") == "qa"
    assert overflow_bucket("llm_error: RuntimeError") == "qa"
    assert overflow_bucket("no_shape_fit") == "qa"


def test_overflow_bucket_routes_chaff_phrasings_to_trimmed() -> None:
    """Sampled directly from the post-#52 cable-id qa file and the earlier
    octopus PDF qa file. Each phrasing here actually appeared in a live
    overflow reason and should route to trimmed."""
    chaff_reasons = [
        # cable-id ingest
        "Passage is a section header with no substantive content to convert into a flashcard.",
        "Passage is a section heading with no factual content to encode.",
        "Passage is a reference list of links without standalone factual content suitable for flashcard format.",
        "Passage is a collection of reference links without structured factual content suitable for flashcard encoding.",
        "Passage is a table of contents—structural metadata rather than a fact suitable for flashcard memorization.",
        "Passage is a document header/overview; no discrete fact suitable for flashcard extraction.",
        # octopus PDF
        "Bibliographic citation does not fit any flashcard shape's instructional purpose.",
        "Bibliographic reference does not fit flashcard format—it is metadata, not a learnable fact or concept.",
        "Citation metadata does not contain a factual claim suitable for flashcard learning.",
        "Passage is a list of author names and affiliations with no factual content suitable for flashcard learning.",
        "Passage is acknowledgments/funding information with no standalone fact suitable for flashcard memorization.",
        "Passage is introductory/transitional text that previews future sections rather than stating a discrete fact suitable for flashcard memorization.",
        "This is a standard boilerplate disclaimer unsuitable for memorization as educational content.",
        # Surfaced on the post-#46 re-ingest of the octopus PDF + mitochondrion
        # (2026-06-09). These leaked into qa because the original chaff list
        # missed "bibliography" (without -ic), "header block", "list of topic
        # headings", "minireview title", "introductory material", fragments,
        # bare taxonomic labels, and "passage is incomplete".
        "Passage is a bibliography citation without a question-answer or fact suitable for flashcard memorization.",
        "Passage is a bibliography entry, not factual content suitable for flashcard conversion.",
        "passage is a header block with affiliations and metadata, not a fact suitable for flashcard encoding",
        "Passage is a list of topic headings without definitions or explanatory content; insufficient factual substance for any card shape.",
        "Passage is a minireview title without substantive factual content suitable for flashcard encoding.",
        "Passage is introductory material about upcoming content structure; contains no standalone factual content suitable for flashcard conversion.",
        "Passage is a fragment lacking definition or context; cannot form a complete flashcard fact.",
        "passage is fragmentary and lacks sufficient content to form a complete flashcard fact",
        "Passage is a bare taxonomic label with no definitional content suitable for flashcard encoding.",
        "Passage is incomplete — it announces a list of transport modes but provides no details to fit within any shape's budget.",
        "Passage is incomplete—it introduces a list of purposes but provides no actual list items to memorize.",
    ]
    for reason in chaff_reasons:
        assert overflow_bucket(reason) == "trimmed", f"expected trimmed for: {reason!r}"


def test_overflow_bucket_chaff_signal_beats_llm_exceeds_rationalization() -> None:
    """The LLM frequently rationalizes a bibliographic-chunk overflow with
    budget language ("exceeds field capacity", "requires multiple metadata
    fields") — but the content itself is still chaff. Chaff signals beat the
    generic 'exceed' substantive signal. The opposite ordering ate the wrong
    bucket on the Cell octopus PDF (2026-06-08) where bibliographic chunks
    leaked into qa.

    Note: the *classifier*-emitted `exceeds_budget:` prefix is still
    unconditionally qa (it fires for real card content failing a shape budget,
    never for chaff). The next test covers that."""
    assert overflow_bucket(
        "Bibliographic reference requires multiple metadata fields (authors, year, title, "
        "journal, pages) that exceed single-field capacity of available shapes."
    ) == "trimmed"
    assert overflow_bucket(
        "Citation with multiple authors and detailed publication metadata exceeds practical "
        "flashcard scope; no single fact isolates well for memorization."
    ) == "trimmed"


def test_overflow_bucket_classifier_prefix_beats_chaff_signal() -> None:
    """An `exceeds_budget:` prefix is classifier-emitted and ALWAYS routes to
    qa, even if a chaff word appears later in the reason. The classifier only
    emits this prefix when actual content failed a shape's budget — never for
    citation metadata."""
    assert overflow_bucket(
        "exceeds_budget: Back is 250 chars, limit 200 (passage was a bibliographic reference)"
    ) == "qa"


def test_overflow_bucket_defaults_unknown_phrasings_to_qa() -> None:
    """Unknown reasons keep content in qa. Better to over-keep and refine
    `_CHAFF_SIGNALS` later than to silently discard new phrasings."""
    assert overflow_bucket("some entirely novel phrasing the LLM just invented") == "qa"
    assert overflow_bucket("") == "qa"


# ---- LLM self-tagged bucket as primary signal (#65) ----


def test_overflow_bucket_llm_trimmed_routes_novel_chaff_without_signal_edits() -> None:
    """The #65 payoff: a chaff phrasing absent from `_CHAFF_SIGNALS` still routes
    to trimmed when the LLM self-tags bucket='trimmed' — zero signal-list edits.

    The reason text below matches no entry in `_CHAFF_SIGNALS` (proven by the
    bucket=None assertion: pattern-match alone defaults it to qa). The LLM bucket
    flips it to trimmed."""
    novel = "this is a pull-quote sidebar repeating a sentence from the body verbatim"
    assert overflow_bucket(novel) == "qa"  # pattern-match alone: no signal hit
    assert overflow_bucket(novel, "trimmed") == "trimmed"  # LLM bucket is primary


def test_overflow_bucket_llm_qa_keeps_borderline_content() -> None:
    """A reason with no chaff signal that the LLM tags qa stays qa (and would
    anyway by default) — the bucket makes the intent explicit rather than relying
    on the default."""
    assert overflow_bucket("a dense paragraph with three interlocking facts", "qa") == "qa"


def test_overflow_bucket_missing_or_garbage_bucket_falls_back_to_pattern_match() -> None:
    """Missing/invalid bucket → pre-#65 behavior. None and out-of-vocabulary
    values both defer to reason pattern-matching, so a run whose model omitted
    the field routes exactly as it did before #65."""
    # chaff signal in the reason, no bucket → trimmed (pattern-match)
    assert overflow_bucket("passage is a bibliography entry", None) == "trimmed"
    assert overflow_bucket("passage is a bibliography entry", "banana") == "trimmed"
    # no signal, no/garbage bucket → qa default
    assert overflow_bucket("a novel phrasing", None) == "qa"
    assert overflow_bucket("a novel phrasing", "") == "qa"


def test_overflow_bucket_classifier_prefix_beats_llm_bucket() -> None:
    """Precedence: classifier-mechanical prefix > LLM bucket. A real card that
    failed a budget must stay qa even if the LLM mislabels it trimmed — the
    prefix is a mechanical fact, not a content judgment."""
    assert overflow_bucket("exceeds_budget: Back is 250 chars, limit 200", "trimmed") == "qa"
    assert overflow_bucket("llm_error: TimeoutError", "trimmed") == "qa"


# ---- _CHAFF_SIGNALS demoted to a conservative safety net + drift telemetry (#66) ----

from anki_translator.queue import classify_overflow_bucket  # noqa: E402


def test_strong_chaff_downgrades_llm_qa_mislabel_to_trimmed() -> None:
    """The S2 backstop: the LLM tagged obvious structural chaff as `qa`, but a
    *strong* chaff signal overrides it to `trimmed` and the disagreement is
    recorded for drift-watching."""
    decision = classify_overflow_bucket("passage is a bibliography entry", "qa")
    assert decision.bucket == "trimmed"
    assert decision.downgraded is True
    assert decision.disagreement == "qa_with_chaff"


def test_llm_qa_without_strong_chaff_is_left_alone() -> None:
    """The backstop is conservative: a `qa` with no strong chaff signal stays qa,
    no disagreement. A merely-fuzzy _CHAFF_SIGNALS hit does NOT override the LLM."""
    decision = classify_overflow_bucket("a dense paragraph holding several interlocking facts", "qa")
    assert decision.bucket == "qa"
    assert decision.downgraded is False
    assert decision.disagreement is None


def test_backstop_never_upgrades_trimmed_to_qa() -> None:
    """Conservative-only: an LLM `trimmed` is always honored, even when no pattern
    chaff signal fires. The disagreement is recorded but not acted on — we never
    resurrect content the LLM chose to trim."""
    decision = classify_overflow_bucket("an unusual sidebar pull-quote", "trimmed")
    assert decision.bucket == "trimmed"
    assert decision.disagreement == "trimmed_without_chaff"

    # LLM trimmed + a pattern chaff signal present → agreement, no disagreement.
    agree = classify_overflow_bucket("passage is a table of contents", "trimmed")
    assert agree.bucket == "trimmed"
    assert agree.disagreement is None


def test_classifier_prefix_records_no_disagreement() -> None:
    """Mechanical prefixes short-circuit to qa before any chaff check, so they
    never register as drift even if a chaff word trails the reason."""
    decision = classify_overflow_bucket(
        "exceeds_budget: Back is 250 chars (was a bibliography reference)", "trimmed"
    )
    assert decision.bucket == "qa"
    assert decision.disagreement is None


def test_write_queue_emits_drift_footer_and_downgrades_in_trimmed_file(tmp_path: Path) -> None:
    """End-to-end: a qa-tagged chaff chunk is downgraded into the trimmed file and
    the drift footer counts the override."""
    overflow = [
        # LLM mislabeled this bibliography as qa; the backstop downgrades it.
        Overflow(chunk=_chunk("Smith J, Jones K. (2019). On cables. J. Cables 4:1-9."),
                 reason="passage is a bibliography entry", bucket="qa"),
        # A genuine substantive qa, left in qa.
        Overflow(chunk=_chunk("A real multi-fact paragraph."),
                 reason="passage holds multiple distinct facts", bucket="qa"),
    ]
    _, qa_path, trimmed_path = write_queue(
        tagged=[],
        overflow=overflow,
        deck="Reading",
        slug="cables",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    trimmed_body = trimmed_path.read_text()
    qa_body = qa_path.read_text()
    # Downgraded chunk moved to trimmed; genuine qa stayed in qa.
    assert "Smith J, Jones K." in trimmed_body and "Smith J, Jones K." not in qa_body
    assert "A real multi-fact paragraph." in qa_body and "A real multi-fact paragraph." not in trimmed_body
    # Drift footer present on the trimmed file, counting exactly one override.
    assert "Routing drift" in trimmed_body
    assert "1 qa→trimmed override" in trimmed_body
    # The qa file carries no drift footer.
    assert "Routing drift" not in qa_body


def test_write_queue_splits_overflow_between_qa_and_trimmed(tmp_path: Path) -> None:
    """End-to-end split: mixed overflow batch routes each chunk to the right
    file. Substantive overflows land in the qa file with their text; trimmed
    chaff lands in the trimmed file. Cross-contamination is the regression
    this test guards against."""
    overflow = [
        Overflow(
            chunk=_chunk("This paragraph has multiple distinct facts and a 250-char back..."),
            reason="exceeds_budget: Back is 250 chars, limit 200",
        ),
        Overflow(
            chunk=_chunk("Smith J, Jones K. (2026) On the mitochondrion. Cell 123: 456-789."),
            reason="Bibliographic reference is metadata, not a learnable fact.",
        ),
        Overflow(
            chunk=_chunk("### Introduction"),
            reason="Passage is a section header with no substantive content to convert into a flashcard.",
        ),
        Overflow(
            chunk=_chunk("Some interesting paragraph that didn't fit a Cloze."),
            reason="Passage holds multiple distinct facts.",
        ),
    ]
    _, qa_path, trimmed_path = write_queue(
        tagged=[],
        overflow=overflow,
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )

    qa_body = qa_path.read_text()
    trimmed_body = trimmed_path.read_text()

    # Substantive overflows in qa only.
    assert "Back is 250 chars" in qa_body
    assert "multiple distinct facts" in qa_body
    assert "This paragraph has multiple" in qa_body
    assert "Some interesting paragraph" in qa_body

    # Chaff in trimmed only.
    assert "Smith J, Jones K." in trimmed_body
    assert "Bibliographic reference" in trimmed_body
    assert "Introduction" in trimmed_body
    assert "section header" in trimmed_body

    # No leakage.
    assert "Smith J, Jones K." not in qa_body
    assert "Introduction" not in qa_body
    assert "Back is 250 chars" not in trimmed_body
    assert "This paragraph has multiple" not in trimmed_body


def test_trimmed_file_uses_trimmed_heading(tmp_path: Path) -> None:
    """The trimmed file's H1 heading distinguishes it from the qa file at a
    glance when both are open in an editor."""
    overflow = [Overflow(chunk=_chunk("x"), reason="Passage is a section header with no factual content.")]
    _, _, trimmed_path = write_queue(
        tagged=[],
        overflow=overflow,
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    body = trimmed_path.read_text()
    assert "# Trimmed — cells" in body
    assert "Q&A" not in body


def test_trimmed_file_empty_still_written_with_placeholder(tmp_path: Path) -> None:
    """Like the qa file, the trimmed file is always written for downstream
    tooling consistency, even with an empty bucket."""
    _, _, trimmed_path = write_queue(
        tagged=[TaggedCandidate(_candidate(), tags=["t"])],
        overflow=[],
        deck="Reading",
        slug="cells",
        queue_dir=tmp_path / "queue",
        qa_dir=tmp_path / "qa",
        trimmed_dir=tmp_path / "trimmed",
        ingestion_date=date(2026, 5, 27),
    )
    assert trimmed_path.exists()
    assert "No overflow chunks" in trimmed_path.read_text()
