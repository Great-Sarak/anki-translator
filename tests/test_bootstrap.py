"""Tests for the starter-note-type bootstrap.

Uses a MagicMock to stand in for AnkiManager so tests don't require a running Anki container.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from anki_translator.bootstrap import (
    STARTER_MODELS,
    bootstrap,
    check_missing,
    starter_model_names,
)


def _fake_mgr(existing_models: dict[str, list[str]] | None = None) -> MagicMock:
    mgr = MagicMock()
    mgr.list_models.return_value = existing_models or {}
    return mgr


def test_starter_set_contains_expected_names() -> None:
    """If this changes, shapes.yaml must change to match."""
    assert starter_model_names() == ["AT Basic", "AT Cloze", "AT List", "AT Steps"]


def test_check_missing_fresh_collection() -> None:
    mgr = _fake_mgr()
    missing = check_missing(mgr)
    assert missing == ["AT Basic", "AT Cloze", "AT List", "AT Steps"]


def test_check_missing_partial_existing() -> None:
    mgr = _fake_mgr({"AT Basic": ["Front", "Back"], "Basic": ["Front", "Back"]})
    missing = check_missing(mgr)
    assert missing == ["AT Cloze", "AT List", "AT Steps"]


def test_check_missing_all_present() -> None:
    mgr = _fake_mgr({name: ["Front"] for name in starter_model_names()})
    assert check_missing(mgr) == []


def test_bootstrap_creates_all_missing() -> None:
    mgr = _fake_mgr()
    result = bootstrap(mgr)
    assert result.created == ["AT Basic", "AT Cloze", "AT List", "AT Steps"]
    assert result.already_present == []
    # call() invoked once per missing model with createModel action
    assert mgr.call.call_count == 4
    actions = [c.args[0] for c in mgr.call.call_args_list]
    assert actions == ["createModel"] * 4


def test_bootstrap_skips_present() -> None:
    mgr = _fake_mgr({"AT Basic": ["Front", "Back"]})
    result = bootstrap(mgr)
    assert "AT Basic" in result.already_present
    assert "AT Basic" not in result.created
    assert mgr.call.call_count == 3  # the other three


def test_bootstrap_idempotent_second_run() -> None:
    """First run creates everything. A second run against a 'now-populated' Anki creates nothing."""
    mgr = _fake_mgr()
    bootstrap(mgr)
    # Pretend Anki now has all four
    mgr.list_models.return_value = {name: ["Front"] for name in starter_model_names()}
    mgr.call.reset_mock()
    result = bootstrap(mgr)
    assert result.created == []
    assert set(result.already_present) == set(starter_model_names())
    mgr.call.assert_not_called()


def test_bootstrap_dry_run_does_not_call() -> None:
    mgr = _fake_mgr()
    result = bootstrap(mgr, dry_run=True)
    assert result.created == starter_model_names()
    mgr.call.assert_not_called()


def test_bootstrap_passes_correct_params_for_cloze() -> None:
    """AT Cloze must be created with isCloze=True; other models with isCloze=False."""
    mgr = _fake_mgr()
    bootstrap(mgr)
    by_name = {c.kwargs["modelName"]: c.kwargs for c in mgr.call.call_args_list}
    assert by_name["AT Cloze"]["isCloze"] is True
    assert by_name["AT Cloze"]["inOrderFields"] == ["Text", "Source", "Position"]
    assert by_name["AT Basic"]["isCloze"] is False
    assert by_name["AT Basic"]["inOrderFields"] == ["Front", "Back", "Source", "Position"]


def test_starter_models_all_include_source_and_position() -> None:
    """The whole point of bootstrapping our own types is that they include Source + Position."""
    for model in STARTER_MODELS:
        fields = model["inOrderFields"]
        assert "Source" in fields, f"{model['modelName']} missing Source field"
        assert "Position" in fields, f"{model['modelName']} missing Position field"
