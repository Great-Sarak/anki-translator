"""Subprocess env hygiene (#101): the launcher's private-runtime variables
(PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH) must not reach spawned children."""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from anki_translator.subproc import clean_env


def _poison_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHOME", "/opt/python3.12")
    monkeypatch.setenv("PYTHONPATH", "/opt/anki-translator/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/python3.12/lib")
    monkeypatch.setenv("ANKI_TRANSLATOR_KEEP_ME", "yes")


def test_clean_env_drops_private_runtime_vars_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _poison_env(monkeypatch)
    env = clean_env()
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert "LD_LIBRARY_PATH" not in env
    assert env["ANKI_TRANSLATOR_KEEP_ME"] == "yes"


def test_clean_env_does_not_mutate_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _poison_env(monkeypatch)
    clean_env()
    import os

    assert os.environ["PYTHONHOME"] == "/opt/python3.12"


def _capture_run(calls: list[dict]) -> object:
    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(kwargs)
        return types.SimpleNamespace(
            stdout=json.dumps({"outputs": [{"text": "qa"}]}), returncode=0
        )

    return fake_run


def test_default_llm_scrubs_subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from anki_translator.classifier import _default_llm

    _poison_env(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(subprocess, "run", _capture_run(calls))
    _default_llm("prompt")
    (kwargs,) = calls
    env = kwargs["env"]
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert "LD_LIBRARY_PATH" not in env
    assert env["ANKI_TRANSLATOR_KEEP_ME"] == "yes"


def test_get_openclaw_agent_names_scrubs_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anki_translator.tagger import _get_openclaw_agent_names

    _poison_env(monkeypatch)
    calls: list[dict] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(kwargs)
        return types.SimpleNamespace(stdout=json.dumps([]), returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    _get_openclaw_agent_names()
    (kwargs,) = calls
    env = kwargs["env"]
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert "LD_LIBRARY_PATH" not in env
