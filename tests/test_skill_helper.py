from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "skills" / "anki-translator" / "scripts" / "ingest_commit_sync.py"


def _write_fake_bins(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"

    (bin_dir / "anki-translator").write_text(
        f"""#!/usr/bin/env python3
import json
import os
import sys

log = {str(log)!r}
with open(log, "a") as fh:
    fh.write(json.dumps({{"bin": "anki-translator", "argv": sys.argv[1:], "concurrency": os.environ.get("ANKI_TRANSLATOR_CONCURRENCY")}}) + "\\n")

cmd = sys.argv[1]
if cmd == "bootstrap":
    print(json.dumps({{"created": [], "already_present": ["AT Basic"]}}))
elif cmd == "ingest":
    print(json.dumps({{"queue_file": "/tmp/fake-queue.md", "candidates": 2, "overflow_qa": 1, "overflow_trimmed": 0}}))
elif cmd == "commit":
    print(json.dumps({{"created": ["anki-manager::1"], "updated": [], "failed": [], "archived_to": "/tmp/committed/fake-queue.md"}}))
else:
    raise SystemExit(9)
"""
    )
    (bin_dir / "anki-manager").write_text(
        f"""#!/usr/bin/env python3
import json
import sys

log = {str(log)!r}
with open(log, "a") as fh:
    fh.write(json.dumps({{"bin": "anki-manager", "argv": sys.argv[1:]}}) + "\\n")

cmd = sys.argv[1]
if cmd == "start":
    print("ready")
elif cmd == "sync":
    print(json.dumps({{"status": "NO_CHANGES"}}))
else:
    raise SystemExit(9)
"""
    )
    (bin_dir / "anki-translator").chmod(0o755)
    (bin_dir / "anki-manager").chmod(0o755)
    return bin_dir


def test_helper_runs_ingest_commit_sync(tmp_path: Path) -> None:
    bin_dir = _write_fake_bins(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.pop("ANKI_TRANSLATOR_CONCURRENCY", None)

    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--deck",
            "Myrzka::Reading",
            "--tag",
            "batch-test",
            "https://example.com/source",
        ],
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["deck"] == "Myrzka::Reading"
    assert parsed["tag"] == "batch-test"
    assert parsed["ingest"]["queue_file"] == "/tmp/fake-queue.md"
    assert parsed["commit"]["created"] == ["anki-manager::1"]
    assert parsed["sync"]["json"] == {"status": "NO_CHANGES"}

    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls == [
        {"bin": "anki-manager", "argv": ["start"]},
        {"bin": "anki-translator", "argv": ["bootstrap"], "concurrency": "1"},
        {
            "bin": "anki-translator",
            "argv": ["ingest", "--deck", "Myrzka::Reading", "--tag", "batch-test", "https://example.com/source"],
            "concurrency": "1",
        },
        {"bin": "anki-translator", "argv": ["commit", "/tmp/fake-queue.md"], "concurrency": "1"},
        {"bin": "anki-manager", "argv": ["sync"]},
    ]


def test_helper_supports_inline_text_and_concurrency_override(tmp_path: Path) -> None:
    bin_dir = _write_fake_bins(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--deck",
            "Myrzka::Reading",
            "--text",
            "A source paragraph.",
            "--label",
            "telegram note",
            "--concurrency",
            "3",
            "--skip-bootstrap",
        ],
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["ok"] is True
    assert parsed["tag"].startswith("anki-translator-")

    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls[1] == {
        "bin": "anki-translator",
        "argv": [
            "ingest",
            "--deck",
            "Myrzka::Reading",
            "--tag",
            parsed["tag"],
            "--label",
            "telegram note",
            "--text",
            "A source paragraph.",
        ],
        "concurrency": "3",
    }


def test_helper_rejects_missing_source_and_text() -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER), "--deck", "Myrzka::Reading"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert "exactly one" in result.stderr
