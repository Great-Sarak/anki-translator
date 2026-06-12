#!/usr/bin/env python3
"""Run the minimal agent workflow: ingest, commit, sync.

The anki-translator CLI intentionally keeps ingest, review, commit, and sync as
separate operations. This helper is the AgentSkill path for v1, where the
approval gate is skipped and cards are committed immediately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Any


class CommandError(RuntimeError):
    """Raised when a child command fails or emits invalid JSON."""

    def __init__(self, message: str, *, command: list[str], returncode: int = 1, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _batch_tag() -> str:
    now = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"anki-translator-{now}"


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise CommandError(
            "command failed",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


def _json_command(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    result = _run(command, env=env)
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"command did not emit JSON: {exc}",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ) from exc
    if not isinstance(parsed, dict):
        raise CommandError(
            "command JSON was not an object",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return parsed


def _plain_command(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    result = _run(command, env=env)
    parsed: Any
    try:
        parsed = json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError:
        parsed = None
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json": parsed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a source, commit generated cards, and sync Anki.")
    parser.add_argument("source", nargs="?", help="URL, PDF path, .md path, or .txt path")
    parser.add_argument("--text", help="Inline text payload instead of a source")
    parser.add_argument("--deck", required=True, help="Target Anki deck")
    parser.add_argument("--tag", default=None, help="Batch tag for rollback/audit. Default: generated timestamp tag")
    parser.add_argument("--label", help="Source label for --text or manual files")
    parser.add_argument("--concurrency", help="ANKI_TRANSLATOR_CONCURRENCY value. Default: preserve env, else 1")
    parser.add_argument("--model", help="ANKI_TRANSLATOR_MODEL override")
    parser.add_argument("--anki-translator-bin", default="anki-translator")
    parser.add_argument("--anki-manager-bin", default="anki-manager")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Do not run anki-translator bootstrap first")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.source) == bool(args.text):
        print("error: provide exactly one of source or --text", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if args.concurrency:
        env["ANKI_TRANSLATOR_CONCURRENCY"] = args.concurrency
    else:
        env.setdefault("ANKI_TRANSLATOR_CONCURRENCY", "1")
    if args.model:
        env["ANKI_TRANSLATOR_MODEL"] = args.model

    tag = args.tag or _batch_tag()

    try:
        start = _plain_command([args.anki_manager_bin, "start"], env=env)
        bootstrap = None
        if not args.skip_bootstrap:
            bootstrap = _json_command([args.anki_translator_bin, "bootstrap"], env=env)

        ingest_cmd = [args.anki_translator_bin, "ingest", "--deck", args.deck, "--tag", tag]
        if args.label:
            ingest_cmd.extend(["--label", args.label])
        if args.text:
            ingest_cmd.extend(["--text", args.text])
        else:
            ingest_cmd.append(args.source)

        ingest = _json_command(ingest_cmd, env=env)
        queue_file = ingest.get("queue_file")
        if not isinstance(queue_file, str) or not queue_file:
            raise CommandError(
                "ingest JSON did not include queue_file",
                command=ingest_cmd,
                stdout=json.dumps(ingest),
            )

        commit = _json_command([args.anki_translator_bin, "commit", queue_file], env=env)
        sync = _plain_command([args.anki_manager_bin, "sync"], env=env)
    except CommandError as exc:
        print(json.dumps({
            "ok": False,
            "error": str(exc),
            "command": exc.command,
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }, indent=2), file=sys.stderr)
        return exc.returncode or 1

    print(json.dumps({
        "ok": True,
        "deck": args.deck,
        "tag": tag,
        "start": start,
        "bootstrap": bootstrap,
        "ingest": ingest,
        "commit": commit,
        "sync": sync,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
