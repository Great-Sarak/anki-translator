"""Create the AT-prefixed starter note types in Anki if they don't already exist.

A fresh Anki collection has only the built-in note types (Basic, Cloze, ...). Those don't
include the Source and Position fields the translator relies on, so we install our own
AT-prefixed set on first run.

The starter set matches what shapes.yaml expects to find — keep these two definitions in
sync. If you add a new shape to shapes.yaml, add the matching model definition here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anki_manager import AnkiManager


# Each starter note type. Fields are ordered; the first one is the duplicate-detection key.
STARTER_MODELS: list[dict[str, Any]] = [
    {
        "modelName": "AT Basic",
        "inOrderFields": ["Front", "Back", "Source", "Position"],
        "isCloze": False,
        "cardTemplates": [
            {
                "Name": "Front→Back",
                "Front": "{{Front}}",
                "Back": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small>{{Source}} {{Position}}</small>",
            },
            {
                "Name": "Back→Front",
                "Front": "{{Back}}",
                "Back": "{{FrontSide}}<hr id=answer>{{Front}}<br><br><small>{{Source}} {{Position}}</small>",
            },
        ],
    },
    {
        "modelName": "AT Cloze",
        "inOrderFields": ["Text", "Source", "Position"],
        "isCloze": True,
        "cardTemplates": [
            {
                "Name": "Cloze",
                "Front": "{{cloze:Text}}",
                "Back": "{{cloze:Text}}<br><br><small>{{Source}} {{Position}}</small>",
            },
        ],
    },
    {
        "modelName": "AT List",
        "inOrderFields": ["Front", "Back", "Source", "Position"],
        "isCloze": False,
        "cardTemplates": [
            {
                "Name": "Front→List",
                "Front": "{{Front}}",
                "Back": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small>{{Source}} {{Position}}</small>",
            },
        ],
    },
    {
        "modelName": "AT Steps",
        "inOrderFields": ["Front", "Back", "Source", "Position"],
        "isCloze": False,
        "cardTemplates": [
            {
                "Name": "Front→Steps",
                "Front": "{{Front}}",
                "Back": "{{FrontSide}}<hr id=answer>{{Back}}<br><br><small>{{Source}} {{Position}}</small>",
            },
        ],
    },
]


@dataclass
class BootstrapResult:
    created: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)


def starter_model_names() -> list[str]:
    """Names of all starter note types — for use by callers that need to check coverage."""
    return [m["modelName"] for m in STARTER_MODELS]


def check_missing(mgr: AnkiManager) -> list[str]:
    """Return the starter model names not yet present in Anki."""
    present = set(mgr.list_models().keys())
    return [m["modelName"] for m in STARTER_MODELS if m["modelName"] not in present]


def bootstrap(mgr: AnkiManager, dry_run: bool = False) -> BootstrapResult:
    """Create any missing starter note types. Idempotent — running twice is a no-op.

    Uses mgr.call('createModel', ...) since anki-manager does not yet expose a typed
    create_model() method. Track that follow-up against Great-Sarak/anki-manager#1 — once
    a typed method lands, swap this passthrough for it.
    """
    result = BootstrapResult()
    present = set(mgr.list_models().keys())

    for model in STARTER_MODELS:
        name = model["modelName"]
        if name in present:
            result.already_present.append(name)
            continue
        if dry_run:
            result.created.append(name)
            continue
        mgr.call(
            "createModel",
            modelName=name,
            inOrderFields=model["inOrderFields"],
            isCloze=model["isCloze"],
            cardTemplates=model["cardTemplates"],
        )
        result.created.append(name)

    return result
