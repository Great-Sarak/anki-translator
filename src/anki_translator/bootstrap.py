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
    {
        # Multi-attribute row card. One row of a reference table is one note;
        # each non-empty attribute slot generates a separate card via Anki's
        # conditional template syntax ({{#FieldName}}...{{/FieldName}}). A row
        # with three populated attribute slots yields three cards: front asks
        # `Key (AttrNName)`, back shows the full row + citation. Slots cap at 4
        # because most reference tables (DisplayPort versions, HDMI, USB) sit
        # at 2–3 value columns and 8 felt too large for card-sized review.
        "modelName": "AT Table",
        "inOrderFields": [
            "Key",
            "Attr1Name", "Attr1Value",
            "Attr2Name", "Attr2Value",
            "Attr3Name", "Attr3Value",
            "Attr4Name", "Attr4Value",
            "Source", "Position",
        ],
        "isCloze": False,
        "cardTemplates": [
            {
                "Name": f"Key→Attr{i}",
                "Front": (
                    "{{#Attr" + str(i) + "Name}}"
                    "{{Key}} ({{Attr" + str(i) + "Name}})"
                    "{{/Attr" + str(i) + "Name}}"
                ),
                "Back": (
                    "{{FrontSide}}<hr id=answer>"
                    "{{Attr" + str(i) + "Value}}<br><br>"
                    "<small>"
                    "{{#Attr1Name}}{{Attr1Name}}: {{Attr1Value}}<br>{{/Attr1Name}}"
                    "{{#Attr2Name}}{{Attr2Name}}: {{Attr2Value}}<br>{{/Attr2Name}}"
                    "{{#Attr3Name}}{{Attr3Name}}: {{Attr3Value}}<br>{{/Attr3Name}}"
                    "{{#Attr4Name}}{{Attr4Name}}: {{Attr4Value}}<br>{{/Attr4Name}}"
                    "{{Source}} {{Position}}"
                    "</small>"
                ),
            }
            for i in (1, 2, 3, 4)
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
