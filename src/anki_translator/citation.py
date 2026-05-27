"""Format Source and Position field values from per-source-type conventions."""

from __future__ import annotations

from .config import CitationConvention


class CitationError(Exception):
    """Raised when a citation cannot be produced from the given metadata."""


class _SafeFormatDict(dict):
    """str.format_map() helper: missing keys render as empty string instead of raising."""

    def __missing__(self, key: str) -> str:  # noqa: ARG002
        return ""


def cite(
    source_type: str,
    metadata: dict[str, object],
    conventions: dict[str, CitationConvention],
) -> tuple[str, str]:
    """Return (source, position) populated from the per-source-type convention.

    Source is required and must render non-empty after templating; any missing key listed in
    source_required, or an empty result, raises CitationError. Position is allowed to be empty
    when its template is empty, when any position_required key is missing or empty, or when the
    rendered string is whitespace.
    """
    if source_type not in conventions:
        raise CitationError(
            f"unknown source type {source_type!r}; "
            f"known types: {sorted(conventions.keys())}"
        )
    conv = conventions[source_type]

    missing_source = [k for k in conv.source_required if _missing(metadata, k)]
    if missing_source:
        raise CitationError(
            f"source for {source_type!r} missing required metadata keys: {missing_source}"
        )
    source = conv.source_template.format_map(_SafeFormatDict(metadata)).strip()
    if not source:
        # Templating-derived emptiness — separate from "key missing" but equally fatal.
        raise CitationError(
            f"source for {source_type!r} rendered empty; metadata={metadata!r}"
        )

    if not conv.position_template:
        return source, ""
    if any(_missing(metadata, k) for k in conv.position_required):
        return source, ""
    position = conv.position_template.format_map(_SafeFormatDict(metadata)).strip()
    return source, position


def _missing(metadata: dict[str, object], key: str) -> bool:
    """A key is missing if absent, None, or whitespace-only after str() conversion."""
    value = metadata.get(key)
    if value is None:
        return True
    return not str(value).strip()
