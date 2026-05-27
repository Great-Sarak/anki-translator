"""The Chunk dataclass produced by extractors and consumed by classifier + queue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A unit of extracted content with citation metadata.

    text:        the chunk content (paragraph or section body)
    source:      canonical identifier for the source artifact (URL, filename, label, ...)
    position:    location within the source (anchor, page N, etc.), or '' if unknown
    source_type: one of the citations.yaml types ('url', 'doi', 'pdf', 'book', 'manual')
    metadata:    extra per-source-type fields the extractor captured (for citation.cite())
    """

    text: str
    source: str
    position: str
    source_type: str
    metadata: dict[str, object]
