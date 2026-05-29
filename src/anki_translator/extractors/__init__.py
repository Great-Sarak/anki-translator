"""Extractors produce a list of Chunks from a given source.

Each module here handles one source type. The CLI's ingest command dispatches to the right
extractor based on the input form (URL string, .pdf path, --text payload, etc.).
"""


class ExtractionError(Exception):
    """Raised when extraction fails (network, parse, unsupported content, ...)."""
