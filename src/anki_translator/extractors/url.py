"""URL extractor: HTML → list[Chunk] with anchor positions.

Trafilatura is great at picking main content but strips heading id attributes during
extraction, which kills our anchor-based Position values. Instead we use trafilatura's
fetcher for the network layer and run our own HTML walk for the content layer: skip the
standard boilerplate tags (nav, header, footer, aside, script, style, form), accumulate
<p> bodies, and tag each paragraph with the nearest preceding <h1..h6 id="..."> anchor.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Optional

import trafilatura

from ..chunk import Chunk
from . import ExtractionError


def fetch(url: str) -> str:
    """Fetch HTML from a URL. Separated from extract() so tests can pass canned HTML."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ExtractionError(f"could not fetch {url}")
    return downloaded


def extract(html: str, url: str) -> list[Chunk]:
    """Extract chunks from an HTML document.

    Returns a Chunk per paragraph. Position is '#anchor-id' if the paragraph follows a
    heading with an id attribute; '' otherwise. Content within nav/header/footer/aside/
    script/style/form tags is skipped.
    """
    if not html or not html.strip():
        raise ExtractionError(f"empty HTML for {url}")

    parser = _ParagraphAndAnchorParser()
    parser.feed(html)
    parser.close()

    chunks: list[Chunk] = []
    for text, anchor in parser.paragraphs:
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                text=text,
                source=url,
                position=f"#{anchor}" if anchor else "",
                source_type="url",
                metadata={"url": url, "anchor": anchor or ""},
            )
        )
    if not chunks:
        raise ExtractionError(f"no paragraphs extracted from {url}")
    return chunks


class _ParagraphAndAnchorParser(HTMLParser):
    """Walks raw HTML, emitting (paragraph_text, anchor_id) pairs.

    State machine:
    - SKIP_TAGS: when inside one of these, ignore everything (boilerplate suppression).
    - HEADING_TAGS: when entering, capture id attribute as the new "current anchor".
    - <p>: collect text content; on </p>, emit the buffered text with the current anchor.
    """

    SKIP_TAGS = frozenset({"nav", "header", "footer", "aside", "script", "style", "form"})
    HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[tuple[str, Optional[str]]] = []
        self._current_anchor: Optional[str] = None
        self._skip_depth = 0
        self._in_paragraph = False
        self._paragraph_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag in self.HEADING_TAGS:
            attr_dict = dict(attrs)
            anchor = attr_dict.get("id")
            if anchor:
                self._current_anchor = anchor
        elif tag == "p":
            self._in_paragraph = True
            self._paragraph_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if tag == "p" and self._in_paragraph:
            text = "".join(self._paragraph_buffer)
            self.paragraphs.append((text, self._current_anchor))
            self._in_paragraph = False
            self._paragraph_buffer = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_paragraph:
            self._paragraph_buffer.append(data)


def extract_url(url: str) -> list[Chunk]:
    """Convenience: fetch + extract in one call. Use extract() directly in tests with canned HTML."""
    return extract(fetch(url), url)
