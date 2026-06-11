"""URL extractor: HTML → list[Chunk] with anchor positions.

Trafilatura is great at picking main content but strips heading id attributes during
extraction, which kills our anchor-based Position values. Instead we use trafilatura's
fetcher for the network layer and run our own HTML walk for the content layer: skip the
standard boilerplate tags (nav, header, footer, aside, script, style, form), accumulate
<p> bodies, and tag each paragraph with the nearest preceding <h1..h6 id="..."> anchor.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Optional

from ..chunk import Chunk
from ..classifier import PREFILTER_METADATA_KEY
from . import ExtractionError

_DEFAULT_USER_AGENT = "anki-translator/0.1 (+https://github.com/Great-Sarak/anki-translator)"

# --- Structural pre-filter heuristics (#70, S6) ----------------------------------
#
# Flag paragraphs that survive extraction but live under a boilerplate article/wiki
# section (references, cited-by, author-info, see-also/external-links) as
# structural chaff (PREFILTER_METADATA_KEY), so they bypass the LLM and route
# straight to trimmed (#67). Driven by the enclosing section heading TEXT rather
# than a fragile id, since wikis vary the anchor. Conservative: only well-known
# boilerplate section titles match; an unrecognized section keeps its paragraphs.


def _boilerplate_kind(section_title: str) -> str | None:
    """Map an enclosing section heading (lowercased) to a chaff kind, or None."""
    t = section_title
    if not t:
        return None
    if "cited by" in t:
        return "cited-by"
    if "author information" in t or "author info" in t:
        return "author-info"
    if "acknowledgment" in t or "acknowledgement" in t:
        return "author-info"
    if any(s in t for s in ("references", "bibliography", "citations", "footnotes", "works cited")):
        return "references"
    # bare "Notes" on its own line — wiki citation/footnote list
    if t == "notes" or t.startswith("notes "):
        return "references"
    if any(s in t for s in ("further reading", "external links", "see also")):
        return "further-reading"
    if "contents" in t:
        return "table-of-contents"
    return None


def fetch_bytes(url: str, *, timeout: float = 30.0) -> tuple[bytes, str]:
    """Fetch a URL and return (body_bytes, content_type).

    Uses urllib.request (stdlib) rather than trafilatura's fetcher so the
    caller can inspect the Content-Type header — needed by the CLI dispatcher
    to route PDF-serving URLs (e.g. PLOS One's `?type=printable` endpoint) to
    the PDF extractor instead of the HTML extractor. See #45.

    `content_type` is the bare media type (e.g. "application/pdf",
    "text/html") with charset stripped; empty string if the server omitted
    the header.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get_content_type() or ""
            body = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise ExtractionError(f"could not fetch {url}: {e}") from e
    return body, content_type


def fetch(url: str) -> str:
    """Fetch URL as HTML text. Convenience wrapper around fetch_bytes for
    callers that know they want HTML (e.g. tests that pass canned HTML)."""
    body, _ = fetch_bytes(url)
    return body.decode("utf-8", errors="replace")


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
    for text, anchor, section_title in parser.paragraphs:
        text = text.strip()
        if not text:
            continue
        metadata: dict[str, object] = {"url": url, "anchor": anchor or ""}
        kind = _boilerplate_kind(section_title)
        if kind:
            metadata[PREFILTER_METADATA_KEY] = kind
        chunks.append(
            Chunk(
                text=text,
                source=url,
                position=f"#{anchor}" if anchor else "",
                source_type="url",
                metadata=metadata,
            )
        )
    if not chunks:
        raise ExtractionError(f"no paragraphs extracted from {url}")
    return chunks


class _ParagraphAndAnchorParser(HTMLParser):
    """Walks raw HTML, emitting (paragraph_text, anchor_id, section_title) triples.

    State machine:
    - SKIP_TAGS: when inside one of these, ignore everything (boilerplate suppression).
    - HEADING_TAGS: when entering, capture id attribute as the new "current anchor"
      and buffer the heading text; on close, set the "current section title" (used
      by the S6 structural pre-filter to flag references/cited-by/author-info).
    - <p>: collect text content; on </p>, emit the buffered text with the current
      anchor and section title.
    """

    SKIP_TAGS = frozenset({"nav", "header", "footer", "aside", "script", "style", "form"})
    HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[tuple[str, Optional[str], str]] = []
        self._current_anchor: Optional[str] = None
        self._current_section_title: str = ""
        self._skip_depth = 0
        self._in_paragraph = False
        self._paragraph_buffer: list[str] = []
        self._in_heading = False
        self._heading_buffer: list[str] = []

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
            self._in_heading = True
            self._heading_buffer = []
        elif tag == "p":
            self._in_paragraph = True
            self._paragraph_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth > 0:
            return
        if tag in self.HEADING_TAGS and self._in_heading:
            self._current_section_title = "".join(self._heading_buffer).strip().lower()
            self._in_heading = False
            self._heading_buffer = []
        elif tag == "p" and self._in_paragraph:
            text = "".join(self._paragraph_buffer)
            self.paragraphs.append((text, self._current_anchor, self._current_section_title))
            self._in_paragraph = False
            self._paragraph_buffer = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_heading:
            self._heading_buffer.append(data)
        elif self._in_paragraph:
            self._paragraph_buffer.append(data)


def extract_url(url: str) -> list[Chunk]:
    """Convenience: fetch + extract in one call. Use extract() directly in tests with canned HTML."""
    return extract(fetch(url), url)
