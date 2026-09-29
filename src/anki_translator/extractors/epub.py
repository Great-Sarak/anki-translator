"""EPUB extractor: .epub file (a zip) → list[Chunk] in OPF spine order.

Reads META-INF/container.xml to locate the OPF package document, then walks the
OPF spine in order, extracting paragraphs from each XHTML spine item with the
same HTML walk as url.py (extended for epub XHTML idioms — see
_XHTMLSectionParser). Position is "<spine-item href>#<anchor>".

Structural pre-filtering (#71, S3) fires at three granularities, all flagging
chunks via PREFILTER_METADATA_KEY so they bypass the LLM and land in trimmed
(#67):

1. Spine-item level — an entire item is structural matter. Signals: the EPUB 3
   manifest ``properties="nav"`` marker, and conservative filename-stem
   heuristics (titlepage/imprint/dedication/preface/colophon/endnotes/…). The
   bare stem "index" is deliberately NOT matched: in journal-article epubs
   (SAGE) the whole article body is ``xhtml/index.xhtml``. Book indexes are
   caught at section level instead (epub:type="index" / a heading).
2. Section level — the enclosing <section>'s epub:type / DPUB ``role="doc-*"``
   token is structural (bibliography, endnotes, abstract, …), or the enclosing
   heading text is recognized by url.py's _boilerplate_kind plus the epub-only
   additions "abstract" and "about the author". This is what reaches structural
   matter inside single-item article epubs, where abstract/references/
   acknowledgments live as sections of the one body file.
3. Element level — <figcaption> text is flagged 'figure-caption'. Figure/table
   wrapper items (common in article epubs) contain only a caption plus an
   <img>/<table>; the caption is never card material, and table cell text is
   not extracted at all (consistent with url.py's paragraph-only walk).

The EPUB 2 toc.ncx is media-type application/x-dtbncx+xml, not XHTML, so it is
never a spine extraction target; only application/xhtml+xml spine items are
read. Chunks shorter than MIN_CHUNK_CHARS are dropped silently, same rationale
as pdf.py (page-number/caption noise below the floor is not worth an auditable
trimmed entry).
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from ..chunk import Chunk
from ..classifier import PREFILTER_METADATA_KEY
from . import ExtractionError
from .url import _ParagraphAndAnchorParser, _boilerplate_kind

# Same floor as pdf.py: shorter fragments are page-number / short-caption noise.
MIN_CHUNK_CHARS = 30

_XHTML_MEDIA_TYPE = "application/xhtml+xml"

# epub:type vocabulary (EPUB 3 semantics / z3998 / DPUB) that marks a section or
# whole item as structural chaff. Deliberately narrow: the generic
# "frontmatter"/"backmatter" classes are NOT here — a translator's introduction
# is frontmatter by class but substantive body text (Sun Tzu sample), and the
# conservative bias is to let borderline content through to the classifier.
# "introduction" and "chapter" are likewise body, not chaff.
_STRUCTURAL_EPUB_TYPES = {
    "titlepage": "title-page",
    "halftitlepage": "title-page",
    "imprint": "copyright",
    "copyright-page": "copyright",
    "dedication": "dedication",
    "preface": "preface",
    "colophon": "colophon",
    "bibliography": "bibliography",
    "endnotes": "endnotes",
    "footnotes": "endnotes",
    "toc": "table-of-contents",
    "index": "index",
    "abstract": "abstract",
    "acknowledgments": "acknowledgments",
}

# DPUB ARIA roles carrying the same semantics on <section role="doc-…">.
_STRUCTURAL_DOC_ROLES = {
    "doc-bibliography": "bibliography",
    "doc-endnotes": "endnotes",
    "doc-footnotes": "endnotes",
    "doc-dedication": "dedication",
    "doc-colophon": "colophon",
    "doc-preface": "preface",
    "doc-toc": "table-of-contents",
    "doc-index": "index",
    "doc-abstract": "abstract",
    "doc-acknowledgments": "acknowledgments",
}


def _localname(tag: str) -> str:
    """Namespace-agnostic XML localname ('{ns}item' → 'item')."""
    return tag.rsplit("}", 1)[-1]


def _href_stem_kind(href: str) -> str | None:
    """Filename-stem heuristics for whole-item structural matter.

    Fallback for epubs whose XHTML carries no epub:type semantics. Conservative:
    matches whole stems or clear suffixes ("translators-preface" → preface), and
    deliberately does NOT match the bare stem "index" — in journal-article
    epubs the article body itself is xhtml/index.xhtml, so name-based index
    filtering would gut the content. Book indexes are caught by epub:type /
    heading-text signals instead.
    """
    stem = href.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    joined = re.sub(r"[^a-z0-9]", "", stem)
    if not joined:
        return None
    if stem in ("toc", "nav"):
        return "table-of-contents"
    if joined.endswith("titlepage"):
        return "title-page"
    if joined in ("imprint", "copyright", "uncopyright", "copyrightpage"):
        return "copyright"
    if joined.endswith("dedication"):
        return "dedication"
    if joined.endswith("colophon"):
        return "colophon"
    if joined.startswith(("endnotes", "footnotes")):
        return "endnotes"
    if joined in ("bibliography", "references"):
        return "bibliography"
    if joined.endswith("preface"):
        return "preface"
    if joined in ("abouttheauthor", "abouttheauthors"):
        return "author-info"
    return None


def _section_title_kind(section_title: str) -> str | None:
    """EPUB-aware extension of url.py's _boilerplate_kind.

    Adds one structural heading common in book back matter that the URL map
    does not cover: 'About the author'. Everything else defers to the shared
    url.py vocabulary, which is the seam that makes section-level structural
    matter reachable inside single-spine-item article epubs.

    Note: 'Abstract' is deliberately NOT matched here. The section-title state
    is sticky (nearest preceding heading), so a heading-text rule flags every
    paragraph until the next heading — in minimal article epubs whose body
    carries no further headings, that sweeps the whole body into trimmed.
    Abstracts are flagged via epub:type="abstract" / role="doc-abstract" on the
    enclosing section instead, which all observed journal epubs carry.
    """
    t = section_title
    if not t:
        return None
    if "about the author" in t:
        return "author-info"
    return _boilerplate_kind(t)


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    """Locate the OPF package document via META-INF/container.xml."""
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError as e:
        raise ExtractionError("not an EPUB: META-INF/container.xml missing") from e
    try:
        root = ElementTree.fromstring(container)
    except ElementTree.ParseError as e:
        raise ExtractionError(f"could not parse META-INF/container.xml: {e}") from e
    for elem in root.iter():
        if _localname(elem.tag) == "rootfile" and elem.get("full-path"):
            return elem.get("full-path") or ""
    raise ExtractionError("not an EPUB: container.xml declares no rootfile")


def _parse_opf(
    zf: zipfile.ZipFile, opf_path: str
) -> tuple[str | None, list[tuple[str, frozenset[str]]]]:
    """Parse the OPF: return (dc:title, [(absolute-href, properties), …]) in spine order.

    Only XHTML spine items are returned — the toc.ncx (media-type
    application/x-dtbncx+xml) and any other non-XHTML spine entries are dropped
    here, which is how the EPUB 2 NCX is 'routed out': it is never read.
    """
    try:
        opf = zf.read(opf_path)
    except KeyError as e:
        raise ExtractionError(f"OPF package document missing from archive: {opf_path}") from e
    try:
        root = ElementTree.fromstring(opf)
    except ElementTree.ParseError as e:
        raise ExtractionError(f"could not parse OPF {opf_path}: {e}") from e

    title: str | None = None
    manifest: dict[str, tuple[str, str, frozenset[str]]] = {}
    idrefs: list[str] = []
    for elem in root.iter():
        name = _localname(elem.tag)
        if name == "title" and title is None:
            text = (elem.text or "").strip()
            if text:
                title = text
        elif name == "item":
            item_id = elem.get("id")
            href = elem.get("href")
            if item_id and href:
                properties = frozenset((elem.get("properties") or "").split())
                manifest[item_id] = (href, elem.get("media-type") or "", properties)
        elif name == "itemref":
            idref = elem.get("idref")
            if idref:
                idrefs.append(idref)

    opf_dir = posixpath.dirname(opf_path)
    spine_items: list[tuple[str, frozenset[str]]] = []
    for idref in idrefs:
        entry = manifest.get(idref)
        if entry is None:
            continue
        href, media_type, properties = entry
        if media_type != _XHTML_MEDIA_TYPE:
            continue
        absolute = posixpath.normpath(posixpath.join(opf_dir, unquote(href)))
        spine_items.append((absolute, properties))
    if not spine_items:
        raise ExtractionError(f"OPF {opf_path} declares no XHTML spine items")
    return title, spine_items


def _item_kind(href: str, properties: frozenset[str]) -> str | None:
    """Whole-spine-item structural kind, or None for ordinary content items."""
    if "nav" in properties:
        return "table-of-contents"
    return _href_stem_kind(href)


class _XHTMLSectionParser(_ParagraphAndAnchorParser):
    """url.py's paragraph walk extended for epub XHTML idioms.

    Differences from the base walk:
    - <div role="paragraph"> and <div role="listitem"> count as paragraph
      containers — journal-article epubs (SAGE) mark up body prose and
      reference entries as role'd divs and carry no <p> tags at all.
    - <figcaption> is captured as a paragraph and marked so the caller can flag
      it 'figure-caption'.
    - <section epub:type="…"> / role="doc-…" tokens are tracked on a stack so
      every paragraph enclosed by a structural section inherits its kind,
      however the publisher organized the files.
    - A <section id="…"> updates the current anchor: in article epubs the id
      lives on the section, not on the heading, and Position wants it.

    Emits self.records: (text, anchor, section_title, type_kind, is_caption).
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, str | None, str, str | None, bool]] = []
        self._section_stack: list[tuple[frozenset[str], str]] = []
        self._paragraph_origin: str | None = None  # None | "div" | "figcaption"
        self._div_depth = 0  # nested plain <div> inside an open div-paragraph

    # -- emit hook: base class calls this for <p> closes; we use it for all --

    def _current_type_kind(self) -> str | None:
        for tokens, role in reversed(self._section_stack):
            for token in tokens:
                kind = _STRUCTURAL_EPUB_TYPES.get(token)
                if kind:
                    return kind
            kind = _STRUCTURAL_DOC_ROLES.get(role)
            if kind:
                return kind
        return None

    def _emit(self, text: str) -> None:
        self.records.append(
            (
                text,
                self._current_anchor,
                self._current_section_title,
                self._current_type_kind(),
                self._paragraph_origin == "figcaption",
            )
        )

    # -- tag handling ---------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            attr = dict(attrs)
            tokens = frozenset((attr.get("epub:type") or "").split())
            self._section_stack.append((tokens, attr.get("role") or ""))
            if attr.get("id"):
                self._current_anchor = attr["id"]
            return  # base ignores <section>
        if self._skip_depth == 0 and tag == "div" and self._in_paragraph and self._paragraph_origin == "div":
            # Nested plain div inside a div-paragraph: balance the close tag.
            self._div_depth += 1
            return
        if self._skip_depth == 0 and not self._in_paragraph:
            if tag == "figcaption":
                self._in_paragraph = True
                self._paragraph_buffer = []
                self._paragraph_origin = "figcaption"
                return
            if tag == "div":
                role = dict(attrs).get("role")
                if role in ("paragraph", "listitem"):
                    self._in_paragraph = True
                    self._paragraph_buffer = []
                    self._paragraph_origin = "div"
                    self._div_depth = 0
                    return
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            if self._section_stack:
                self._section_stack.pop()
            return  # base ignores </section>
        if self._in_paragraph and self._paragraph_origin == "div" and tag == "div":
            if self._div_depth > 0:
                self._div_depth -= 1
                return
            self._emit("".join(self._paragraph_buffer))
            self._in_paragraph = False
            self._paragraph_buffer = []
            self._paragraph_origin = None
            return
        if self._in_paragraph and self._paragraph_origin == "figcaption" and tag == "figcaption":
            self._emit("".join(self._paragraph_buffer))
            self._in_paragraph = False
            self._paragraph_buffer = []
            self._paragraph_origin = None
            return
        super().handle_endtag(tag)


def extract(path: Path | str) -> list[Chunk]:
    """Extract chunks from an EPUB file, in OPF spine order.

    Returns one Chunk per paragraph (or role-paragraph div / figcaption) of
    each XHTML spine item. Source is the OPF dc:title, falling back to the
    filename; Position is "<spine-item href>#<anchor>" (anchor optional).
    Structural matter (title/copyright/dedication pages, nav/TOC, indexes,
    bibliographies, endnotes, about-the-author, colophons, abstracts, figure
    captions) is flagged via PREFILTER_METADATA_KEY so the pipeline routes it
    to trimmed without an LLM call (#67/#71); fragments shorter than
    MIN_CHUNK_CHARS are dropped.
    """
    p = Path(path)
    if not p.exists():
        raise ExtractionError(f"EPUB not found: {p}")
    try:
        zf = zipfile.ZipFile(p)
    except zipfile.BadZipFile as e:
        raise ExtractionError(f"could not open EPUB {p}: not a zip archive ({e})") from e

    with zf:
        opf_path = _find_opf_path(zf)
        title, spine_items = _parse_opf(zf, opf_path)
        source_label = title or p.name

        chunks: list[Chunk] = []
        for href, properties in spine_items:
            item_kind = _item_kind(href, properties)
            try:
                data = zf.read(href)
            except KeyError:
                continue  # malformed epub: spine item absent from the archive
            parser = _XHTMLSectionParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            parser.close()

            for text, anchor, section_title, type_kind, is_caption in parser.records:
                text = text.strip()
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                kind = (
                    item_kind
                    or type_kind
                    or ("figure-caption" if is_caption else None)
                    or _section_title_kind(section_title)
                )
                metadata: dict[str, object] = {
                    "title": source_label,
                    "filename": p.name,
                    "spine_item": href,
                    "anchor": f"#{anchor}" if anchor else "",
                }
                if kind:
                    metadata[PREFILTER_METADATA_KEY] = kind
                chunks.append(
                    Chunk(
                        text=text,
                        source=source_label,
                        position=f"{href}#{anchor}" if anchor else href,
                        source_type="epub",
                        metadata=metadata,
                    )
                )

    if not chunks:
        raise ExtractionError(
            f"no paragraphs extracted from {p} — EPUB may contain no XHTML text content"
        )
    return chunks
