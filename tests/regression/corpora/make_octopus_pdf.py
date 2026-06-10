"""Regenerate octopus.pdf, the PDF-extractor regression corpus.

Run from anywhere:  python tests/regression/corpora/make_octopus_pdf.py

The PDF is committed as a binary fixture; this generator exists so the binary is
auditable and reproducible rather than opaque. Each text block is laid out in its
own rect so pymupdf's `get_text("blocks")` returns them as separate chunks,
mirroring the real Octopus PDF's title-page / body / references structure:

  block 0 — title + author affiliation  → trimmed (structural chaff)
  block 1 — a real body fact            → card candidate
  block 2 — a References entry          → trimmed (bibliography)
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

BLOCKS = [
    (
        "I know my neighbour: individual recognition in Octopus vulgaris. "
        "Gabriella Scata, Department of Marine Biology, "
        "University of Naples Federico II, Naples, Italy."
    ),
    (
        "Octopus vulgaris individuals can recognise and remember other octopuses, "
        "distinguishing familiar neighbours from unfamiliar strangers in the wild."
    ),
    (
        "References. Boal JG (2006) Social recognition: a top-down view of "
        "cephalopod behaviour. Vie et Milieu 56:69-79."
    ),
]


def build(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    top = 72.0
    for text in BLOCKS:
        rect = pymupdf.Rect(72, top, 540, top + 90)
        page.insert_textbox(rect, text, fontsize=11, fontname="helv")
        top += 120.0
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    out = Path(__file__).with_name("octopus.pdf")
    build(out)
    print(f"wrote {out}")
