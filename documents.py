"""Turning an uploaded PDF into text the extractor can read, with page numbers.

A real user uploads a PDF, not a .txt. This module pulls out the text layer page by page,
remembers where each page starts, and can say which page a quote came from.
Scanned PDFs (photos of pages, no text layer) are detected and refused: they need OCR,
which this project does not do.
"""
import io
import unicodedata
from dataclasses import dataclass

from pypdf import PdfReader

MIN_CHARS_PER_PAGE = 200    # below this on average, the PDF is almost certainly a scan


@dataclass
class Document:
    text: str               # all pages joined with a blank line between them
    pages: list[str]        # one string per page


class NoTextLayer(ValueError):
    """The PDF has no extractable text (probably scanned)."""


def clean(text: str) -> str:
    """Small, safe fixes only. NFKC turns typographic ligatures like 'ﬁ' into 'fi';
    soft hyphens are invisible marks some PDFs put inside words."""
    return unicodedata.normalize("NFKC", text).replace("\u00ad", "")


def pdf_to_document(data: bytes) -> Document:
    reader = PdfReader(io.BytesIO(data))
    pages = [clean(page.extract_text() or "") for page in reader.pages]
    chars = sum(len(p.strip()) for p in pages)
    if not pages or chars / len(pages) < MIN_CHARS_PER_PAGE:
        raise NoTextLayer(f"{len(pages)} pages but only {chars} characters of text: "
                          f"looks like a scan, which needs OCR")
    return Document(text="\n\n".join(pages), pages=pages)


def page_of_quote(doc: Document, quote: str | None) -> int | None:
    """1-based page where the quote starts, or None if it cannot be placed.
    Uses the first 80 characters, so a quote that runs onto the next page still
    lands on the page where it begins."""
    if not quote:
        return None
    probe = " ".join(quote.split())[:80]
    for number, page in enumerate(doc.pages, start=1):
        if probe in " ".join(page.split()):
            return number
    return None