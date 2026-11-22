"""Turn a file on disk into plain text.

This is the "plumbing" stage of a RAG: no understanding of meaning yet,
but every later stage depends on it. Garbage text in = garbage answers out.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PasswordType, PdfReader
from pypdf.errors import PdfReadError

from localmind.ocr import OcrPage

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
# Images are not readable yet, but they are not "junk" either: stage 2b (OCR)
# will read them, so ingestion sets them aside as "needs OCR".
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})

# Order matters. UTF-8 is strict: random bytes almost never form valid UTF-8,
# so if decoding succeeds it is very likely right. cp1252 (old Windows default)
# accepts nearly any byte, so it must come after. latin-1 accepts *every* byte
# and is the last-resort fallback that never fails.
_FALLBACK_ENCODINGS = ("utf-8", "cp1252", "latin-1")
_BOMS = {
    b"\xef\xbb\xbf": "utf-8-sig",
    b"\xff\xfe": "utf-16",
    b"\xfe\xff": "utf-16",
}


class UnsupportedFormat(Exception):
    """The file type is not handled (yet)."""


class ExtractionError(Exception):
    """The file type is supported but this file could not be read."""


class EncryptedPdf(ExtractionError):
    """The PDF is password-protected and the empty password does not open it."""


@dataclass(frozen=True)
class ExtractedText:
    """Extracted content, one string per page (a text file is a single page)."""

    pages: tuple[str, ...]
    kind: str  # "text", "pdf" or "image"
    encoding: str | None = None
    # 1-based numbers of pages that draw a raster image (scans, photos).
    image_pages: frozenset[int] = frozenset()
    # 1-based page -> what OCR read there (filled by ingestion, never by extract).
    ocr: Mapping[int, OcrPage] = field(default_factory=dict)

    @property
    def text(self) -> str:
        # Form feed = page break: later stages can still cite page numbers.
        return "\f".join(self.pages)

    @property
    def page_count(self) -> int | None:
        return len(self.pages) if self.kind in ("pdf", "image") else None


def extract(path: Path) -> ExtractedText:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _extract_text_file(path)
    if suffix in PDF_SUFFIXES:
        return _extract_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _extract_image(path)
    raise UnsupportedFormat(f"unsupported file type: {suffix or '(none)'}")


def _decode(raw: bytes) -> tuple[str, str]:
    """Guess the encoding of a text file and decode it."""
    for bom, encoding in _BOMS.items():
        if raw.startswith(bom):
            return raw.decode(encoding), encoding
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 decodes any byte sequence")


def _extract_text_file(path: Path) -> ExtractedText:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"cannot read file: {exc}") from exc
    text, encoding = _decode(raw)
    # Normalise the reported name: "utf-8-sig" is still UTF-8 for the user.
    reported = "utf-8" if encoding == "utf-8-sig" else encoding
    return ExtractedText(pages=(text,), kind="text", encoding=reported)


def _extract_pdf(path: Path) -> ExtractedText:
    try:
        reader = PdfReader(path)
        # Many PDFs are "encrypted" only to restrict printing/copying: they
        # open with an empty user password. Only a real password stops us.
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise EncryptedPdf("PDF is password-protected")
        pages = tuple(page.extract_text() or "" for page in reader.pages)
        image_pages = frozenset(
            n for n, page in enumerate(reader.pages, start=1) if _has_image(page)
        )
    except EncryptedPdf:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise ExtractionError(f"cannot parse PDF: {exc}") from exc
    return ExtractedText(pages=pages, kind="pdf", image_pages=image_pages)


def _has_image(page) -> bool:
    """True if the page's resources hold an image XObject.

    A scanner app often adds a tiny text layer ("Scanned with ...") on top of
    the picture: counting characters alone would mistake the scan for text.
    """
    resources = page.get("/Resources")
    if resources is None:
        return False
    xobjects = resources.get_object().get("/XObject")
    if xobjects is None:
        return False
    xobjects = xobjects.get_object()
    return any(xobjects[name].get_object().get("/Subtype") == "/Image" for name in xobjects)


def _extract_image(path: Path) -> ExtractedText:
    """A photo or a scan saved as an image: no text yet, one page per frame
    (a multi-page TIFF fax has several). OCR fills the text in later."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.load()  # decode now: a truncated file fails here, not later
            frames = getattr(img, "n_frames", 1)
    except (UnidentifiedImageError, OSError, ValueError, TypeError, SyntaxError, EOFError,
            Image.DecompressionBombError) as exc:  # too large, truncated, malformed
        raise ExtractionError(f"unreadable image: {exc.__class__.__name__}") from exc
    return ExtractedText(
        pages=("",) * frames, kind="image", image_pages=frozenset(range(1, frames + 1))
    )
