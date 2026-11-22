"""Turn pages into pictures, for OCR.

An OCR engine reads images, not PDFs. pypdf extracts text but cannot *draw* a
page; pypdfium2 (the PDF engine of Chrome, PDFium) can. Image files are opened
with Pillow.

Pages are rendered one at a time (a generator): a 300 dpi A4 page is about
25 MB in memory, so a 100-page scan must never be rendered all at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image, ImageOps

from localmind.extractors import PDF_SUFFIXES

_POINTS_PER_INCH = 72  # PDF sizes are in points: 1/72 inch


def render_pages(
    path: Path, page_numbers: Iterable[int], dpi: int = 300
) -> Iterator[tuple[int, Image.Image]]:
    """Yield (page number, RGB image) for each requested 1-based page."""
    if path.suffix.lower() in PDF_SUFFIXES:
        yield from _render_pdf(path, page_numbers, dpi)
    else:
        yield from _render_image(path, page_numbers)


def _render_pdf(path: Path, page_numbers: Iterable[int], dpi: int) -> Iterator[tuple[int, Image.Image]]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        for page_no in page_numbers:
            page = pdf[page_no - 1]
            try:
                bitmap = page.render(scale=dpi / _POINTS_PER_INCH)
                yield page_no, bitmap.to_pil().convert("RGB")
            finally:
                page.close()
    finally:
        pdf.close()


def _render_image(path: Path, page_numbers: Iterable[int]) -> Iterator[tuple[int, Image.Image]]:
    with Image.open(path) as img:
        for page_no in page_numbers:
            img.seek(page_no - 1)  # multi-page TIFF: one frame per page
            # Phones store "rotate me" in EXIF instead of rotating the pixels.
            yield page_no, ImageOps.exif_transpose(img).convert("RGB")
