"""OCR: the engine interface and its results.

This module must stay light: it is imported by ``localmind`` itself, so it
never imports torch or docTR at module level. The real engine (docTR) lives
behind ``load_engine()``, which imports them only when called.

An OCR engine looks at a page *image* and returns the text it believes is
there, plus how sure it is. That confidence is the point: OCR text is a guess,
and localmind keeps the quality of the guess next to the text.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class OcrPage:
    """What OCR read on one page. ``confidence < 0`` means the engine failed."""

    text: str
    confidence: float
    engine: str


class OcrEngine(Protocol):
    name: str  # includes version and models, e.g. "doctr-1.1.0/db_resnet50+crnn_vgg16_bn"
    device: str  # "GPU" or "CPU"

    def read(self, image: Image.Image) -> tuple[str, float]:
        """Return (text, page confidence in [0, 1]) for one page image."""
        ...


class OcrUnavailable(RuntimeError):
    """The OCR extra is not installed (or the engine cannot start)."""


def page_confidence(words: Sequence[tuple[str, float]]) -> float:
    """Average word confidence, weighted by word length.

    A misread one-letter word ("a") should weigh less than a misread
    "Facture": long words carry most of the meaning.
    """
    total = sum(len(word) for word, _ in words)
    if total == 0:
        return 0.0
    return sum(len(word) * conf for word, conf in words) / total


def ocr_available() -> bool:
    """True if docTR is installed. Checks without importing it (importing
    torch takes seconds and hundreds of MB)."""
    return importlib.util.find_spec("doctr") is not None


def load_engine() -> OcrEngine:
    if not ocr_available():
        raise OcrUnavailable("OCR not installed: uv sync --extra ocr")
    raise OcrUnavailable("docTR engine not implemented yet")
