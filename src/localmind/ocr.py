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
import sys
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


DET_ARCH = "db_resnet50"  # finds where the text is (boxes around words)
RECO_ARCH = "crnn_vgg16_bn"  # reads each box (French + English characters)


class DoctrEngine:
    """docTR: a two-stage neural OCR running on PyTorch.

    1. *Detection*: a segmentation network looks at the whole page and outputs
       boxes around words.
    2. *Recognition*: each box is cropped and a second network reads it,
       character by character, with a confidence per word.

    The model is loaded once (slow: weights to read, GPU to warm up) and then
    used for every page.
    """

    def __init__(self) -> None:
        import doctr
        import torch
        from doctr.models import ocr_predictor

        version = doctr.__version__.removeprefix("v")  # docTR reports "v1.1.0"
        self.name = f"doctr-{version}/{DET_ARCH}+{RECO_ARCH}"
        self._torch = torch
        model = ocr_predictor(det_arch=DET_ARCH, reco_arch=RECO_ARCH, pretrained=True)
        if torch.cuda.is_available():
            self.device = "GPU"
            model = model.to(torch.device("cuda"))
        else:
            self.device = "CPU"
            print("note: no CUDA GPU found, OCR runs on the CPU (slower).", file=sys.stderr)
        self._model = model.eval()

    def read(self, image: Image.Image) -> tuple[str, float]:
        import numpy as np

        array = np.asarray(image.convert("RGB"))  # (height, width, 3) uint8
        with self._torch.inference_mode():
            page = self._model([array]).pages[0]
        words: list[tuple[str, float]] = []
        blocks = []
        for block in page.blocks:
            lines = []
            for line in block.lines:
                line_words = [(w.value, float(w.confidence)) for w in line.words]
                words.extend(line_words)
                lines.append(" ".join(value for value, _ in line_words))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), page_confidence(words)


def load_engine() -> OcrEngine:
    if not ocr_available():
        raise OcrUnavailable("OCR not installed: uv sync --extra ocr")
    try:
        return DoctrEngine()
    except Exception as exc:  # e.g. weights download failed, CUDA error
        raise OcrUnavailable(f"OCR engine failed to start: {exc}") from exc
