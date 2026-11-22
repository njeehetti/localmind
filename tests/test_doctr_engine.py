"""The real docTR engine. Skipped when the `ocr` extra is not installed.

The first run downloads docTR's model weights once (the only network access
localmind makes); afterwards everything runs offline.
"""

import pytest

pytest.importorskip("doctr")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from localmind.ocr import load_engine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return load_engine()


def _font(size):
    """A real TrueType font: Pillow's built-in font draws accents as empty boxes."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    pytest.skip("no TrueType font with accents available")


def test_doctr_reads_a_rendered_french_sentence(engine):
    img = Image.new("RGB", (1700, 200), "white")
    ImageDraw.Draw(img).text(
        (20, 60), "Facture numéro 42 payée le 15 mars, reçu à Orléans", fill="black", font=_font(44)
    )
    text, conf = engine.read(img)
    assert "numéro" in text and "Orléans" in text and "42" in text
    assert conf > 0.7
    assert engine.name == f"doctr-{engine.name.split('-', 1)[1]}"
    assert not engine.name.startswith("doctr-v")
    assert engine.name.endswith("/db_resnet50+crnn_vgg16_bn")
    assert engine.device in {"GPU", "CPU"}


def test_doctr_finds_nothing_on_a_blank_page(engine):
    text, conf = engine.read(Image.new("RGB", (800, 1000), "white"))
    assert text.strip() == "" and conf == 0.0
