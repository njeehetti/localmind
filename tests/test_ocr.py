import subprocess
import sys

import pytest

from localmind.ocr import OcrUnavailable, load_engine, ocr_available, page_confidence


def test_page_confidence_weights_long_words():
    assert page_confidence([("Facture", 0.9), ("a", 0.1)]) == pytest.approx((7 * 0.9 + 0.1) / 8)


def test_page_confidence_without_words_is_zero():
    assert page_confidence([]) == 0.0


def test_ocr_available_is_a_bool():
    assert isinstance(ocr_available(), bool)


def test_load_engine_without_doctr_raises(monkeypatch):
    monkeypatch.setattr("localmind.ocr.ocr_available", lambda: False)
    with pytest.raises(OcrUnavailable):
        load_engine()


def test_importing_localmind_does_not_import_torch():
    code = (
        "import sys, localmind, localmind.cli; "
        "print('torch' in sys.modules, 'doctr' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
