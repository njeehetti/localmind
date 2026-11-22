import pytest

from localmind.extractors import ExtractionError, UnsupportedFormat, extract
from tests.pdf_factory import make_pdf


def test_txt_utf8(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Facture d'électricité", encoding="utf-8")
    result = extract(f)
    assert result.text == "Facture d'électricité"
    assert result.kind == "text"
    assert result.encoding == "utf-8"


def test_txt_utf8_with_bom(tmp_path):
    f = tmp_path / "bom.txt"
    f.write_bytes("﻿Bonjour".encode("utf-8"))
    result = extract(f)
    assert result.text == "Bonjour"  # the BOM must not leak into the text


def test_txt_windows_1252_is_not_garbled(tmp_path):
    f = tmp_path / "old.txt"
    f.write_bytes("Reçu n°12 : 30 €".encode("cp1252"))
    result = extract(f)
    assert result.text == "Reçu n°12 : 30 €"  # and not "ReÃ§u"
    assert result.encoding == "cp1252"


def test_txt_utf16(tmp_path):
    f = tmp_path / "notepad.txt"
    f.write_text("Hé ho", encoding="utf-16")
    assert extract(f).text == "Hé ho"


def test_markdown_is_read_as_text(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title", encoding="utf-8")
    assert extract(f).kind == "text"


def test_pdf_text_and_page_count(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(make_pdf(["Invoice 2024", "Total 42 EUR"]))
    result = extract(f)
    assert result.kind == "pdf"
    assert result.page_count == 2
    assert "Invoice 2024" in result.text
    assert "Total 42 EUR" in result.text


def test_pdf_without_text_layer_gives_empty_text(tmp_path):
    # Simulates a scanned PDF: pages exist but contain no text.
    f = tmp_path / "scan.pdf"
    f.write_bytes(make_pdf(["", ""]))
    result = extract(f)
    assert result.page_count == 2
    assert result.text.strip() == ""


def test_corrupted_pdf_raises_extraction_error(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf at all")
    with pytest.raises(ExtractionError):
        extract(f)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(UnsupportedFormat):
        extract(f)
