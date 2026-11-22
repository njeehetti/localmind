import pytest
from PIL import Image

from localmind.extractors import EncryptedPdf, ExtractionError, UnsupportedFormat, extract
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
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK\x03\x04")
    with pytest.raises(UnsupportedFormat):
        extract(f)


def test_text_file_is_one_page(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x y z", encoding="utf-8")
    assert extract(f).pages == ("x y z",)


def test_pdf_pages_are_separate(tmp_path):
    f = tmp_path / "d.pdf"
    f.write_bytes(make_pdf(["One", "Two"]))
    r = extract(f)
    assert len(r.pages) == 2 and "One" in r.pages[0] and "Two" in r.pages[1]


def test_pdf_pages_with_images_are_reported(tmp_path):
    f = tmp_path / "scan.pdf"
    f.write_bytes(make_pdf(["Scanned with CamScanner", "Plain text page here"], image_pages={1}))
    assert extract(f).image_pages == frozenset({1})


def test_text_file_has_no_image_pages(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x y z", encoding="utf-8")
    assert extract(f).image_pages == frozenset()


def test_encrypted_pdf_raises_encrypted(tmp_path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(100, 100)
    w.encrypt(user_password="secret", algorithm="AES-256")
    f = tmp_path / "locked.pdf"
    with f.open("wb") as s:
        w.write(s)
    with pytest.raises(EncryptedPdf):
        extract(f)


def test_pdf_encrypted_with_empty_password_is_readable(tmp_path):
    from pypdf import PdfReader, PdfWriter

    src = tmp_path / "src.pdf"
    src.write_bytes(make_pdf(["Hello world text"]))
    w = PdfWriter(clone_from=PdfReader(src))
    w.encrypt(user_password="", owner_password="o", algorithm="AES-256")
    f = tmp_path / "open.pdf"
    with f.open("wb") as s:
        w.write(s)
    assert "Hello world text" in extract(f).text


def test_png_is_a_one_page_image(tmp_path):
    f = tmp_path / "ticket.png"
    Image.new("RGB", (40, 20), "white").save(f)
    r = extract(f)
    assert (r.kind, r.page_count, r.image_pages) == ("image", 1, frozenset({1}))


def test_multi_page_tiff_has_one_page_per_frame(tmp_path):
    f = tmp_path / "fax.tif"
    frames = [Image.new("L", (40, 20), c) for c in (0, 128, 255)]
    frames[0].save(f, save_all=True, append_images=frames[1:])
    assert extract(f).page_count == 3


def test_corrupted_image_raises_extraction_error(tmp_path):
    f = tmp_path / "broken.jpg"
    f.write_bytes(b"\xff\xd8\xff not really")
    with pytest.raises(ExtractionError):
        extract(f)
