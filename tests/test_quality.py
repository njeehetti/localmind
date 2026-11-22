from dataclasses import replace

from localmind.extractors import ExtractedText
from localmind.ocr import OcrPage
from localmind.quality import (
    QualityConfig,
    Reason,
    assess,
    is_garbled,
    is_machine_log,
    needs_ocr_pages,
)


def pdf(*pages):
    return ExtractedText(pages=pages, kind="pdf")


def txt(s):
    return ExtractedText(pages=(s,), kind="text")


GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."
IBAN = " IBAN FR7630006000011234567890189"


def test_clean_document_accepted():
    a = assess(pdf(GOOD, GOOD), QualityConfig())
    assert a.quarantine_reason is None and not a.has_set_aside


def test_scanned_page_set_aside_others_kept():
    a = assess(pdf(GOOD, "", GOOD), QualityConfig())
    assert [p.reason for p in a.pages] == [None, Reason.NEEDS_OCR, None]
    assert [p.page_no for p in a.pages] == [1, 2, 3]
    assert a.quarantine_reason is None and a.has_set_aside
    assert a.usable_text == GOOD + "\f" + GOOD


def test_page_with_only_page_number_needs_ocr():  # Review Focus 1
    assert assess(pdf(GOOD, "12"), QualityConfig()).pages[1].reason is Reason.NEEDS_OCR


def test_fully_scanned_pdf_quarantined_needs_ocr():
    assert assess(pdf("", " "), QualityConfig()).quarantine_reason is Reason.NEEDS_OCR


def test_whitespace_text_file_is_empty():
    assert assess(txt(" \n\t "), QualityConfig()).quarantine_reason is Reason.EMPTY


def test_too_short():
    assert assess(txt("appeler"), QualityConfig()).quarantine_reason is Reason.TOO_SHORT


def test_garbled_page():
    junk = "\ufffd\ufffd\x07\x08\ufffd\ufffd\x01\x02\x03\ufffd\ufffd\x04\x05\x06\ufffd\ufffd\x0e\x0f\x10\x11 ab"
    assert assess(pdf(GOOD, junk), QualityConfig()).pages[1].reason is Reason.GARBLED


def test_symbols_are_not_garbled():  # Review Focus 2
    assert not is_garbled("Total : 30 € — TVA 20 % — 25 °C — n° 12 (réf. A/B)", QualityConfig())


def test_replacement_chars_are_garbled():
    assert is_garbled("\ufffd" * 10 + "ab", QualityConfig())


def test_machine_log():
    log = "\n".join(f"2024-03-01 12:00:{i:02d} INFO worker started job {i}" for i in range(25))
    assert assess(txt(log), QualityConfig()).quarantine_reason is Reason.MACHINE_LOG


def test_diary_with_dated_lines_is_not_a_log():  # final review Important #4
    diary = "\n".join(f"2024-03-{i + 1:02d} Courses au marché, 45,20 €, avec Marie" for i in range(24))
    assert not is_machine_log(diary, QualityConfig())


def test_scan_with_watermark_text_layer_needs_ocr():  # final review Important #1
    pages = ("Scanned with CamScanner",) * 3
    scan = ExtractedText(pages=pages, kind="pdf", image_pages=frozenset({1, 2, 3}))
    assert assess(scan, QualityConfig()).quarantine_reason is Reason.NEEDS_OCR


def test_image_page_with_real_text_layer_is_usable():
    long_text = GOOD * 5  # an OCR-ed scan or a letter with a logo
    doc = ExtractedText(pages=(long_text,), kind="pdf", image_pages=frozenset({1}))
    assert assess(doc, QualityConfig()).quarantine_reason is None


def test_short_log_like_text_is_not_a_log():
    log = "\n".join(f"2024-03-01 INFO line {i}" for i in range(5))
    assert not is_machine_log(log, QualityConfig())


def test_accept_skips_quality_rules_but_not_technical():
    assert assess(txt("appeler"), QualityConfig(), accepted=True).quarantine_reason is None
    assert assess(pdf(""), QualityConfig(), accepted=True).quarantine_reason is Reason.NEEDS_OCR


def test_sensitive_flagged_by_default():
    a = assess(txt(GOOD + IBAN), QualityConfig())
    assert a.quarantine_reason is None and a.pages[0].spans[0].kind == "iban"


def test_sensitive_quarantine_policy_and_accept_override():
    doc = txt(GOOD + IBAN)
    cfg = QualityConfig(sensitive_policy="quarantine")
    assert assess(doc, cfg).quarantine_reason is Reason.SENSITIVE
    assert assess(doc, cfg, accepted=True).quarantine_reason is None


def test_set_aside_pages_are_not_scanned_for_sensitive_data():
    a = assess(pdf(GOOD, "\x01" * 30 + IBAN), QualityConfig())
    assert a.pages[1].reason is Reason.GARBLED and a.pages[1].spans == ()


# --- stage 2b: OCR page rules ---------------------------------------------

SCAN = ExtractedText(pages=("",), kind="pdf", image_pages=frozenset({1}))


def test_ocr_text_replaces_needs_ocr_page():
    a = assess(replace(SCAN, ocr={1: OcrPage(GOOD, 0.95, "fake")}), QualityConfig())
    assert a.quarantine_reason is None
    assert a.pages[0].source == "ocr" and a.pages[0].ocr_confidence == 0.95
    assert a.pages[0].ocr_engine == "fake"
    assert a.usable_text == GOOD


def test_ocr_low_confidence_page_is_set_aside():
    a = assess(replace(SCAN, ocr={1: OcrPage(GOOD, 0.58, "fake")}), QualityConfig())
    assert a.quarantine_reason is Reason.OCR_LOW_CONFIDENCE


def test_ocr_failure_page_is_set_aside():
    a = assess(replace(SCAN, ocr={1: OcrPage("", -1.0, "fake")}), QualityConfig())
    assert a.quarantine_reason is Reason.OCR_FAILED


def test_ocr_short_text_is_blank_and_does_not_make_partial():
    doc = ExtractedText(
        pages=(GOOD, ""), kind="pdf", image_pages=frozenset({2}),
        ocr={2: OcrPage("3", 0.99, "fake")},
    )
    a = assess(doc, QualityConfig())
    assert a.pages[1].is_blank and not a.has_set_aside and a.quarantine_reason is None
    assert a.usable_text == GOOD


def test_all_blank_pages_quarantine_as_ocr_no_text():
    a = assess(replace(SCAN, ocr={1: OcrPage("", 0.0, "fake")}), QualityConfig())
    assert a.quarantine_reason is Reason.OCR_NO_TEXT


def test_blank_and_low_confidence_quarantine_with_the_real_problem():
    doc = ExtractedText(
        pages=("", "", ""), kind="pdf", image_pages=frozenset({1, 2, 3}),
        ocr={1: OcrPage("", 0.0, "f"), 2: OcrPage("", 0.0, "f"), 3: OcrPage(GOOD, 0.4, "f")},
    )
    assert assess(doc, QualityConfig()).quarantine_reason is Reason.OCR_LOW_CONFIDENCE


def test_sensitive_data_found_in_ocr_text():
    rib = GOOD + IBAN
    a = assess(replace(SCAN, ocr={1: OcrPage(rib, 0.9, "fake")}), QualityConfig())
    assert [s.kind for s in a.pages[0].spans] == ["iban"]


def test_image_kind_without_ocr_needs_ocr():
    img = ExtractedText(pages=("",), kind="image", image_pages=frozenset({1}))
    assert assess(img, QualityConfig()).quarantine_reason is Reason.NEEDS_OCR


def test_needs_ocr_pages_lists_scanned_pages_only():
    doc = ExtractedText(
        pages=(GOOD * 5, "Scanned with CamScanner"), kind="pdf", image_pages=frozenset({2})
    )
    assert needs_ocr_pages(doc, QualityConfig()) == frozenset({2})


def test_text_files_never_need_ocr():
    assert needs_ocr_pages(txt(""), QualityConfig()) == frozenset()
