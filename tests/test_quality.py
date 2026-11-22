from localmind.extractors import ExtractedText
from localmind.quality import QualityConfig, Reason, assess, is_garbled, is_machine_log


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
