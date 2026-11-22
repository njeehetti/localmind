"""Stage 2b: OCR inside the ingestion path, with a scripted fake engine."""

import pytest
from PIL import Image

from localmind.catalog import Catalog, Decision
from localmind.hashing import file_sha256
from localmind.ingest import Outcome, apply_decision, ingest, retry, run_ocr
from localmind.quality import Reason
from tests.pdf_factory import make_pdf

GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."
GOOD2 = "Contrat de location signé le 12 avril 2023 entre les parties."
GARBLED = "\x07\x08\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a ab"


class FakeEngine:
    name = "fake-1"
    device = "GPU"

    def __init__(self, script, raise_on=None, exc=RuntimeError):
        self.script = list(script)
        self.raise_on = raise_on
        self.exc = exc
        self.calls = 0

    def read(self, image):
        self.calls += 1
        if self.calls == self.raise_on:
            raise self.exc("boom")
        return self.script.pop(0)


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "catalog.db") as cat:
        yield cat


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def outcomes(report):
    return {r.path.name: r.outcome for r in report.results}


def reasons(report):
    return {r.path.name: r.reason for r in report.results}


def test_scanned_pdf_is_read_and_indexed(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf(["", ""]))
    engine = FakeEngine([(GOOD, 0.95), (GOOD2, 0.9)])
    report = ingest(inbox, catalog, engine=engine)
    assert outcomes(report) == {"scan.pdf": Outcome.NEW} and engine.calls == 2
    assert catalog.stats().ocr_pages == 2
    text = catalog.document_for_path((inbox / "scan.pdf").resolve()).text
    assert GOOD in text and GOOD2 in text


def test_only_needs_ocr_pages_are_read(catalog, inbox):
    (inbox / "mixed.pdf").write_bytes(make_pdf([GOOD * 3, ""]))
    engine = FakeEngine([(GOOD2, 0.95)])
    report = ingest(inbox, catalog, engine=engine)
    assert engine.calls == 1 and outcomes(report) == {"mixed.pdf": Outcome.NEW}


def test_photo_becomes_a_one_page_document(catalog, inbox):
    Image.new("RGB", (80, 40), "white").save(inbox / "ticket.png")
    report = ingest(inbox, catalog, engine=FakeEngine([(GOOD, 0.91)]))
    assert outcomes(report) == {"ticket.png": Outcome.NEW}
    assert catalog.document_for_path((inbox / "ticket.png").resolve()).kind == "image"


def test_without_engine_scans_stay_quarantined(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf(["", "", ""]))
    report = ingest(inbox, catalog)
    assert reasons(report) == {"scan.pdf": Reason.NEEDS_OCR}
    assert catalog.ocr_pending().pages == 3


def test_cache_is_reused_by_copies_and_retry(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf([""]))
    engine = FakeEngine([(GOOD, 0.95)])
    ingest(inbox, catalog, engine=engine)
    (inbox / "copy").mkdir()
    (inbox / "copy" / "scan.pdf").write_bytes((inbox / "scan.pdf").read_bytes())
    ingest(inbox, catalog, engine=engine)
    retry(catalog, engine=engine)
    assert engine.calls == 1


def test_cached_ocr_applies_without_engine(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf([""]))
    ingest(inbox, catalog)  # quarantined: no engine
    run_ocr(catalog, FakeEngine([(GOOD, 0.95)]))
    retry(catalog)  # no engine: the cached text still applies
    assert catalog.stats().ocr_pages == 1
    assert catalog.quarantined_files() == []


def test_accept_keeps_ocr_pages(catalog, inbox):
    path = inbox / "contract.pdf"
    path.write_bytes(make_pdf([GOOD + " " + GARBLED * 3, ""]))
    ingest(inbox, catalog, engine=FakeEngine([(GOOD2, 0.95)]))
    apply_decision(path, Decision.ACCEPT, catalog)
    assert GOOD2 in catalog.document_for_path(path.resolve()).text


def test_low_confidence_page_is_set_aside(catalog, inbox):
    (inbox / "blurry.pdf").write_bytes(make_pdf([GOOD, ""]))
    report = ingest(inbox, catalog, engine=FakeEngine([(GOOD2, 0.58)]))
    assert outcomes(report) == {"blurry.pdf": Outcome.PARTIAL}
    [page] = catalog.set_aside_pages()
    assert page.reason is Reason.OCR_LOW_CONFIDENCE and page.ocr_confidence == 0.58


def test_new_set_aside_reason_replaces_the_old_one(catalog, inbox):
    # Same usable text before and after OCR (page 2 still set aside), but the
    # reason changed: the stored pages must follow.
    (inbox / "blurry.pdf").write_bytes(make_pdf([GOOD, ""]))
    ingest(inbox, catalog)
    run_ocr(catalog, FakeEngine([(GOOD2, 0.58)]))
    [page] = catalog.set_aside_pages()
    assert page.reason is Reason.OCR_LOW_CONFIDENCE
    assert catalog.stats().documents == 1


def test_engine_failure_sets_page_aside_and_continues(catalog, inbox):
    (inbox / "a.pdf").write_bytes(make_pdf([""]))
    (inbox / "b.pdf").write_bytes(make_pdf(["", ""]))  # different bytes: not a copy of a.pdf
    engine = FakeEngine([(GOOD2, 0.9), (GOOD, 0.9)], raise_on=1)
    report = ingest(inbox, catalog, engine=engine)
    assert reasons(report)["a.pdf"] is Reason.OCR_FAILED
    assert outcomes(report)["b.pdf"] is Outcome.NEW
    assert catalog.ocr_cached(file_sha256(inbox / "a.pdf"), 1) is None


def test_interrupted_run_keeps_pages_already_read(catalog, inbox):
    (inbox / "long.pdf").write_bytes(make_pdf(["", "", ""]))
    engine = FakeEngine([(GOOD, 0.9), (GOOD2, 0.9)], raise_on=3, exc=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        ingest(inbox, catalog, engine=engine)
    h = file_sha256(inbox / "long.pdf")
    assert catalog.ocr_cached(h, 1) and catalog.ocr_cached(h, 2)
    assert catalog.ocr_cached(h, 3) is None


def test_run_ocr_only_touches_ocr_candidates_under_path(catalog, inbox):
    (inbox / "a").mkdir()
    (inbox / "b").mkdir()
    (inbox / "a" / "scan.pdf").write_bytes(make_pdf([""]))
    (inbox / "b" / "scan.pdf").write_bytes(make_pdf(["", ""]))
    (inbox / "a" / "todo.txt").write_text("appeler", encoding="utf-8")  # too_short
    ingest(inbox, catalog)
    engine = FakeEngine([(GOOD, 0.95)])
    report = run_ocr(catalog, engine, under=(inbox / "a").resolve())
    assert [r.path.name for r in report.results] == ["scan.pdf"]
    assert engine.calls == 1
    assert catalog.ocr_pending().pages == 2


def test_corrupted_image_is_quarantined(catalog, inbox):
    (inbox / "broken.jpg").write_bytes(b"\xff\xd8\xff not really")
    report = ingest(inbox, catalog, engine=FakeEngine([]))
    assert reasons(report) == {"broken.jpg": Reason.CORRUPTED}


def test_watermark_layer_is_replaced_by_ocr_text(catalog, inbox):
    stamp = "Scanned with CamScanner"
    (inbox / "w.pdf").write_bytes(make_pdf([stamp], image_pages={1}))
    ingest(inbox, catalog, engine=FakeEngine([(GOOD, 0.95)]))
    text = catalog.document_for_path((inbox / "w.pdf").resolve()).text
    assert text == GOOD and stamp not in text


def test_progress_callback_sees_each_page(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf(["", ""]))
    seen = []
    ingest(
        inbox,
        catalog,
        engine=FakeEngine([(GOOD, 0.9), (GOOD2, 0.9)]),
        on_ocr_page=lambda path, n: seen.append((path.name, n)),
    )
    assert seen == [("scan.pdf", 1), ("scan.pdf", 2)]


def test_quarantined_scan_records_its_page_count(catalog, inbox):
    (inbox / "scan.pdf").write_bytes(make_pdf(["", "", "", ""]))
    ingest(inbox, catalog)
    assert catalog.ocr_pending((inbox / "scan.pdf").resolve()).pages == 4
