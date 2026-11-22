import pytest

from localmind.catalog import Catalog
from localmind.ingest import Outcome, ingest
from localmind.quality import Reason
from tests.pdf_factory import make_pdf

GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."


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


def test_new_files_are_cataloged(catalog, inbox):
    (inbox / "a.txt").write_text("Alpha bravo charlie", encoding="utf-8")
    (inbox / "b.pdf").write_bytes(make_pdf([GOOD]))

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"a.txt": Outcome.NEW, "b.pdf": Outcome.NEW}
    assert catalog.stats().documents == 2
    assert catalog.stats().files == 2


def test_exact_copy_is_a_duplicate_pointing_to_same_document(catalog, inbox):
    (inbox / "facture.txt").write_text("Facture numéro 42", encoding="utf-8")
    (inbox / "facture (1).txt").write_text("Facture numéro 42", encoding="utf-8")

    report = ingest(inbox, catalog)

    assert sorted(outcomes(report).values()) == [Outcome.DUPLICATE, Outcome.NEW]
    assert catalog.stats().documents == 1
    assert catalog.stats().files == 2


def test_same_content_different_encoding_is_a_duplicate(catalog, inbox):
    (inbox / "utf8.txt").write_bytes("Reçu de caisse : 30 €".encode("utf-8"))
    (inbox / "win.txt").write_bytes("Reçu de caisse : 30 €\r\n".encode("cp1252"))

    report = ingest(inbox, catalog)

    assert sorted(outcomes(report).values()) == [Outcome.DUPLICATE, Outcome.NEW]
    assert catalog.stats().documents == 1


def test_duplicate_groups_are_listed(catalog, inbox):
    for name in ("x.txt", "x_copy.txt", "x_copy2.txt"):
        (inbox / name).write_text("Same text here", encoding="utf-8")
    (inbox / "other.txt").write_text("Different text there", encoding="utf-8")

    ingest(inbox, catalog)
    groups = catalog.duplicate_groups()

    assert len(groups) == 1
    assert sorted(p.name for p in groups[0].paths) == ["x.txt", "x_copy.txt", "x_copy2.txt"]


def test_reingesting_unchanged_files_does_nothing(catalog, inbox):
    (inbox / "a.txt").write_text("Alpha bravo charlie", encoding="utf-8")
    ingest(inbox, catalog)

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"a.txt": Outcome.UNCHANGED}
    assert catalog.stats().documents == 1


def test_modified_file_is_updated_and_old_content_dropped(catalog, inbox):
    f = inbox / "notes.txt"
    f.write_text("Notes version one", encoding="utf-8")
    ingest(inbox, catalog)
    f.write_text("Notes version two", encoding="utf-8")

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"notes.txt": Outcome.UPDATED}
    assert catalog.stats().documents == 1
    assert catalog.document_for_path(f.resolve()).text == "Notes version two"


def test_unreadable_files_are_quarantined_with_reason(catalog, inbox):
    (inbox / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (inbox / "broken.pdf").write_bytes(b"not a pdf")
    (inbox / "archive.zip").write_bytes(b"PK\x03\x04")
    (inbox / "zero.txt").write_bytes(b"")

    report = ingest(inbox, catalog)

    assert set(outcomes(report).values()) == {Outcome.QUARANTINED}
    assert reasons(report) == {
        "photo.jpg": Reason.NEEDS_OCR,
        "broken.pdf": Reason.CORRUPTED,
        "archive.zip": Reason.UNSUPPORTED_FORMAT,
        "zero.txt": Reason.EMPTY,
    }
    assert catalog.stats().documents == 0
    assert len(catalog.quarantined_files()) == 4


def test_encrypted_pdf_is_quarantined(catalog, inbox):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(100, 100)
    w.encrypt(user_password="secret", algorithm="AES-256")
    with (inbox / "locked.pdf").open("wb") as s:
        w.write(s)

    assert reasons(ingest(inbox, catalog)) == {"locked.pdf": Reason.ENCRYPTED}


def test_subfolders_are_scanned(catalog, inbox):
    sub = inbox / "2024" / "march"
    sub.mkdir(parents=True)
    (sub / "deep.txt").write_text("Deep inside folders", encoding="utf-8")

    assert outcomes(ingest(inbox, catalog)) == {"deep.txt": Outcome.NEW}


def test_single_file_can_be_ingested(catalog, inbox):
    f = inbox / "solo.txt"
    f.write_text("Solo file content", encoding="utf-8")

    assert outcomes(ingest(f, catalog)) == {"solo.txt": Outcome.NEW}


def test_original_files_are_never_modified(catalog, inbox):
    f = inbox / "keep.txt"
    f.write_bytes(b"untouched file content")
    before = f.stat().st_mtime_ns

    ingest(inbox, catalog)

    assert f.read_bytes() == b"untouched file content"
    assert f.stat().st_mtime_ns == before


def test_two_different_scans_are_not_duplicates(catalog, inbox):  # stage 1 bug
    (inbox / "a.pdf").write_bytes(make_pdf(["", ""]))
    (inbox / "b.pdf").write_bytes(make_pdf([""]))

    assert set(outcomes(ingest(inbox, catalog)).values()) == {Outcome.QUARANTINED}
    assert catalog.duplicate_groups() == []


def test_partial_pdf(catalog, inbox):
    (inbox / "c.pdf").write_bytes(make_pdf([GOOD, ""]))

    assert outcomes(ingest(inbox, catalog)) == {"c.pdf": Outcome.PARTIAL}
    assert [p.page_no for p in catalog.set_aside_pages()] == [2]


def test_same_good_page_different_scans_not_merged(catalog, inbox):
    (inbox / "a.pdf").write_bytes(make_pdf([GOOD, ""]))
    (inbox / "b.pdf").write_bytes(make_pdf([GOOD, "", ""]))

    report = ingest(inbox, catalog)

    assert sorted(outcomes(report).values()) == [Outcome.PARTIAL, Outcome.PARTIAL]
    assert catalog.stats().documents == 2


def test_copy_of_quarantined_file_is_quarantined_too(catalog, inbox):
    (inbox / "x.jpg").write_bytes(b"\xff\xd8\xff")
    (inbox / "x (1).jpg").write_bytes(b"\xff\xd8\xff")

    ingest(inbox, catalog)

    assert [r.reason for r in catalog.quarantined_files()] == [Reason.NEEDS_OCR] * 2


def test_quality_rules_apply(catalog, inbox):
    (inbox / "todo.txt").write_text("appeler", encoding="utf-8")
    log = "\n".join(f"2024-03-01 12:00:{i:02d} INFO job {i} done" for i in range(25))
    (inbox / "app.log.txt").write_text(log, encoding="utf-8")

    assert reasons(ingest(inbox, catalog)) == {
        "todo.txt": Reason.TOO_SHORT,
        "app.log.txt": Reason.MACHINE_LOG,
    }


def test_sensitive_data_is_flagged_not_refused(catalog, inbox):
    (inbox / "rib.txt").write_text(GOOD + " IBAN FR7630006000011234567890189", encoding="utf-8")

    assert outcomes(ingest(inbox, catalog)) == {"rib.txt": Outcome.NEW}
    assert [f.kind for f in catalog.sensitive_flags()] == ["iban"]
