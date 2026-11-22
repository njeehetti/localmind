import pytest

from localmind.catalog import Catalog
from localmind.ingest import Outcome, ingest
from tests.pdf_factory import make_pdf


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


def test_new_files_are_cataloged(catalog, inbox):
    (inbox / "a.txt").write_text("Alpha", encoding="utf-8")
    (inbox / "b.pdf").write_bytes(make_pdf(["Bravo"]))

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"a.txt": Outcome.NEW, "b.pdf": Outcome.NEW}
    assert catalog.stats().documents == 2
    assert catalog.stats().files == 2


def test_exact_copy_is_a_duplicate_pointing_to_same_document(catalog, inbox):
    (inbox / "facture.txt").write_text("Facture 42", encoding="utf-8")
    (inbox / "facture (1).txt").write_text("Facture 42", encoding="utf-8")

    report = ingest(inbox, catalog)

    assert sorted(outcomes(report).values()) == [Outcome.DUPLICATE, Outcome.NEW]
    assert catalog.stats().documents == 1  # one piece of information...
    assert catalog.stats().files == 2  # ...two pointers to it


def test_same_content_different_encoding_is_a_duplicate(catalog, inbox):
    (inbox / "utf8.txt").write_bytes("Reçu : 30 €".encode("utf-8"))
    (inbox / "win.txt").write_bytes("Reçu : 30 €\r\n".encode("cp1252"))

    report = ingest(inbox, catalog)

    assert sorted(outcomes(report).values()) == [Outcome.DUPLICATE, Outcome.NEW]
    assert catalog.stats().documents == 1


def test_duplicate_groups_are_listed(catalog, inbox):
    for name in ("x.txt", "x_copy.txt", "x_copy2.txt"):
        (inbox / name).write_text("Same", encoding="utf-8")
    (inbox / "other.txt").write_text("Different", encoding="utf-8")

    ingest(inbox, catalog)
    groups = catalog.duplicate_groups()

    assert len(groups) == 1
    assert sorted(p.name for p in groups[0].paths) == [
        "x.txt",
        "x_copy.txt",
        "x_copy2.txt",
    ]


def test_reingesting_unchanged_files_does_nothing(catalog, inbox):
    (inbox / "a.txt").write_text("Alpha", encoding="utf-8")
    ingest(inbox, catalog)

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"a.txt": Outcome.UNCHANGED}
    assert catalog.stats().documents == 1


def test_modified_file_is_updated_and_old_content_dropped(catalog, inbox):
    f = inbox / "notes.txt"
    f.write_text("Version 1", encoding="utf-8")
    ingest(inbox, catalog)
    f.write_text("Version 2", encoding="utf-8")

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"notes.txt": Outcome.UPDATED}
    assert catalog.stats().documents == 1
    assert catalog.document_for_path(f).text == "Version 2"


def test_unsupported_and_broken_files_are_reported_not_cataloged(catalog, inbox):
    (inbox / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (inbox / "broken.pdf").write_bytes(b"not a pdf")

    report = ingest(inbox, catalog)

    assert outcomes(report) == {
        "photo.jpg": Outcome.UNSUPPORTED,
        "broken.pdf": Outcome.FAILED,
    }
    assert catalog.stats().files == 0


def test_subfolders_are_scanned(catalog, inbox):
    sub = inbox / "2024" / "march"
    sub.mkdir(parents=True)
    (sub / "deep.txt").write_text("Deep", encoding="utf-8")

    report = ingest(inbox, catalog)

    assert outcomes(report) == {"deep.txt": Outcome.NEW}


def test_single_file_can_be_ingested(catalog, inbox):
    f = inbox / "solo.txt"
    f.write_text("Solo", encoding="utf-8")

    report = ingest(f, catalog)

    assert outcomes(report) == {"solo.txt": Outcome.NEW}


def test_original_files_are_never_modified(catalog, inbox):
    f = inbox / "keep.txt"
    f.write_bytes(b"untouched")
    before = f.stat().st_mtime_ns

    ingest(inbox, catalog)

    assert f.read_bytes() == b"untouched"
    assert f.stat().st_mtime_ns == before
