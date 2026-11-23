"""Stage 4: forgetting files deleted from disk (catalog rows only)."""

import shutil

import pytest

from localmind.catalog import Catalog, Classification, Decision
from localmind.folders import FolderName
from localmind.indexing import index
from localmind.ocr import OcrPage
from localmind.prune import PruneReport, missing_files, prune
from tests.fake_classifier import FakeClassifier
from tests.stage4_data import add_files, count

TEXT = "Facture EDF de mars, montant total 84,20 euros."


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def test_missing_files_lists_only_vanished_paths(catalog, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    add_files(catalog, a, {"x.txt": TEXT, "court.txt": "Bonjour"})  # court.txt is quarantined
    add_files(catalog, b, {"y.txt": "Bail de location appartement"})
    assert missing_files(catalog) == []
    for p in (a / "x.txt", a / "court.txt", b / "y.txt"):
        p.unlink()
    assert missing_files(catalog) == sorted(
        [(a / "court.txt").resolve(), (a / "x.txt").resolve(), (b / "y.txt").resolve()]
    )
    assert missing_files(catalog, under=b.resolve()) == [(b / "y.txt").resolve()]


def test_prune_forgets_document_with_its_pages_chunks_vectors_and_folder(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"x.txt": TEXT})
    index(catalog, FakeClassifier())
    folder_id = catalog.ensure_folder(FolderName("Maison", "Factures"), "user", True)
    catalog.set_classification(Classification(ids["x.txt"], folder_id, "placed", "user", 1.0, None, None), "m")
    (tmp_path / "in" / "x.txt").unlink()
    report = prune(catalog, missing_files(catalog))
    assert report == PruneReport(files=1, documents=1)
    for table in ("files", "documents", "pages", "chunks", "chunk_vectors", "classifications"):
        assert count(catalog, f"SELECT COUNT(*) FROM {table}") == 0, table
    assert missing_files(catalog) == []


def test_prune_keeps_a_document_that_still_has_a_copy(catalog, tmp_path):
    inbox = tmp_path / "in"
    add_files(catalog, inbox, {"x.txt": TEXT})
    shutil.copy(inbox / "x.txt", inbox / "x_copie.txt")
    add_files(catalog, inbox, {})
    index(catalog, FakeClassifier())
    (inbox / "x.txt").unlink()
    assert prune(catalog, missing_files(catalog)) == PruneReport(files=1, documents=0)
    assert catalog.stats().documents == 1 and catalog.stats().chunks >= 1


def test_prune_keeps_user_choices(catalog, tmp_path):
    inbox = tmp_path / "in"
    add_files(catalog, inbox, {"x.txt": TEXT})
    file_hash = catalog.file_by_path((inbox / "x.txt").resolve()).file_hash
    catalog.record_decision(file_hash, Decision.ACCEPT)
    catalog.record_folder_decision(file_hash, (inbox / "x.txt").resolve(), FolderName("Maison", "Factures"))
    catalog.store_ocr(file_hash, 1, OcrPage("Texte lu", 0.9, "doctr-x"))
    (inbox / "x.txt").unlink()
    prune(catalog, missing_files(catalog))
    assert catalog.decision_for(file_hash) is Decision.ACCEPT
    assert count(catalog, "SELECT COUNT(*) FROM folder_decisions") == 1
    assert catalog.ocr_cached(file_hash, 1) is not None


def test_prune_never_touches_disk(catalog, tmp_path):
    inbox = tmp_path / "in"
    add_files(catalog, inbox, {"x.txt": TEXT, "y.txt": "Bail de location appartement"})
    before = (inbox / "y.txt").read_bytes(), (inbox / "y.txt").stat().st_mtime_ns
    (inbox / "x.txt").unlink()
    prune(catalog, missing_files(catalog))
    assert ((inbox / "y.txt").read_bytes(), (inbox / "y.txt").stat().st_mtime_ns) == before
    assert sorted(p.name for p in inbox.iterdir()) == ["y.txt"]


def test_prune_refuses_a_file_that_still_exists(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"x.txt": TEXT})
    assert prune(catalog, [(tmp_path / "in" / "x.txt").resolve()]) == PruneReport(0, 0)
    assert catalog.stats().documents == 1
