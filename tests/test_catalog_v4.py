import sqlite3

import pytest

from localmind.catalog import SCHEMA_VERSION, Catalog, Classification, Decision
from localmind.folders import FolderName
from localmind.hashing import file_sha256
from localmind.ingest import ingest

TEXT = "Facture EDF de mars 2024, consommation 450 kWh, montant total 84,20 euros TTC."
TEXT2 = "Bulletin de paie de mars 2024, salaire de base 2100 euros, net a payer 1650 euros."
FACTURES = FolderName("Maison", "Factures")
PAIE = FolderName("Travail", "Bulletins de paie")


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "catalog.db") as cat:
        yield cat


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def _doc(catalog, inbox, name, text):
    (inbox / name).write_text(text, encoding="utf-8")
    ingest(inbox / name, catalog)
    return catalog.document_for_path((inbox / name).resolve()).id


def _place(catalog, doc_id, name, method="embedding", confirmed=True, status="placed"):
    folder_id = catalog.ensure_folder(name, "llm" if method != "user" else "user", confirmed)
    catalog.set_classification(
        Classification(doc_id, folder_id, status, method, 0.9, None, None), "m"
    )
    return folder_id


def test_v3_catalog_upgrades_to_v4_keeping_everything(tmp_path, inbox):
    db = tmp_path / "c.db"
    with Catalog(db) as cat:
        _doc(cat, inbox, "a.txt", TEXT)
        cat.record_decision("abc", Decision.ACCEPT)
    con = sqlite3.connect(db)
    for table in ("classifications", "embeddings", "folder_decisions", "folders"):
        con.execute(f"DROP TABLE {table}")
    con.execute("PRAGMA user_version = 3")
    con.commit()
    con.close()
    with Catalog(db) as cat:
        assert cat._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 6
        assert cat.document_for_path((inbox / "a.txt").resolve()).text.startswith("Facture")
        assert cat.decision_for("abc") is Decision.ACCEPT
        assert cat.folders() == []


def test_masked_text_hides_sensitive_values(catalog, inbox):
    (inbox / "rib.txt").write_text(
        "Mon IBAN FR76 3000 6000 0112 3456 7890 189 merci de payer", encoding="utf-8"
    )
    ingest(inbox, catalog)
    [doc] = catalog.documents_to_classify(None, redo=False)
    assert "[IBAN]" in doc.text and "3000" not in doc.text and doc.name == "rib.txt"
    assert doc.text.endswith("merci de payer")


def test_classified_documents_are_not_returned_again_unless_redo(catalog, inbox):
    doc_id = _doc(catalog, inbox, "a.txt", TEXT)
    _place(catalog, doc_id, FACTURES)
    assert catalog.documents_to_classify(None, redo=False) == []
    assert [d.id for d in catalog.documents_to_classify(None, redo=True)] == [doc_id]


def test_redo_never_returns_user_classifications(catalog, inbox):
    doc_id = _doc(catalog, inbox, "a.txt", TEXT)
    _place(catalog, doc_id, FACTURES, method="user")
    assert catalog.documents_to_classify(None, redo=True) == []


def test_documents_to_classify_only_under_path(catalog, inbox):
    (inbox / "sub").mkdir()
    _doc(catalog, inbox, "a.txt", TEXT)
    sub_id = _doc(catalog, inbox / "sub", "b.txt", TEXT2)
    found = catalog.documents_to_classify((inbox / "sub").resolve(), redo=False)
    assert [d.id for d in found] == [sub_id]
    assert found[0].paths == ((inbox / "sub" / "b.txt").resolve(),)


def test_examples_weight_user_and_skip_hesitations(catalog, inbox):
    ids = [_doc(catalog, inbox, f"{n}.txt", f"{TEXT} {n}") for n in "abcd"]
    for doc_id in ids:
        catalog.store_embedding(doc_id, "emb", [1.0, 0.0])
    confirmed = _place(catalog, ids[0], FACTURES, method="user")
    _place(catalog, ids[1], FACTURES)
    _place(catalog, ids[2], FACTURES, method="llm", status="to_confirm")  # hesitation
    proposed = _place(catalog, ids[3], PAIE, method="llm", confirmed=False, status="to_confirm")
    examples = sorted((e.folder_id, e.weight) for e in catalog.examples("emb", user_weight=2))
    assert examples == sorted([(confirmed, 2), (confirmed, 1), (proposed, 1)])
    assert catalog.examples("other-model", user_weight=2) == []


def test_embedding_round_trip_per_model(catalog, inbox):
    doc_id = _doc(catalog, inbox, "a.txt", TEXT)
    catalog.store_embedding(doc_id, "emb", [0.6, 0.8])
    assert catalog.embedding(doc_id, "emb") == pytest.approx([0.6, 0.8])
    assert catalog.embedding(doc_id, "other") is None


def test_folder_decision_matches_hash_then_path(catalog, tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    catalog.record_folder_decision("h1", a, FACTURES)
    assert catalog.folder_decision_for(("h1",), (b,)) == (FACTURES, "h1")
    assert catalog.folder_decision_for(("h2",), (a,)) == (FACTURES, "h1")  # edited in place
    assert catalog.folder_decision_for(("h2",), (b,)) is None
    catalog.rekey_folder_decision("h1", "h2", a)
    assert catalog.folder_decision_for(("h2",), (b,)) == (FACTURES, "h2")


def test_rename_folder_decisions(catalog, tmp_path):
    catalog.record_folder_decision("h1", tmp_path / "a.pdf", FACTURES)
    catalog.rename_folder_decisions(FACTURES, FolderName("Maison", "Energie"))
    assert catalog.folder_decision_for(("h1",), ()) == (FolderName("Maison", "Energie"), "h1")


def test_ensure_folder_matches_ignoring_case_and_accents(catalog):
    first = catalog.ensure_folder(FolderName("Santé", "Remboursements"), "llm", False)
    again = catalog.ensure_folder(FolderName("Sante", "remboursements"), "user", True)
    assert first == again
    [folder] = catalog.folders()
    assert folder.confirmed and folder.name == FolderName("Santé", "Remboursements")
    assert catalog.folder_by_name(FolderName("SANTE", "Remboursements")).id == first


def test_folders_count_documents(catalog, inbox):
    folder_id = _place(catalog, _doc(catalog, inbox, "a.txt", TEXT), FACTURES)
    _place(catalog, _doc(catalog, inbox, "b.txt", TEXT2), FACTURES)
    [folder] = catalog.folders()
    assert (folder.id, folder.documents, folder.created_by) == (folder_id, 2, "llm")
    assert sorted(d.name for d in catalog.documents_in(folder_id)) == ["a.txt", "b.txt"]


def test_pending_reviews(catalog, inbox):
    doc_id = _doc(catalog, inbox, "a.txt", TEXT)
    _place(catalog, doc_id, FACTURES, method="llm", confirmed=False, status="to_confirm")
    [pending] = catalog.pending_reviews()
    assert pending.document_id == doc_id and pending.status == "to_confirm"
    assert catalog.classification(doc_id) == pending


def test_empty_llm_folders_disappear_user_folders_stay(catalog, inbox):
    catalog.ensure_folder(FACTURES, "llm", False)
    catalog.ensure_folder(PAIE, "user", True)
    kept = _place(catalog, _doc(catalog, inbox, "a.txt", TEXT), FolderName("Santé", "Ordonnances"))
    catalog.delete_empty_llm_folders()
    assert sorted(str(f.name) for f in catalog.folders()) == [
        "Santé/Ordonnances", "Travail/Bulletins de paie"]
    assert kept in [f.id for f in catalog.folders()]


def test_editing_a_file_drops_the_old_classification_and_embedding(catalog, inbox):
    doc_id = _doc(catalog, inbox, "a.txt", TEXT)
    catalog.store_embedding(doc_id, "emb", [1.0, 0.0])
    _place(catalog, doc_id, FACTURES)
    new_id = _doc(catalog, inbox, "a.txt", TEXT2)
    assert new_id != doc_id
    assert catalog.classification(doc_id) is None and catalog.embedding(doc_id, "emb") is None
    assert catalog.classification(new_id) is None
    assert file_sha256(inbox / "a.txt")  # the original file is untouched by all this
