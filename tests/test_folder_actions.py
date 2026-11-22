import shutil

import pytest

from localmind.catalog import Catalog
from localmind.classify import classify
from localmind.folder_actions import choose_folder, confirm, move, rename_folder
from localmind.folders import FolderName, InvalidFolderName
from localmind.ingest import ingest
from tests.fake_classifier import FakeClassifier, answer

FACTURES = FolderName("Maison", "Factures")


def invoice(i):
    return f"Facture EDF numero {i}, consommation {400 + i} kWh, montant total {80 + i} euros TTC."


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "catalog.db") as cat:
        yield cat


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def add(catalog, inbox, name, text):
    (inbox / name).write_text(text, encoding="utf-8")
    ingest(inbox, catalog)
    return catalog.document_for_path((inbox / name).resolve()).id


def proposal(catalog, inbox, n=2, folder="Maison/Factures"):
    ids = [add(catalog, inbox, f"facture_{i}.txt", invoice(i)) for i in range(n)]
    classify(catalog, FakeClassifier(lambda p: answer(folder, new_folder=True)))
    return ids


def test_confirming_a_proposal_places_all_its_documents(catalog, inbox):
    ids = proposal(catalog, inbox)
    assert confirm(catalog, "maison/factures") == 2
    assert catalog.folder_by_name(FACTURES).confirmed
    for doc_id in ids:
        c = catalog.classification(doc_id)
        assert (c.status, c.method) == ("placed", "user")
    assert catalog.pending_reviews() == []


def test_confirm_unknown_folder_raises(catalog):
    with pytest.raises(LookupError):
        confirm(catalog, "Maison/Inconnu")


def test_renaming_a_proposal_moves_its_documents_and_confirms(catalog, inbox):
    ids = proposal(catalog, inbox)
    rename_folder(catalog, "Maison/Factures", "Maison/Énergie")
    [folder] = catalog.folders()
    assert str(folder.name) == "Maison/Énergie" and folder.confirmed and folder.documents == 2
    assert all(catalog.classification(i).status == "placed" for i in ids)


def test_choose_folder_answers_a_hesitation(catalog, inbox):
    doc_id = add(catalog, inbox, "a.txt", invoice(1))
    choose_folder(catalog, doc_id, "Santé/Contrats")
    c = catalog.classification(doc_id)
    assert (c.status, c.method) == ("placed", "user")
    assert catalog.folder(c.folder_id).created_by == "user"


def test_move_creates_a_confirmed_user_folder(catalog, inbox):
    doc_id = add(catalog, inbox, "a.txt", invoice(1))
    move(catalog, inbox / "a.txt", "maison / énergie")
    folder = catalog.folder(catalog.classification(doc_id).folder_id)
    assert str(folder.name) == "Maison/Énergie"
    assert folder.confirmed and folder.created_by == "user"


def test_move_rejects_bad_names_and_unknown_paths(catalog, inbox):
    add(catalog, inbox, "a.txt", invoice(1))
    with pytest.raises(InvalidFolderName):
        move(catalog, inbox / "a.txt", "Maison/Factures/EDF")
    with pytest.raises(LookupError):
        move(catalog, inbox / "nope.txt", "Maison/Factures")


def test_correction_follows_edited_file_and_copies(catalog, inbox):
    add(catalog, inbox, "a.txt", invoice(1))
    move(catalog, inbox / "a.txt", "Maison/Factures")
    (inbox / "sub").mkdir()
    shutil.copy(inbox / "a.txt", inbox / "sub" / "b.txt")  # exact copy
    (inbox / "a.txt").write_text(invoice(1) + " Rectificatif.", encoding="utf-8")  # edited
    ingest(inbox, catalog)
    report = classify(catalog, FakeClassifier([answer("Divers/Autres", new_folder=True)] * 3))
    for name in ("a.txt", "sub/b.txt"):
        doc = catalog.document_for_path((inbox / name).resolve())
        c = catalog.classification(doc.id)
        assert (str(catalog.folder(c.folder_id).name), c.method) == ("Maison/Factures", "user")
    assert all(r.method == "user" for r in report.results)
    edited = catalog.file_by_path((inbox / "a.txt").resolve())
    assert catalog.folder_decision_for((edited.file_hash,), ()) == (FACTURES, edited.file_hash)


def test_rename_into_existing_merges_and_keeps_decisions(catalog, inbox):
    a = add(catalog, inbox, "a.txt", invoice(1))
    b = add(catalog, inbox, "b.txt", invoice(2))
    move(catalog, inbox / "a.txt", "Maison/Factures")
    move(catalog, inbox / "b.txt", "Maison/Energie")
    rename_folder(catalog, "Maison/Energie", "Maison/Factures")
    assert [str(f.name) for f in catalog.folders()] == ["Maison/Factures"]
    classify(catalog, FakeClassifier(lambda p: answer("Divers/Autres", new_folder=True)), redo=True)
    for doc_id in (a, b):
        c = catalog.classification(doc_id)
        assert (str(catalog.folder(c.folder_id).name), c.method) == ("Maison/Factures", "user")


def test_user_folders_are_examples_for_the_next_document(catalog, inbox):
    for i in range(3):
        add(catalog, inbox, f"seed_{i}.txt", invoice(i))
        move(catalog, inbox / f"seed_{i}.txt", "Maison/Factures")
    add(catalog, inbox, "new.txt", invoice(9))
    fake = FakeClassifier([])
    [result] = classify(catalog, fake).results
    assert (result.folder, result.method, result.status) == ("Maison/Factures", "embedding", "placed")
    assert fake.chats == []


def test_empty_llm_folder_disappears_after_moves(catalog, inbox):
    proposal(catalog, inbox, n=1, folder="Divers/Autres")
    move(catalog, inbox / "facture_0.txt", "Maison/Factures")
    assert [str(f.name) for f in catalog.folders()] == ["Maison/Factures"]


def test_correction_survives_editing_one_of_two_existing_copies(catalog, inbox):
    (inbox / "sub").mkdir()
    add(catalog, inbox, "a.txt", invoice(1))
    shutil.copy(inbox / "a.txt", inbox / "sub" / "b.txt")  # the copy exists BEFORE the correction
    ingest(inbox, catalog)
    move(catalog, inbox / "a.txt", "Maison/Factures")
    (inbox / "a.txt").write_text(invoice(1) + " Rectificatif.", encoding="utf-8")
    ingest(inbox, catalog)
    classify(catalog, FakeClassifier(lambda p: answer("Divers/Autres", new_folder=True)))
    for name in ("a.txt", "sub/b.txt"):
        c = catalog.classification(catalog.document_for_path((inbox / name).resolve()).id)
        assert (str(catalog.folder(c.folder_id).name), c.method) == ("Maison/Factures", "user")


def test_rename_and_merge_keep_unanswered_hesitations_open(catalog, inbox):
    from localmind.catalog import Classification

    add(catalog, inbox, "a.txt", invoice(1))
    move(catalog, inbox / "a.txt", "Maison/Factures")
    b = add(catalog, inbox, "b.txt", "Contrat d'electricite, offre verte.")
    energie = catalog.ensure_folder(FolderName("Maison", "Energie"), "user", True)
    contrats = catalog.ensure_folder(FolderName("Santé", "Contrats"), "user", True)
    catalog.set_classification(
        Classification(b, energie, "to_confirm", "llm", 0.5, contrats, None), "m")
    rename_folder(catalog, "Maison/Energie", "Maison/Électricité")
    c = catalog.classification(b)
    assert (c.status, c.method, c.alternative_folder_id) == ("to_confirm", "llm", contrats)
    rename_folder(catalog, "Maison/Électricité", "Maison/Factures")  # merge
    c = catalog.classification(b)
    assert (c.status, c.method, c.alternative_folder_id) == ("to_confirm", "llm", contrats)
    assert str(catalog.folder(c.folder_id).name) == "Maison/Factures"
