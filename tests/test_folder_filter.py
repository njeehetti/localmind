"""Stage 5: the folder of each document, and the --folder filter."""

import pytest

from localmind.catalog import Catalog, Classification
from localmind.catalog_folders import DocFolder
from localmind.folders import (
    FolderFilter,
    FolderName,
    InvalidFolderName,
    UnknownFolder,
    parse_folder_filter,
)
from tests.stage4_data import add_files


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def _docs(catalog, tmp_path, *names):
    ids = add_files(catalog, tmp_path / "in", {n: f"Document {n} avec assez de texte" for n in names})
    return [ids[n] for n in names]


def _place(catalog, doc_id, folder, status="placed", alternative=None, confirmed=True):
    folder_id = catalog.ensure_folder(folder, "user" if confirmed else "llm", confirmed) if folder else None
    alt_id = catalog.ensure_folder(alternative, "user", True) if alternative else None
    catalog.set_classification(
        Classification(doc_id, folder_id, status, "llm", 0.8, alt_id, None), "m"
    )


FACTURES = FolderName("Maison", "Factures")
CONTRATS = FolderName("Maison", "Contrats")
SANTE = FolderName("Santé", "Ordonnances")


def test_parse_folder_filter_domain_or_folder():
    assert parse_folder_filter("  maison ") == FolderFilter("Maison", None)
    assert parse_folder_filter("maison / factures") == FolderFilter("Maison", "Factures")
    assert str(FolderFilter("Maison", None)) == "Maison"
    assert str(FolderFilter("Maison", "Factures")) == "Maison/Factures"
    for bad in ["", "a/b/c", "/Factures", "Maison/", "Fac<tures>"]:
        with pytest.raises(InvalidFolderName):
            parse_folder_filter(bad)


def test_placed_and_to_confirm_are_listed_marked(catalog, tmp_path):
    a, b = _docs(catalog, tmp_path, "a.txt", "b.txt")
    _place(catalog, a, FACTURES)
    _place(catalog, b, FACTURES, status="to_confirm")
    assert catalog.document_folders() == {
        a: DocFolder("Maison/Factures", False),
        b: DocFolder("Maison/Factures", True),
    }


def test_hesitation_matches_its_chosen_folder_not_the_alternative(catalog, tmp_path):
    (a,) = _docs(catalog, tmp_path, "a.txt")
    _place(catalog, a, FACTURES, status="to_confirm", alternative=CONTRATS)
    assert catalog.document_folders(parse_folder_filter("Maison/Factures")) == {
        a: DocFolder("Maison/Factures", True)
    }
    assert catalog.document_folders(parse_folder_filter("Maison/Contrats")) == {}


def test_unclassified_and_invalid_llm_answer_never_match(catalog, tmp_path):
    a, b, c = _docs(catalog, tmp_path, "a.txt", "b.txt", "c.txt")
    _place(catalog, a, FACTURES)
    _place(catalog, b, None, status="to_confirm")  # invalid LLM answer: folder_id NULL
    assert catalog.document_folders() == {a: DocFolder("Maison/Factures", False)}
    assert set(catalog.document_folders(parse_folder_filter("Maison"))) == {a}


def test_domain_filter_covers_its_folders(catalog, tmp_path):
    a, b, c = _docs(catalog, tmp_path, "a.txt", "b.txt", "c.txt")
    _place(catalog, a, FACTURES)
    _place(catalog, b, CONTRATS)
    _place(catalog, c, SANTE)
    assert set(catalog.document_folders(parse_folder_filter("Maison"))) == {a, b}
    assert set(catalog.document_folders(parse_folder_filter("sante"))) == {c}


def test_filter_is_case_and_space_insensitive(catalog, tmp_path):  # Review Focus 4
    (a,) = _docs(catalog, tmp_path, "a.txt")
    _place(catalog, a, FACTURES)
    for text in ["maison/factures", "Maison / Factures", "MAISON/FACTURES", " maison "]:
        assert set(catalog.document_folders(parse_folder_filter(text))) == {a}, text


def test_domain_prefix_is_not_a_domain(catalog, tmp_path):
    (a,) = _docs(catalog, tmp_path, "a.txt")
    _place(catalog, a, FolderName("Maisonnette", "Factures"))
    with pytest.raises(UnknownFolder):
        catalog.document_folders(parse_folder_filter("Maison"))


def test_unknown_folder_raises_naming_it(catalog, tmp_path):
    (a,) = _docs(catalog, tmp_path, "a.txt")
    _place(catalog, a, FACTURES)
    with pytest.raises(UnknownFolder) as exc:
        catalog.document_folders(parse_folder_filter("maison/factur"))
    assert str(exc.value) == "no folder named 'Maison/Factur'"
    with pytest.raises(UnknownFolder, match="no folder named 'Travail'"):
        catalog.document_folders(parse_folder_filter("travail"))


def test_known_folder_without_documents_is_empty(catalog):  # Review Focus 4
    catalog.ensure_folder(CONTRATS, "user", True)
    assert catalog.document_folders(parse_folder_filter("Maison/Contrats")) == {}
    assert catalog.document_folders(parse_folder_filter("Maison")) == {}


def test_llm_proposed_folder_is_known(catalog, tmp_path):
    (a,) = _docs(catalog, tmp_path, "a.txt")
    _place(catalog, a, FACTURES, status="to_confirm", confirmed=False)
    assert catalog.document_folders(parse_folder_filter("Maison/Factures")) == {
        a: DocFolder("Maison/Factures", True)
    }
