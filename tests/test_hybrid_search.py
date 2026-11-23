"""Stage 5: hybrid search (meaning + keywords, fused) with a folder filter, fake embedder.

The fake embedder gives a text with a known keyword ("facture", "bail"...) that
keyword's axis, and any other text an axis of its own: a question about a plate
number "AB-123-CD" is orthogonal to every chunk, so meaning alone ranks at random
and only keywords can find the right chunk.
"""

import pytest

from localmind.catalog import Catalog, Classification
from localmind.folders import FolderName, InvalidFolderName, UnknownFolder
from localmind.indexing import index
from localmind.prune import missing_files, prune
from localmind.search import VectorIndex, search
from tests.fake_classifier import FakeClassifier
from tests.stage4_data import add_files

REASON = "Ollama not reachable at http://fake: start it (ollama serve)"
FACTURES = FolderName("Maison", "Factures")


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


@pytest.fixture
def inbox(tmp_path):
    return tmp_path / "in"


@pytest.fixture
def fake():
    return FakeClassifier()


def _filler(i: int) -> str:
    return f"Courrier numero {i} sans rapport, texte de remplissage pour le test."


@pytest.fixture
def plates(catalog, inbox, fake):
    """Five unrelated letters ingested first, then the insurance certificate."""
    files = {f"courrier_{i}.txt": _filler(i) for i in range(5)}
    files["zz_assurance.txt"] = "Attestation d'assurance du véhicule immatriculé AB-123-CD."
    ids = add_files(catalog, inbox, files)
    index(catalog, fake)
    return ids


def _place(catalog, doc_id, status="placed", folder=FACTURES):
    folder_id = catalog.ensure_folder(folder, "user", True)
    catalog.set_classification(Classification(doc_id, folder_id, status, "llm", 0.8, None, None), "m")


def test_exact_word_found_even_when_meaning_misses(catalog, plates, fake):
    hits = search(catalog, "véhicule AB-123-CD", k=1, embedder=fake)
    assert [h.path.name for h in hits] == ["zz_assurance.txt"]
    assert hits[0].matched == frozenset({"meaning", "keywords"})
    assert hits[0].score == pytest.approx(0.0)  # the cosine is kept, even when it is low


def test_both_sides_agreeing_rank_first(catalog, inbox, fake):
    add_files(catalog, inbox, {
        "bail.txt": "Le bail de location est consenti pour trois ans.",
        "bail_garage.txt": "Bail du garage, loyer mensuel.",
        "note.txt": "Note sur la location saisonniere a la mer.",
    })
    index(catalog, fake)
    hits = search(catalog, "durée du bail de location", k=3, embedder=fake)
    assert hits[0].path.name == "bail.txt"  # meaning (axis "bail") and both keywords agree
    assert hits[0].matched == frozenset({"meaning", "keywords"})


def test_hit_carries_folder_mark_and_matched(catalog, inbox, fake):
    ids = add_files(catalog, inbox, {
        "facture_edf.txt": "Facture EDF de mars, montant 84,20 euros.",
        "facture_eau.txt": "Facture d'eau du trimestre, montant 40 euros.",
        "lettre.txt": "Une lettre sans aucun rapport avec le reste.",
    })
    index(catalog, fake)
    _place(catalog, ids["facture_edf.txt"], status="to_confirm")
    _place(catalog, ids["facture_eau.txt"])
    hits = {h.path.name: h for h in search(catalog, "facture EDF", k=3, embedder=fake)}
    assert hits["facture_edf.txt"].folder == "Maison/Factures"
    assert hits["facture_edf.txt"].to_confirm is True
    assert hits["facture_edf.txt"].matched == frozenset({"meaning", "keywords"})
    assert hits["facture_eau.txt"].to_confirm is False
    assert hits["lettre.txt"].folder is None and hits["lettre.txt"].to_confirm is False
    assert hits["lettre.txt"].matched == frozenset({"meaning"})


def test_folder_filter_applies_before_ranking(catalog, inbox, fake):
    # 40 invoices outside the folder are as close as the 2 inside: more than
    # the 30 candidates a search looks at first
    files = {f"facture_{i:02}.txt": f"Facture numero {i} du fournisseur, montant {i} euros." for i in range(40)}
    files.update({"zz_in_1.txt": "Facture du plombier, montant 90 euros.",
                  "zz_in_2.txt": "Facture du peintre, montant 300 euros."})
    ids = add_files(catalog, inbox, files)
    index(catalog, fake)
    _place(catalog, ids["zz_in_1.txt"])
    _place(catalog, ids["zz_in_2.txt"])
    hits = search(catalog, "facture montant", k=2, embedder=fake, folder="maison/factures")
    assert sorted(h.path.name for h in hits) == ["zz_in_1.txt", "zz_in_2.txt"]
    assert search(catalog, "facture montant", k=5, embedder=fake, folder="Maison") == hits


def test_keyword_only_hit_of_deleted_or_edited_file_is_never_cited(catalog, inbox, fake):  # Review Focus 2
    add_files(catalog, inbox, {
        "a_assurance.txt": "Attestation du véhicule AB-123-CD, assureur Durand.",
        "b_carte_grise.txt": "Carte grise du véhicule AB-123-CD.",
        "c_autre.txt": _filler(1),
    })
    index(catalog, fake)
    (inbox / "a_assurance.txt").unlink()
    (inbox / "b_carte_grise.txt").write_text("Carte grise modifiée du véhicule AB-123-CD, nouveau titulaire.", encoding="utf-8")
    hits = search(catalog, "AB-123-CD", k=3, embedder=fake)
    assert [h.path.name for h in hits] == ["c_autre.txt"]
    hits = search(catalog, "AB-123-CD", k=3, embedder=FakeClassifier(unavailable=True))
    assert hits == []


def test_long_lived_index_sees_keywords_from_another_connection(catalog, inbox, fake, tmp_path):  # Review Focus 3
    add_files(catalog, inbox, {"a.txt": _filler(1), "b_garage.txt": "Bail du garage, loyer Durand SARL."})
    index(catalog, fake)
    vi = VectorIndex(catalog, fake)
    assert [h.path.name for h in vi.search("Durand", k=1)] == ["b_garage.txt"]
    with Catalog(tmp_path / "c.db") as other:
        add_files(other, inbox, {"c_paie.txt": "Bulletin de paie de Martin Dupont, employeur Durand SARL."})
        index(other, fake)
        (inbox / "b_garage.txt").unlink()
        prune(other, missing_files(other))
    names = [h.path.name for h in vi.search("Durand SARL", k=3)]
    assert names[0] == "c_paie.txt" and "b_garage.txt" not in names


def test_ollama_down_falls_back_to_keywords(catalog, plates):
    vi = VectorIndex(catalog, FakeClassifier(unavailable=True))
    hits = vi.search("véhicule AB-123-CD", k=3)
    assert [h.path.name for h in hits] == ["zz_assurance.txt"]
    assert all(h.matched == frozenset({"keywords"}) and h.score is None for h in hits)
    assert vi.last_warning == REASON
    assert vi.search("le de la", k=3) == []  # nothing left to look for: no hit, still a warning
    assert vi.last_warning == REASON


def test_warning_is_cleared_when_ollama_answers_again(catalog, plates, fake):
    vi = VectorIndex(catalog, FakeClassifier(unavailable=True))
    vi.search("AB-123-CD")
    vi.embedder = fake
    vi.search("AB-123-CD")
    assert vi.last_warning is None


def test_ollama_down_with_folder_keeps_the_filter(catalog, inbox, fake):  # Review Focus 5
    ids = add_files(catalog, inbox, {
        "facture_edf.txt": "Facture EDF de mars, montant 84,20 euros.",
        "facture_garage.txt": "Facture du garage pour la vidange.",
    })
    index(catalog, fake)
    _place(catalog, ids["facture_edf.txt"])
    _place(catalog, ids["facture_garage.txt"], folder=FolderName("Voiture", "Entretien"))
    hits = search(catalog, "facture", k=5, embedder=FakeClassifier(unavailable=True), folder="Voiture")
    assert [h.path.name for h in hits] == ["facture_garage.txt"]
    assert hits[0].folder == "Voiture/Entretien"


def test_unknown_or_invalid_folder_raises(catalog, plates, fake):
    with pytest.raises(UnknownFolder, match="no folder named 'Maison/Factur'"):
        search(catalog, "facture", embedder=fake, folder="maison/factur")
    with pytest.raises(InvalidFolderName):
        search(catalog, "facture", embedder=fake, folder="a/b/c")


def test_keyword_hit_without_vector_has_no_score(catalog, plates, fake):
    other = FakeClassifier()
    other.embed_model = "other-embed"
    vi = VectorIndex(catalog, other)
    hits = vi.search("AB-123-CD", k=3)
    assert [h.path.name for h in hits] == ["zz_assurance.txt"]
    assert hits[0].score is None and hits[0].matched == frozenset({"keywords"})
    assert vi.last_warning is None and other.embed_batches == []  # no vector: Ollama not asked
