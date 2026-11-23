"""Stage 4: vector search over chunks (numpy brute force), fake embedder."""

import shutil
import sqlite3

import pytest

from localmind.catalog import Catalog
from localmind.indexing import index
from localmind.ollama import ClassifierUnavailable
from localmind.search import CatalogBusy, Hit, VectorIndex, search
from tests.fake_classifier import FakeClassifier
from tests.stage4_data import add_files

BAIL = "Le bail de location est consenti pour trois ans, préavis d'un mois."


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


@pytest.fixture
def indexed(catalog, inbox, fake):
    add_files(catalog, inbox, {
        "abonnement_tel.txt": "Abonnement telephone mobile : prélèvement le 5 de chaque mois.",
        "bail.txt": BAIL,
        "releve_electricite.txt": "Releve electricite de mars : 342 kWh consommés.",
    })
    index(catalog, fake)
    return catalog


def test_best_chunk_first_with_file_and_page(indexed, inbox, fake):
    hits = search(indexed, "jour du prélèvement telephone", k=2, embedder=fake)
    assert len(hits) == 2 and isinstance(hits[0], Hit)
    assert hits[0].path == (inbox / "abonnement_tel.txt").resolve() and hits[0].page_no == 1
    assert hits[0].score == pytest.approx(1.0) and hits[0].score >= hits[1].score
    assert "prélèvement le 5" in hits[0].text
    assert fake.prefixes[-1] == "search_query: " and fake.keep_alives[-1] == "30m"


def test_k_larger_than_the_catalog_returns_everything(indexed, fake):
    hits = search(indexed, "bail", k=50, embedder=fake)
    assert len(hits) == 3 and hits[0].path.name == "bail.txt"


def test_nothing_indexed_returns_empty_list_without_calling_ollama(catalog):
    assert search(catalog, "bail", embedder=FakeClassifier(unavailable=True)) == []


@pytest.mark.parametrize("question, k", [("", 5), ("   ", 5), ("bail", 0)])
def test_bad_question_or_k_is_rejected(indexed, fake, question, k):
    with pytest.raises(ValueError):
        search(indexed, question, k=k, embedder=fake)


def test_unavailable_ollama_points_at_search(indexed):
    with pytest.raises(ClassifierUnavailable) as err:
        search(indexed, "bail", embedder=FakeClassifier(unavailable=True))
    assert err.value.hint.endswith("then: localmind search")


def test_only_vectors_of_the_current_model_are_searched(indexed, fake):
    other = FakeClassifier()
    other.embed_model = "other-embed"
    assert search(indexed, "bail", embedder=other) == []


def test_deleted_file_is_skipped_and_copy_is_named(indexed, inbox, fake):
    shutil.copy(inbox / "bail.txt", inbox / "bail_copie.txt")
    add_files(indexed, inbox, {})  # ingest the copy: same document
    (inbox / "bail.txt").unlink()
    (inbox / "abonnement_tel.txt").unlink()
    hits = search(indexed, "bail", k=5, embedder=fake)
    assert [h.path.name for h in hits] == ["bail_copie.txt", "releve_electricite.txt"]
    assert all(h.path.exists() for h in hits)


def test_skipped_hits_are_replaced_to_keep_k(indexed, inbox, fake):
    (inbox / "bail.txt").unlink()
    hits = search(indexed, "bail", k=2, embedder=fake)
    assert len(hits) == 2 and "bail.txt" not in [h.path.name for h in hits]


def test_index_reloads_after_catalog_change(indexed, inbox, fake):
    vi = VectorIndex(indexed, fake)
    assert vi.search("bail", k=1)[0].path.name == "bail.txt"
    add_files(indexed, inbox, {"rdv_medecin.txt": "Votre rendez-vous chez le cardiologue est le 12 janvier."})
    index(indexed, fake)
    assert vi.search("rendez-vous", k=1)[0].path.name == "rdv_medecin.txt"


def test_locked_catalog_raises_catalog_busy(indexed, tmp_path, fake):
    indexed._conn.execute("PRAGMA busy_timeout = 50")
    other = sqlite3.connect(tmp_path / "c.db")
    other.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(CatalogBusy, match="the catalog is busy, try again"):
            VectorIndex(indexed, fake).search("bail")
    finally:
        other.rollback()
        other.close()


def test_warm_embeds_once_with_keep_alive(indexed, fake):
    calls = len(fake.embed_batches)
    VectorIndex(indexed, fake).warm()
    assert len(fake.embed_batches) == calls + 1
    assert fake.prefixes[-1] == "search_query: " and fake.keep_alives[-1] == "30m"
