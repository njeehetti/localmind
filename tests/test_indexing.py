"""Stage 4: the indexing run (chunks + vectors), with the fake embedder."""

import pytest

from localmind.catalog import Catalog
from localmind.chunking import ChunkConfig
from localmind.indexing import IndexReport, index
from tests.fake_classifier import FakeClassifier
from tests.stage4_data import IBAN, add_files, count, long_text


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def _all_rows(catalog, model="fake-embed"):
    return catalog.chunk_rows([cid for cid, _ in catalog.vector_matrix_rows(model)])


def test_index_chunks_and_embeds_every_accepted_document(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {
        "contrat_tel.txt": "Contrat telephone : prélèvement le 5 de chaque mois.",
        "bail.txt": long_text("bail", 4),
    })
    fake = FakeClassifier()
    report = index(catalog, fake)
    assert report.documents == 2 and report.unavailable is None
    assert report.chunks == catalog.stats().chunks > 2
    assert set(fake.prefixes) == {"search_document: "} and set(fake.keep_alives) == {"30m"}
    assert any(t.startswith("contrat_tel.txt p. 1\nContrat telephone") for t in fake.embedded)
    assert count(catalog, "SELECT COUNT(*) FROM chunk_vectors") == report.chunks


def test_all_chunks_of_a_run_go_in_batches_of_32(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {f"d{i}.txt": long_text(f"sujet{i}", 20) for i in range(4)})
    fake = FakeClassifier()
    report = index(catalog, fake)
    sizes = [len(b) for b in fake.embed_batches]
    assert sum(sizes) == report.chunks > 32
    assert all(s == 32 for s in sizes[:-1]) and 0 < sizes[-1] <= 32


def test_second_run_does_nothing_and_redo_does_all(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location appartement"})
    index(catalog, FakeClassifier())
    assert index(catalog, FakeClassifier()) == IndexReport(0, 0, 0, None)
    assert index(catalog, FakeClassifier(), redo=True).documents == 2
    assert catalog.stats().chunks == 2


def test_new_chunker_version_reindexes(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": long_text("bail", 6)})
    first = index(catalog, FakeClassifier()).chunks
    report = index(catalog, FakeClassifier(), ChunkConfig(chunk_chars=300, overlap_chars=0))
    assert report.documents == 1 and report.chunks > first
    assert catalog.stats().chunks == report.chunks  # the old chunks are gone


def test_under_limits_the_run_and_none_covers_everything(catalog, tmp_path):
    add_files(catalog, tmp_path / "a", {"x.txt": "Facture EDF de mars"})
    add_files(catalog, tmp_path / "b", {"y.txt": "Bail de location appartement"})
    assert index(catalog, FakeClassifier(), under=(tmp_path / "a").resolve()).documents == 1
    assert index(catalog, FakeClassifier()).documents == 1  # the one outside "a"


def test_chunks_never_hold_sensitive_values(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"rib.txt": f"Pour le paiement, mon IBAN {IBAN} merci."})
    fake = FakeClassifier()
    index(catalog, fake)
    texts = [r.text for r in _all_rows(catalog).values()]
    assert texts and all("3000" not in t and "7890" not in t for t in texts)
    assert all("3000" not in t for t in fake.embedded) and any("[IBAN]" in t for t in fake.embedded)


def test_document_with_no_ok_page_is_skipped_quietly(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location appartement"})
    catalog._conn.execute("UPDATE pages SET status = 'set_aside' WHERE document_id = ?", (ids["b.txt"],))
    catalog._conn.commit()
    assert index(catalog, FakeClassifier()) == IndexReport(1, 1, 0, None)
    assert index(catalog, FakeClassifier()) == IndexReport(0, 0, 0, None)  # no nagging


def test_ollama_dropping_mid_run_keeps_whole_documents(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {f"doc{i}.txt": long_text(f"sujet{i}", 40) for i in (1, 2, 3)})
    fake = FakeClassifier(fail_embed_on="doc3.txt")
    report = index(catalog, fake)
    assert report.unavailable == (
        "Ollama not reachable at http://fake: start it (ollama serve), then: localmind index"
    )
    assert report.documents >= 1
    for doc_id in ids.values():
        chunks = count(catalog, "SELECT COUNT(*) FROM chunks WHERE document_id = ?", doc_id)
        vectors = count(
            catalog,
            "SELECT COUNT(*) FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id WHERE c.document_id = ?",
            doc_id,
        )
        assert chunks == vectors  # whole document, or nothing
    assert count(catalog, "SELECT COUNT(DISTINCT document_id) FROM chunks") == report.documents
    assert index(catalog, FakeClassifier()).documents == 3 - report.documents  # the rest, later


def test_edited_file_replaces_its_chunks(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Ancienne phrase du contrat."})
    index(catalog, FakeClassifier())
    add_files(catalog, tmp_path / "in", {"a.txt": "Nouvelle phrase du contrat."})
    index(catalog, FakeClassifier())
    texts = " ".join(r.text for r in _all_rows(catalog).values())
    assert "Nouvelle" in texts and "Ancienne" not in texts


def test_on_document_is_called_per_stored_document(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location appartement"})
    seen = []
    index(catalog, FakeClassifier(), on_document=seen.append)
    assert sorted(seen) == sorted(ids.values())


class _BrokenAfterFirstBatch(FakeClassifier):
    """Ollama answers once, then sends a malformed answer (wrong vector count)."""

    def embed(self, texts, prefix="classification: ", keep_alive=None):
        if self.embed_batches:
            raise ValueError("expected 32 embeddings, got 31")
        return super().embed(texts, prefix, keep_alive)


def test_malformed_ollama_answer_stops_the_run_with_a_hint(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {f"doc{i}.txt": long_text(f"sujet{i}", 20) for i in (1, 2, 3)})
    report = index(catalog, _BrokenAfterFirstBatch())  # 20 chunks each: doc1 fits in batch 1
    assert report.unavailable == (
        "Ollama gave an unexpected answer (expected 32 embeddings, got 31), then: localmind index"
    )
    assert report.documents >= 1  # what was complete is kept
