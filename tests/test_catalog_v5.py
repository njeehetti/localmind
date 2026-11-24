"""Schema v5: chunks and chunk vectors."""

import sqlite3

import pytest

from localmind.catalog import SCHEMA_VERSION, Catalog, Decision
from localmind.chunking import Chunk
from tests.stage4_data import add_files, count


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def _store(catalog, doc_id, model="m1", n=2, chunker="v"):
    chunks = [Chunk(1, i, f"morceau {i}") for i in range(n)]
    catalog.store_chunks(doc_id, chunker, model, chunks, [[1.0, 0.0]] * n)


def test_v4_catalog_gains_chunk_tables_and_keeps_data(tmp_path):
    db = tmp_path / "c.db"
    with Catalog(db) as cat:
        add_files(cat, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
        cat.record_decision("abc", Decision.ACCEPT)
    con = sqlite3.connect(db)
    con.executescript("DROP TABLE chunk_vectors; DROP TABLE chunks; PRAGMA user_version = 4;")
    con.close()
    with Catalog(db) as cat:
        assert cat._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 7
        assert cat.decision_for("abc") is Decision.ACCEPT
        assert cat.stats().documents == 1 and cat.stats().chunks == 0


def test_store_chunks_replaces_and_counts(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], n=3)
    _store(catalog, ids["a.txt"], n=2)
    assert catalog.stats().chunks == 2 and catalog.stats().indexed_documents == 1
    rows = catalog.chunk_rows([cid for cid, _ in catalog.vector_matrix_rows("m1")])
    assert sorted(r.text for r in rows.values()) == ["morceau 0", "morceau 1"]
    assert all(r.paths == ((tmp_path / "in" / "a.txt").resolve(),) for r in rows.values())


def test_orphan_document_takes_its_chunks_and_vectors(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"])
    add_files(catalog, tmp_path / "in", {"a.txt": "Un tout autre texte"})  # edited in place
    assert count(catalog, "SELECT COUNT(*) FROM chunks") == 0
    assert count(catalog, "SELECT COUNT(*) FROM chunk_vectors") == 0


def test_vectors_are_kept_per_model(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location appartement"})
    _store(catalog, ids["a.txt"], model="m1")
    _store(catalog, ids["b.txt"], model="m2")
    assert len(catalog.vector_matrix_rows("m1")) == 2
    assert len(catalog.vector_matrix_rows("m2")) == 2
    assert catalog.documents_to_index("v", "m1") == [ids["b.txt"]]


def test_marker_changes_whenever_chunks_change(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    before = catalog.vector_marker()
    _store(catalog, ids["a.txt"], n=2)
    middle = catalog.vector_marker()
    _store(catalog, ids["a.txt"], n=2)  # same count, same rowids reused
    assert before != middle != catalog.vector_marker()


def test_masked_pages_hide_sensitive_values(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"rib.txt": "Mon IBAN FR76 3000 6000 0112 3456 7890 189 merci"})
    [(page_no, text)] = catalog.masked_pages(ids["rib.txt"])
    assert page_no == 1 and "[IBAN]" in text and "3000" not in text


def test_v1_rebuild_drops_chunks_but_never_decisions(tmp_path):
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY, junk TEXT);"
        "CREATE TABLE decisions (file_hash TEXT PRIMARY KEY, decision TEXT NOT NULL, "
        "decided_at TEXT NOT NULL);"
        "INSERT INTO decisions VALUES ('abc', 'ignore', '2026-01-01');"
    )
    con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        assert cat.decision_for("abc") is Decision.IGNORE
        assert cat.stats().chunks == 0  # the new chunks table, not the junk one
