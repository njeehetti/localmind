"""Schema v6: the keyword index (FTS5) over chunks, kept in step by triggers."""

import sqlite3

import pytest

from localmind.catalog import SCHEMA_VERSION, Catalog, Decision
from localmind.chunking import Chunk
from localmind.prune import missing_files, prune
from tests.stage4_data import IBAN, add_files, count


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def _store(catalog, doc_id, texts, model="m1"):
    chunks = [Chunk(1, i, t) for i, t in enumerate(texts)]
    catalog.store_chunks(doc_id, "v", model, chunks, [[1.0, 0.0]] * len(chunks))


def _ids(catalog, doc_id):
    rows = catalog._conn.execute("SELECT id FROM chunks WHERE document_id = ? ORDER BY seq", (doc_id,))
    return [r[0] for r in rows]


def test_keyword_rows_find_accentless_words(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], ["Relevé électricité de mars", "Conditions générales"])
    assert catalog.keyword_rows('"electricite"', 10) == [_ids(catalog, ids["a.txt"])[0]]


def test_keyword_rows_rank_by_bm25(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], ["mars", "facture mars électricité", "rien"])
    first, second, _ = _ids(catalog, ids["a.txt"])
    assert catalog.keyword_rows('"facture" OR "mars"', 10) == [second, first]
    assert catalog.keyword_rows('"facture" OR "mars"', 1) == [second]


def test_replacing_chunks_updates_the_keyword_index(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], ["ancien mot girafe"])
    _store(catalog, ids["a.txt"], ["nouveau mot zebre"])
    assert catalog.keyword_rows('"girafe"', 10) == []
    assert catalog.keyword_rows('"zebre"', 10) == _ids(catalog, ids["a.txt"])


def test_updating_chunk_text_updates_the_keyword_index(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], ["girafe"])
    catalog._conn.execute("UPDATE chunks SET text = 'zebre'")
    assert catalog.keyword_rows('"girafe"', 10) == []
    assert catalog.keyword_rows('"zebre"', 10) == _ids(catalog, ids["a.txt"])


def test_orphan_cascade_and_prune_remove_keywords(catalog, tmp_path):
    inbox = tmp_path / "in"
    ids = add_files(catalog, inbox, {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location"})
    _store(catalog, ids["a.txt"], ["girafe"])
    _store(catalog, ids["b.txt"], ["zebre"])
    add_files(catalog, inbox, {"a.txt": "Un tout autre texte"})  # edited in place: orphan cascade
    assert catalog.keyword_rows('"girafe"', 10) == []
    (inbox / "b.txt").unlink()
    prune(catalog, missing_files(catalog))
    assert catalog.keyword_rows('"zebre"', 10) == []
    assert count(catalog, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebre OR girafe'") == 0


def test_masked_iban_is_not_findable(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"rib.txt": f"Mon IBAN {IBAN} merci"})
    [(page_no, text)] = catalog.masked_pages(ids["rib.txt"])
    _store(catalog, ids["rib.txt"], [text])
    assert catalog.keyword_rows('"30006000011234567890189"', 10) == []
    assert catalog.keyword_rows('"3000"', 10) == []
    assert catalog.keyword_rows('"iban"', 10) == _ids(catalog, ids["rib.txt"])


def test_keyword_rows_respect_document_ids(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars", "b.txt": "Bail de location"})
    _store(catalog, ids["a.txt"], ["girafe a"])
    _store(catalog, ids["b.txt"], ["girafe b"])
    assert catalog.keyword_rows('"girafe"', 10, document_ids=[ids["b.txt"]]) == _ids(catalog, ids["b.txt"])
    assert catalog.keyword_rows('"girafe"', 10, document_ids=set()) == []
    assert len(catalog.keyword_rows('"girafe"', 10)) == 2


def test_v5_catalog_is_migrated_with_its_chunks_searchable(tmp_path):
    db = tmp_path / "c.db"
    with Catalog(db) as cat:
        ids = add_files(cat, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
        cat.record_decision("abc", Decision.ACCEPT)
        _store(cat, ids["a.txt"], ["girafe", "zebre"])
        vectors = cat.vector_matrix_rows("m1")
    con = sqlite3.connect(db)
    con.executescript(
        "DROP TRIGGER chunks_ai; DROP TRIGGER chunks_ad; DROP TRIGGER chunks_au;"
        "DROP TABLE chunks_fts; PRAGMA user_version = 5;"
    )
    con.close()
    with Catalog(db) as cat:
        assert cat._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 6
        assert cat.keyword_rows('"girafe"', 10) == _ids(cat, ids["a.txt"])[:1]
        assert cat.vector_matrix_rows("m1") == vectors
        assert cat.decision_for("abc") is Decision.ACCEPT


def test_v1_rebuild_drops_keyword_index_too(tmp_path):
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
        "CREATE TABLE chunks_fts (junk TEXT);"
        "INSERT INTO chunks_fts VALUES ('girafe');"
    )
    con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        assert cat.keyword_rows('"girafe"', 10) == []
        ids = add_files(cat, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
        _store(cat, ids["a.txt"], ["girafe"])
        assert cat.keyword_rows('"girafe"', 10) == _ids(cat, ids["a.txt"])


def test_fts_syntax_in_questions_never_raises(catalog, tmp_path):  # Review Focus 1
    from localmind.keyword import keyword_query
    from tests.test_keyword import FTS_SYNTAX

    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars"})
    _store(catalog, ids["a.txt"], ["Relevé électricité de mars"])
    for q in FTS_SYNTAX:
        match = keyword_query(q)
        if match:
            catalog.keyword_rows(match, 10)
    assert catalog.keyword_rows(keyword_query("l'électricité"), 10) == _ids(catalog, ids["a.txt"])
