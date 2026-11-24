"""Schema v7: facts, fact runs and fact pages, cascading from documents."""

import sqlite3

import pytest

from localmind.catalog import SCHEMA_VERSION, Catalog
from localmind.catalog_facts import NewFact, PageResult, counterparty_key
from localmind.prune import missing_files, prune
from tests.stage4_data import add_files, count


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def fact(page_no=1, kind="payment", cents=8420, iso="2026-03-15", who="EDF SA"):
    return NewFact(
        page_no=page_no, kind=kind, amount_cents=cents, currency="EUR", date=iso,
        day_of_month=None, time=None, counterparty=who, method="debit", place=None,
        with_whom=None, label="Facture", amount_span=(10, 17), date_span=(30, 40),
    )


def done(page_no, *facts):
    return PageResult(page_no, "done", tuple(facts))


def run(catalog, doc_id):
    return catalog._conn.execute(
        "SELECT status, capped, facts, rejected FROM fact_runs WHERE document_id = ?", (doc_id,)
    ).fetchone()


def test_v6_catalog_migrates_to_v7_keeping_its_data(tmp_path):
    db = tmp_path / "c.db"
    with Catalog(db) as cat:
        ids = add_files(cat, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    con = sqlite3.connect(db)
    con.executescript("DROP TABLE facts; DROP TABLE fact_pages; DROP TABLE fact_runs; PRAGMA user_version = 6;")
    con.close()
    with Catalog(db) as cat:
        assert cat._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 7
        assert cat.document_for_path((tmp_path / "in" / "a.txt").resolve()).id == ids["a.txt"]
        cat.store_extraction(ids["a.txt"], "qwen3:8b", "1", [done(1, fact())], rejected=0, capped=False)
        assert count(cat, "SELECT COUNT(*) FROM facts") == 1


def test_v1_rebuild_drops_fact_tables_too(tmp_path):
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
        "CREATE TABLE facts (junk TEXT);"
        "PRAGMA user_version = 1;"
    )
    con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        columns = {r[1] for r in cat._conn.execute("PRAGMA table_info(facts)")}
        assert "amount_cents" in columns


def test_store_extraction_writes_facts_and_run(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    catalog.store_extraction(ids["a.txt"], "qwen3:8b", "1", [done(1, fact())], rejected=2, capped=False)
    row = catalog._conn.execute(
        "SELECT kind, amount_cents, date, counterparty, counterparty_key, amount_start, date_end FROM facts"
    ).fetchone()
    assert row == ("payment", 8420, "2026-03-15", "EDF SA", "edf", 10, 40)
    assert run(catalog, ids["a.txt"]) == ("done", 0, 1, 2)


def test_facts_cascade_when_the_file_is_edited(catalog, tmp_path):
    inbox = tmp_path / "in"
    ids = add_files(catalog, inbox, {"a.txt": "Facture EDF de mars 84,20 €"})
    catalog.store_extraction(ids["a.txt"], "qwen3:8b", "1", [done(1, fact())], rejected=0, capped=False)
    add_files(catalog, inbox, {"a.txt": "Facture EDF d'avril 91,00 €"})
    assert count(catalog, "SELECT COUNT(*) FROM facts") == 0
    assert count(catalog, "SELECT COUNT(*) FROM fact_runs") == 0
    assert count(catalog, "SELECT COUNT(*) FROM fact_pages") == 0


def test_facts_cascade_when_the_file_is_pruned(catalog, tmp_path):
    inbox = tmp_path / "in"
    ids = add_files(catalog, inbox, {"a.txt": "Facture EDF de mars 84,20 €"})
    catalog.store_extraction(ids["a.txt"], "qwen3:8b", "1", [done(1, fact())], rejected=0, capped=False)
    (inbox / "a.txt").unlink()
    prune(catalog, missing_files(catalog))
    assert count(catalog, "SELECT COUNT(*) FROM facts") == 0


def test_store_extraction_replaces_only_the_given_pages(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    doc = ids["a.txt"]
    catalog.store_extraction(
        doc, "m", "1", [done(1, fact(1)), PageResult(2, "failed", ())], rejected=1, capped=False
    )
    assert run(catalog, doc) == ("incomplete", 0, 1, 1)
    assert catalog.failed_pages(doc) == [2]
    catalog.store_extraction(doc, "m", "1", [done(2, fact(2, cents=500))], rejected=1, capped=False)
    pages = [r[0] for r in catalog._conn.execute("SELECT page_no FROM facts ORDER BY page_no")]
    assert pages == [1, 2]
    assert run(catalog, doc) == ("done", 0, 2, 2)  # rejected accumulates within one run
    assert catalog.failed_pages(doc) == []


def test_a_new_prompt_version_starts_the_run_afresh(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    doc = ids["a.txt"]
    catalog.store_extraction(doc, "m", "1", [done(1, fact(1))], rejected=3, capped=False)
    catalog.store_extraction(doc, "m", "2", [done(1, fact(1, cents=1))], rejected=0, capped=False)
    assert run(catalog, doc) == ("done", 0, 1, 0)
    assert count(catalog, "SELECT amount_cents FROM facts") == 1


@pytest.mark.parametrize(
    "pages, capped, status",
    [
        ([PageResult(1, "failed", ()), PageResult(2, "failed", ())], False, "failed"),
        ([PageResult(1, "no_candidates", ()), PageResult(2, "no_candidates", ())], False, "no_candidates"),
        ([PageResult(1, "no_candidates", ()), PageResult(2, "done", ())], False, "done"),
        ([PageResult(1, "done", ()), PageResult(2, "failed", ())], False, "incomplete"),
        ([PageResult(1, "done", ())], True, "incomplete"),
    ],
)
def test_run_status_rules(catalog, tmp_path, pages, capped, status):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    catalog.store_extraction(ids["a.txt"], "m", "1", pages, rejected=0, capped=capped)
    assert run(catalog, ids["a.txt"])[0] == status


def test_documents_to_extract(catalog, tmp_path):
    ids = add_files(
        catalog,
        tmp_path / "in",
        {n: f"Document {n} avec assez de texte, 84,20 €" for n in ("new.txt", "done.txt", "old.txt", "inc.txt", "cap.txt")},
    )
    catalog.store_extraction(ids["done.txt"], "m", "1", [done(1)], rejected=0, capped=False)
    catalog.store_extraction(ids["old.txt"], "m", "0", [done(1)], rejected=0, capped=False)
    catalog.store_extraction(
        ids["inc.txt"], "m", "1", [done(1), PageResult(2, "failed", ())], rejected=0, capped=False
    )
    catalog.store_extraction(ids["cap.txt"], "m", "1", [done(1)], rejected=0, capped=True)
    todo = catalog.documents_to_extract("m", "1")
    assert sorted(todo) == sorted([ids["new.txt"], ids["old.txt"], ids["inc.txt"]])
    assert sorted(catalog.documents_to_extract("other-model", "1")) == sorted(ids.values())
    assert sorted(catalog.documents_to_extract("m", "1", redo=True)) == sorted(ids.values())


def test_fact_run_reports_the_current_run(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF de mars 84,20 €"})
    assert catalog.fact_run(ids["a.txt"]) is None
    catalog.store_extraction(ids["a.txt"], "m", "1", [done(1)], rejected=0, capped=False)
    current = catalog.fact_run(ids["a.txt"])
    assert (current.model, current.prompt_version, current.status, current.capped) == ("m", "1", "done", False)


@pytest.mark.parametrize(
    "name, key",
    [
        ("EDF SA", "edf"),
        ("Électricité de France", "electricite de france"),
        ("  Free   Mobile ", "free mobile"),
        ("Plomberie Durand SARL", "plomberie durand"),
        ("S.N.C.F.", "s n c f"),
        (None, ""),
        ("", ""),
    ],
)
def test_counterparty_key(name, key):
    assert counterparty_key(name) == key
