"""Schema v3 (stage 2b): OCR provenance, OCR cache, blank pages, safe migrations."""

import sqlite3
from pathlib import Path

import pytest

from localmind.catalog import Catalog, Decision, PendingOcr
from localmind.ocr import OcrPage
from localmind.quality import PageVerdict, Reason

# Exact stage 2a schema (v2), frozen here to test the migration.
V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    content_key  TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    page_count   INTEGER,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('ok', 'set_aside')),
    reason       TEXT,
    UNIQUE (document_id, page_no)
);
CREATE TABLE IF NOT EXISTS sensitive_spans (
    id       INTEGER PRIMARY KEY,
    page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL,
    start    INTEGER NOT NULL,
    "end"    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    file_hash    TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    encoding     TEXT,
    status       TEXT NOT NULL CHECK (status IN ('indexed', 'quarantined', 'ignored')),
    reason       TEXT,
    document_id  INTEGER REFERENCES documents(id),
    ingested_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    file_hash   TEXT PRIMARY KEY,
    decision    TEXT NOT NULL CHECK (decision IN ('accept', 'ignore')),
    decided_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
"""


@pytest.fixture
def cat(tmp_path):
    with Catalog(tmp_path / "c.db") as c:
        yield c


def user_version(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def make_v2_catalog(db):
    con = sqlite3.connect(db)
    con.executescript(V2_SCHEMA)
    con.executescript(
        "INSERT INTO documents VALUES (1, 'k1', 'pdf', 2, '2026-01-01');"
        "INSERT INTO pages VALUES (1, 1, 1, 'Facture EDF de mars', 'ok', NULL);"
        "INSERT INTO pages VALUES (2, 1, 2, '', 'set_aside', 'needs_ocr');"
        "INSERT INTO files VALUES (1, '/x/a.pdf', 'h1', 10, NULL, 'indexed', NULL, 1, '2026-01-01');"
        "INSERT INTO decisions VALUES ('abc', 'accept', '2026-01-01');"
        "PRAGMA user_version = 2;"
    )
    con.commit()
    con.close()


def test_v2_catalog_migrates_in_place_keeping_everything(tmp_path):
    db = tmp_path / "c.db"
    make_v2_catalog(db)
    with Catalog(db) as cat:
        assert not cat.rebuilt_from_v1
        assert cat.decision_for("abc") is Decision.ACCEPT
        assert cat.stats().files == 1 and cat.stats().documents == 1
        assert cat.stats().partial == 1
        assert cat.ocr_pending() == PendingOcr(pages=1, files=1)
    assert user_version(db) == 3


def test_v1_rebuild_keeps_decisions(tmp_path):
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
        "CREATE TABLE decisions (file_hash TEXT PRIMARY KEY, decision TEXT NOT NULL, "
        "decided_at TEXT NOT NULL);"
        "INSERT INTO decisions VALUES ('abc', 'ignore', '2026-01-01');"
    )
    con.commit()
    con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        assert cat.decision_for("abc") is Decision.IGNORE


def test_ocr_cache_roundtrip(cat):
    cat.store_ocr("h", 2, OcrPage("Bonjour", 0.9, "doctr-x"))
    assert cat.ocr_cached("h", 2) == OcrPage("Bonjour", 0.9, "doctr-x")
    assert cat.ocr_cached("h", 3) is None
    cat.store_ocr("h", 2, OcrPage("Bonjour tout", 0.95, "doctr-x"))
    assert cat.ocr_cached("h", 2).text == "Bonjour tout"


def test_blank_and_ocr_pages_are_stored_and_counted(cat):
    pages = [
        PageVerdict(1, "Texte lu par OCR ici", None, (), "ocr", "doctr-x", 0.93),
        PageVerdict(2, "", Reason.OCR_NO_TEXT, (), "ocr", "doctr-x", 0.0),
    ]
    doc = cat.add_document("k", "pdf", 2, pages)
    cat.link_file(Path("/x/scan.pdf"), "h1", 10, doc)
    s = cat.stats()
    assert (s.ocr_pages, s.blank_pages, s.partial, s.accepted) == (1, 1, 0, 1)
    assert cat.set_aside_pages() == []
    assert cat.retry_candidates() == []
    assert cat.document_for_path(Path("/x/scan.pdf")).status == "accepted"


def test_set_aside_page_carries_ocr_confidence(cat):
    pages = [
        PageVerdict(1, "Texte correct ici pour la page", None),
        PageVerdict(2, "fl0u", Reason.OCR_LOW_CONFIDENCE, (), "ocr", "doctr-x", 0.58),
    ]
    doc = cat.add_document("k", "pdf", 2, pages)
    cat.link_file(Path("/x/f.pdf"), "h1", 10, doc)
    [page] = cat.set_aside_pages()
    assert page.reason is Reason.OCR_LOW_CONFIDENCE and page.ocr_confidence == 0.58


def test_pending_counts_unknown_page_count_as_one(cat):
    cat.record_quarantine(Path("/s/a.pdf"), "h1", 10, Reason.NEEDS_OCR, page_count=4)
    cat.record_quarantine(Path("/s/b.jpg"), "h2", 10, Reason.NEEDS_OCR)
    cat.record_quarantine(Path("/s/c.txt"), "h3", 10, Reason.TOO_SHORT, page_count=1)
    assert cat.ocr_pending() == PendingOcr(pages=5, files=2)
    assert cat.ocr_pending(Path("/s/a.pdf")) == PendingOcr(pages=4, files=1)
    assert cat.ocr_pending(Path("/elsewhere")) == PendingOcr(pages=0, files=0)


def test_ocr_candidates_exclude_other_reasons(cat):
    garbled = [PageVerdict(1, "Texte correct ici pour la page", None),
               PageVerdict(2, "@@##", Reason.GARBLED)]
    scanned = [PageVerdict(1, "Texte correct ici pour la page", None),
               PageVerdict(2, "", Reason.NEEDS_OCR)]
    cat.link_file(Path("/x/g.pdf"), "h1", 10, cat.add_document("k1", "pdf", 2, garbled))
    cat.link_file(Path("/x/s.pdf"), "h2", 10, cat.add_document("k2", "pdf", 2, scanned))
    cat.record_quarantine(Path("/x/photo.jpg"), "h3", 10, Reason.NEEDS_OCR, page_count=1)
    cat.record_quarantine(Path("/x/log.txt"), "h4", 10, Reason.MACHINE_LOG)
    assert [r.path.name for r in cat.ocr_candidates()] == ["photo.jpg", "s.pdf"]
    assert cat.ocr_pending() == PendingOcr(pages=2, files=2)
