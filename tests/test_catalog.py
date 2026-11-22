import sqlite3
from pathlib import Path

import pytest

from localmind.catalog import (
    Catalog,
    Decision,
    FileStatus,
    SensitiveFlag,
    SetAsidePage,
)
from localmind.quality import PageVerdict, Reason
from localmind.sensitive import SensitiveSpan


@pytest.fixture
def cat(tmp_path):
    with Catalog(tmp_path / "c.db") as c:
        yield c


def ok(n, text, spans=()):
    return PageVerdict(n, text, None, spans)


def store(cat, path, pages, key="k1", file_hash="h1"):
    doc = cat.add_document(key, "pdf", len(pages), pages)
    cat.link_file(path, file_hash, 10, doc)
    return doc


def test_round_trip_partial_document(cat, tmp_path):
    p = tmp_path / "a.pdf"
    store(cat, p, [ok(1, "page one text"), PageVerdict(2, "", Reason.NEEDS_OCR)])
    doc = cat.document_for_path(p)
    assert doc.status == "partial"
    assert doc.text == "page one text"
    assert cat.set_aside_pages() == [SetAsidePage(p, 2, Reason.NEEDS_OCR)]
    assert cat.file_by_path(p).status is FileStatus.INDEXED


def test_accepted_document(cat, tmp_path):
    p = tmp_path / "a.txt"
    store(cat, p, [ok(1, "fine text here")])
    assert cat.document_for_path(p).status == "accepted"
    assert cat.stats().accepted == 1


def test_record_quarantine(cat, tmp_path):
    p = tmp_path / "scan.pdf"
    cat.record_quarantine(p, "h9", 5, Reason.NEEDS_OCR)
    (rec,) = cat.quarantined_files()
    assert rec.path == p and rec.reason is Reason.NEEDS_OCR and rec.document_id is None


def test_quarantining_indexed_path_drops_orphan_document(cat, tmp_path):
    p = tmp_path / "a.txt"
    store(cat, p, [ok(1, "some text here")])
    cat.record_quarantine(p, "h2", 5, Reason.TOO_SHORT)
    assert cat.stats().documents == 0
    assert cat.set_aside_pages() == []


def test_decisions_round_trip(cat):
    assert cat.decision_for("h1") is None
    cat.record_decision("h1", Decision.ACCEPT)
    assert cat.decision_for("h1") is Decision.ACCEPT
    cat.record_decision("h1", Decision.IGNORE)
    assert cat.decision_for("h1") is Decision.IGNORE


def test_sensitive_flags_expose_kind_and_page_only(cat, tmp_path):
    p = tmp_path / "rib.txt"
    store(cat, p, [ok(1, "IBAN FR76...", (SensitiveSpan("iban", 5, 12),))])
    assert cat.sensitive_flags() == [SensitiveFlag(p, 1, "iban")]
    assert cat.stats().sensitive_flags == 1


def test_retry_candidates_filtered_by_folder(cat, tmp_path):
    sub = tmp_path / "sub"
    cat.record_quarantine(sub / "a.pdf", "h1", 1, Reason.NEEDS_OCR)
    cat.record_quarantine(tmp_path / "b.pdf", "h2", 1, Reason.NEEDS_OCR)
    store(cat, sub / "c.pdf", [ok(1, "good text page"), PageVerdict(2, "", Reason.NEEDS_OCR)], key="k3", file_hash="h3")
    store(cat, sub / "d.txt", [ok(1, "fully fine text")], key="k4", file_hash="h4")
    assert sorted(r.path.name for r in cat.retry_candidates()) == ["a.pdf", "b.pdf", "c.pdf"]
    assert sorted(r.path.name for r in cat.retry_candidates(under=sub)) == ["a.pdf", "c.pdf"]


def test_record_ignored_and_stats(cat, tmp_path):
    p = tmp_path / "a.txt"
    store(cat, p, [ok(1, "some text here")])
    cat.record_ignored(p, "h1", 10)
    s = cat.stats()
    assert (s.documents, s.files, s.ignored) == (0, 1, 1)
    assert cat.file_by_path(p).status is FileStatus.IGNORED


def test_file_by_hash_prefers_indexed(cat, tmp_path):
    cat.record_quarantine(tmp_path / "q.pdf", "same", 1, Reason.NEEDS_OCR)
    store(cat, tmp_path / "i.pdf", [ok(1, "text text text")], file_hash="same")
    assert cat.file_by_hash("same").status is FileStatus.INDEXED
    assert cat.file_by_hash("nope") is None


def test_duplicate_groups_only_indexed(cat, tmp_path):
    doc = store(cat, tmp_path / "a.txt", [ok(1, "same text here")])
    cat.link_file(tmp_path / "b.txt", "h2", 10, doc)
    cat.record_quarantine(tmp_path / "c.jpg", "h3", 1, Reason.NEEDS_OCR)
    (group,) = cat.duplicate_groups()
    assert sorted(p.name for p in group.paths) == ["a.txt", "b.txt"]
    assert cat.stats().duplicate_files == 1


def test_v1_catalog_is_rebuilt(tmp_path):  # Review Focus 5
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
        "INSERT INTO documents VALUES (1, 'x', 'old');"
    )
    con.commit()
    con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        assert cat.stats().documents == 0
    with Catalog(db) as cat:
        assert not cat.rebuilt_from_v1
