import sqlite3

import pytest

from localmind.cli import main
from tests.pdf_factory import make_pdf

GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."
IBAN_DIGITS = "30006000011234567890189"


@pytest.fixture
def env(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return tmp_path / "c.db", inbox


def run(db, *args):
    return main(["--db", str(db), *map(str, args)])


def test_ingest_then_stats_and_duplicates(env, capsys):
    db, inbox = env
    (inbox / "a.txt").write_text("Same text here", encoding="utf-8")
    (inbox / "b.txt").write_text("Same text here", encoding="utf-8")

    assert run(db, "ingest", inbox) == 0
    out = capsys.readouterr().out
    assert "1 new" in out and "1 duplicate" in out

    assert run(db, "stats") == 0
    assert "duplicate files: 1" in capsys.readouterr().out

    assert run(db, "duplicates") == 0
    out = capsys.readouterr().out
    assert "a.txt" in out and "b.txt" in out


def test_missing_path_returns_error_code(env, capsys):
    db, inbox = env
    assert run(db, "ingest", inbox / "nope") == 2
    assert "path not found" in capsys.readouterr().err


def test_db_path_can_come_from_environment(tmp_path, monkeypatch):
    from localmind.config import default_db_path

    monkeypatch.setenv("LOCALMIND_DB", str(tmp_path / "x.db"))
    assert default_db_path() == tmp_path / "x.db"


def test_quarantine_lists_files_pages_and_sensitive_without_values(env, capsys):
    db, inbox = env
    (inbox / "scan.pdf").write_bytes(make_pdf([""]))
    (inbox / "mixed.pdf").write_bytes(make_pdf([GOOD, ""]))
    (inbox / "rib.txt").write_text(GOOD + " IBAN FR76" + IBAN_DIGITS, encoding="utf-8")
    run(db, "ingest", inbox)
    capsys.readouterr()

    assert run(db, "quarantine") == 0
    out = capsys.readouterr().out
    assert "needs_ocr" in out and "scan.pdf" in out
    assert "mixed.pdf" in out and "page 2" in out
    assert "iban @ page 1" in out
    assert IBAN_DIGITS not in out


def test_ingest_output_never_shows_sensitive_values(env, capsys):
    db, inbox = env
    (inbox / "rib.txt").write_text(GOOD + " IBAN FR76" + IBAN_DIGITS, encoding="utf-8")
    run(db, "ingest", "-v", inbox)
    run(db, "stats")
    captured = capsys.readouterr()
    assert IBAN_DIGITS not in captured.out + captured.err


def test_accept_then_stats(env, capsys):
    db, inbox = env
    (inbox / "todo.txt").write_text("appeler", encoding="utf-8")
    run(db, "ingest", inbox)

    assert run(db, "quarantine", "accept", inbox / "todo.txt") == 0
    run(db, "stats")
    assert "accepted:        1" in capsys.readouterr().out


def test_ignore_command(env, capsys):
    db, inbox = env
    (inbox / "a.txt").write_text(GOOD, encoding="utf-8")
    run(db, "ingest", inbox)

    assert run(db, "quarantine", "ignore", inbox / "a.txt") == 0
    run(db, "stats")
    assert "ignored:         1" in capsys.readouterr().out


def test_decision_on_unknown_path(env, capsys):
    db, inbox = env
    f = inbox / "never.txt"
    f.write_text(GOOD, encoding="utf-8")
    assert run(db, "quarantine", "accept", f) == 2
    assert "not in catalog" in capsys.readouterr().err


def test_retry_command(env, capsys):
    db, inbox = env
    f = inbox / "scan.pdf"
    f.write_bytes(make_pdf([""]))
    run(db, "ingest", inbox)
    f.unlink()
    capsys.readouterr()

    assert run(db, "quarantine", "retry") == 0
    assert "missing" in capsys.readouterr().out


def test_sensitive_option_quarantines(env, capsys):
    db, inbox = env
    (inbox / "rib.txt").write_text(GOOD + " IBAN FR76" + IBAN_DIGITS, encoding="utf-8")
    run(db, "ingest", "--sensitive", "quarantine", inbox)
    assert "1 quarantined" in capsys.readouterr().out


def test_v1_catalog_prints_rebuild_notice(env, capsys):  # Review Focus 5
    db, inbox = env
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    assert run(db, "stats") == 0
    assert "rebuilt" in capsys.readouterr().err
