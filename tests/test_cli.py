from localmind.cli import main


def test_ingest_then_stats_and_duplicates(tmp_path, capsys):
    db = tmp_path / "c.db"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text("Same", encoding="utf-8")
    (inbox / "b.txt").write_text("Same", encoding="utf-8")

    assert main(["--db", str(db), "ingest", str(inbox)]) == 0
    out = capsys.readouterr().out
    assert "1 new" in out and "1 duplicate" in out

    assert main(["--db", str(db), "stats"]) == 0
    assert "duplicate files: 1" in capsys.readouterr().out

    assert main(["--db", str(db), "duplicates"]) == 0
    out = capsys.readouterr().out
    assert "a.txt" in out and "b.txt" in out


def test_missing_path_returns_error_code(tmp_path, capsys):
    code = main(["--db", str(tmp_path / "c.db"), "ingest", str(tmp_path / "nope")])
    assert code == 2
    assert "path not found" in capsys.readouterr().err


def test_db_path_can_come_from_environment(tmp_path, monkeypatch):
    from localmind.config import default_db_path

    monkeypatch.setenv("LOCALMIND_DB", str(tmp_path / "x.db"))
    assert default_db_path() == tmp_path / "x.db"
