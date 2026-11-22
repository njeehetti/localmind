"""Stage 2b CLI: the OCR offer at the end of ingest, retry with OCR, listings."""

import pytest

from localmind import cli
from localmind.cli import _estimate, main
from localmind.ocr import OcrUnavailable
from tests.pdf_factory import make_pdf
from tests.test_ocr_ingest import GOOD, FakeEngine


@pytest.fixture
def env(tmp_path, monkeypatch):
    """OCR installed, interactive terminal, user answers no; override per test."""
    state = {"asked": 0, "answer": False, "engine": FakeEngine([(GOOD, 0.95)] * 5)}

    def ask(prompt):
        state["asked"] += 1
        assert prompt == "Run OCR now? [y/N] "
        return state["answer"]

    monkeypatch.setattr(cli, "ocr_available", lambda: True)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_ask", ask)
    monkeypatch.setattr(cli, "load_engine", lambda: state["engine"])
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    state["db"] = str(tmp_path / "c.db")
    state["inbox"] = inbox
    return state


def never(*_args):
    raise AssertionError("must not be called")


def run(env, *args):
    return main(["--db", env["db"], *args])


def test_offer_yes_runs_ocr(env, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    env["answer"] = True
    assert run(env, "ingest", str(env["inbox"])) == 0
    out = capsys.readouterr().out
    assert "1 scanned page(s) in 1 file(s) can be read by OCR (~1 s on GPU)." in out
    assert "OCR page 1/1: scan.pdf p.1" in out
    run(env, "stats")
    assert "ocr pages:       1" in capsys.readouterr().out


def test_offer_no_does_nothing_and_asks_again_next_time(env, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    run(env, "ingest", str(env["inbox"]))
    run(env, "ingest", str(env["inbox"]))
    assert env["asked"] == 2 and env["engine"].calls == 0


def test_non_interactive_never_asks(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    monkeypatch.setattr(cli, "_ask", never)
    assert run(env, "ingest", str(env["inbox"])) == 0
    out = capsys.readouterr().out
    assert "1 scanned page(s) can be read by OCR: localmind quarantine retry" in out
    assert env["engine"].calls == 0


def test_ocr_flag_runs_without_asking(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    monkeypatch.setattr(cli, "_ask", never)
    run(env, "ingest", str(env["inbox"]), "--ocr")
    assert env["engine"].calls == 1


def test_no_ocr_flag_skips(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    monkeypatch.setattr(cli, "_ask", never)
    run(env, "ingest", str(env["inbox"]), "--no-ocr")
    assert "scanned page" not in capsys.readouterr().out


def test_not_installed_prints_install_hint(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf(["", ""]))
    monkeypatch.setattr(cli, "ocr_available", lambda: False)
    monkeypatch.setattr(cli, "_ask", never)
    run(env, "ingest", str(env["inbox"]))
    assert "2 scanned page(s) could be read with OCR: uv sync --extra ocr" in capsys.readouterr().out


def test_offer_counts_only_ingested_paths_and_reminds_others(env, capsys):
    a, b = env["inbox"] / "a", env["inbox"] / "b"
    a.mkdir()
    b.mkdir()
    (a / "old.pdf").write_bytes(make_pdf([""]))
    (b / "new.pdf").write_bytes(make_pdf(["", ""]))
    run(env, "ingest", str(a), "--no-ocr")
    capsys.readouterr()
    run(env, "ingest", str(b))
    out = capsys.readouterr().out
    assert "2 scanned page(s) in 1 file(s) can be read by OCR" in out
    assert "(1 more scanned page(s) are waiting: localmind quarantine retry)" in out


def test_nothing_to_read_prints_nothing(env, capsys):
    (env["inbox"] / "a.txt").write_text(GOOD, encoding="utf-8")
    run(env, "ingest", str(env["inbox"]))
    assert "scanned page" not in capsys.readouterr().out and env["asked"] == 0


def test_retry_uses_engine_when_available(env, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    run(env, "ingest", str(env["inbox"]), "--no-ocr")
    run(env, "quarantine", "retry")
    assert env["engine"].calls == 1


def test_retry_no_ocr_flag_skips_engine(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))
    run(env, "ingest", str(env["inbox"]), "--no-ocr")
    monkeypatch.setattr(cli, "load_engine", never)
    assert run(env, "quarantine", "retry", "--no-ocr") == 0


def test_engine_that_cannot_start_is_reported(env, monkeypatch, capsys):
    (env["inbox"] / "scan.pdf").write_bytes(make_pdf([""]))

    def broken():
        raise OcrUnavailable("CUDA error")

    monkeypatch.setattr(cli, "load_engine", broken)
    assert run(env, "ingest", str(env["inbox"]), "--ocr") == 0
    assert "CUDA error" in capsys.readouterr().err


def test_quarantine_lists_ocr_confidence(env, capsys):
    (env["inbox"] / "blurry.pdf").write_bytes(make_pdf([GOOD, ""]))
    env["engine"] = FakeEngine([(GOOD, 0.58)])
    run(env, "ingest", str(env["inbox"]), "--ocr")
    run(env, "quarantine")
    assert "page 2  [ocr_low_confidence 58 %]" in capsys.readouterr().out


def test_stats_shows_ocr_and_blank_pages(env, capsys):
    run(env, "stats")
    out = capsys.readouterr().out
    assert "ocr pages:       0" in out and "blank pages:     0" in out


def test_estimate_formats():
    assert _estimate(1, "GPU") == "1 s"
    assert _estimate(10, "CPU") == "50 s"
    assert _estimate(14, "CPU") == "2 min"
    assert _estimate(300, "GPU") == "3 min"
