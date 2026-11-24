"""Stage 5b CLI: ``facts extract``, ``facts list``, analysis after ingest, stats."""

from __future__ import annotations

import re
from datetime import date

import pytest

import localmind
from localmind import cli_facts, cli_search
from localmind.catalog import Catalog
from localmind.catalog_facts import NewFact, PageResult
from localmind.cli import main
from localmind.facts import PROMPT_VERSION
from tests.fake_classifier import FakeClassifier
from tests.stage4_data import add_files


def item(kind, amount=None, date_=None, who="EDF"):
    return {"kind": kind, "amount": amount, "date": date_, "time": None, "counterparty": who,
            "method": None, "place": None, "with_whom": None, "label": None}


def labeller(payload):
    """One payment per tagged amount, with the first date of the page."""
    text = payload["text"]
    dates = re.findall(r"\[(D\w+)\]", text)
    facts = [item("payment", m, dates[0] if dates else None) for m in re.findall(r"\[(M\w+)\]", text)]
    return {"document_type": "invoice", "facts": facts}


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = {"llm": FakeClassifier(answers=labeller), "db": str(tmp_path / "c.db")}
    monkeypatch.setattr(cli_facts, "make_llm", lambda: state["llm"])
    monkeypatch.setattr(cli_facts, "_today", lambda: date(2026, 11, 24))
    return state


def nf(kind="payment", cents=8420, iso="2026-03-15", who="EDF", dom=None):
    return NewFact(1, kind, cents, "EUR" if cents is not None else None, iso, dom, None, who,
                   None, None, None, None, None, None)


def stored(env, tmp_path, **facts_by_name):
    with Catalog(tmp_path / "c.db") as catalog:
        ids = add_files(catalog, tmp_path / "in",
                        {f"{n}.txt": f"Document {n} avec du texte suffisant." for n in facts_by_name})
        for name, fs in facts_by_name.items():
            catalog.store_extraction(ids[f"{name}.txt"], "m", PROMPT_VERSION,
                                     [PageResult(1, "done", tuple(fs))], rejected=0, capped=False)


def run(env, capsys, *args):
    code = main(["--db", env["db"], *args])
    out, err = capsys.readouterr()
    return code, out, err


def test_euros_format():
    assert cli_facts.euros(8420) == "84,20 €"
    assert cli_facts.euros(107800) == "1 078,00 €"
    assert cli_facts.euros(-8420) == "-84,20 €"
    assert cli_facts.euros(5) == "0,05 €"
    assert cli_facts.euros(123456789) == "1 234 567,89 €"


def test_list_prints_one_exact_line_per_fact(env, tmp_path, capsys):
    stored(env, tmp_path, facture=[nf()])
    code, out, _ = run(env, capsys, "facts", "list")
    assert code == 0
    assert out == "15/03/2026  payment            84,20 €  EDF  facture.txt p.1\n"


def test_list_day_of_month_duplicate_and_missing_values(env, tmp_path, capsys):
    stored(env, tmp_path,
           free=[nf("due", 1999, None, "Free", dom=5)],
           a=[nf(iso="2026-03-15")], b=[nf(iso="2026-03-16")],
           rdv=[nf("appointment", None, "2026-12-12", None)])
    _, out, _ = run(env, capsys, "facts", "list")
    lines = out.splitlines()
    assert lines[0] == "15/03/2026  payment            84,20 €  EDF  a.txt p.1"
    assert lines[1] == "16/03/2026  payment            84,20 €  EDF  b.txt p.1  (duplicate)"
    assert lines[2] == "12/12/2026  appointment             --  --  rdv.txt p.1"
    assert lines[3] == "le 5 du mois  due                19,99 €  Free  free.txt p.1"


def test_sum_prints_totals(env, tmp_path, capsys):
    stored(env, tmp_path,
           ech=[nf("due", 9800, f"2026-{m:02d}-05") for m in range(1, 12)],
           mars=[nf(cents=8420, iso="2026-03-20")])
    _, out, _ = run(env, capsys, "facts", "list", "--who", "edf", "--sum")
    assert out.splitlines()[-1] == "Total: 1 162,20 € (confirmed 84,20 €, unconfirmed 1 078,00 €)"


def test_filters_reach_the_query(env, tmp_path, capsys):
    stored(env, tmp_path,
           a=[nf(iso="2026-03-15")], b=[nf(iso="2025-03-15", cents=1)],
           c=[nf("appointment", None, "2026-12-12", None)], d=[nf("appointment", None, "2026-01-12", None)])
    _, out, _ = run(env, capsys, "facts", "list", "--from", "2026-01-01", "--to", "2026-12-31", "--kind", "payment")
    assert [line.split("  ")[-1] for line in out.splitlines()] == ["a.txt p.1"]
    _, out, _ = run(env, capsys, "facts", "list", "--upcoming", "--kind", "appointment")
    assert [line.split("  ")[-1] for line in out.splitlines()] == ["c.txt p.1"]


def test_coverage_line(env, tmp_path, capsys):
    stored(env, tmp_path, a=[nf()])
    with Catalog(tmp_path / "c.db") as catalog:
        add_files(catalog, tmp_path / "in2", {"new.txt": "Un document jamais analysé ici."})
    _, out, _ = run(env, capsys, "facts", "list")
    assert out.splitlines()[-1] == "  1 document(s) not analysed yet, 0 incomplete: localmind facts extract"


def test_no_fact(env, tmp_path, capsys):
    stored(env, tmp_path, a=[])
    _, out, _ = run(env, capsys, "facts", "list")
    assert out == "No fact.\n"


@pytest.mark.parametrize(
    "args, message",
    [
        (["--folder", "Nulle/Part"], "error: no folder named 'Nulle/Part'"),
        (["--from", "15/03/2026"], "error: bad date '15/03/2026' (expected YYYY-MM-DD)"),
        (["--to", "2026-02-30"], "error: bad date '2026-02-30' (expected YYYY-MM-DD)"),
    ],
)
def test_list_errors_exit_2(env, tmp_path, capsys, args, message):
    stored(env, tmp_path, a=[nf()])
    code, out, err = run(env, capsys, "facts", "list", *args)
    assert (code, out, err.strip()) == (2, "", message)


def test_extract_reports(env, tmp_path, capsys):
    with Catalog(tmp_path / "c.db") as catalog:
        add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF 84,20 € prélevée le 15/03/2026."})
    code, out, _ = run(env, capsys, "facts", "extract")
    assert (code, out) == (0, "Analysed 1 document(s): 1 fact(s)\n")
    env["llm"] = FakeClassifier(answers=lambda p: {"document_type": "invoice", "facts": [item("gift"), item("due")]})
    _, out, _ = run(env, capsys, "facts", "extract", "--all")
    assert out == "Analysed 1 document(s): 0 fact(s)\n  2 answer(s) rejected\n"
    env["llm"] = FakeClassifier(answers=[ValueError("x"), ValueError("x")])
    _, out, _ = run(env, capsys, "facts", "extract", "--all")
    assert out == "Analysed 1 document(s): 0 fact(s)\n  1 document(s) incomplete\n"


def test_extract_without_ollama_exits_2_with_hint(env, tmp_path, capsys):
    with Catalog(tmp_path / "c.db") as catalog:
        add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF 84,20 € prélevée le 15/03/2026."})
    env["llm"] = FakeClassifier(unavailable=True)
    code, out, err = run(env, capsys, "facts", "extract")
    assert code == 2
    assert "localmind facts extract" in err


def test_ingest_analyses_new_documents(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli_search, "make_embedder", lambda: FakeClassifier())
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "a.txt").write_text("Facture EDF 84,20 € prélevée le 15/03/2026.", encoding="utf-8")
    _, out, _ = run(env, capsys, "ingest", str(inbox), "--no-classify", "--no-ocr")
    assert "Analysed 1 document(s): 1 fact(s)" in out
    assert len(env["llm"].chats) == 1


def test_ingest_no_facts_makes_no_llm_call(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli_search, "make_embedder", lambda: FakeClassifier())
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "a.txt").write_text("Facture EDF 84,20 € prélevée le 15/03/2026.", encoding="utf-8")
    _, out, _ = run(env, capsys, "ingest", str(inbox), "--no-classify", "--no-ocr", "--no-facts")
    assert env["llm"].chats == []
    assert "Analysed" not in out


def test_stats_rows(env, tmp_path, capsys):
    stored(env, tmp_path, a=[nf(), nf(iso="2026-04-15")], b=[])
    _, out, _ = run(env, capsys, "stats")
    assert "facts:           2\n" in out and "analysed docs:   2\n" in out


def test_public_api():
    for name in ("extract_facts", "ExtractReport", "facts", "Fact", "FactsResult", "Coverage"):
        assert name in localmind.__all__ and hasattr(localmind, name)
    assert localmind.__version__ == "0.5.0"
