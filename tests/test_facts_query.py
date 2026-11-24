"""Stage 5b: answering JARVIS from stored facts (filters, kinds, duplicates, totals)."""

from __future__ import annotations

from datetime import date

import pytest

from localmind.catalog import Catalog, Classification
from localmind.catalog_facts import NewFact, PageResult
from localmind.facts import PROMPT_VERSION
from localmind.facts_query import Coverage, facts
from localmind.folder_actions import choose_folder
from localmind.folders import UnknownFolder, parse_folder
from tests.stage4_data import add_files

TODAY = date(2026, 11, 24)


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def nf(kind="payment", cents=8420, iso="2026-03-15", who="EDF", page_no=1, dom=None, time=None,
       currency="EUR", place=None):
    return NewFact(page_no, kind, cents, currency if cents is not None else None, iso, dom, time, who,
                   None, place, None, "label", None, None)


def docs(catalog, tmp_path, **facts_by_name):
    """One document per name (text made unique), its facts stored."""
    ids = add_files(catalog, tmp_path / "in",
                    {f"{n}.txt": f"Document {n} avec du texte suffisant." for n in facts_by_name})
    for name, fs in facts_by_name.items():
        catalog.store_extraction(ids[f"{name}.txt"], "m", PROMPT_VERSION,
                                 [PageResult(1, "done", tuple(fs))], rejected=0, capped=False)
    return {n: ids[f"{n}.txt"] for n in facts_by_name}


def test_question_1_electricity_2026(catalog, tmp_path):
    docs(catalog, tmp_path,
         mars=[nf(cents=8420, iso="2026-03-15")],
         avril=[nf(cents=8420, iso="2026-04-15")],
         ech=[nf("due", 9800, "2026-10-05"), nf("due", 9800, "2026-12-05")],
         gaz=[nf(cents=5000, iso="2026-03-15", who="Engie")],
         old=[nf(cents=7000, iso="2025-12-15")])
    r = facts(catalog, kind=("payment", "due"), counterparty="edf",
              date_from=date(2026, 1, 1), date_to=date(2026, 12, 31), today=TODAY)
    assert [(f.kind, f.amount_cents, f.date) for f in r.facts] == [
        ("payment", 8420, date(2026, 3, 15)), ("payment", 8420, date(2026, 4, 15)),
        ("due", 9800, date(2026, 10, 5)), ("due", 9800, date(2026, 12, 5))]
    assert (r.confirmed_cents, r.unconfirmed_cents, r.total_cents) == (16840, 9800, 26640)


def test_counterparty_matches_by_substring_of_the_key(catalog, tmp_path):
    docs(catalog, tmp_path, a=[nf(who="EDF SA")], b=[nf(who="Électricité de France", cents=1)])
    assert [f.counterparty for f in facts(catalog, counterparty="EDF").facts] == ["EDF SA"]
    assert [f.amount_cents for f in facts(catalog, counterparty="electricite").facts] == [1]


def test_payment_and_due_same_amount_4_days_apart_counted_once(catalog, tmp_path):
    docs(catalog, tmp_path, facture=[nf("due", 9800, "2026-03-05")], recu=[nf("payment", 9800, "2026-03-09")])
    r = facts(catalog, today=TODAY)
    due = next(f for f in r.facts if f.kind == "due")
    payment = next(f for f in r.facts if f.kind == "payment")
    assert due.duplicate_of == payment.id and payment.duplicate_of is None
    assert (r.confirmed_cents, r.unconfirmed_cents) == (9800, 0)


def test_two_identical_payments_keep_the_lower_id(catalog, tmp_path):
    docs(catalog, tmp_path, a=[nf(iso="2026-03-15")], b=[nf(iso="2026-03-16")])
    first, second = facts(catalog, today=TODAY).facts
    assert (first.duplicate_of, second.duplicate_of) == (None, first.id)
    assert facts(catalog, today=TODAY).confirmed_cents == 8420


@pytest.mark.parametrize("other", [nf(iso="2026-03-21"), nf(cents=8421), nf(who="Engie"), nf(who=None)])
def test_not_duplicates(catalog, tmp_path, other):
    first = nf(who=other.counterparty) if other.counterparty is None else nf()
    docs(catalog, tmp_path, a=[first], b=[other])
    r = facts(catalog, today=TODAY)
    assert all(f.duplicate_of is None for f in r.facts)
    assert r.confirmed_cents == first.amount_cents + other.amount_cents


def test_statement_folder_makes_bank_movements_left_out_of_totals(catalog, tmp_path):
    ids = docs(catalog, tmp_path, rel=[nf(cents=-8420)], fac=[nf()])
    assert facts(catalog, today=TODAY).confirmed_cents == 0  # -8420 + 8420 as payments
    choose_folder(catalog, ids["rel"], "Banque/Relevés")
    r = facts(catalog, today=TODAY)
    assert {f.kind for f in r.facts} == {"bank_movement", "payment"}
    assert r.confirmed_cents == 8420
    assert [f.kind for f in facts(catalog, kind="bank_movement", today=TODAY).facts] == ["bank_movement"]


def test_a_statement_folder_still_to_confirm_counts_too(catalog, tmp_path):
    ids = docs(catalog, tmp_path, rel=[nf(cents=-8420)])
    folder_id = catalog.ensure_folder(parse_folder("Banque/Releves bancaires"), "llm", False)
    catalog.set_classification(Classification(ids["rel"], folder_id, "to_confirm", None, None, None, None), "m")
    (f,) = facts(catalog, today=TODAY).facts
    assert (f.kind, f.folder) == ("bank_movement", "Banque/Releves bancaires")


def test_upcoming_appointments_sorted_from_today(catalog, tmp_path):
    docs(catalog, tmp_path,
         a=[nf("appointment", None, "2026-12-12", who=None, time="14:30", place="Hôpital Nord")],
         b=[nf("appointment", None, "2026-11-30", who=None)],
         c=[nf("appointment", None, "2026-11-01", who=None)],
         d=[nf("appointment", None, "2026-11-24", who=None)])
    r = facts(catalog, kind="appointment", upcoming=True, today=TODAY)
    assert [f.date for f in r.facts] == [date(2026, 11, 24), date(2026, 11, 30), date(2026, 12, 12)]
    assert (r.facts[-1].time, r.facts[-1].place) == ("14:30", "Hôpital Nord")


def test_recurring_due_found_without_dates_and_never_summed(catalog, tmp_path):
    docs(catalog, tmp_path, free=[nf("due", 1999, None, who="Free Mobile", dom=5)])
    (f,) = facts(catalog, kind="due", counterparty="free", today=TODAY).facts
    assert (f.day_of_month, f.date) == (5, None)
    assert facts(catalog, today=TODAY).unconfirmed_cents == 0
    assert facts(catalog, upcoming=True, today=TODAY).facts == ()
    assert facts(catalog, date_from=date(2026, 1, 1), today=TODAY).facts == ()


def test_future_dues_listed_not_summed_and_non_euro_never_summed(catalog, tmp_path):
    docs(catalog, tmp_path, a=[nf("due", 9800, "2027-01-05"), nf(cents=5000, currency="USD", who="Shop")])
    r = facts(catalog, today=TODAY)
    assert len(r.facts) == 2
    assert (r.confirmed_cents, r.unconfirmed_cents) == (0, 0)


def test_rows_carry_file_and_page_for_citing(catalog, tmp_path):
    docs(catalog, tmp_path, mars=[nf(page_no=1)])
    (f,) = facts(catalog, today=TODAY).facts
    assert (f.file, f.page_no, f.folder) == ("mars.txt", 1, None)


def test_folder_filter_and_coverage_in_folder_scope(catalog, tmp_path):
    ids = add_files(catalog, tmp_path / "in", {
        n: f"Document {n} avec du texte suffisant." for n in ("done.txt", "new.txt", "inc.txt", "other.txt", "old.txt")})
    catalog.store_extraction(ids["done.txt"], "m", PROMPT_VERSION, [PageResult(1, "done", (nf(),))], rejected=0, capped=False)
    catalog.store_extraction(ids["inc.txt"], "m", PROMPT_VERSION,
                             [PageResult(1, "done", ()), PageResult(2, "failed", ())], rejected=0, capped=False)
    catalog.store_extraction(ids["old.txt"], "m", "0", [PageResult(1, "done", ())], rejected=0, capped=False)
    for name in ("done.txt", "new.txt", "inc.txt", "old.txt"):
        choose_folder(catalog, ids[name], "Maison/Factures")
    choose_folder(catalog, ids["other.txt"], "Santé/Ordonnances")
    r = facts(catalog, folder="maison", today=TODAY)
    assert [f.file for f in r.facts] == ["done.txt"]
    assert r.coverage == Coverage(missing=2, incomplete=1, failed=0)  # new + old prompt
    assert facts(catalog, today=TODAY).coverage == Coverage(missing=3, incomplete=1, failed=0)


def test_unknown_folder_and_kind_raise(catalog):
    with pytest.raises(UnknownFolder):
        facts(catalog, folder="Nulle/Part")
    with pytest.raises(ValueError):
        facts(catalog, kind="gift")
