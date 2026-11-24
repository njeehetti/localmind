"""Stage 5b: amounts, dates and times found by rules, and their inline tags."""

from __future__ import annotations

import re

import pytest

from localmind.candidates import Amount, DateCandidate, TimeCandidate, annotate, find_candidates


@pytest.mark.parametrize(
    "text, cents",
    [
        ("84,20 €", 8420),
        ("84,20 EUR", 8420),
        ("1 234,56 €", 123456),
        ("1 234,56 €", 123456),
        ("1 234,56 €", 123456),
        ("1.234,56 €", 123456),
        ("€ 84,20", 8420),
        ("€ 1.234,56", 123456),
        ("-84,20 €", -8420),
        ("−84,20 €", -8420),
        ("+2 145,30 €", 214530),
        ("84.20 €", 8420),
        ("1.234 €", 123400),
        ("98 euros", 9800),
        ("1 euro", 100),
        ("Montant : 98,00 €", 9800),
        ("[IBAN] 84,20 €", 8420),
        ("84,5 €", 8450),
    ],
)
def test_amounts(text, cents):
    found = find_candidates(text)
    assert [a.cents for a in found.amounts] == [cents]
    assert found.amounts[0].currency == "EUR"


@pytest.mark.parametrize(
    "text",
    ["01 23 45 67 89", "N° 2026-03-15-A", "[IBAN] ", "Quantité 3", "12345", "Contrat n° 4 556 778", "412 kWh"],
)
def test_not_amounts(text):
    assert find_candidates(text).amounts == ()


@pytest.mark.parametrize(
    "text, iso",
    [
        ("15/03/2026", "2026-03-15"),
        ("15/03/26", "2026-03-15"),
        ("5/3/2026", "2026-03-05"),
        ("15-03-2026", "2026-03-15"),
        ("15.03.2026", "2026-03-15"),
        ("15.03.26", "2026-03-15"),
        ("01/01/99", "1999-01-01"),
        ("2026-03-15", "2026-03-15"),
        ("15 mars 2026", "2026-03-15"),
        ("1er mars 2026", "2026-03-01"),
        ("12 décembre 2026", "2026-12-12"),
        ("12 decembre 2026", "2026-12-12"),
        ("3 Août 2026", "2026-08-03"),
    ],
)
def test_dates(text, iso):
    dates = find_candidates(text).dates
    assert [(d.iso, d.day_of_month) for d in dates] == [(iso, None)]


@pytest.mark.parametrize(
    "text", ["prélevé le 5 de chaque mois", "le 5 du mois", "Prélèvement le 1er de chaque mois"]
)
def test_recurring_day(text):
    dates = find_candidates(text).dates
    expected = 1 if "1er" in text else 5
    assert [(d.iso, d.day_of_month) for d in dates] == [(None, expected)]


@pytest.mark.parametrize("text", ["mars 2027", "Période : mars 2026", "en décembre 2026"])
def test_month_only_is_not_a_candidate(text):
    assert find_candidates(text).dates == ()


@pytest.mark.parametrize("text", ["31/02/2026", "00/01/2026", "15/13/2026", "le 32 de chaque mois"])
def test_impossible_dates_dropped(text):
    assert find_candidates(text).dates == ()


@pytest.mark.parametrize(
    "text, hhmm", [("14h30", "14:30"), ("14:30", "14:30"), ("14 h", "14:00"), ("à 9h15", "09:15"), ("8 h 05", "08:05")]
)
def test_times(text, hhmm):
    assert [t.hhmm for t in find_candidates(text).times] == [hhmm]


@pytest.mark.parametrize("text", ["84,20 €", "14 heures", "15/03/2026", "TVA 20 %"])
def test_not_times(text):
    assert find_candidates(text).times == ()


@pytest.mark.parametrize(
    "text, cents",
    [  # final review: a number before an amount must not become its thousands group
        ("Qté 2 450,00 €", 45000),
        ("Échéance n° 4 250,00 €", 25000),
        ("Mensualité 3 150,00 €", 15000),
        ("15/03 150,00 €", 15000),
        ("Article 12 345,00 €", 34500),
        ("TOTAL 2 450,00 €", 245000),
        ("Total 1 234 567,89 €", 123456789),
    ],
)
def test_a_number_before_an_amount_is_not_its_thousands(text, cents):
    assert [a.cents for a in find_candidates(text).amounts] == [cents]


def test_a_date_is_never_also_an_amount_or_a_time():
    found = find_candidates("Intervention du 15.03.26 à 14 h, total € 1.234,56")
    assert [d.iso for d in found.dates] == ["2026-03-15"]
    assert [a.cents for a in found.amounts] == [123456]
    assert [t.hhmm for t in found.times] == ["14:00"]


def test_spans_point_at_the_text():
    text = "Total TTC : 84,20 € prélevé le 15/03/2026 à 14h30"
    found = find_candidates(text)
    assert "84,20" in text[found.amounts[0].start : found.amounts[0].end]
    assert text[found.dates[0].start : found.dates[0].end] == "15/03/2026"
    assert text[found.times[0].start : found.times[0].end] == "14h30"


def test_ids_are_unique_typed_and_not_sequential():
    text = "\n".join(f"05/{m:02d}/2026  Prélèvement  98,00 €" for m in range(1, 13))
    found = find_candidates(text)
    ids = [c.id for c in (*found.amounts, *found.dates, *found.times)]
    assert len(ids) == len(set(ids)) == 24
    assert all(re.fullmatch(r"M\d[A-Z]+", a.id) for a in found.amounts)
    assert all(re.fullmatch(r"D\d[A-Z]+", d.id) for d in found.dates)
    assert {a.id for a in found.amounts}.isdisjoint({f"M{i}" for i in range(30)})


def test_ids_stay_unique_on_a_huge_page():
    text = " ".join(f"{i % 28 + 1:02d}/01/2026 {i},00 €" for i in range(400))
    found = find_candidates(text)
    ids = [c.id for c in (*found.amounts, *found.dates)]
    assert len(ids) == len(set(ids)) == 800


def test_get_returns_the_candidate_or_none():
    found = find_candidates("84,20 € le 15/03/2026 à 14h30")
    amount, day, hour = found.amounts[0], found.dates[0], found.times[0]
    assert isinstance(amount, Amount) and found.get(amount.id) is amount
    assert isinstance(day, DateCandidate) and found.get(day.id) is day
    assert isinstance(hour, TimeCandidate) and found.get(hour.id) is hour
    assert found.get("M999Z") is None


def test_annotate_puts_each_tag_before_its_value():
    text = "Total TTC : 84,20 € le 15/03/2026"
    found = find_candidates(text)
    m, d = found.amounts[0].id, found.dates[0].id
    assert annotate(text, found) == f"Total TTC : [{m}] 84,20 € le [{d}] 15/03/2026"


def test_annotate_without_candidates_is_the_text():
    assert annotate("Aucun montant.", find_candidates("Aucun montant.")) == "Aucun montant."
