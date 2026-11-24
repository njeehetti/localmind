"""Stage 5b: answer questions from stored facts, for JARVIS and the CLI.

    facts(catalog, kind=("payment", "due"), counterparty="edf",
          date_from=date(2026, 1, 1), date_to=date(2026, 12, 31))

returns every matching fact (with file and page, to cite them), two totals
and the coverage. Decided here, at query time, never stored:

* **Effective kind**: a fact of a document whose folder type starts with
  "relev" ("Relevés", "Relevés bancaires"), placed or still to confirm, is a
  ``bank_movement``. Moving a document changes its facts with no new reading.
* **Probable duplicates**: same counterparty, same amount, dates at most 5
  days apart = one payment (a ``payment`` wins over a ``due``, else the older
  fact). The other row stays in the list, marked, but leaves the totals.
* **Totals**, euro only: ``confirmed`` = payments; ``unconfirmed`` = dues dated
  today or before. Future dues, day-of-month dues and bank lines are listed,
  never summed.
* **Coverage**: documents not read yet (or read by an older prompt), or read
  with failures, so a sum can say what it may be missing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import date as Day

from localmind.catalog import Catalog
from localmind.catalog_facts import FactRow, counterparty_key
from localmind.facts import KINDS, PROMPT_VERSION
from localmind.folders import fold, parse_folder_filter
from localmind.search import CatalogBusy

DUPLICATE_DAYS = 5
SUMMED_CURRENCY = "EUR"


@dataclass(frozen=True)
class Fact:
    id: int
    kind: str  # effective kind
    amount_cents: int | None
    currency: str | None
    date: Day | None
    day_of_month: int | None
    time: str | None
    counterparty: str | None
    method: str | None
    place: str | None
    with_whom: str | None
    label: str | None
    file: str
    page_no: int
    folder: str | None
    duplicate_of: int | None = None


@dataclass(frozen=True)
class Coverage:
    missing: int  # no reading by the current prompt
    incomplete: int
    failed: int


@dataclass(frozen=True)
class FactsResult:
    facts: tuple[Fact, ...]
    confirmed_cents: int
    unconfirmed_cents: int
    coverage: Coverage

    @property
    def total_cents(self) -> int:
        return self.confirmed_cents + self.unconfirmed_cents


def _kinds(kind: str | Collection[str] | None) -> set[str] | None:
    if kind is None:
        return None
    wanted = {kind} if isinstance(kind, str) else set(kind)
    unknown = wanted - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kind: {', '.join(sorted(unknown))} (expected {', '.join(KINDS)})")
    return wanted


def _is_statement(folder: str | None) -> bool:
    return folder is not None and fold(folder.split("/", 1)[1]).startswith("relev")


def _fact(row: FactRow, folder: str | None, file: str) -> Fact:
    kind = "bank_movement" if _is_statement(folder) else row.kind
    return Fact(
        id=row.id, kind=kind, amount_cents=row.amount_cents, currency=row.currency,
        date=Day.fromisoformat(row.date) if row.date else None, day_of_month=row.day_of_month,
        time=row.time, counterparty=row.counterparty, method=row.method, place=row.place,
        with_whom=row.with_whom, label=row.label, file=file, page_no=row.page_no, folder=folder,
    )


def _mark_duplicates(found: list[Fact]) -> list[Fact]:
    """Probable duplicates among payments and dues (see module docstring)."""
    duplicate_of: dict[int, int] = {}
    groups: dict[tuple[str, int], list[Fact]] = {}
    for f in found:
        key = counterparty_key(f.counterparty)
        if f.kind in ("payment", "due") and key and f.amount_cents is not None and f.date is not None:
            groups.setdefault((key, f.amount_cents), []).append(f)
    for group in groups.values():
        primaries: list[Fact] = []
        for f in sorted(group, key=lambda f: (f.date, f.id)):
            near = next((p for p in primaries if abs((f.date - p.date).days) <= DUPLICATE_DAYS), None)
            if near is None:
                primaries.append(f)
                continue
            f_wins = (f.kind, near.kind) == ("payment", "due") or (f.kind == near.kind and f.id < near.id)
            winner, loser = (f, near) if f_wins else (near, f)
            duplicate_of[loser.id] = winner.id
            if f_wins:
                primaries[primaries.index(near)] = f
                for lost, won in duplicate_of.items():  # rows that pointed at the loser
                    if won == near.id:
                        duplicate_of[lost] = f.id
    return [replace(f, duplicate_of=duplicate_of.get(f.id)) for f in found]


def _totals(found: list[Fact], today: Day) -> tuple[int, int]:
    confirmed = unconfirmed = 0
    for f in found:
        if f.duplicate_of is not None or f.currency != SUMMED_CURRENCY or f.amount_cents is None:
            continue
        if f.kind == "payment":
            confirmed += f.amount_cents
        elif f.kind == "due" and f.date is not None and f.date <= today:
            unconfirmed += f.amount_cents
    return confirmed, unconfirmed


def facts(
    catalog: Catalog,
    kind: str | Collection[str] | None = None,
    counterparty: str | None = None,
    folder: str | None = None,
    date_from: Day | None = None,
    date_to: Day | None = None,
    upcoming: bool = False,
    today: Day | None = None,
) -> FactsResult:
    """Facts matching every given filter, sorted by date (undated last).
    ``folder``: ``"Maison"`` or ``"Maison/Factures"`` (``UnknownFolder`` if no
    such folder). A date filter or ``upcoming`` (date >= today) drops undated
    facts. Raises ``CatalogBusy`` while an ingest holds the catalog."""
    wanted = _kinds(kind)
    today = today or Day.today()
    try:
        scope = catalog.document_folders(parse_folder_filter(folder)) if folder else None
        folders = catalog.document_folders()
        if upcoming and (date_from is None or date_from < today):
            date_from = today
        rows = catalog.fact_rows(
            document_ids=set(scope) if scope is not None else None,
            counterparty=counterparty_key(counterparty) if counterparty else None,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            dated_only=bool(date_from or date_to),
        )
        names: dict[int, str] = {}
        found = []
        for row in rows:
            if row.document_id not in names:
                names[row.document_id] = catalog.file_name(row.document_id)
            doc_folder = folders.get(row.document_id)
            fact = _fact(row, doc_folder.name if doc_folder else None, names[row.document_id])
            if wanted is None or fact.kind in wanted:
                found.append(fact)
        coverage = Coverage(*catalog.coverage_counts(PROMPT_VERSION, set(scope) if scope is not None else None))
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc):
            raise CatalogBusy() from exc
        raise
    found = _mark_duplicates(found)
    found.sort(key=lambda f: (f.date is None, f.date or Day.min, f.id))
    confirmed, unconfirmed = _totals(found, today)
    return FactsResult(tuple(found), confirmed, unconfirmed, coverage)
