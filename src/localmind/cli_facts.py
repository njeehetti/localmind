"""Command-line side of stage 5b: ``facts extract``, ``facts list`` and the
analysis after ingest. It only prints: the work is done by ``facts`` and
``facts_query``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

from localmind.catalog import Catalog
from localmind.facts import KINDS, ExtractReport, extract_facts
from localmind.facts_query import Fact, FactsResult, facts
from localmind.folders import UnknownFolder
from localmind.ollama import Classifier, OllamaClassifier
from localmind.search import CatalogBusy

EXTRACT = "localmind facts extract"


def make_llm() -> Classifier:
    return OllamaClassifier()


def _today() -> date:
    return date.today()


def _open(db: Path) -> Catalog:
    from localmind import cli  # late import: cli imports this module

    return cli._open_catalog(db)


def euros(cents: int) -> str:
    """``107800`` -> ``"1 078,00 €"`` (French: space thousands, comma decimals)."""
    sign = "-" if cents < 0 else ""
    whole, frac = divmod(abs(cents), 100)
    return f"{sign}{whole:,}".replace(",", " ") + f",{frac:02d} €"


def print_extract(report: ExtractReport, always: bool = True) -> None:
    if always or report.facts or report.rejected or report.incomplete or report.failed:
        print(f"Analysed {report.documents} document(s): {report.facts} fact(s)")
        if report.rejected:
            print(f"  {report.rejected} answer(s) rejected")
        if report.incomplete or report.failed:
            print(f"  {report.incomplete + report.failed} document(s) incomplete")
    if report.unavailable:
        print(report.unavailable)


def facts_after_ingest(catalog: Catalog) -> None:
    """Read the facts of every document not read yet (not only the ingested
    ones). Quiet when nothing was found."""
    print_extract(extract_facts(catalog, make_llm()), always=False)


def fact_line(f: Fact) -> str:
    """``15/03/2026  payment            84,20 €  EDF  facture.txt p.1``"""
    if f.date is not None:
        when = f.date.strftime("%d/%m/%Y")
    elif f.day_of_month is not None:
        when = f"le {f.day_of_month} du mois"
    else:
        when = "--"
    amount = euros(f.amount_cents) if f.amount_cents is not None else "--"
    who = f.counterparty or f.place or f.label or "--"
    line = f"{when}  {f.kind:<13} {amount:>12}  {who}  {f.file} p.{f.page_no}"
    return line + ("  (duplicate)" if f.duplicate_of is not None else "")


def _print_result(result: FactsResult, with_sum: bool) -> None:
    if not result.facts:
        print("No fact.")
    for f in result.facts:
        print(fact_line(f))
    if with_sum:
        print(
            f"Total: {euros(result.total_cents)} (confirmed {euros(result.confirmed_cents)}, "
            f"unconfirmed {euros(result.unconfirmed_cents)})"
        )
    cov = result.coverage
    if cov.missing or cov.incomplete or cov.failed:
        print(f"  {cov.missing} document(s) not analysed yet, {cov.incomplete + cov.failed} incomplete: {EXTRACT}")


class _BadDate(ValueError):
    pass


def _day(text: str | None) -> date | None:
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise _BadDate(f"bad date '{text}' (expected YYYY-MM-DD)") from None


# -- commands -------------------------------------------------------------
def _cmd_extract(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        report = extract_facts(catalog, make_llm(), redo=args.all)
    if report.unavailable and not report.documents:
        print(report.unavailable, file=sys.stderr)
        return 2
    print_extract(report)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        date_from, date_to = _day(args.date_from), _day(args.date_to)
        with _open(args.db) as catalog:
            result = facts(
                catalog, kind=args.kind, counterparty=args.who, folder=args.folder,
                date_from=date_from, date_to=date_to, upcoming=args.upcoming, today=_today(),
            )
    except (_BadDate, UnknownFolder) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (CatalogBusy, sqlite3.OperationalError) as exc:
        if isinstance(exc, sqlite3.OperationalError) and "locked" not in str(exc):
            raise
        print(f"error: {CatalogBusy()}", file=sys.stderr)
        return 2
    _print_result(result, args.sum)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p_facts = sub.add_parser("facts", help="payments, dues and appointments read from documents")
    f_sub = p_facts.add_subparsers(dest="facts_command", required=True)

    p_extract = f_sub.add_parser("extract", help="read the facts of documents not read yet")
    p_extract.add_argument("--all", action="store_true", help="read every document again")
    p_extract.set_defaults(func=_cmd_extract)

    p_list = f_sub.add_parser("list", help="list facts, with filters and totals")
    p_list.add_argument("--kind", action="append", choices=KINDS, help="repeatable")
    p_list.add_argument("--who", help="counterparty (part of its name, accents ignored)")
    p_list.add_argument("--folder", help="only this domain or folder (Maison, Maison/Factures)")
    p_list.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD")
    p_list.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD")
    p_list.add_argument("--upcoming", action="store_true", help="from today on, soonest first")
    p_list.add_argument("--sum", action="store_true", help="print the totals")
    p_list.set_defaults(func=_cmd_list)
