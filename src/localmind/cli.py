"""Command-line interface: a thin layer over the public API, no logic here."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localmind.catalog import Catalog
from localmind.config import default_db_path
from localmind.ingest import IngestReport, Outcome, ingest

_SYMBOLS = {
    Outcome.NEW: "+",
    Outcome.UPDATED: "~",
    Outcome.DUPLICATE: "=",
    Outcome.UNCHANGED: ".",
    Outcome.UNSUPPORTED: "?",
    Outcome.FAILED: "!",
}


def _print_report(report: IngestReport, verbose: bool) -> None:
    for result in report.results:
        if result.outcome is Outcome.UNCHANGED and not verbose:
            continue
        detail = f"  ({result.detail})" if result.detail else ""
        print(f" {_SYMBOLS[result.outcome]} {result.outcome:<11} {result.path}{detail}")
    counts = report.counts()
    summary = ", ".join(f"{counts[o]} {o}" for o in Outcome if counts[o])
    print(f"\n{len(report.results)} file(s): {summary or 'nothing to do'}")


def _cmd_ingest(args: argparse.Namespace) -> int:
    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: path not found: {p}", file=sys.stderr)
        return 2
    with Catalog(args.db) as catalog:
        for path in args.paths:
            _print_report(ingest(path, catalog), args.verbose)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with Catalog(args.db) as catalog:
        s = catalog.stats()
    print(f"catalog:         {args.db}")
    print(f"documents:       {s.documents}")
    print(f"files:           {s.files}")
    print(f"duplicate files: {s.duplicate_files}")
    return 0


def _cmd_duplicates(args: argparse.Namespace) -> int:
    with Catalog(args.db) as catalog:
        groups = catalog.duplicate_groups()
    if not groups:
        print("No duplicates.")
        return 0
    for group in groups:
        print(f"Document #{group.document_id} ({len(group.paths)} files):")
        for path in group.paths:
            print(f"   {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localmind", description="A fully local RAG over your documents."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_path(),
        help="catalog database (default: $LOCALMIND_DB or ~/.localmind/catalog.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="read files and add them to the catalog")
    p_ingest.add_argument("paths", nargs="+", type=Path, help="files or folders")
    p_ingest.add_argument("-v", "--verbose", action="store_true", help="show unchanged files")
    p_ingest.set_defaults(func=_cmd_ingest)

    sub.add_parser("stats", help="catalog summary").set_defaults(func=_cmd_stats)
    sub.add_parser("duplicates", help="list files sharing the same content").set_defaults(
        func=_cmd_duplicates
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
