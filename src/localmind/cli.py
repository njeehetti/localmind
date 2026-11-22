"""Command-line interface: a thin layer over the public API, no logic here.

Sensitive values are never printed: only their kind and page number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

from localmind.catalog import Catalog, Decision
from localmind.config import default_db_path, sensitive_policy_from_env
from localmind.ingest import IngestReport, Outcome, apply_decision, ingest, retry
from localmind.quality import QualityConfig, SensitivePolicy

_SYMBOLS = {
    Outcome.NEW: "+",
    Outcome.PARTIAL: "~",
    Outcome.UPDATED: "^",
    Outcome.DUPLICATE: "=",
    Outcome.UNCHANGED: ".",
    Outcome.QUARANTINED: "!",
    Outcome.IGNORED: "x",
    Outcome.MISSING: "?",
}


def _open_catalog(db: Path) -> Catalog:
    catalog = Catalog(db)
    if catalog.rebuilt_from_v1:
        print(
            "note: catalog rebuilt for the new format (v2); run `localmind ingest` again.",
            file=sys.stderr,
        )
    return catalog


def _config(args: argparse.Namespace) -> QualityConfig:
    return QualityConfig(sensitive_policy=args.sensitive)


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
    with _open_catalog(args.db) as catalog:
        for path in args.paths:
            _print_report(ingest(path, catalog, _config(args)), args.verbose)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with _open_catalog(args.db) as catalog:
        s = catalog.stats()
    rows = [
        ("catalog", args.db),
        ("documents", s.documents),
        ("accepted", s.accepted),
        ("partial", s.partial),
        ("files", s.files),
        ("duplicate files", s.duplicate_files),
        ("quarantined", s.quarantined),
        ("ignored", s.ignored),
        ("sensitive flags", s.sensitive_flags),
    ]
    for label, value in rows:
        print(f"{label + ':':<17}{value}")
    return 0


def _cmd_duplicates(args: argparse.Namespace) -> int:
    with _open_catalog(args.db) as catalog:
        groups = catalog.duplicate_groups()
    if not groups:
        print("No duplicates.")
        return 0
    for group in groups:
        print(f"Document #{group.document_id} ({len(group.paths)} files):")
        for path in group.paths:
            print(f"   {path}")
    return 0


def _cmd_quarantine_list(args: argparse.Namespace) -> int:
    with _open_catalog(args.db) as catalog:
        files = catalog.quarantined_files()
        pages = catalog.set_aside_pages()
        flags = catalog.sensitive_flags()

    print(f"Quarantined files ({len(files)}):")
    current = None
    for record in files:  # sorted by reason, then path
        if record.reason != current:
            current = record.reason
            print(f"  [{current}]")
        print(f"    {record.path}")

    print(f"\nSet-aside pages ({len(pages)}):")
    for page in pages:
        print(f"    {page.path}  page {page.page_no}  [{page.reason}]")

    print(f"\nSensitive data, kept and flagged ({len(flags)}):")
    for flag in flags:
        print(f"    {flag.path}  {flag.kind} @ page {flag.page_no}")
    return 0


def _cmd_retry(args: argparse.Namespace) -> int:
    under = args.path.resolve() if args.path else None
    with _open_catalog(args.db) as catalog:
        _print_report(retry(catalog, _config(args), under), verbose=True)
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    with _open_catalog(args.db) as catalog:
        try:
            result = apply_decision(args.path, args.decision, catalog)
        except KeyError:
            print(f"error: not in catalog: {args.path} (run `ingest` first)", file=sys.stderr)
            return 2
    detail = f"  ({result.detail})" if result.detail else ""
    print(f" {_SYMBOLS[result.outcome]} {result.outcome:<11} {result.path}{detail}")
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

    policy = argparse.ArgumentParser(add_help=False)
    policy.add_argument(
        "--sensitive",
        choices=get_args(SensitivePolicy),
        default=sensitive_policy_from_env(),
        help="what to do with sensitive data (default: $LOCALMIND_SENSITIVE_POLICY or flag)",
    )

    p_ingest = sub.add_parser(
        "ingest", parents=[policy], help="read files and add them to the catalog"
    )
    p_ingest.add_argument("paths", nargs="+", type=Path, help="files or folders")
    p_ingest.add_argument("-v", "--verbose", action="store_true", help="show unchanged files")
    p_ingest.set_defaults(func=_cmd_ingest)

    sub.add_parser("stats", help="catalog summary").set_defaults(func=_cmd_stats)
    sub.add_parser("duplicates", help="list files sharing the same content").set_defaults(
        func=_cmd_duplicates
    )

    p_q = sub.add_parser("quarantine", help="list and manage what was set aside")
    p_q.set_defaults(func=_cmd_quarantine_list)
    q_sub = p_q.add_subparsers(dest="action")

    p_retry = q_sub.add_parser(
        "retry", parents=[policy], help="re-judge quarantined and partial files"
    )
    p_retry.add_argument("path", nargs="?", type=Path, help="only files under this path")
    p_retry.set_defaults(func=_cmd_retry)

    for name, decision, help_text in (
        ("accept", Decision.ACCEPT, "keep this content despite quality rules"),
        ("ignore", Decision.IGNORE, "never propose this content again"),
    ):
        p = q_sub.add_parser(name, help=help_text)
        p.add_argument("path", type=Path)
        p.set_defaults(func=_cmd_decide, decision=decision)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
