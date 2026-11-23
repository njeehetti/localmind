"""Command-line interface: a thin layer over the public API, no logic here.

Sensitive values are never printed: only their kind and page number.

OCR is offered, never imposed: at the end of an interactive ``ingest`` that
left scanned pages behind, the user is asked. When nobody can answer (a
script, a voice assistant), nothing is asked and nothing slow is started.
The same goes for classification into folders (``cli_folders``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import get_args

from localmind import cli_folders, cli_search
from localmind.catalog import Catalog, Decision, PendingOcr
from localmind.config import default_db_path, sensitive_policy_from_env
from localmind.ingest import IngestReport, Outcome, apply_decision, ingest, retry, run_ocr
from localmind.ocr import OcrEngine, OcrUnavailable, load_engine, ocr_available
from localmind.quality import QualityConfig, SensitivePolicy

# Rough OCR speed (seconds per page), only used to warn before a long run.
SECONDS_PER_PAGE = {"GPU": 0.5, "CPU": 5.0}

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


def _is_interactive() -> bool:
    """Can someone answer a question? (False in scripts, pipes, JARVIS.)

    On Windows the NUL device (``< NUL``, ``< /dev/null``) claims to be a
    terminal: only a real console counts."""
    if not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    except (OSError, ValueError):
        return False
    mode = ctypes.c_uint32()
    return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))


def _ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes", "o", "oui"}
    except EOFError:
        return False


def _estimate(pages: int, device: str) -> str:
    seconds = pages * SECONDS_PER_PAGE[device]
    if seconds < 60:
        return f"{max(1, math.ceil(seconds))} s"
    return f"{math.ceil(seconds / 60)} min"


def _load_engine() -> OcrEngine | None:
    try:
        return load_engine()
    except OcrUnavailable as exc:
        print(f"OCR unavailable: {exc}", file=sys.stderr)
        return None


def _run_ocr(catalog: Catalog, engine: OcrEngine, paths: list[Path], total: int, args) -> None:
    done = 0

    def progress(path: Path, page_no: int) -> None:
        nonlocal done
        done += 1
        print(f"  OCR page {done}/{total}: {path.name} p.{page_no}", flush=True)

    for path in paths:
        report = run_ocr(catalog, engine, _config(args), under=path, on_ocr_page=progress)
        _print_report(report, verbose=True)


def _offer_ocr(catalog: Catalog, paths: list[Path], args: argparse.Namespace) -> None:
    """After ingest: offer to read the scanned pages of the paths just ingested,
    and remind (one line) of scanned pages waiting elsewhere in the catalog."""
    counts = [catalog.ocr_pending(p) for p in paths]
    pending = PendingOcr(sum(c.pages for c in counts), sum(c.files for c in counts))
    others = catalog.ocr_pending().pages - pending.pages
    if pending.pages > 0 and args.ocr is not False:
        _offer_pending(catalog, paths, pending, args)
    if others > 0:
        print(f"({others} more scanned page(s) are waiting: localmind quarantine retry)")


def _offer_pending(
    catalog: Catalog, paths: list[Path], pending: PendingOcr, args: argparse.Namespace
) -> None:
    if not ocr_available():
        print(f"{pending.pages} scanned page(s) could be read with OCR: uv sync --extra ocr")
        return
    if args.ocr is None and not _is_interactive():
        print(f"{pending.pages} scanned page(s) can be read by OCR: localmind quarantine retry")
        return

    engine = _load_engine()
    if engine is None:
        return
    if args.ocr is None:
        print(
            f"{pending.pages} scanned page(s) in {pending.files} file(s) can be read by OCR "
            f"(~{_estimate(pending.pages, engine.device)} on {engine.device})."
        )
        if not _ask("Run OCR now? [y/N] "):
            return
    _run_ocr(catalog, engine, paths, pending.pages, args)


def _cmd_ingest(args: argparse.Namespace) -> int:
    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: path not found: {p}", file=sys.stderr)
        return 2
    with _open_catalog(args.db) as catalog:
        for path in args.paths:
            _print_report(ingest(path, catalog, _config(args)), args.verbose)
        resolved = [p.resolve() for p in args.paths]
        _offer_ocr(catalog, resolved, args)
        reachable = True
        if not args.no_classify:
            then = "localmind classify" if args.no_index else "localmind classify; localmind index"
            reachable = cli_folders.classify_and_review(catalog, resolved, then=then)
        if not args.no_index and reachable:  # Ollama down: the hint above said it all
            cli_search.index_after_ingest(catalog)
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
        ("ocr pages", s.ocr_pages),
        ("blank pages", s.blank_pages),
        ("chunks", s.chunks),
        ("indexed docs", s.indexed_documents),
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
        conf = f" {page.ocr_confidence:.0%}".replace("%", " %") if page.ocr_confidence is not None else ""
        print(f"    {page.path}  page {page.page_no}  [{page.reason}{conf}]")

    print(f"\nSensitive data, kept and flagged ({len(flags)}):")
    for flag in flags:
        print(f"    {flag.path}  {flag.kind} @ page {flag.page_no}")
    return 0


def _cmd_retry(args: argparse.Namespace) -> int:
    under = args.path.resolve() if args.path else None
    engine = None
    if not args.no_ocr and ocr_available():
        engine = _load_engine()
    with _open_catalog(args.db) as catalog:
        total = catalog.ocr_pending(under).pages
        done = 0

        def progress(path: Path, page_no: int) -> None:
            nonlocal done
            done += 1
            print(f"  OCR page {done}/{total}: {path.name} p.{page_no}", flush=True)

        report = retry(catalog, _config(args), under, engine=engine, on_ocr_page=progress)
        _print_report(report, verbose=True)
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
    ocr_choice = p_ingest.add_mutually_exclusive_group()
    ocr_choice.add_argument(
        "--ocr", dest="ocr", action="store_true", default=None,
        help="read scanned pages now, without asking",
    )
    ocr_choice.add_argument(
        "--no-ocr", dest="ocr", action="store_false", help="do not offer OCR"
    )
    p_ingest.add_argument(
        "--no-classify", action="store_true", help="do not sort the documents into folders"
    )
    p_ingest.add_argument(
        "--no-index", action="store_true", help="do not prepare the documents for search"
    )
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
    p_retry.add_argument(
        "--no-ocr", action="store_true", help="do not run OCR (cached OCR text still applies)"
    )
    p_retry.set_defaults(func=_cmd_retry)

    for name, decision, help_text in (
        ("accept", Decision.ACCEPT, "keep this content despite quality rules"),
        ("ignore", Decision.IGNORE, "never propose this content again"),
    ):
        p = q_sub.add_parser(name, help=help_text)
        p.add_argument("path", type=Path)
        p.set_defaults(func=_cmd_decide, decision=decision)

    cli_folders.register(sub)
    cli_search.register(sub)
    return parser


def _utf8_output() -> None:
    """Write UTF-8 even when the output is redirected on Windows (where the
    default is a legacy code page that cannot encode "→" or many names)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
