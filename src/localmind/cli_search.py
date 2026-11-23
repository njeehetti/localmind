"""Command-line side of stages 4-5: indexing after ingest, and the ``index``,
``search`` and ``prune`` commands. Like the rest of the CLI, it only prints:
the work is done by ``indexing``, ``search`` and ``prune``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from localmind.catalog import Catalog
from localmind.indexing import IndexReport, index
from localmind.ollama import Classifier, ClassifierUnavailable, OllamaClassifier
from localmind.prune import missing_files, prune
from localmind.folders import UnknownFolder
from localmind.search import CatalogBusy, Hit, VectorIndex

EXCERPT_CHARS = 100
_YES = {"y", "yes", "o", "oui"}


def make_embedder() -> Classifier:
    return OllamaClassifier()


def _open(db: Path) -> Catalog:
    from localmind import cli  # late import: cli imports this module

    return cli._open_catalog(db)


def _interactive() -> bool:
    from localmind import cli

    return cli._is_interactive()


def print_index(report: IndexReport, always: bool = True) -> None:
    if always or report.documents or report.empty:
        print(f"Indexed {report.documents} document(s): {report.chunks} chunk(s)")
    if report.empty:
        print(f"  {report.empty} document(s) with no usable text")
    if report.unavailable:
        print(report.unavailable)


def index_after_ingest(catalog: Catalog) -> None:
    """Index every document still missing chunks (not only the ingested ones:
    files accepted or OCR-ed since are picked up too)."""
    print_index(index(catalog, make_embedder()), always=False)


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= EXCERPT_CHARS else flat[:EXCERPT_CHARS] + "…"


# -- commands -------------------------------------------------------------
def _cmd_index(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        under = args.path.resolve() if args.path else None
        print_index(index(catalog, make_embedder(), under=under, redo=args.all))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        with _open(args.db) as catalog:
            if catalog.stats().chunks == 0:
                print("Nothing indexed yet: localmind index")
                return 0
            finder = VectorIndex(catalog, make_embedder())
            hits = finder.search(args.question, k=args.k, folder=args.folder)
    except ClassifierUnavailable as exc:
        print(exc.hint, file=sys.stderr)
        return 2
    except (ValueError, UnknownFolder) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (CatalogBusy, sqlite3.OperationalError) as exc:
        if isinstance(exc, sqlite3.OperationalError) and "locked" not in str(exc):
            raise
        print(f"error: {CatalogBusy()}", file=sys.stderr)
        return 2
    if finder.last_warning:
        print(f"meaning search unavailable ({finder.last_warning}): keyword results only", file=sys.stderr)
    if not hits:
        print("No match.")
    for h in hits:
        print(hit_line(h))
    return 0


def hit_line(h: Hit) -> str:
    """``0.70  facture.txt p.1  Maison/Factures (to confirm)  "excerpt"``; the
    score reads ``--`` without a vector, the folder is left out when there is none."""
    score = "  --" if h.score is None else f"{h.score:.2f}"
    folder = ""
    if h.folder is not None:
        folder = f"{h.folder} (to confirm)  " if h.to_confirm else f"{h.folder}  "
    return f'{score}  {h.path.name} p.{h.page_no}  {folder}"{_excerpt(h.text)}"'


def _cmd_prune(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        gone = missing_files(catalog, args.path.resolve() if args.path else None)
        if not gone:
            print("Nothing to prune.")
            return 0
        if not args.yes and not _interactive():
            print(f"{len(gone)} file(s) no longer exist: localmind prune --yes")
            return 0
        print(f"{len(gone)} file(s) no longer exist:")
        for path in gone:
            print(f"  {path}")
        if not args.yes:
            try:
                answer = input("Forget them? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if answer not in _YES:
                return 0
        report = prune(catalog, gone)
        print(f"Forgot {report.files} file(s) and {report.documents} document(s).")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p_index = sub.add_parser("index", help="cut documents into chunks and embed them for search")
    p_index.add_argument("path", nargs="?", type=Path, help="only files under this path")
    p_index.add_argument("--all", action="store_true", help="index everything again")
    p_index.set_defaults(func=_cmd_index)

    p_search = sub.add_parser("search", help="find the passages that best answer a question")
    p_search.add_argument("question")
    p_search.add_argument("-k", type=int, default=5, help="number of passages (default 5)")
    p_search.add_argument(
        "--folder", help="only documents of this domain or folder (Maison, Maison/Factures)"
    )
    p_search.set_defaults(func=_cmd_search)

    p_prune = sub.add_parser("prune", help="forget files deleted from disk (never touches files)")
    p_prune.add_argument("path", nargs="?", type=Path, help="only files under this path")
    p_prune.add_argument("--yes", action="store_true", help="do not ask")
    p_prune.set_defaults(func=_cmd_prune)
