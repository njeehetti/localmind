"""Command-line side of stage 3: the summary printed after a classification,
the review of what waits for the user, and the folder commands.

Like the OCR offer, questions are only asked when someone can answer: in a
script or from JARVIS, one line says what waits. Ctrl+C or EOF at any prompt
means "later": nothing is lost, the queue stays.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from localmind.catalog import Catalog, Classification
from localmind.classify import LLM_INVALID, UNAVAILABLE, ClassifyReport, DocClassification, classify
from localmind.folder_actions import choose_folder, confirm, move, rename_folder
from localmind.folders import InvalidFolderName, fold, parse_folder
from localmind.ollama import Classifier, OllamaClassifier

UNSORTED = "Unsorted"
_REASONS = {
    UNAVAILABLE: "the models could not be reached",
    LLM_INVALID: "the LLM gave no usable answer",
}


def make_classifier() -> Classifier:
    return OllamaClassifier()


def _interactive() -> bool:
    from localmind import cli  # late import: cli imports this module

    return cli._is_interactive()


def _open(db: Path) -> Catalog:
    from localmind import cli

    return cli._open_catalog(db)


def _prompt(text: str) -> str | None:
    """One answer, or None for "later" (Ctrl+C, EOF)."""
    try:
        return input(text).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


# -- after a classification ----------------------------------------------
def _line(r: DocClassification) -> str:
    suffix = " (your choice)" if r.method == "user" else " (proposed)" if r.new_folder else ""
    return f"  {r.name}  → {r.folder or UNSORTED}{suffix}"


def print_summary(report: ClassifyReport) -> None:
    for r in report.results:
        if r.reason == UNAVAILABLE:
            continue  # one hint line below says it for all of them
        print(_line(r))
        if r.proposed_name:
            print(f"  ({r.name}: proposed {r.proposed_name}, used existing {r.folder})")
    if not report.results:
        return
    print(f"{len(report.results)} document(s) classified:")
    placed = report.placed()
    if placed:
        top = Counter(r.folder for r in placed).most_common(5)
        print(f"  {len(placed)} placed  " + ", ".join(f"{f} ({n})" for f, n in top))
    for folder, docs in sorted(report.proposals().items()):
        print(f"  {len(docs)} to confirm  new folder proposed: {folder} ({len(docs)} document(s))")
    for r in report.hesitations():
        print(f"  1 hesitation  {r.name} → {r.folder} or {r.alternative}?")
    for reason, n in Counter(r.reason for r in report.unsorted()).items():
        print(f"  {n} unsorted  {_REASONS.get(reason or '', reason)}")
    if report.unavailable:
        print(report.unavailable)


def classify_and_review(
    catalog: Catalog,
    paths: list[Path] | None,
    redo: bool = False,
    then: str = "localmind classify",
) -> bool:
    """Classify (only under ``paths`` when given), print the summary, then
    offer the review of what waits for the user. Returns False when Ollama
    could not be reached; the hint then names ``then`` as the command to run."""
    classifier = make_classifier()
    results: list[DocClassification] = []
    unavailable = None
    for under in paths or [None]:
        report = classify(catalog, classifier, under=under, redo=redo)
        results.extend(report.results)
        unavailable = unavailable or report.unavailable
        if unavailable:
            unavailable = unavailable.removesuffix("localmind classify") + then
            break
    print_summary(ClassifyReport(tuple(results), unavailable))
    waiting = [c for c in catalog.pending_reviews() if c.document_id in {r.document_id for r in results}]
    if not waiting:
        return unavailable is None
    if not _interactive():
        print(f"{len(waiting)} classification(s) to confirm: localmind folders review")
        return unavailable is None
    answer = _prompt(f"Confirm the {len(waiting)} now? [y/N] ")
    if answer is not None and answer.lower() in {"y", "yes", "o", "oui"}:
        review(catalog, waiting)
    return unavailable is None


# -- review ---------------------------------------------------------------
def _name(catalog: Catalog, document_id: int) -> str:
    return catalog.document_to_classify(document_id).name


def _groups(catalog: Catalog, waiting: list[Classification]):
    """Proposed folders (one question for all their documents), then single
    documents (hesitations, unsorted)."""
    proposals: dict[int, list[Classification]] = {}
    singles: list[Classification] = []
    for c in waiting:
        if c.folder_id is not None and not catalog.folder(c.folder_id).confirmed:
            proposals.setdefault(c.folder_id, []).append(c)
        else:
            singles.append(c)
    ordered = sorted(proposals.items(), key=lambda item: str(catalog.folder(item[0]).name))
    return ordered, singles


def _apply(action, *args) -> bool:
    try:
        action(*args)
    except (InvalidFolderName, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    return True


def review(catalog: Catalog, waiting: list[Classification]) -> None:
    """Ask about each proposal and hesitation; stop at the first Ctrl+C."""
    proposals, singles = _groups(catalog, waiting)
    ask = _interactive()
    for folder_id, docs in proposals:
        folder = str(catalog.folder(folder_id).name)
        names = ", ".join(_name(catalog, c.document_id) for c in docs)
        print(f"New folder {folder} for {len(docs)} document(s): {names}")
        if not ask:
            continue
        choice = _prompt("  [y] yes  [r] rename  [o] other existing folder  [l] later ")
        if choice is None:
            return
        choice = choice.lower()
        if choice == "y":
            confirm(catalog, folder)
        elif choice in {"r", "o"}:
            target = _prompt("New name: " if choice == "r" else "Folder: ")
            if target is None:
                return
            if choice == "r":
                _apply(rename_folder, catalog, folder, target)
            else:
                for c in docs:
                    if not _apply(choose_folder, catalog, c.document_id, target):
                        break
    for c in singles:
        a = str(catalog.folder(c.folder_id).name) if c.folder_id else UNSORTED
        b = str(catalog.folder(c.alternative_folder_id).name) if c.alternative_folder_id else None
        print(_name(catalog, c.document_id))
        if not ask:
            continue
        if b:
            options = f"  [1] {a}  [2] {b}  [o] other  [l] later "
        elif c.folder_id:  # its proposal was confirmed meanwhile
            options = f"  [1] {a}  [o] other  [l] later "
        else:
            options = "  [o] other  [l] later "
        choice = _prompt(options)
        if choice is None:
            return
        if choice == "1" and c.folder_id:
            _apply(choose_folder, catalog, c.document_id, a)
        elif choice == "2" and b:
            _apply(choose_folder, catalog, c.document_id, b)
        elif choice.lower() == "o":
            target = _prompt("Folder: ")
            if target is None:
                return
            _apply(choose_folder, catalog, c.document_id, target)


# -- commands -------------------------------------------------------------
def _cmd_folders(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        domain = None
        for f in catalog.folders():
            if f.name.domain != domain:
                domain = f.name.domain
                print(domain)
            proposed = "" if f.confirmed else " (proposed)"
            print(f"  {f.name.type} ({f.documents}){proposed}")
        unsorted = len(catalog.unsorted_documents())
        if unsorted:
            print(f"{UNSORTED} ({unsorted})")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        under = args.path.resolve() if args.path else None
        waiting = catalog.pending_reviews(under)
        if not waiting:
            print("Nothing to confirm.")
            return 0
        review(catalog, waiting)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        if fold(args.folder) == fold(UNSORTED):
            docs, title = catalog.unsorted_documents(), UNSORTED
        else:
            try:
                found = catalog.folder_by_name(parse_folder(args.folder))
            except InvalidFolderName as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if found is None:
                print(f"error: no folder named {args.folder}", file=sys.stderr)
                return 2
            docs, title = catalog.documents_in(found.id), str(found.name)
        print(f"{title} ({len(docs)})")
        for doc in docs:
            print(f"  {doc.paths[0] if doc.paths else doc.name}")
    return 0


def _cmd_rename(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        if not _apply(rename_folder, catalog, args.old, args.new):
            return 2
        print(f"{parse_folder(args.old)} → {parse_folder(args.new)}")
    return 0


def _cmd_move(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        if not _apply(move, catalog, args.file, args.folder):
            return 2
        print(f"  {args.file.name}  → {parse_folder(args.folder)} (your choice)")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    with _open(args.db) as catalog:
        classify_and_review(catalog, [args.path.resolve()] if args.path else None, redo=args.all)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p_folders = sub.add_parser("folders", help="list the virtual folders")
    p_folders.set_defaults(func=_cmd_folders)
    f_sub = p_folders.add_subparsers(dest="action")
    p_review = f_sub.add_parser("review", help="confirm proposed folders and hesitations")
    p_review.add_argument("path", nargs="?", type=Path, help="only files under this path")
    p_review.set_defaults(func=_cmd_review)
    p_show = f_sub.add_parser("show", help="list the documents of a folder")
    p_show.add_argument("folder", help='"Domain/Type", or Unsorted')
    p_show.set_defaults(func=_cmd_show)
    p_rename = f_sub.add_parser("rename", help="rename a folder, or merge it into another")
    p_rename.add_argument("old")
    p_rename.add_argument("new")
    p_rename.set_defaults(func=_cmd_rename)

    p_move = sub.add_parser("move", help="put a document into a folder (your choice)")
    p_move.add_argument("file", type=Path)
    p_move.add_argument("folder", help='"Domain/Type"')
    p_move.set_defaults(func=_cmd_move)

    p_classify = sub.add_parser("classify", help="sort documents into folders")
    p_classify.add_argument("path", nargs="?", type=Path, help="only files under this path")
    p_classify.add_argument(
        "--all", action="store_true", help="classify again everything except your own choices"
    )
    p_classify.set_defaults(func=_cmd_classify)
