"""Stage 4: forget the files you deleted.

The catalog keeps a file it cannot find (it may sit on an unplugged drive), so
forgetting is an explicit step: ``missing_files`` lists them, ``prune`` removes
their rows and the documents left without any file, with everything derived
from them (pages, chunks, vectors, folder). Nothing on disk is ever touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from localmind.catalog import Catalog


@dataclass(frozen=True)
class PruneReport:
    files: int
    documents: int


def missing_files(catalog: Catalog, under: Path | None = None) -> list[Path]:
    """Catalog files (any status) whose path no longer exists, sorted."""
    return sorted(
        p
        for p in catalog.file_paths()
        if (under is None or p == under or under in p.parents) and not p.is_file()
    )


def prune(catalog: Catalog, paths: Sequence[Path]) -> PruneReport:
    """Forget these files. A path that exists again is kept (never forget a
    file that is there)."""
    gone = [p for p in paths if not p.is_file()]
    if not gone:
        return PruneReport(0, 0)
    documents = catalog.forget_files(gone)
    return PruneReport(len(gone), documents)
