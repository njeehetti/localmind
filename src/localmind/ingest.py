"""Ingestion: walk files, extract text, record them in the catalog.

Decision path for each file (cheapest checks first):

    1. same path, same bytes as last time?  -> UNCHANGED  (nothing to do)
    2. same bytes as another known file?    -> DUPLICATE  (no extraction needed)
    3. extract text, same normalized text?  -> DUPLICATE  (different bytes, same info)
    4. otherwise                            -> NEW, or UPDATED if the path was known
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from localmind.catalog import Catalog
from localmind.extractors import ExtractionError, UnsupportedFormat, extract
from localmind.hashing import content_hash, file_sha256

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    NEW = "new"
    UPDATED = "updated"
    DUPLICATE = "duplicate"
    UNCHANGED = "unchanged"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class FileResult:
    path: Path
    outcome: Outcome
    detail: str = ""


@dataclass(frozen=True)
class IngestReport:
    results: tuple[FileResult, ...]

    def counts(self) -> Counter[Outcome]:
        return Counter(r.outcome for r in self.results)


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(p for p in root.rglob("*") if p.is_file())


def ingest(root: Path, catalog: Catalog) -> IngestReport:
    """Ingest a file or every file under a folder. Never modifies the files."""
    results = tuple(_ingest_file(path.resolve(), catalog) for path in iter_files(root))
    return IngestReport(results=results)


def _ingest_file(path: Path, catalog: Catalog) -> FileResult:
    try:
        file_hash = file_sha256(path)
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return FileResult(path, Outcome.FAILED, f"cannot read file: {exc}")

    previous = catalog.file_by_path(path)
    if previous is not None and previous.file_hash == file_hash:
        return FileResult(path, Outcome.UNCHANGED)

    # Fast path: identical bytes already known -> reuse, skip extraction.
    known_doc = catalog.document_id_by_file_hash(file_hash)
    if known_doc is not None:
        catalog.link_file(path, file_hash, size, known_doc)
        return FileResult(path, Outcome.DUPLICATE, "exact copy of a known file")

    try:
        extracted = extract(path)
    except UnsupportedFormat as exc:
        return FileResult(path, Outcome.UNSUPPORTED, str(exc))
    except ExtractionError as exc:
        logger.warning("extraction failed for %s: %s", path, exc)
        return FileResult(path, Outcome.FAILED, str(exc))

    text_hash = content_hash(extracted.text)
    same_content = catalog.document_id_by_content_hash(text_hash)
    if same_content is not None:
        catalog.link_file(path, file_hash, size, same_content, extracted.encoding)
        return FileResult(path, Outcome.DUPLICATE, "same text as a known file")

    document_id = catalog.add_document(text_hash, extracted)
    catalog.link_file(path, file_hash, size, document_id, extracted.encoding)
    outcome = Outcome.UPDATED if previous is not None else Outcome.NEW
    return FileResult(path, outcome)
