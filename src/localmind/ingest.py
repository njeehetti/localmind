"""Ingestion: walk files, extract, judge quality, record everything in the catalog.

Decision path for each file (cheapest checks first):

    1. same path, same bytes as last time?        -> UNCHANGED
    2. user said "ignore" for these bytes?        -> IGNORED
    3. same bytes as another known file?          -> DUPLICATE (or QUARANTINED, like it)
    4. empty / image / unreadable / encrypted?    -> QUARANTINED (level A, whole file)
    5. judge each page, then the document         -> QUARANTINED if nothing usable
    6. same usable text as a known document?      -> DUPLICATE
    7. otherwise                                  -> NEW / UPDATED / PARTIAL

`retry` and `apply_decision` run the same path but skip steps 1 and 3 (force):
the point is to re-judge the file, not to recognise it.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from localmind.catalog import Catalog, Decision, FileRecord, FileStatus
from localmind.extractors import (
    IMAGE_SUFFIXES,
    EncryptedPdf,
    ExtractionError,
    UnsupportedFormat,
    extract,
)
from localmind.hashing import content_hash, file_sha256
from localmind.quality import QualityConfig, Reason, assess

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    NEW = "new"
    PARTIAL = "partial"
    UPDATED = "updated"
    DUPLICATE = "duplicate"
    UNCHANGED = "unchanged"
    QUARANTINED = "quarantined"
    IGNORED = "ignored"
    MISSING = "missing"


@dataclass(frozen=True)
class FileResult:
    path: Path
    outcome: Outcome
    detail: str = ""
    reason: Reason | None = None


@dataclass(frozen=True)
class IngestReport:
    results: tuple[FileResult, ...]

    def counts(self) -> Counter[Outcome]:
        return Counter(r.outcome for r in self.results)


_DEFAULT_CONFIG = QualityConfig()


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(p for p in root.rglob("*") if p.is_file())


def ingest(root: Path, catalog: Catalog, config: QualityConfig = _DEFAULT_CONFIG) -> IngestReport:
    """Ingest a file or every file under a folder. Never modifies the files."""
    results = tuple(
        _process(path.resolve(), catalog, config, force=False) for path in iter_files(root)
    )
    return IngestReport(results=results)


def retry(
    catalog: Catalog, config: QualityConfig = _DEFAULT_CONFIG, under: Path | None = None
) -> IngestReport:
    """Re-judge quarantined files and files of partial documents (optionally only
    under a folder), with the current config and decisions."""
    results = []
    for record in catalog.retry_candidates(under):
        if not record.path.is_file():
            # Keep the catalog entry: the file may be on an unplugged drive.
            results.append(FileResult(record.path, Outcome.MISSING, "file not found"))
            continue
        results.append(_process(record.path, catalog, config, force=True))
    return IngestReport(results=tuple(results))


def apply_decision(
    path: Path, decision: Decision, catalog: Catalog, config: QualityConfig = _DEFAULT_CONFIG
) -> FileResult:
    """Record the user's decision for this content and apply it right away,
    to this file and to every exact copy already in the catalog (decisions are
    about content, not about one path)."""
    path = path.resolve()
    record = catalog.file_by_path(path)
    if record is None:
        raise KeyError(f"not in catalog: {path}")
    exists = path.is_file()
    file_hash = file_sha256(path) if exists else record.file_hash
    catalog.record_decision(file_hash, decision)

    copies = [r for r in catalog.files_by_hash(file_hash) if r.path != path]
    result = _apply_to(path, exists, record.size_bytes, file_hash, decision, catalog, config)
    for copy in copies:
        _apply_to(copy.path, copy.path.is_file(), copy.size_bytes, file_hash, decision, catalog, config)
    return result


def _apply_to(
    path: Path,
    exists: bool,
    known_size: int,
    file_hash: str,
    decision: Decision,
    catalog: Catalog,
    config: QualityConfig,
) -> FileResult:
    if decision is Decision.IGNORE:
        size = path.stat().st_size if exists else known_size
        catalog.record_ignored(path, file_hash, size)
        return FileResult(path, Outcome.IGNORED, "ignored by user")
    if not exists:
        return FileResult(path, Outcome.MISSING, "file not found")
    return _process(path, catalog, config, force=True)


# --- the decision path ------------------------------------------------------


def _quarantine(
    path: Path, file_hash: str, size: int, reason: Reason, catalog: Catalog, detail: str = ""
) -> FileResult:
    catalog.record_quarantine(path, file_hash, size, reason)
    return FileResult(path, Outcome.QUARANTINED, detail or reason.value, reason)


def _process(path: Path, catalog: Catalog, config: QualityConfig, *, force: bool) -> FileResult:
    try:
        file_hash = file_sha256(path)
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return FileResult(path, Outcome.MISSING, f"cannot read file: {exc}")

    previous = catalog.file_by_path(path)
    if not force and previous is not None and previous.file_hash == file_hash:
        return FileResult(path, Outcome.UNCHANGED)

    decision = catalog.decision_for(file_hash)
    if decision is Decision.IGNORE:
        catalog.record_ignored(path, file_hash, size)
        return FileResult(path, Outcome.IGNORED, "ignored by user")

    if not force:
        copy_result = _copy_of_known_file(path, file_hash, size, catalog)
        if copy_result is not None:
            return copy_result

    file_level = _file_level_reason(path, size)
    if file_level is not None:
        return _quarantine(path, file_hash, size, file_level, catalog)

    try:
        extracted = extract(path)
    except UnsupportedFormat as exc:
        return _quarantine(path, file_hash, size, Reason.UNSUPPORTED_FORMAT, catalog, str(exc))
    except EncryptedPdf as exc:
        return _quarantine(path, file_hash, size, Reason.ENCRYPTED, catalog, str(exc))
    except ExtractionError as exc:
        logger.warning("extraction failed for %s: %s", path, exc)
        return _quarantine(path, file_hash, size, Reason.CORRUPTED, catalog, str(exc))

    assessment = assess(extracted, config, accepted=decision is Decision.ACCEPT)
    if assessment.quarantine_reason is not None:
        return _quarantine(path, file_hash, size, assessment.quarantine_reason, catalog)

    # Dedup key: the usable text. A document with set-aside pages also carries
    # its file hash, otherwise two PDFs sharing a cover page but with different
    # scanned pages would be merged and the second one's pages lost.
    key = content_hash(assessment.usable_text)
    if assessment.has_set_aside:
        key = f"{key}:{file_hash}"

    existing = catalog.document_id_by_content_key(key)
    if existing is not None:
        catalog.link_file(path, file_hash, size, existing, extracted.encoding)
        if previous is not None and previous.document_id == existing:
            return FileResult(path, Outcome.UNCHANGED, "re-assessed, no change")
        return FileResult(path, Outcome.DUPLICATE, "same text as a known file")

    document_id = catalog.add_document(key, extracted.kind, extracted.page_count, assessment.pages)
    catalog.link_file(path, file_hash, size, document_id, extracted.encoding)
    return FileResult(path, _stored_outcome(previous, assessment.has_set_aside))


def _copy_of_known_file(
    path: Path, file_hash: str, size: int, catalog: Catalog
) -> FileResult | None:
    """Fast path: identical bytes already judged -> reuse the verdict, no extraction."""
    known = catalog.file_by_hash(file_hash)
    if known is None or known.path == path:
        return None
    if known.status is FileStatus.INDEXED and known.document_id is not None:
        catalog.link_file(path, file_hash, size, known.document_id)
        return FileResult(path, Outcome.DUPLICATE, "exact copy of a known file")
    if known.status is FileStatus.QUARANTINED and known.reason is not None:
        return _quarantine(
            path, file_hash, size, known.reason, catalog, "exact copy of a quarantined file"
        )
    return None


def _file_level_reason(path: Path, size: int) -> Reason | None:
    """Level A rules that need no extraction."""
    if size == 0:
        return Reason.EMPTY
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return Reason.NEEDS_OCR  # stage 2b (OCR) will pick these up
    return None


def _stored_outcome(previous: FileRecord | None, partial: bool) -> Outcome:
    if partial:
        return Outcome.PARTIAL
    if previous is not None and previous.status is FileStatus.INDEXED:
        return Outcome.UPDATED
    return Outcome.NEW
