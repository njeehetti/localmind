"""Ingestion: walk files, extract, judge quality, record everything in the catalog.

Decision path for each file (cheapest checks first):

    1. same path, same bytes as last time?        -> UNCHANGED
    2. user said "ignore" for these bytes?        -> IGNORED
    3. same bytes as another known file?          -> DUPLICATE (or QUARANTINED, like it)
    4. empty / unreadable / encrypted?            -> QUARANTINED (level A, whole file)
    5. pages without a text layer: OCR text from the cache, or from the engine
       if one was given (stage 2b)
    6. judge each page, then the document         -> QUARANTINED if nothing usable
    7. same usable text as a known document?      -> DUPLICATE
    8. otherwise                                  -> NEW / UPDATED / PARTIAL

`retry`, `run_ocr` and `apply_decision` run the same path but skip steps 1
and 3 (force): the point is to re-judge the file, not to recognise it.

OCR results are cached by (file hash, page): a cached page is always used,
the engine (slow, GPU) runs only when the caller passes one.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from localmind.catalog import Catalog, Decision, FileRecord, FileStatus
from localmind.extractors import (
    EncryptedPdf,
    ExtractedText,
    ExtractionError,
    UnsupportedFormat,
    extract,
)
from localmind.hashing import content_hash, file_sha256
from localmind.ocr import OcrEngine, OcrPage
from localmind.quality import QualityConfig, Reason, assess, needs_ocr_pages

# Called just before the engine reads a page: (file, 1-based page number).
OcrProgress = Callable[[Path, int], None]

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


@dataclass(frozen=True)
class _Ocr:
    """What _process needs to read pages: an engine (or None) and a progress hook."""

    engine: OcrEngine | None = None
    on_page: OcrProgress | None = None


_NO_OCR = _Ocr()


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(p for p in root.rglob("*") if p.is_file())


def ingest(
    root: Path,
    catalog: Catalog,
    config: QualityConfig = _DEFAULT_CONFIG,
    engine: OcrEngine | None = None,
    on_ocr_page: OcrProgress | None = None,
) -> IngestReport:
    """Ingest a file or every file under a folder. Never modifies the files.
    Pages without text are read by ``engine`` if one is given."""
    ocr = _Ocr(engine, on_ocr_page)
    results = tuple(
        _process(path.resolve(), catalog, config, force=False, ocr=ocr)
        for path in iter_files(root)
    )
    return IngestReport(results=results)


def retry(
    catalog: Catalog,
    config: QualityConfig = _DEFAULT_CONFIG,
    under: Path | None = None,
    engine: OcrEngine | None = None,
    on_ocr_page: OcrProgress | None = None,
) -> IngestReport:
    """Re-judge quarantined files and files of partial documents (optionally only
    under a folder), with the current config and decisions."""
    return _reprocess(catalog.retry_candidates(under), catalog, config, _Ocr(engine, on_ocr_page))


def run_ocr(
    catalog: Catalog,
    engine: OcrEngine,
    config: QualityConfig = _DEFAULT_CONFIG,
    under: Path | None = None,
    on_ocr_page: OcrProgress | None = None,
) -> IngestReport:
    """Read the pages waiting for OCR (optionally only under a path)."""
    return _reprocess(catalog.ocr_candidates(under), catalog, config, _Ocr(engine, on_ocr_page))


def _reprocess(
    records: list[FileRecord], catalog: Catalog, config: QualityConfig, ocr: _Ocr
) -> IngestReport:
    results = []
    for record in records:
        if not record.path.is_file():
            # Keep the catalog entry: the file may be on an unplugged drive.
            results.append(FileResult(record.path, Outcome.MISSING, "file not found"))
            continue
        results.append(_process(record.path, catalog, config, force=True, ocr=ocr))
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
    path: Path,
    file_hash: str,
    size: int,
    reason: Reason,
    catalog: Catalog,
    detail: str = "",
    page_count: int | None = None,
) -> FileResult:
    catalog.record_quarantine(path, file_hash, size, reason, page_count)
    return FileResult(path, Outcome.QUARANTINED, detail or reason.value, reason)


def _with_ocr(
    path: Path,
    file_hash: str,
    extracted: ExtractedText,
    catalog: Catalog,
    config: QualityConfig,
    ocr: _Ocr,
) -> ExtractedText:
    """Attach OCR text to the pages that need it: from the cache, else from the
    engine (one page rendered at a time, each result saved immediately)."""
    wanted = sorted(needs_ocr_pages(extracted, config))
    if not wanted:
        return extracted
    found: dict[int, OcrPage] = {}
    to_read = []
    for page_no in wanted:
        cached = catalog.ocr_cached(file_hash, page_no)
        if cached is not None:
            found[page_no] = cached
        else:
            to_read.append(page_no)
    if to_read and ocr.engine is not None:
        found.update(_read_pages(path, file_hash, to_read, catalog, ocr))
    return replace(extracted, ocr=found) if found else extracted


def _read_pages(
    path: Path, file_hash: str, page_numbers: list[int], catalog: Catalog, ocr: _Ocr
) -> dict[int, OcrPage]:
    from localmind.rendering import render_pages

    engine = ocr.engine
    assert engine is not None
    results: dict[int, OcrPage] = {}
    for page_no, image in render_pages(path, page_numbers):
        if ocr.on_page is not None:
            ocr.on_page(path, page_no)
        try:
            text, confidence = engine.read(image)
        except Exception as exc:  # one bad page must not stop a long run
            logger.warning("OCR failed on %s page %d: %s", path, page_no, exc.__class__.__name__)
            results[page_no] = OcrPage("", -1.0, engine.name)  # not cached: retry later
            continue
        page = OcrPage(text, confidence, engine.name)
        catalog.store_ocr(file_hash, page_no, page)
        results[page_no] = page
    return results


def _process(
    path: Path, catalog: Catalog, config: QualityConfig, *, force: bool, ocr: _Ocr = _NO_OCR
) -> FileResult:
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

    extracted = _with_ocr(path, file_hash, extracted, catalog, config, ocr)
    assessment = assess(extracted, config, accepted=decision is Decision.ACCEPT)
    if assessment.quarantine_reason is not None:
        return _quarantine(
            path,
            file_hash,
            size,
            assessment.quarantine_reason,
            catalog,
            page_count=extracted.page_count,
        )

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
    return None


def _stored_outcome(previous: FileRecord | None, partial: bool) -> Outcome:
    if partial:
        return Outcome.PARTIAL
    if previous is not None and previous.status is FileStatus.INDEXED:
        return Outcome.UPDATED
    return Outcome.NEW
