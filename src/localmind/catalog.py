"""The catalog: a SQLite database of *pointers* to your files (schema v5).

    files                       documents                 pages
    -------------------------   -----------------------   -------------------------
    path (unique)               content_key (unique)      document_id, page_no
    file_hash  (bytes)          kind, page_count          text
    status: indexed |           <-- document_id           status: ok | set_aside | blank
      quarantined | ignored                               reason, source (text_layer | ocr)
    page_count                                            ocr_engine, ocr_confidence
    reason                                                --> sensitive_spans
    document_id (NULL unless indexed) ----------------->      (kind, start, end)

    decisions: file_hash -> accept | ignore   (keyed by content: survives moves)
    ocr_cache: (file_hash, page_no, engine) -> text, confidence
               (OCR is slow: never read the same page twice, even for a copy)

* A **document** is a piece of information; a **file** is where it lives.
  Many files can point to one document: that is duplicate detection.
* A quarantined file has no document: text-less files are never "duplicates"
  of each other just because their (empty) text is identical.
* Original files are never moved, copied or modified; only their path is stored.
* Migrations never touch ``decisions``, ``ocr_cache`` or ``folder_decisions``:
  they hold the user's choices and hours of OCR work, which a new ``ingest``
  cannot recreate.
* Stage 3 (folders, classifications, embeddings) lives in ``catalog_folders``,
  stage 4 (chunks and their vectors) in ``catalog_chunks``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal

from localmind.catalog_chunks import CHUNKS_SCHEMA, ChunkStore
from localmind.catalog_folders import (
    FOLDERS_SCHEMA,
    Classification,
    DocumentToClassify,
    Folder,
    FolderStore,
)
from localmind.ocr import OcrPage
from localmind.quality import PageVerdict, Reason

SCHEMA_VERSION = 6

_PAGES_COLUMNS = """
    id              INTEGER PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('ok', 'set_aside', 'blank')),
    reason          TEXT,
    source          TEXT NOT NULL DEFAULT 'text_layer' CHECK (source IN ('text_layer', 'ocr')),
    ocr_engine      TEXT,
    ocr_confidence  REAL,
    UNIQUE (document_id, page_no)
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    content_key  TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    page_count   INTEGER,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages ({pages_columns});
CREATE TABLE IF NOT EXISTS sensitive_spans (
    id       INTEGER PRIMARY KEY,
    page_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL,
    start    INTEGER NOT NULL,
    "end"    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    file_hash    TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    encoding     TEXT,
    status       TEXT NOT NULL CHECK (status IN ('indexed', 'quarantined', 'ignored')),
    reason       TEXT,
    document_id  INTEGER REFERENCES documents(id),
    ingested_at  TEXT NOT NULL,
    page_count   INTEGER
);
CREATE TABLE IF NOT EXISTS decisions (
    file_hash   TEXT PRIMARY KEY,
    decision    TEXT NOT NULL CHECK (decision IN ('accept', 'ignore')),
    decided_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ocr_cache (
    file_hash   TEXT NOT NULL,
    page_no     INTEGER NOT NULL,
    engine      TEXT NOT NULL,
    text        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (file_hash, page_no, engine)
);
CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
""".replace("{pages_columns}", _PAGES_COLUMNS) + FOLDERS_SCHEMA + CHUNKS_SCHEMA

# Tables a v1 catalog rebuild may drop. Never "decisions", "ocr_cache" nor
# "folder_decisions".
_REBUILT_TABLES = (
    "chunks_fts", "chunk_vectors", "chunks", "classifications", "embeddings", "sensitive_spans", "pages", "files", "documents"
)

# v2 -> v3 in place. SQLite cannot change a CHECK constraint with ALTER TABLE,
# so "pages" is copied into a new table with the new definition, then renamed.
_V2_TO_V3 = """
CREATE TABLE pages_new ({pages_columns});
INSERT INTO pages_new (id, document_id, page_no, text, status, reason)
    SELECT id, document_id, page_no, text, status, reason FROM pages;
DROP TABLE pages;
ALTER TABLE pages_new RENAME TO pages;
ALTER TABLE files ADD COLUMN page_count INTEGER;
PRAGMA user_version = 3;
""".replace("{pages_columns}", _PAGES_COLUMNS)


class FileStatus(StrEnum):
    INDEXED = "indexed"
    QUARANTINED = "quarantined"
    IGNORED = "ignored"


class Decision(StrEnum):
    ACCEPT = "accept"
    IGNORE = "ignore"


@dataclass(frozen=True)
class FileRecord:
    path: Path
    file_hash: str
    size_bytes: int
    status: FileStatus
    reason: Reason | None
    document_id: int | None


@dataclass(frozen=True)
class Document:
    id: int
    content_key: str
    kind: str
    page_count: int | None
    status: Literal["accepted", "partial"]
    text: str  # usable pages only, joined by form feeds


@dataclass(frozen=True)
class SetAsidePage:
    path: Path
    page_no: int
    reason: Reason
    ocr_confidence: float | None = None


@dataclass(frozen=True)
class PendingOcr:
    """Pages OCR could read: set aside "needs_ocr" or in quarantined scans."""

    pages: int
    files: int


@dataclass(frozen=True)
class SensitiveFlag:
    path: Path
    page_no: int
    kind: str


@dataclass(frozen=True)
class Stats:
    documents: int
    files: int
    duplicate_files: int
    accepted: int
    partial: int
    quarantined: int
    ignored: int
    sensitive_flags: int
    ocr_pages: int = 0
    blank_pages: int = 0
    chunks: int = 0
    indexed_documents: int = 0


@dataclass(frozen=True)
class DuplicateGroup:
    document_id: int
    paths: tuple[Path, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reason(value: str | None) -> Reason | None:
    return Reason(value) if value else None


_FILE_COLUMNS = "path, file_hash, size_bytes, status, reason, document_id"


def _is_under(path: Path, under: Path | None) -> bool:
    return under is None or path == under or path.is_relative_to(under)


def _file_record(row: tuple) -> FileRecord:
    path, file_hash, size, status, reason, document_id = row
    return FileRecord(Path(path), file_hash, size, FileStatus(status), _reason(reason), document_id)


class Catalog(FolderStore, ChunkStore):
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.rebuilt_from_v1 = self._migrate()
        older = self._conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION
        self._conn.executescript(_SCHEMA)  # commits on its own: the rebuild below must
        with self._conn:  # not depend on whether chunks_fts existed before it
            if older:  # v5 -> v6: index the chunks already there ('rebuild' is idempotent)
                self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate(self) -> bool:
        """Bring an older catalog to the current schema. Returns True if it was
        a v1 catalog that had to be rebuilt.

        * v1 (stage 1) is a rebuildable index: its tables are dropped and a new
          ``ingest`` recreates them (originals are untouched). ``decisions`` and
          ``ocr_cache`` are never dropped.
        * v2 (stage 2a) is upgraded in place: nothing is lost.
        * v3 (stage 2b) -> v4 -> v5 only add the stage 3 and stage 4 tables;
          v5 -> v6 adds the keyword index and fills it from the existing chunks.
        """
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()[0]
        if version >= SCHEMA_VERSION or not has_tables:
            return False
        if version in (3, 4, 5):
            return False  # only new tables: _SCHEMA creates them, __init__ fills chunks_fts
        if version == 2:
            self._conn.commit()
            self._conn.execute("PRAGMA foreign_keys = OFF")  # no effect inside a transaction
            try:
                self._conn.executescript("BEGIN;" + _V2_TO_V3 + "COMMIT;")
            except sqlite3.Error:
                self._conn.rollback()
                raise
            finally:
                self._conn.execute("PRAGMA foreign_keys = ON")
            return False
        with self._conn:
            for table in _REBUILT_TABLES:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        return True

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- lookups ---------------------------------------------------------
    def file_by_path(self, path: Path) -> FileRecord | None:
        row = self._conn.execute(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        return _file_record(row) if row else None

    def file_by_hash(self, file_hash: str) -> FileRecord | None:
        row = self._conn.execute(
            f"""SELECT {_FILE_COLUMNS} FROM files WHERE file_hash = ?
                ORDER BY status = 'indexed' DESC, id LIMIT 1""",
            (file_hash,),
        ).fetchone()
        return _file_record(row) if row else None

    def files_by_hash(self, file_hash: str) -> list[FileRecord]:
        rows = self._conn.execute(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE file_hash = ? ORDER BY path", (file_hash,)
        ).fetchall()
        return [_file_record(r) for r in rows]

    def document_id_by_content_key(self, content_key: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE content_key = ?", (content_key,)
        ).fetchone()
        return row[0] if row else None

    def document_for_path(self, path: Path) -> Document | None:
        row = self._conn.execute(
            """SELECT d.id, d.content_key, d.kind, d.page_count
               FROM documents d JOIN files f ON f.document_id = d.id
               WHERE f.path = ?""",
            (str(path),),
        ).fetchone()
        if not row:
            return None
        pages = self._conn.execute(
            "SELECT text, status FROM pages WHERE document_id = ? ORDER BY page_no", (row[0],)
        ).fetchall()
        status = "partial" if any(s == "set_aside" for _, s in pages) else "accepted"
        text = "\f".join(t for t, s in pages if s == "ok")
        return Document(row[0], row[1], row[2], row[3], status, text)

    def decision_for(self, file_hash: str) -> Decision | None:
        row = self._conn.execute(
            "SELECT decision FROM decisions WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return Decision(row[0]) if row else None

    def quarantined_files(self) -> list[FileRecord]:
        rows = self._conn.execute(
            f"SELECT {_FILE_COLUMNS} FROM files WHERE status = 'quarantined' ORDER BY reason, path"
        ).fetchall()
        return [_file_record(r) for r in rows]

    def set_aside_pages(self) -> list[SetAsidePage]:
        rows = self._conn.execute(
            """SELECT f.path, p.page_no, p.reason, p.ocr_confidence FROM pages p
               JOIN files f ON f.document_id = p.document_id
               WHERE p.status = 'set_aside' AND f.status = 'indexed'
               ORDER BY f.path, p.page_no"""
        ).fetchall()
        return [SetAsidePage(Path(path), n, Reason(r), conf) for path, n, r, conf in rows]

    def sensitive_flags(self) -> list[SensitiveFlag]:
        rows = self._conn.execute(
            """SELECT DISTINCT f.path, p.page_no, s.kind FROM sensitive_spans s
               JOIN pages p ON p.id = s.page_id
               JOIN files f ON f.document_id = p.document_id
               WHERE f.status = 'indexed'
               ORDER BY f.path, p.page_no, s.kind"""
        ).fetchall()
        return [SensitiveFlag(Path(path), n, kind) for path, n, kind in rows]

    def retry_candidates(self, under: Path | None = None) -> list[FileRecord]:
        rows = self._conn.execute(
            f"""SELECT {_FILE_COLUMNS} FROM files
                WHERE status = 'quarantined'
                   OR (status = 'indexed' AND document_id IN (
                        SELECT document_id FROM pages WHERE status = 'set_aside'))
                ORDER BY path"""
        ).fetchall()
        return [r for r in map(_file_record, rows) if _is_under(r.path, under)]

    def ocr_candidates(self, under: Path | None = None) -> list[FileRecord]:
        """Files with pages OCR could read: quarantined "needs_ocr" files and
        indexed files whose document has pages set aside as "needs_ocr"."""
        rows = self._conn.execute(
            f"""SELECT {_FILE_COLUMNS} FROM files
                WHERE (status = 'quarantined' AND reason = ?)
                   OR (status = 'indexed' AND document_id IN (
                        SELECT document_id FROM pages
                        WHERE status = 'set_aside' AND reason = ?))
                ORDER BY path""",
            (Reason.NEEDS_OCR, Reason.NEEDS_OCR),
        ).fetchall()
        return [r for r in map(_file_record, rows) if _is_under(r.path, under)]

    def ocr_pending(self, under: Path | None = None) -> PendingOcr:
        """How many pages OCR could read (a stage 2a quarantined file has no
        stored page count: it counts as one page)."""
        rows = self._conn.execute(
            """SELECT path, COALESCE(page_count, 1) FROM files
                   WHERE status = 'quarantined' AND reason = ?
               UNION ALL
               SELECT f.path, COUNT(*) FROM pages p
                   JOIN files f ON f.document_id = p.document_id
                   WHERE f.status = 'indexed' AND p.status = 'set_aside' AND p.reason = ?
                   GROUP BY f.path""",
            (Reason.NEEDS_OCR, Reason.NEEDS_OCR),
        ).fetchall()
        counts = [n for path, n in rows if _is_under(Path(path), under)]
        return PendingOcr(pages=sum(counts), files=len(counts))

    def ocr_cached(
        self, file_hash: str, page_no: int, engine: str | None = None
    ) -> OcrPage | None:
        """The OCR result of that page by ``engine``; with no engine given, the
        latest result of any engine (so cached text applies without the extra)."""
        row = self._conn.execute(
            """SELECT text, confidence, engine FROM ocr_cache
               WHERE file_hash = ? AND page_no = ? AND (? IS NULL OR engine = ?)
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (file_hash, page_no, engine, engine),
        ).fetchone()
        return OcrPage(*row) if row else None

    # -- writes ----------------------------------------------------------
    def add_document(
        self, content_key: str, kind: str, page_count: int | None, pages: Sequence[PageVerdict]
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO documents (content_key, kind, page_count, created_at) VALUES (?, ?, ?, ?)",
                (content_key, kind, page_count, _now()),
            )
            document_id = int(cursor.lastrowid)
            for page in pages:
                if page.reason is None:
                    status = "ok"
                elif page.is_blank:
                    status = "blank"
                else:
                    status = "set_aside"
                page_id = self._conn.execute(
                    """INSERT INTO pages (document_id, page_no, text, status, reason,
                                          source, ocr_engine, ocr_confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (document_id, page.page_no, page.text, status, page.reason,
                     page.source, page.ocr_engine, page.ocr_confidence),
                ).lastrowid
                self._conn.executemany(
                    'INSERT INTO sensitive_spans (page_id, kind, start, "end") VALUES (?, ?, ?, ?)',
                    [(page_id, s.kind, s.start, s.end) for s in page.spans],
                )
        return document_id

    def _upsert_file(
        self,
        path: Path,
        file_hash: str,
        size_bytes: int,
        status: FileStatus,
        reason: Reason | None,
        document_id: int | None,
        encoding: str | None,
        page_count: int | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO files
                   (path, file_hash, size_bytes, encoding, status, reason, document_id,
                    ingested_at, page_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       file_hash = excluded.file_hash,
                       size_bytes = excluded.size_bytes,
                       encoding = excluded.encoding,
                       status = excluded.status,
                       reason = excluded.reason,
                       document_id = excluded.document_id,
                       ingested_at = excluded.ingested_at,
                       page_count = excluded.page_count""",
                (str(path), file_hash, size_bytes, encoding, status, reason, document_id,
                 _now(), page_count),
            )
            # A document nobody points to anymore is garbage. Its pages and
            # sensitive spans go with it (ON DELETE CASCADE).
            self._conn.execute(
                """DELETE FROM documents WHERE id NOT IN
                   (SELECT document_id FROM files WHERE document_id IS NOT NULL)"""
            )

    def link_file(
        self,
        path: Path,
        file_hash: str,
        size_bytes: int,
        document_id: int,
        encoding: str | None = None,
    ) -> None:
        self._upsert_file(path, file_hash, size_bytes, FileStatus.INDEXED, None, document_id, encoding)

    def record_quarantine(
        self,
        path: Path,
        file_hash: str,
        size_bytes: int,
        reason: Reason,
        page_count: int | None = None,
    ) -> None:
        self._upsert_file(
            path, file_hash, size_bytes, FileStatus.QUARANTINED, reason, None, None, page_count
        )

    def record_ignored(self, path: Path, file_hash: str, size_bytes: int) -> None:
        self._upsert_file(path, file_hash, size_bytes, FileStatus.IGNORED, None, None, None)

    def store_ocr(self, file_hash: str, page_no: int, page: OcrPage) -> None:
        """Save one OCR-ed page right away: an interrupted run keeps its work."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO ocr_cache (file_hash, page_no, engine, text, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash, page_no, engine) DO UPDATE SET
                       text = excluded.text, confidence = excluded.confidence,
                       created_at = excluded.created_at""",
                (file_hash, page_no, page.engine, page.text, page.confidence, _now()),
            )

    def record_decision(self, file_hash: str, decision: Decision) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO decisions (file_hash, decision, decided_at) VALUES (?, ?, ?)
                   ON CONFLICT(file_hash) DO UPDATE SET
                       decision = excluded.decision, decided_at = excluded.decided_at""",
                (file_hash, decision, _now()),
            )

    # -- reporting -------------------------------------------------------
    def stats(self) -> Stats:
        def count(sql: str) -> int:
            return self._conn.execute(sql).fetchone()[0]

        documents = count("SELECT COUNT(*) FROM documents")
        partial = count("SELECT COUNT(DISTINCT document_id) FROM pages WHERE status = 'set_aside'")
        indexed = count("SELECT COUNT(*) FROM files WHERE status = 'indexed'")
        return Stats(
            documents=documents,
            files=count("SELECT COUNT(*) FROM files"),
            duplicate_files=indexed - documents,
            accepted=documents - partial,
            partial=partial,
            quarantined=count("SELECT COUNT(*) FROM files WHERE status = 'quarantined'"),
            ignored=count("SELECT COUNT(*) FROM files WHERE status = 'ignored'"),
            sensitive_flags=len(self.sensitive_flags()),
            ocr_pages=count("SELECT COUNT(*) FROM pages WHERE source = 'ocr' AND status = 'ok'"),
            blank_pages=count("SELECT COUNT(*) FROM pages WHERE status = 'blank'"),
            chunks=count("SELECT COUNT(*) FROM chunks"),
            indexed_documents=count("SELECT COUNT(DISTINCT document_id) FROM chunks"),
        )

    def duplicate_groups(self) -> list[DuplicateGroup]:
        rows = self._conn.execute(
            """SELECT document_id, path FROM files
               WHERE status = 'indexed' AND document_id IN (
                   SELECT document_id FROM files WHERE status = 'indexed'
                   GROUP BY document_id HAVING COUNT(*) > 1)
               ORDER BY document_id, path"""
        ).fetchall()
        grouped: dict[int, list[Path]] = {}
        for document_id, path in rows:
            grouped.setdefault(document_id, []).append(Path(path))
        return [DuplicateGroup(doc_id, tuple(paths)) for doc_id, paths in grouped.items()]
