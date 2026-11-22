"""The catalog: a SQLite database of *pointers* to your files (schema v2).

    files                       documents                 pages
    -------------------------   -----------------------   -------------------------
    path (unique)               content_key (unique)      document_id, page_no
    file_hash  (bytes)          kind, page_count          text
    status: indexed |           <-- document_id           status: ok | set_aside
      quarantined | ignored                               reason
    reason                                                --> sensitive_spans
    document_id (NULL unless indexed) ----------------->      (kind, start, end)

    decisions: file_hash -> accept | ignore   (keyed by content: survives moves)

* A **document** is a piece of information; a **file** is where it lives.
  Many files can point to one document: that is duplicate detection.
* A quarantined file has no document: text-less files are never "duplicates"
  of each other just because their (empty) text is identical.
* Original files are never moved, copied or modified; only their path is stored.
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

from localmind.quality import PageVerdict, Reason

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    content_key  TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    page_count   INTEGER,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('ok', 'set_aside')),
    reason       TEXT,
    UNIQUE (document_id, page_no)
);
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
    ingested_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    file_hash   TEXT PRIMARY KEY,
    decision    TEXT NOT NULL CHECK (decision IN ('accept', 'ignore')),
    decided_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id);
"""

_ALL_TABLES = ("sensitive_spans", "pages", "files", "documents", "decisions")


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


@dataclass(frozen=True)
class DuplicateGroup:
    document_id: int
    paths: tuple[Path, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _reason(value: str | None) -> Reason | None:
    return Reason(value) if value else None


_FILE_COLUMNS = "path, file_hash, size_bytes, status, reason, document_id"


def _file_record(row: tuple) -> FileRecord:
    path, file_hash, size, status, reason, document_id = row
    return FileRecord(Path(path), file_hash, size, FileStatus(status), _reason(reason), document_id)


class Catalog:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.rebuilt_from_v1 = self._migrate()
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _migrate(self) -> bool:
        """A v1 catalog (stage 1) is a rebuildable index: drop it and start over.
        Originals are untouched and v1 had no user decisions, so nothing is lost
        that a new `ingest` cannot recreate."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()[0]
        if version >= SCHEMA_VERSION or not has_tables:
            return False
        with self._conn:
            for table in _ALL_TABLES:
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
            """SELECT f.path, p.page_no, p.reason FROM pages p
               JOIN files f ON f.document_id = p.document_id
               WHERE p.status = 'set_aside' AND f.status = 'indexed'
               ORDER BY f.path, p.page_no"""
        ).fetchall()
        return [SetAsidePage(Path(path), n, Reason(r)) for path, n, r in rows]

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
        records = [_file_record(r) for r in rows]
        if under is None:
            return records
        return [r for r in records if r.path == under or r.path.is_relative_to(under)]

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
                status = "ok" if page.reason is None else "set_aside"
                page_id = self._conn.execute(
                    """INSERT INTO pages (document_id, page_no, text, status, reason)
                       VALUES (?, ?, ?, ?, ?)""",
                    (document_id, page.page_no, page.text, status, page.reason),
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
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO files
                   (path, file_hash, size_bytes, encoding, status, reason, document_id, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       file_hash = excluded.file_hash,
                       size_bytes = excluded.size_bytes,
                       encoding = excluded.encoding,
                       status = excluded.status,
                       reason = excluded.reason,
                       document_id = excluded.document_id,
                       ingested_at = excluded.ingested_at""",
                (str(path), file_hash, size_bytes, encoding, status, reason, document_id, _now()),
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

    def record_quarantine(self, path: Path, file_hash: str, size_bytes: int, reason: Reason) -> None:
        self._upsert_file(path, file_hash, size_bytes, FileStatus.QUARANTINED, reason, None, None)

    def record_ignored(self, path: Path, file_hash: str, size_bytes: int) -> None:
        self._upsert_file(path, file_hash, size_bytes, FileStatus.IGNORED, None, None, None)

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
