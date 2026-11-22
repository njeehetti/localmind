"""The catalog: a SQLite database of *pointers* to your files.

Two tables, on purpose:

    files                              documents
    ---------------------------        ---------------------------
    path (unique)  ──────────────┐     content_hash (unique)
    file_hash (bytes fingerprint) └──► text, kind, page_count...
    document_id  ────────────────────► id

* A **document** is a piece of information (unique normalized text).
* A **file** is a place on disk where that information lives.

Many files can point to the same document: that *is* duplicate detection.
Original files are never moved, copied or modified; only their path is stored.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from localmind.extractors import ExtractedText

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY,
    content_hash  TEXT NOT NULL UNIQUE,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    page_count    INTEGER,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    file_hash    TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    encoding     TEXT,
    document_id  INTEGER NOT NULL REFERENCES documents(id),
    ingested_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
"""


@dataclass(frozen=True)
class FileRecord:
    path: Path
    file_hash: str
    document_id: int


@dataclass(frozen=True)
class Document:
    id: int
    content_hash: str
    kind: str
    text: str
    char_count: int
    page_count: int | None


@dataclass(frozen=True)
class Stats:
    documents: int
    files: int
    duplicate_files: int


@dataclass(frozen=True)
class DuplicateGroup:
    document_id: int
    paths: tuple[Path, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Catalog:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

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
            "SELECT path, file_hash, document_id FROM files WHERE path = ?",
            (str(path),),
        ).fetchone()
        return FileRecord(Path(row[0]), row[1], row[2]) if row else None

    def document_id_by_file_hash(self, file_hash: str) -> int | None:
        row = self._conn.execute(
            "SELECT document_id FROM files WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()
        return row[0] if row else None

    def document_id_by_content_hash(self, content_hash: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row[0] if row else None

    def document_for_path(self, path: Path) -> Document | None:
        row = self._conn.execute(
            """SELECT d.id, d.content_hash, d.kind, d.text, d.char_count, d.page_count
               FROM documents d JOIN files f ON f.document_id = d.id
               WHERE f.path = ?""",
            (str(path),),
        ).fetchone()
        return Document(*row) if row else None

    # -- writes ----------------------------------------------------------
    def add_document(self, content_hash: str, extracted: ExtractedText) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """INSERT INTO documents
                   (content_hash, kind, text, char_count, page_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    content_hash,
                    extracted.kind,
                    extracted.text,
                    len(extracted.text),
                    extracted.page_count,
                    _now(),
                ),
            )
        return int(cursor.lastrowid)

    def link_file(
        self,
        path: Path,
        file_hash: str,
        size_bytes: int,
        document_id: int,
        encoding: str | None = None,
    ) -> None:
        """Create or repoint the file -> document pointer, then drop orphans."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO files
                   (path, file_hash, size_bytes, encoding, document_id, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       file_hash = excluded.file_hash,
                       size_bytes = excluded.size_bytes,
                       encoding = excluded.encoding,
                       document_id = excluded.document_id,
                       ingested_at = excluded.ingested_at""",
                (str(path), file_hash, size_bytes, encoding, document_id, _now()),
            )
            # A document nobody points to anymore (e.g. old version of an
            # edited file) is garbage: remove it.
            self._conn.execute(
                """DELETE FROM documents
                   WHERE id NOT IN (SELECT DISTINCT document_id FROM files)"""
            )

    # -- reporting -------------------------------------------------------
    def stats(self) -> Stats:
        documents = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return Stats(documents=documents, files=files, duplicate_files=files - documents)

    def duplicate_groups(self) -> list[DuplicateGroup]:
        rows = self._conn.execute(
            """SELECT document_id, path FROM files
               WHERE document_id IN (
                   SELECT document_id FROM files
                   GROUP BY document_id HAVING COUNT(*) > 1)
               ORDER BY document_id, path"""
        ).fetchall()
        grouped: dict[int, list[Path]] = {}
        for document_id, path in rows:
            grouped.setdefault(document_id, []).append(Path(path))
        return [DuplicateGroup(doc_id, tuple(paths)) for doc_id, paths in grouped.items()]
