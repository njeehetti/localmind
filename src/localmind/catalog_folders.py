"""Catalog storage for stage 3: virtual folders, classifications, document
embeddings and the user's folder decisions (schema v4).

    folders            classifications (1 per document)     embeddings
    ----------------   ----------------------------------   -------------------------
    domain, type       document_id --> documents            document_id, model
    key (folded,       folder_id --> folders (NULL: none)   vector (float32 bytes)
      unique)          status: placed | to_confirm
    created_by:        method: embedding | llm | user       folder_decisions
      user | llm       score, alternative_folder_id,        -------------------------
    confirmed          reason, model                        file_hash (unique), path
                                                            domain, type

``folder_decisions`` holds the user's choices, keyed by file hash (an exact copy
follows) and by path (a file edited in place follows). Like ``decisions`` and
``ocr_cache`` it is never dropped by a migration.

Everything sent to a model goes through ``documents_to_classify``, whose text
has its sensitive spans masked (``[IBAN]``...).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from localmind.folders import FolderName, fold
from localmind.scoring import Example, from_bytes, to_bytes

FOLDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id          INTEGER PRIMARY KEY,
    domain      TEXT NOT NULL,
    type        TEXT NOT NULL,
    key         TEXT NOT NULL UNIQUE,
    created_by  TEXT NOT NULL CHECK (created_by IN ('user', 'llm')),
    confirmed   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS classifications (
    document_id            INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    folder_id              INTEGER REFERENCES folders(id),
    status                 TEXT NOT NULL CHECK (status IN ('placed', 'to_confirm')),
    method                 TEXT CHECK (method IN ('embedding', 'llm', 'user')),
    score                  REAL,
    alternative_folder_id  INTEGER REFERENCES folders(id),
    reason                 TEXT,
    model                  TEXT,
    classified_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    model        TEXT NOT NULL,
    vector       BLOB NOT NULL,
    PRIMARY KEY (document_id, model)
);
CREATE TABLE IF NOT EXISTS folder_decisions (
    file_hash   TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    domain      TEXT NOT NULL,
    type        TEXT NOT NULL,
    decided_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_folder_decisions_path ON folder_decisions(path);
CREATE INDEX IF NOT EXISTS idx_classifications_folder ON classifications(folder_id);
"""

MASKS = {"iban": "[IBAN]", "card": "[CARD]", "fr_nir": "[NIR]", "password": "[PASSWORD]"}


@dataclass(frozen=True)
class Folder:
    id: int
    name: FolderName
    created_by: Literal["user", "llm"]
    confirmed: bool
    documents: int


@dataclass(frozen=True)
class Classification:
    document_id: int
    folder_id: int | None
    status: Literal["placed", "to_confirm"]
    method: Literal["embedding", "llm", "user"] | None
    score: float | None
    alternative_folder_id: int | None
    reason: str | None


@dataclass(frozen=True)
class DocumentToClassify:
    id: int
    name: str  # file name of its first file
    text: str  # usable text, sensitive spans masked
    file_hashes: tuple[str, ...]
    paths: tuple[Path, ...]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _key(name: FolderName) -> str:
    return f"{fold(name.domain)}/{fold(name.type)}"


def _under(paths: Sequence[Path], under: Path | None) -> bool:
    return under is None or any(p == under or under in p.parents for p in paths)


_CLASSIFICATION_COLUMNS = (
    "document_id, folder_id, status, method, score, alternative_folder_id, reason"
)


class FolderStore:
    """Mixin for ``Catalog``: needs ``self._conn``."""

    _conn: sqlite3.Connection

    # -- folders ---------------------------------------------------------
    def folders(self) -> list[Folder]:
        rows = self._conn.execute(
            """SELECT f.id, f.domain, f.type, f.created_by, f.confirmed,
                      (SELECT COUNT(*) FROM classifications c WHERE c.folder_id = f.id)
               FROM folders f ORDER BY f.key"""
        ).fetchall()
        return [Folder(r[0], FolderName(r[1], r[2]), r[3], bool(r[4]), r[5]) for r in rows]

    def folder(self, folder_id: int) -> Folder:
        return next(f for f in self.folders() if f.id == folder_id)

    def folder_by_name(self, name: FolderName) -> Folder | None:
        row = self._conn.execute("SELECT id FROM folders WHERE key = ?", (_key(name),)).fetchone()
        return self.folder(row[0]) if row else None

    def ensure_folder(
        self, name: FolderName, created_by: Literal["user", "llm"], confirmed: bool
    ) -> int:
        with self._conn:
            row = self._conn.execute(
                "SELECT id FROM folders WHERE key = ?", (_key(name),)
            ).fetchone()
            if row:
                if confirmed:
                    self._conn.execute("UPDATE folders SET confirmed = 1 WHERE id = ?", (row[0],))
                return int(row[0])
            cursor = self._conn.execute(
                """INSERT INTO folders (domain, type, key, created_by, confirmed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name.domain, name.type, _key(name), created_by, int(confirmed), _now()),
            )
            return int(cursor.lastrowid)

    def rename_folder_row(self, folder_id: int, name: FolderName) -> None:
        """Rename in place; the folder becomes confirmed (the user named it)."""
        with self._conn:
            self._conn.execute(
                "UPDATE folders SET domain = ?, type = ?, key = ?, confirmed = 1 WHERE id = ?",
                (name.domain, name.type, _key(name), folder_id),
            )

    def merge_folders(self, source_id: int, target_id: int) -> None:
        """Move everything from ``source`` into ``target`` and delete ``source``."""
        with self._conn:
            self._conn.execute(
                "UPDATE classifications SET folder_id = ? WHERE folder_id = ?", (target_id, source_id)
            )
            self._conn.execute(
                """UPDATE classifications SET alternative_folder_id = NULL
                   WHERE alternative_folder_id IN (?, ?) AND folder_id = ?""",
                (source_id, target_id, target_id),
            )
            self._conn.execute(
                "UPDATE classifications SET alternative_folder_id = ? WHERE alternative_folder_id = ?",
                (target_id, source_id),
            )
            self._conn.execute("UPDATE folders SET confirmed = 1 WHERE id = ?", (target_id,))
            self._conn.execute("DELETE FROM folders WHERE id = ?", (source_id,))

    def delete_empty_llm_folders(self) -> None:
        with self._conn:
            self._conn.execute(
                """DELETE FROM folders WHERE created_by = 'llm'
                   AND id NOT IN (SELECT folder_id FROM classifications
                                  WHERE folder_id IS NOT NULL)
                   AND id NOT IN (SELECT alternative_folder_id FROM classifications
                                  WHERE alternative_folder_id IS NOT NULL)"""
            )

    # -- documents -------------------------------------------------------
    def _masked_text(self, document_id: int) -> str:
        pages = self._conn.execute(
            "SELECT id, text FROM pages WHERE document_id = ? AND status = 'ok' ORDER BY page_no",
            (document_id,),
        ).fetchall()
        texts = []
        for page_id, text in pages:
            spans = self._conn.execute(
                'SELECT kind, start, "end" FROM sensitive_spans WHERE page_id = ? ORDER BY start DESC',
                (page_id,),
            ).fetchall()
            for kind, start, end in spans:  # from the end: earlier offsets stay valid
                text = text[:start] + MASKS.get(kind, "[SENSITIVE]") + text[end:]
            texts.append(text)
        return "\f".join(texts)

    def _document_to_classify(self, document_id: int) -> DocumentToClassify:
        files = self._conn.execute(
            """SELECT path, file_hash FROM files
               WHERE document_id = ? AND status = 'indexed' ORDER BY id""",
            (document_id,),
        ).fetchall()
        paths = tuple(Path(p) for p, _ in files)
        return DocumentToClassify(
            id=document_id,
            name=paths[0].name if paths else "",
            text=self._masked_text(document_id),
            file_hashes=tuple(h for _, h in files),
            paths=paths,
        )

    def document_to_classify(self, document_id: int) -> DocumentToClassify:
        return self._document_to_classify(document_id)

    def unsorted_documents(self) -> list[DocumentToClassify]:
        """Documents with an indexed file but no folder (never classified, or
        classification failed)."""
        ids = [
            r[0]
            for r in self._conn.execute(
                """SELECT d.id FROM documents d
                   LEFT JOIN classifications c ON c.document_id = d.id
                   WHERE c.folder_id IS NULL ORDER BY d.id"""
            )
        ]
        return [d for d in map(self._document_to_classify, ids) if d.paths]

    def documents_to_classify(self, under: Path | None, redo: bool) -> list[DocumentToClassify]:
        condition = "c.method IS NULL OR c.method != 'user'" if redo else "c.document_id IS NULL"
        ids = [
            r[0]
            for r in self._conn.execute(
                f"""SELECT d.id FROM documents d
                    LEFT JOIN classifications c ON c.document_id = d.id
                    WHERE {condition} ORDER BY d.id"""
            )
        ]
        docs = [self._document_to_classify(i) for i in ids]
        return [d for d in docs if d.paths and _under(d.paths, under)]

    def documents_in(self, folder_id: int) -> list[DocumentToClassify]:
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT document_id FROM classifications WHERE folder_id = ? ORDER BY document_id",
                (folder_id,),
            )
        ]
        return [self._document_to_classify(i) for i in ids]

    # -- classifications -------------------------------------------------
    def set_classification(self, c: Classification, model: str) -> None:
        with self._conn:
            self._conn.execute(
                f"""INSERT INTO classifications ({_CLASSIFICATION_COLUMNS}, model, classified_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        folder_id = excluded.folder_id, status = excluded.status,
                        method = excluded.method, score = excluded.score,
                        alternative_folder_id = excluded.alternative_folder_id,
                        reason = excluded.reason, model = excluded.model,
                        classified_at = excluded.classified_at""",
                (c.document_id, c.folder_id, c.status, c.method, c.score,
                 c.alternative_folder_id, c.reason, model, _now()),
            )

    def classification(self, document_id: int) -> Classification | None:
        row = self._conn.execute(
            f"SELECT {_CLASSIFICATION_COLUMNS} FROM classifications WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return Classification(*row) if row else None

    def pending_reviews(self, under: Path | None = None) -> list[Classification]:
        rows = self._conn.execute(
            f"""SELECT {_CLASSIFICATION_COLUMNS} FROM classifications
                WHERE status = 'to_confirm' ORDER BY document_id"""
        ).fetchall()
        pending = [Classification(*r) for r in rows]
        if under is None:
            return pending
        return [c for c in pending if _under(self._document_to_classify(c.document_id).paths, under)]

    # -- embeddings ------------------------------------------------------
    def store_embedding(self, document_id: int, model: str, vector: Sequence[float]) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO embeddings (document_id, model, vector) VALUES (?, ?, ?)
                   ON CONFLICT(document_id, model) DO UPDATE SET vector = excluded.vector""",
                (document_id, model, to_bytes(vector)),
            )

    def embedding(self, document_id: int, model: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE document_id = ? AND model = ?",
            (document_id, model),
        ).fetchone()
        return from_bytes(row[0]) if row else None

    def examples(
        self, model: str, user_weight: int, exclude: int | None = None
    ) -> list[Example]:
        """Documents a new one is compared with: placed in a confirmed folder, or
        waiting in a proposed (unconfirmed) one. Hesitations do not count.
        ``exclude``: the document being re-classified never matches itself."""
        rows = self._conn.execute(
            """SELECT c.folder_id, c.method, e.vector FROM classifications c
               JOIN folders f ON f.id = c.folder_id
               JOIN embeddings e ON e.document_id = c.document_id AND e.model = ?
               WHERE ((c.status = 'placed' AND f.confirmed = 1)
                      OR (c.status = 'to_confirm' AND f.confirmed = 0))
                 AND c.document_id IS NOT ?""",
            (model, exclude),
        ).fetchall()
        return [
            Example(folder_id, from_bytes(vec), user_weight if method == "user" else 1)
            for folder_id, method, vec in rows
        ]

    def documents_missing_embedding(self, model: str) -> list[DocumentToClassify]:
        """Classified documents with no vector for ``model`` (new embedding model,
        or placed by a user decision without any model call)."""
        ids = [
            r[0]
            for r in self._conn.execute(
                """SELECT c.document_id FROM classifications c
                   LEFT JOIN embeddings e ON e.document_id = c.document_id AND e.model = ?
                   WHERE c.folder_id IS NOT NULL AND e.document_id IS NULL
                   ORDER BY c.document_id""",
                (model,),
            )
        ]
        return [self._document_to_classify(i) for i in ids]

    def user_examples(self, model: str) -> list[tuple[str, FolderName, list[float]]]:
        """(file name, folder, vector) of documents placed by the user."""
        rows = self._conn.execute(
            """SELECT c.document_id, f.domain, f.type, e.vector FROM classifications c
               JOIN folders f ON f.id = c.folder_id
               JOIN embeddings e ON e.document_id = c.document_id AND e.model = ?
               WHERE c.method = 'user' ORDER BY c.document_id""",
            (model,),
        ).fetchall()
        return [
            (self._document_to_classify(doc_id).name, FolderName(d, t), from_bytes(v))
            for doc_id, d, t, v in rows
        ]

    # -- user decisions --------------------------------------------------
    def record_folder_decision(self, file_hash: str, path: Path, name: FolderName) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO folder_decisions (file_hash, path, domain, type, decided_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(file_hash) DO UPDATE SET path = excluded.path,
                       domain = excluded.domain, type = excluded.type,
                       decided_at = excluded.decided_at""",
                (file_hash, str(path), name.domain, name.type, _now()),
            )

    def folder_decision_for(
        self, file_hashes: Sequence[str], paths: Sequence[Path]
    ) -> tuple[FolderName, str] | None:
        """The user's folder for these files: by hash (copies), else by path
        (edited in place). Returns the folder and the matched row's hash."""
        for column, values in (("file_hash", list(file_hashes)), ("path", [str(p) for p in paths])):
            if not values:
                continue
            marks = ", ".join("?" * len(values))
            row = self._conn.execute(
                f"""SELECT domain, type, file_hash FROM folder_decisions
                    WHERE {column} IN ({marks}) ORDER BY decided_at DESC, rowid DESC LIMIT 1""",
                values,
            ).fetchone()
            if row:
                return FolderName(row[0], row[1]), row[2]
        return None

    def rekey_folder_decision(self, old_hash: str, new_hash: str, path: Path) -> None:
        """A file edited in place: its decision also covers the new content.
        The old row stays: an exact copy of the old version keeps its folder."""
        with self._conn:
            row = self._conn.execute(
                "SELECT domain, type, decided_at FROM folder_decisions WHERE file_hash = ?",
                (old_hash,),
            ).fetchone()
            if row is None or old_hash == new_hash:
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO folder_decisions
                   (file_hash, path, domain, type, decided_at) VALUES (?, ?, ?, ?, ?)""",
                (new_hash, str(path), row[0], row[1], row[2]),
            )

    def rename_folder_decisions(self, old: FolderName, new: FolderName) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE folder_decisions SET domain = ?, type = ? WHERE domain = ? AND type = ?",
                (new.domain, new.type, old.domain, old.type),
            )
