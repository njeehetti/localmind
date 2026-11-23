"""Catalog storage for stages 4-5: chunks, their vectors (v5) and keywords (v6).

    chunks                                   chunk_vectors
    --------------------------------------   ------------------------------
    document_id --> documents (cascade)      chunk_id --> chunks (cascade)
    page_no, seq (unique per document)       model
    text (masked: never a sensitive value)   vector (float32, normalised)
    chunker (the parameters that cut it)

One source of truth: a document that disappears (edited file, pruned file)
takes its chunks and vectors with it, in the same transaction. Copies of a file
share one document, so they share chunks: no duplicate hits.

``chunks_fts`` (stage 5) is an FTS5 keyword index over ``chunks.text``. It is
"external content": the text lives once, in ``chunks``, and triggers keep the
index in step on every insert, delete (cascades included) and update.
Accents are ignored ("electricite" finds "électricité").
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from localmind.chunking import Chunk
from localmind.scoring import to_bytes

CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      INTEGER NOT NULL,
    seq          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    chunker      TEXT NOT NULL,
    UNIQUE (document_id, seq)
);
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_model ON chunk_vectors(model);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    document_id: int
    page_no: int
    text: str
    paths: tuple[Path, ...]  # the document's indexed files, oldest first
    sizes: tuple[int, ...] = ()  # their sizes when last ingested (same order)


def _under(path: str, under: Path | None) -> bool:
    p = Path(path)
    return under is None or p == under or under in p.parents


class ChunkStore:
    """Mixin for ``Catalog`` (needs ``self._conn`` and ``self._masked_pages``)."""

    def masked_pages(self, document_id: int) -> list[tuple[int, str]]:
        """``(page_no, text)`` of the document's usable pages, sensitive spans masked."""
        return self._masked_pages(document_id)  # type: ignore[attr-defined]

    def documents_to_index(
        self, chunker: str, model: str, under: Path | None = None, redo: bool = False
    ) -> list[int]:
        """Documents with usable text whose chunks are missing, were cut with
        other parameters, or lack a vector from ``model`` (all of them if ``redo``)."""
        stale = "1" if redo else """(
            NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
            OR EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.chunker != :chunker)
            OR EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND NOT EXISTS (
                SELECT 1 FROM chunk_vectors v WHERE v.chunk_id = c.id AND v.model = :model)))"""
        rows = self._conn.execute(  # type: ignore[attr-defined]
            f"""SELECT d.id, f.path FROM documents d JOIN files f ON f.document_id = d.id
                WHERE f.status = 'indexed'
                  AND EXISTS (SELECT 1 FROM pages p WHERE p.document_id = d.id AND p.status = 'ok')
                  AND {stale}
                ORDER BY d.id, f.id""",
            {"chunker": chunker, "model": model},
        ).fetchall()
        ids: list[int] = []
        for doc_id, path in rows:
            if _under(path, under) and doc_id not in ids:
                ids.append(doc_id)
        return ids

    def store_chunks(
        self,
        document_id: int,
        chunker: str,
        model: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Replace the document's chunks and vectors, all at once or not at all."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._conn:  # type: ignore[attr-defined]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))  # type: ignore[attr-defined]
            for chunk, vector in zip(chunks, vectors):
                cursor = self._conn.execute(  # type: ignore[attr-defined]
                    "INSERT INTO chunks (document_id, page_no, seq, text, chunker) VALUES (?, ?, ?, ?, ?)",
                    (document_id, chunk.page_no, chunk.seq, chunk.text, chunker),
                )
                self._conn.execute(  # type: ignore[attr-defined]
                    "INSERT INTO chunk_vectors (chunk_id, model, vector) VALUES (?, ?, ?)",
                    (cursor.lastrowid, model, to_bytes(vector)),
                )

    def vector_matrix_rows(self, model: str) -> list[tuple[int, bytes]]:
        """``(chunk_id, vector bytes)`` of every chunk embedded by ``model``."""
        return self._conn.execute(  # type: ignore[attr-defined]
            "SELECT chunk_id, vector FROM chunk_vectors WHERE model = ? ORDER BY chunk_id", (model,)
        ).fetchall()

    def keyword_rows(
        self, match: str, limit: int, document_ids: Collection[int] | None = None
    ) -> list[int]:
        """Chunk ids matching the FTS5 query ``match``, best BM25 first, at most
        ``limit``; only chunks of ``document_ids`` when given (empty: none)."""
        if document_ids is not None and not document_ids:
            return []
        where, args = "chunks_fts MATCH ?", [match]
        if document_ids is not None:
            ids = sorted(document_ids)
            where += f" AND c.document_id IN ({','.join('?' * len(ids))})"
            args += ids
        rows = self._conn.execute(  # type: ignore[attr-defined]
            f"""SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE {where} ORDER BY bm25(chunks_fts), c.id LIMIT ?""",
            [*args, limit],
        )
        return [r[0] for r in rows]

    def vector_marker(self) -> tuple[int, int]:
        """Changes whenever the catalog may have changed: commits by another
        connection (``data_version``) or by this one (``total_changes``).
        Row ids are reused after a delete, so counting rows would not do."""
        version = self._conn.execute("PRAGMA data_version").fetchone()[0]  # type: ignore[attr-defined]
        return version, self._conn.total_changes  # type: ignore[attr-defined]

    def chunk_rows(self, chunk_ids: Sequence[int]) -> dict[int, ChunkRow]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(  # type: ignore[attr-defined]
            f"SELECT id, document_id, page_no, text FROM chunks WHERE id IN ({marks})", list(chunk_ids)
        ).fetchall()
        doc_ids = sorted({r[1] for r in rows})
        files: dict[int, list[tuple[Path, int]]] = {d: [] for d in doc_ids}
        if doc_ids:
            for doc_id, path, size in self._conn.execute(  # type: ignore[attr-defined]
                f"""SELECT document_id, path, size_bytes FROM files WHERE status = 'indexed'
                    AND document_id IN ({",".join("?" * len(doc_ids))}) ORDER BY id""",
                doc_ids,
            ):
                files[doc_id].append((Path(path), size))
        return {
            r[0]: ChunkRow(
                r[0], r[1], r[2], r[3],
                tuple(p for p, _ in files[r[1]]), tuple(s for _, s in files[r[1]]),
            )
            for r in rows
        }

    # -- forgetting files deleted from disk (prune) -------------------------
    def file_paths(self) -> list[Path]:
        """Every file the catalog knows, whatever its status."""
        rows = self._conn.execute("SELECT path FROM files ORDER BY path")  # type: ignore[attr-defined]
        return [Path(r[0]) for r in rows]

    def forget_files(self, paths: Sequence[Path]) -> int:
        """Delete these files' rows, then the documents no file points to any more
        (their pages, chunks, vectors and classification go with them).
        Returns the number of documents forgotten. ``decisions``, ``ocr_cache``
        and ``folder_decisions`` stay: they are keyed by content, and a file put
        back later finds them again."""
        with self._conn:  # type: ignore[attr-defined]
            self._conn.executemany(  # type: ignore[attr-defined]
                "DELETE FROM files WHERE path = ?", [(str(p),) for p in paths]
            )
            cursor = self._conn.execute(  # type: ignore[attr-defined]
                """DELETE FROM documents WHERE id NOT IN
                   (SELECT document_id FROM files WHERE document_id IS NOT NULL)"""
            )
            return cursor.rowcount
