"""Stage 4: find the chunks closest to a question.

A small vector database, built by hand: every chunk vector of the current
embedding model sits in one numpy matrix (one row per chunk, float32, length 1).
A question becomes a vector too (``"search_query: "`` prefix), and its cosine
with every chunk is one matrix-vector product. Brute force, on purpose: 10 000
chunks take a few milliseconds, far below what a voice answer can feel, and no
second store has to be kept in sync with the catalog.

The matrix stays in memory between questions (JARVIS asks many in one process)
and is rebuilt when the catalog changed. A file deleted from disk is never
cited, nor a file edited since it was ingested (its stored text is outdated):
an unchanged copy is named instead, or the chunk is skipped.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localmind.catalog import Catalog
from localmind.ollama import (
    QUERY_PREFIX,
    SEARCH_KEEP_ALIVE,
    Classifier,
    ClassifierUnavailable,
    OllamaClassifier,
)

_CANDIDATES = 4  # look at 4 * k best chunks first; more only if files are gone


@dataclass(frozen=True)
class Hit:
    path: Path  # an existing file holding the chunk
    page_no: int
    score: float  # cosine similarity with the question, 1 = same direction
    text: str  # masked: never a sensitive value
    document_id: int
    chunk_id: int


class CatalogBusy(RuntimeError):
    """Another process is writing the catalog (an ingest): ask again later."""

    def __init__(self) -> None:
        super().__init__("the catalog is busy, try again")


class VectorIndex:
    def __init__(self, catalog: Catalog, embedder: Classifier, model: str | None = None) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.model = model or embedder.embed_model
        self._marker: tuple[int, int] | None = None
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def _load(self) -> None:
        marker = self.catalog.vector_marker()
        if marker == self._marker:
            return
        rows = self.catalog.vector_matrix_rows(self.model)
        self._ids = np.array([chunk_id for chunk_id, _ in rows], dtype=np.int64)
        if rows:
            flat = np.frombuffer(b"".join(vec for _, vec in rows), dtype=np.float32)
            self._matrix = flat.reshape(len(rows), -1)
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        self._marker = marker

    def _embed(self, question: str) -> np.ndarray:
        try:
            (vector,) = self.embedder.embed(
                [question], prefix=QUERY_PREFIX, keep_alive=SEARCH_KEEP_ALIVE
            )
        except ClassifierUnavailable as exc:
            raise exc.with_command("localmind search") from exc
        return np.asarray(vector, dtype=np.float32)

    def warm(self) -> None:
        """Load the embedding model now, so the first real question is fast."""
        self._embed("warm-up")

    def search(self, question: str, k: int = 5) -> list[Hit]:
        if not question.strip():
            raise ValueError("the question is empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        try:
            self._load()
            if not len(self._ids):
                return []
            scores = self._matrix @ self._embed(question.strip())
            want = min(len(scores), _CANDIDATES * k)
            while True:
                hits = self._existing_hits(self._best(scores, want), scores, k)
                if len(hits) >= k or want >= len(scores):
                    return hits
                want = len(scores)  # files were gone: look at everything
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc):
                raise CatalogBusy() from exc
            raise

    @staticmethod
    def _best(scores: np.ndarray, want: int) -> np.ndarray:
        """Indexes of the ``want`` best scores, best first."""
        if want < len(scores):
            top = np.argpartition(-scores, want - 1)[:want]
        else:
            top = np.arange(len(scores))
        return top[np.argsort(-scores[top], kind="stable")]

    def _existing_hits(self, order: np.ndarray, scores: np.ndarray, k: int) -> list[Hit]:
        rows = self.catalog.chunk_rows([int(self._ids[i]) for i in order])
        current: dict[Path, bool] = {}
        hits: list[Hit] = []
        for i in order:
            row = rows.get(int(self._ids[i]))
            if row is None:
                continue
            path = next(
                (p for p, size in zip(row.paths, row.sizes) if _unchanged(p, size, current)), None
            )
            if path is None:
                continue  # every copy was deleted or edited since: never cite stale text
            hits.append(Hit(path, row.page_no, float(scores[i]), row.text, row.document_id, row.chunk_id))
            if len(hits) == k:
                break
        return hits


def _unchanged(path: Path, size: int, cache: dict[Path, bool]) -> bool:
    """The file still exists and still has the size it had when ingested (an
    edit not yet re-ingested must not be quoted with its old text)."""
    if path not in cache:
        try:
            cache[path] = path.stat().st_size == size
        except OSError:
            cache[path] = False
    return cache[path]


def search(
    catalog: Catalog, question: str, k: int = 5, embedder: Classifier | None = None
) -> list[Hit]:
    """The ``k`` chunks closest to ``question``, best first (one-off search;
    keep a ``VectorIndex`` to ask several questions)."""
    return VectorIndex(catalog, embedder or OllamaClassifier()).search(question, k)
