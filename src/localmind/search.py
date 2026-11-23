"""Stages 4-5: find the chunks that best answer a question (hybrid search).

Two searches run side by side and their rankings are fused:

* **meaning** (stage 4): a small vector database built by hand. Every chunk
  vector of the current embedding model sits in one numpy matrix (one row per
  chunk, float32, length 1). The question becomes a vector too
  (``"search_query: "`` prefix) and its cosine with every chunk is one
  matrix-vector product. Brute force on purpose: 10 000 chunks take a few
  milliseconds, and no second store has to be kept in sync with the catalog.
* **keywords** (stage 5): BM25 over the SQLite FTS5 index ``chunks_fts``,
  which triggers keep in step with ``chunks`` (``keyword.py`` builds the query).

Reciprocal rank fusion merges the two rankings (``keyword.fuse``). A folder
filter (``"Maison"`` or ``"Maison/Factures"``) is applied before ranking, on
both sides. If the question cannot be embedded (Ollama down), the keyword side
answers alone and ``VectorIndex.last_warning`` says why.

The matrix stays in memory between questions (JARVIS asks many in one process)
and is rebuilt when the catalog changed. A file deleted from disk is never
cited, nor a file edited since it was ingested (its stored text is outdated):
an unchanged copy is named instead, or the chunk is skipped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localmind.catalog import Catalog
from localmind.catalog_folders import DocFolder
from localmind.folders import parse_folder_filter
from localmind.keyword import fuse, keyword_query
from localmind.ollama import (
    QUERY_PREFIX,
    SEARCH_KEEP_ALIVE,
    Classifier,
    ClassifierUnavailable,
    OllamaClassifier,
)

_CANDIDATES = 4  # each side looks at its max(30, 4 * k) best chunks first
_MIN_CANDIDATES = 30
_ALL = 2**31 - 1  # files were gone: look at every chunk


@dataclass(frozen=True)
class Hit:
    path: Path  # an existing file holding the chunk
    page_no: int
    score: float | None  # cosine with the question (1 = same direction); None without a vector
    text: str  # masked: never a sensitive value
    document_id: int
    chunk_id: int
    folder: str | None = None  # "Maison/Factures"; None when the document is unclassified
    to_confirm: bool = False  # placed by the LLM, not reviewed by the user yet
    matched: frozenset[str] = frozenset()  # {"meaning"}, {"keywords"} or both


class CatalogBusy(RuntimeError):
    """Another process is writing the catalog (an ingest): ask again later."""

    def __init__(self) -> None:
        super().__init__("the catalog is busy, try again")


class VectorIndex:
    """Hybrid search over the catalog (the name is stage 4's, kept for callers)."""

    def __init__(self, catalog: Catalog, embedder: Classifier, model: str | None = None) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.model = model or embedder.embed_model
        self.last_warning: str | None = None  # why the last search used keywords only
        self._marker: tuple[int, int] | None = None
        self._ids = np.empty(0, dtype=np.int64)
        self._docs = np.empty(0, dtype=np.int64)
        self._row: dict[int, int] = {}
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def _load(self) -> None:
        marker = self.catalog.vector_marker()
        if marker == self._marker:
            return
        rows = self.catalog.vector_rows(self.model)
        self._ids = np.array([chunk_id for chunk_id, _, _ in rows], dtype=np.int64)
        self._docs = np.array([doc_id for _, doc_id, _ in rows], dtype=np.int64)
        self._row = {chunk_id: i for i, (chunk_id, _, _) in enumerate(rows)}
        if rows:
            flat = np.frombuffer(b"".join(vec for _, _, vec in rows), dtype=np.float32)
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

    def search(self, question: str, k: int = 5, folder: str | None = None) -> list[Hit]:
        """The ``k`` chunks that best answer ``question``, best first, from the
        documents of ``folder`` only when given (``UnknownFolder`` or
        ``InvalidFolderName`` otherwise)."""
        if not question.strip():
            raise ValueError("the question is empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        only = parse_folder_filter(folder) if folder is not None else None
        try:
            folders = self.catalog.document_folders()
            allowed = set(self.catalog.document_folders(only)) if only else None
            self._load()
            scores = self._scores(question.strip())
            match = keyword_query(question)
            want = max(_MIN_CANDIDATES, _CANDIDATES * k)
            while True:
                by_meaning = self._by_meaning(scores, allowed, want)
                by_keywords = self.catalog.keyword_rows(match, want, allowed) if match else []
                hits = self._existing_hits(
                    fuse(by_meaning, by_keywords), scores, k,
                    set(by_meaning), set(by_keywords), folders,
                )
                if len(hits) >= k or want == _ALL:
                    return hits
                if len(by_meaning) < want and len(by_keywords) < want:
                    return hits  # both sides already gave everything they had
                want = _ALL
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc):
                raise CatalogBusy() from exc
            raise

    def _scores(self, question: str) -> np.ndarray | None:
        """Cosine of every matrix row with the question; None when there is no
        vector to compare with, or Ollama cannot embed it (``last_warning``)."""
        self.last_warning = None
        if not len(self._ids):
            return None  # nothing embedded by this model: do not wake Ollama
        try:
            return self._matrix @ self._embed(question)
        except ClassifierUnavailable as exc:
            self.last_warning = exc.reason
            return None

    def _by_meaning(
        self, scores: np.ndarray | None, allowed: Collection[int] | None, want: int
    ) -> list[int]:
        """Chunk ids of the ``want`` best cosines, best first, in ``allowed`` documents."""
        if scores is None:
            return []
        rows = np.arange(len(scores))
        if allowed is not None:
            rows = rows[np.isin(self._docs, list(allowed))]
        if want < len(rows):
            rows = rows[np.argpartition(-scores[rows], want - 1)[:want]]
        rows = rows[np.argsort(-scores[rows], kind="stable")]
        return [int(self._ids[i]) for i in rows]

    def _existing_hits(
        self,
        order: list[int],
        scores: np.ndarray | None,
        k: int,
        by_meaning: set[int],
        by_keywords: set[int],
        folders: dict[int, DocFolder],
    ) -> list[Hit]:
        rows = self.catalog.chunk_rows(order)
        current: dict[Path, bool] = {}
        hits: list[Hit] = []
        for chunk_id in order:
            row = rows.get(chunk_id)
            if row is None:
                continue
            path = next(
                (p for p, size in zip(row.paths, row.sizes) if _unchanged(p, size, current)), None
            )
            if path is None:
                continue  # every copy was deleted or edited since: never cite stale text
            i = self._row.get(chunk_id)
            folder = folders.get(row.document_id)
            hits.append(Hit(
                path, row.page_no,
                None if scores is None or i is None else float(scores[i]),
                row.text, row.document_id, chunk_id,
                folder.name if folder else None,
                folder.to_confirm if folder else False,
                frozenset(
                    side for side, found in (("meaning", by_meaning), ("keywords", by_keywords))
                    if chunk_id in found
                ),
            ))
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
    catalog: Catalog,
    question: str,
    k: int = 5,
    embedder: Classifier | None = None,
    folder: str | None = None,
) -> list[Hit]:
    """The ``k`` chunks that best answer ``question``, best first (one-off
    search; keep a ``VectorIndex`` to ask several questions)."""
    return VectorIndex(catalog, embedder or OllamaClassifier()).search(question, k, folder)
