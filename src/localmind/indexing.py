"""Stage 4: index documents for search (chunks + vectors).

Every accepted document is cut into chunks (``chunking``), each chunk is
embedded as ``"search_document: " + "file p. N" + "\\n" + text``, and the vectors
are stored in the catalog next to the chunks.

All chunks of a run are embedded in batches of 32, one model only (no switching
with an LLM, see stage 3). A document is stored as soon as all of its vectors
are in, in one transaction: if Ollama stops halfway, every document is either
fully indexed or untouched, and the next run picks up the rest.
"""

from __future__ import annotations

import http.client
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from localmind.catalog import Catalog
from localmind.chunking import Chunk, ChunkConfig, chunk_pages, embed_text
from localmind.ollama import (
    BATCH,
    DOCUMENT_PREFIX,
    SEARCH_KEEP_ALIVE,
    Classifier,
    ClassifierUnavailable,
)


@dataclass(frozen=True)
class IndexReport:
    documents: int  # documents indexed by this run
    chunks: int  # chunks stored by this run
    empty: int  # documents whose usable pages gave no chunk
    unavailable: str | None = None  # Ollama hint when the run stopped early


def index(
    catalog: Catalog,
    embedder: Classifier,
    config: ChunkConfig = ChunkConfig(),
    under: Path | None = None,
    redo: bool = False,
    on_document: Callable[[int], None] | None = None,
) -> IndexReport:
    """Chunk and embed the documents that need it (all of them if ``redo``)."""
    model = embedder.embed_model
    todo: list[tuple[int, list[Chunk]]] = []
    texts: list[str] = []
    empty = 0
    for doc_id in catalog.documents_to_index(config.version, model, under, redo):
        chunks = chunk_pages(catalog.masked_pages(doc_id), config)
        if not chunks:
            empty += 1
            continue
        name = catalog.file_name(doc_id)
        todo.append((doc_id, chunks))
        texts.extend(embed_text(name, c) for c in chunks)

    vectors: list[list[float]] = []
    stored_docs = stored_chunks = 0
    offset = 0  # position in ``vectors`` of the next document to store
    try:
        for start in range(0, len(texts), BATCH):
            vectors.extend(
                embedder.embed(
                    texts[start : start + BATCH], prefix=DOCUMENT_PREFIX, keep_alive=SEARCH_KEEP_ALIVE
                )
            )
            while stored_docs < len(todo) and offset + len(todo[stored_docs][1]) <= len(vectors):
                doc_id, chunks = todo[stored_docs]
                catalog.store_chunks(
                    doc_id, config.version, model, chunks, vectors[offset : offset + len(chunks)]
                )
                offset += len(chunks)
                stored_docs += 1
                stored_chunks += len(chunks)
                if on_document:
                    on_document(doc_id)
    except ClassifierUnavailable as exc:
        hint = exc.with_command("localmind index").hint
        return IndexReport(stored_docs, stored_chunks, empty, hint)
    except (ValueError, http.client.HTTPException) as exc:  # wrong count, truncated answer
        reason = f"Ollama gave an unexpected answer ({exc})"
        return IndexReport(stored_docs, stored_chunks, empty, f"{reason}, then: localmind index")
    return IndexReport(stored_docs, stored_chunks, empty)
