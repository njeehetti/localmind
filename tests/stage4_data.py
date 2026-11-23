"""Scratch documents for the stage 4 tests (chunks, search, prune)."""

from __future__ import annotations

from pathlib import Path

from localmind.catalog import Catalog
from localmind.ingest import ingest

IBAN = "FR76 3000 6000 0112 3456 7890 189"


def long_text(word: str, paragraphs: int) -> str:
    """Several chunks' worth of text, every paragraph mentioning ``word``."""
    return "\n\n".join(
        f"Paragraphe {i} au sujet du {word}. " + f"Ceci est une phrase de remplissage numero {i}. " * 8
        for i in range(paragraphs)
    )


def add_files(catalog: Catalog, folder: Path, files: dict[str, str]) -> dict[str, int | None]:
    """Write the files, ingest them, return name -> document id."""
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    ingest(folder, catalog)
    docs = {name: catalog.document_for_path((folder / name).resolve()) for name in files}
    return {name: doc.id if doc else None for name, doc in docs.items()}  # None: quarantined


def count(catalog: Catalog, sql: str, *args) -> int:
    return catalog._conn.execute(sql, args).fetchone()[0]
