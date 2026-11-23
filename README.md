# localmind

**A fully local RAG over your own documents, built step by step to learn how RAG really works.**

Drop any pile of TXT / Markdown / PDF files and photos of documents in. localmind reads them, removes duplicates,
sorts them into folders on its own, and lets you ask questions about them. Everything runs
on your machine: no cloud, no API key, your files never leave your disk.

> 🎓 This is a learning project. Each stage is built from scratch (no LangChain magic),
> tested, and explained in [`docs/`](docs/).

## Roadmap

| # | Stage | What you learn | Status |
|---|-------|----------------|--------|
| 1 | [Ingestion](docs/01-ingestion.md) | Text extraction, encodings, PDF internals, content hashing, deduplication | ✅ |
| 2a | [Filtering & quarantine](docs/02-quality-filtering.md) | Page-level quality rules, thresholds, checksums (Luhn, mod 97), user decisions | ✅ |
| 2b | [OCR](docs/02b-ocr.md) | Neural OCR (docTR, PyTorch, GPU) for scans and photos, confidence, caching | ✅ |
| 3 | [Classification](docs/03-classification.md) | Embeddings, cosine similarity, a local LLM as second judge, virtual folders | ✅ |
| 4 | [Chunking & vector search](docs/04-chunking-vectors.md) | Chunks, overlap, asymmetric prefixes, a hand-made vector store (SQLite + numpy) | ✅ |
| 5 | [Hybrid search](docs/05-hybrid-search.md) | BM25 with SQLite FTS5, reciprocal rank fusion, folder filter, keyword fallback | ✅ |
| 6 | Generation | Grounded answers with citations (Ollama) | ⏳ |
| 7 | Evaluation | Measuring whether answers are right | ⏳ |

## Quick start

```bash
uv sync
uv run localmind ingest ~/Documents/inbox   # read + deduplicate
uv run localmind stats
uv run localmind duplicates
uv run localmind quarantine                 # what was set aside, and why
uv run localmind quarantine accept FILE     # keep it anyway
```

The catalog lives in `~/.localmind/catalog.db` (override with `--db` or `LOCALMIND_DB`).
**Your original files are never moved, modified or deleted**: localmind only stores pointers to them.

### Reading scans and photos (optional OCR)

```bash
uv sync --extra ocr                         # docTR + PyTorch with CUDA (~2 GB download)
uv cache prune                              # optional: reclaim the download cache
uv run localmind ingest ~/Documents/inbox   # asks "Run OCR now? [y/N]" if scans were found
uv run localmind quarantine retry           # or read waiting scans later
```

OCR runs on your machine (NVIDIA GPU if available, otherwise CPU, about 10× slower).
The only network access is docTR downloading its model weights once.
No NVIDIA GPU? Delete the `[tool.uv.sources]` and `[[tool.uv.index]]` sections of
`pyproject.toml` before `uv sync --extra ocr` to get the smaller CPU build of PyTorch.

### Sorting into folders (local LLM)

```bash
ollama pull qwen3:8b && ollama pull nomic-embed-text
uv run localmind ingest ~/Documents/inbox   # classifies new documents into Domain/Type folders
uv run localmind folders                    # the tree: Maison / Factures (12) ...
uv run localmind folders review             # confirm proposed folders, answer hesitations
uv run localmind move FILE Maison/Contrats  # your choice, remembered for good
```

Folders are virtual (rows in the catalog): nothing is moved. Models run in
[Ollama](https://ollama.com) on `localhost`; sensitive values are masked before anything
reaches them. Without Ollama, `ingest` still works (`--no-classify` skips the step).

### Searching your documents

```bash
uv run localmind ingest ~/Documents/inbox                 # also cuts documents into chunks and embeds them
uv run localmind search "quel est le préavis de mon bail ?"
# 0.71  bail_appartement.txt p.1  Maison/Contrats  "...le délai de préavis est d'un mois..."
uv run localmind search "facture 2026-04-5120" --folder Maison   # only one domain (or Maison/Factures)
uv run localmind prune                                    # forget files you deleted from disk
```

Search combines meaning (vectors) and exact words (BM25), with file, page and folder
(`(to confirm)` until you review the LLM's choice); it never cites a deleted file and
never shows a masked value. Without Ollama it falls back to keywords. Answers written by
a local LLM come in stage 6.

## Use it as a library

```python
from pathlib import Path
from localmind import Catalog, default_db_path, ingest

with Catalog(default_db_path()) as catalog:
    report = ingest(Path("~/Documents/inbox").expanduser(), catalog)
    print(report.counts())
```

## Development

```bash
uv run pytest --cov=localmind
```

## License

[MIT](LICENSE)
