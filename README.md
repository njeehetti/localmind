# localmind

**A fully local RAG over your own documents, built step by step to learn how RAG really works.**

Drop any pile of TXT / Markdown / PDF files in. localmind reads them, removes duplicates,
sorts them into folders on its own, and lets you ask questions about them. Everything runs
on your machine: no cloud, no API key, your files never leave your disk.

> 🎓 This is a learning project. Each stage is built from scratch (no LangChain magic),
> tested, and explained in [`docs/`](docs/).

## Roadmap

| # | Stage | What you learn | Status |
|---|-------|----------------|--------|
| 1 | [Ingestion](docs/01-ingestion.md) | Text extraction, encodings, PDF internals, content hashing, deduplication | ✅ |
| 2 | Filtering & quarantine | Detecting empty, scanned or unreadable files | ⏳ |
| 3 | Classification | Zero-shot sorting into folders with a local LLM | ⏳ |
| 4 | Chunking & embeddings | Turning text into vectors | ⏳ |
| 5 | Retrieval | Vector search, BM25, hybrid search | ⏳ |
| 6 | Generation | Grounded answers with citations (Ollama) | ⏳ |
| 7 | Evaluation | Measuring whether answers are right | ⏳ |

## Quick start

```bash
uv sync
uv run localmind ingest ~/Documents/inbox   # read + deduplicate
uv run localmind stats
uv run localmind duplicates
```

The catalog lives in `~/.localmind/catalog.db` (override with `--db` or `LOCALMIND_DB`).
**Your original files are never moved, modified or deleted**: localmind only stores pointers to them.

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
