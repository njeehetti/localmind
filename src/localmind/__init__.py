"""localmind: a fully local RAG over your own documents.

Public API (the CLI, the future web UI and external tools such as a voice
assistant only use these names):

    >>> from pathlib import Path
    >>> from localmind import Catalog, default_db_path, ingest
    >>> with Catalog(default_db_path()) as catalog:
    ...     report = ingest(Path("~/Documents/inbox").expanduser(), catalog)
"""

from localmind.catalog import Catalog
from localmind.config import default_db_path
from localmind.ingest import IngestReport, Outcome, ingest

__all__ = ["Catalog", "IngestReport", "Outcome", "default_db_path", "ingest"]
__version__ = "0.1.0"
