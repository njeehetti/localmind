"""localmind: a fully local RAG over your own documents.

Public API (the CLI, the future web UI and external tools such as a voice
assistant only use these names):

    >>> from pathlib import Path
    >>> from localmind import Catalog, default_db_path, ingest
    >>> with Catalog(default_db_path()) as catalog:
    ...     report = ingest(Path("~/Documents/inbox").expanduser(), catalog)
"""

from localmind.catalog import Catalog, Decision, FileStatus
from localmind.config import default_db_path, sensitive_policy_from_env
from localmind.ingest import IngestReport, Outcome, apply_decision, ingest, retry, run_ocr
from localmind.ocr import OcrEngine, OcrPage, load_engine, ocr_available
from localmind.quality import QualityConfig, Reason

__all__ = [
    "Catalog",
    "Decision",
    "FileStatus",
    "IngestReport",
    "OcrEngine",
    "OcrPage",
    "Outcome",
    "QualityConfig",
    "Reason",
    "apply_decision",
    "default_db_path",
    "ingest",
    "load_engine",
    "ocr_available",
    "retry",
    "run_ocr",
    "sensitive_policy_from_env",
]
__version__ = "0.2.0"
