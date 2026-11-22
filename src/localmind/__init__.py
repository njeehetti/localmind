"""localmind: a fully local RAG over your own documents.

Public API (the CLI, the future web UI and external tools such as a voice
assistant only use these names):

    >>> from pathlib import Path
    >>> from localmind import Catalog, default_db_path, ingest
    >>> with Catalog(default_db_path()) as catalog:
    ...     report = ingest(Path("~/Documents/inbox").expanduser(), catalog)
"""

from localmind.catalog import Catalog, Decision, FileStatus
from localmind.classify import ClassifyReport, classify
from localmind.config import default_db_path, sensitive_policy_from_env
from localmind.folder_actions import choose_folder, confirm, move, rename_folder
from localmind.ingest import IngestReport, Outcome, apply_decision, ingest, retry, run_ocr
from localmind.ocr import OcrEngine, OcrPage, load_engine, ocr_available
from localmind.ollama import ClassifierUnavailable, OllamaClassifier
from localmind.quality import QualityConfig, Reason
from localmind.scoring import ClassifyConfig

__all__ = [
    "Catalog",
    "ClassifierUnavailable",
    "ClassifyConfig",
    "ClassifyReport",
    "Decision",
    "FileStatus",
    "IngestReport",
    "OcrEngine",
    "OcrPage",
    "Outcome",
    "QualityConfig",
    "Reason",
    "OllamaClassifier",
    "apply_decision",
    "choose_folder",
    "classify",
    "confirm",
    "default_db_path",
    "ingest",
    "load_engine",
    "move",
    "rename_folder",
    "ocr_available",
    "retry",
    "run_ocr",
    "sensitive_policy_from_env",
]
__version__ = "0.2.0"
