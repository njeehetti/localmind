"""Where localmind keeps its data."""

from __future__ import annotations

import os
from pathlib import Path

ENV_DB = "LOCALMIND_DB"


def default_db_path() -> Path:
    """``$LOCALMIND_DB`` if set, otherwise ``~/.localmind/catalog.db``."""
    override = os.environ.get(ENV_DB)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".localmind" / "catalog.db"
