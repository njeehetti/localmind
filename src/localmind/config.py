"""Where localmind keeps its data, and settings read from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import get_args

from localmind.quality import SensitivePolicy

ENV_DB = "LOCALMIND_DB"
ENV_SENSITIVE_POLICY = "LOCALMIND_SENSITIVE_POLICY"


def default_db_path() -> Path:
    """``$LOCALMIND_DB`` if set, otherwise ``~/.localmind/catalog.db``."""
    override = os.environ.get(ENV_DB)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".localmind" / "catalog.db"


def sensitive_policy_from_env() -> SensitivePolicy:
    """``$LOCALMIND_SENSITIVE_POLICY`` ("flag" or "quarantine"), default "flag"."""
    value = os.environ.get(ENV_SENSITIVE_POLICY, "flag").strip().lower()
    allowed = get_args(SensitivePolicy)
    if value not in allowed:
        raise ValueError(f"{ENV_SENSITIVE_POLICY} must be one of {allowed}, got {value!r}")
    return value  # type: ignore[return-value]
