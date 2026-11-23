"""Virtual folder names: ``Domaine/Type`` (area of life / document type).

A folder is a row in the catalog, never a directory on disk. Names come from
the user or from an LLM, so they are always normalised and checked here: an
LLM answer like "Maison/Factures/EDF" or "Fac<tures>" never becomes a folder.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_PART = 40
_ALLOWED = re.compile(r"^[\w '&-]+$")  # \w: letters (accents included), digits, "_"


class InvalidFolderName(ValueError):
    """Not a valid ``Domaine/Type`` name."""


class UnknownFolder(LookupError):
    """A folder filter that names no folder of the catalog."""

    def __init__(self, text: str) -> None:
        super().__init__(f"no folder named '{text}'")


@dataclass(frozen=True, order=True)
class FolderName:
    domain: str
    type: str

    def __str__(self) -> str:
        return f"{self.domain}/{self.type}"


def _clean_part(part: str, text: str) -> str:
    part = unicodedata.normalize("NFC", " ".join(part.split()))
    if not part or len(part) > MAX_PART or "_" in part or not _ALLOWED.match(part):
        raise InvalidFolderName(f"invalid folder name: {text!r} (expected Domaine/Type)")
    return part[0].upper() + part[1:]


def parse_folder(text: str) -> FolderName:
    """Normalise ``"  maison / factures "`` into ``Maison/Factures``."""
    parts = text.split("/")
    if len(parts) != 2:
        raise InvalidFolderName(f"invalid folder name: {text!r} (expected Domaine/Type)")
    return FolderName(_clean_part(parts[0], text), _clean_part(parts[1], text))


@dataclass(frozen=True)
class FolderFilter:
    """A whole domain (``Maison``, type None) or one folder (``Maison/Factures``)."""

    domain: str
    type: str | None = None

    def __str__(self) -> str:
        return self.domain if self.type is None else f"{self.domain}/{self.type}"


def parse_folder_filter(text: str) -> FolderFilter:
    """``" maison "`` -> ``Maison``; ``"maison / factures"`` -> ``Maison/Factures``."""
    parts = text.split("/")
    if len(parts) == 1:
        return FolderFilter(_clean_part(parts[0], text))
    folder = parse_folder(text)
    return FolderFilter(folder.domain, folder.type)


def fold(text: str) -> str:
    """Compare names ignoring case and accents: "Santé" == "SANTE"."""
    decomposed = unicodedata.normalize("NFD", " ".join(text.split()).casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))
