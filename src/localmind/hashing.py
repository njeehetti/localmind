"""Fingerprints used to identify files and detect duplicates.

Two different hashes, for two different questions:

* ``file_sha256``  -> "are these two files byte-for-byte identical?"
  (exact copies: ``invoice.pdf`` vs ``invoice (1).pdf``)
* ``content_hash`` -> "do these two files hold the same text?"
  (same note saved as UTF-8 and as Windows-1252, with different line endings...)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # read 1 MiB at a time: constant memory, any file size
_WHITESPACE = re.compile(r"\s+")


def file_sha256(path: Path) -> str:
    """Hash the raw bytes of a file, streaming it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Remove differences that change the bytes but not the information.

    * Unicode NFC: "é" can be stored as 1 code point or as "e" + accent (2).
    * Whitespace: CRLF vs LF, tabs, repeated spaces and blank lines.

    Case is kept on purpose: "Apple" (company) and "apple" (fruit) differ.
    """
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Hash the normalized text: equal for files that say the same thing."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
