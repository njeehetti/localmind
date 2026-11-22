"""Quality rules: decide, page by page, what is usable.

Two levels of rules (see docs/02-quality-filtering.md):

* **A, technical**: the page or file cannot be used at all (no text layer,
  empty file...). A user "accept" cannot override these: there is no text.
* **B, quality**: there is text, but it is worthless for search (too short,
  garbled, machine logs). A user "accept" overrides these.

Every rule is a small pure function and every threshold is a named field of
``QualityConfig``, so they can be read, tested and tuned one by one.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from localmind.extractors import ExtractedText
from localmind.sensitive import SensitiveSpan, find_sensitive


class Reason(StrEnum):
    UNSUPPORTED_FORMAT = "unsupported_format"
    NEEDS_OCR = "needs_ocr"
    CORRUPTED = "corrupted"
    ENCRYPTED = "encrypted"
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    GARBLED = "garbled"
    MACHINE_LOG = "machine_log"
    SENSITIVE = "sensitive"


SensitivePolicy = Literal["flag", "quarantine"]


@dataclass(frozen=True)
class QualityConfig:
    # A: a PDF page with fewer visible characters has no real text layer
    # (a scan, maybe with a page number): it needs OCR.
    min_page_chars: int = 20
    # B: fewer words than this in the whole document is not worth indexing.
    min_doc_words: int = 3
    # B: below this share of "normal" characters, a page is garbage.
    min_normal_char_ratio: float = 0.70
    # B: a document with at least this many lines, most of them log lines.
    log_min_lines: int = 20
    log_line_ratio: float = 0.60
    # Sensitive data: "flag" keeps the document and records where it is.
    sensitive_policy: SensitivePolicy = "flag"


@dataclass(frozen=True)
class PageVerdict:
    page_no: int  # 1-based
    text: str
    reason: Reason | None  # None = usable
    spans: tuple[SensitiveSpan, ...] = ()


@dataclass(frozen=True)
class Assessment:
    pages: tuple[PageVerdict, ...]
    quarantine_reason: Reason | None

    @property
    def usable_text(self) -> str:
        return "\f".join(p.text for p in self.pages if p.reason is None)

    @property
    def has_set_aside(self) -> bool:
        return any(p.reason is not None for p in self.pages)


# --- pure rules -------------------------------------------------------------

# Symbols that belong in real documents: currency (€ = Sc), math (+ = Sm),
# modifiers (^ = Sk), other symbols (° = So). "%" is punctuation (Po).
_NORMAL_SYMBOLS = {"Sc", "Sm", "Sk", "So"}
_LOG_LINE = re.compile(
    r"^\s*\[?\s*(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|"
    r"(?:DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|TRACE)\b)"
)


def _is_normal_char(ch: str) -> bool:
    if ch == "\ufffd":  # the "replacement character": a decoding already failed
        return False
    if ch in "\n\t\r\f":
        return True
    category = unicodedata.category(ch)
    return category[0] in "LNZP" or category in _NORMAL_SYMBOLS


def is_garbled(text: str, config: QualityConfig) -> bool:
    if not text:
        return False
    normal = sum(1 for ch in text if _is_normal_char(ch))
    return normal / len(text) < config.min_normal_char_ratio


def is_machine_log(text: str, config: QualityConfig) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < config.log_min_lines:
        return False
    log_lines = sum(1 for line in lines if _LOG_LINE.match(line))
    return log_lines / len(lines) >= config.log_line_ratio


def _visible_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


# --- assessment -------------------------------------------------------------


def _page_reason(text: str, kind: str, config: QualityConfig, accepted: bool) -> Reason | None:
    if kind == "pdf" and _visible_chars(text) < config.min_page_chars:
        return Reason.NEEDS_OCR  # level A
    if not accepted and is_garbled(text, config):
        return Reason.GARBLED  # level B
    return None


def assess(extracted: ExtractedText, config: QualityConfig, *, accepted: bool = False) -> Assessment:
    """Judge each page, then the document as a whole."""
    if extracted.kind == "text" and not extracted.text.strip():
        page = PageVerdict(1, extracted.text, Reason.EMPTY)
        return Assessment(pages=(page,), quarantine_reason=Reason.EMPTY)

    pages = []
    for page_no, text in enumerate(extracted.pages, start=1):
        reason = _page_reason(text, extracted.kind, config, accepted)
        spans = find_sensitive(text) if reason is None else ()
        pages.append(PageVerdict(page_no, text, reason, spans))
    verdicts = tuple(pages)
    assessment = Assessment(pages=verdicts, quarantine_reason=None)

    if all(p.reason is not None for p in verdicts):
        # Counter keeps first-seen order on ties: the first page's reason wins.
        most_common = Counter(p.reason for p in verdicts).most_common(1)[0][0]
        return Assessment(pages=verdicts, quarantine_reason=most_common)

    if not accepted:
        usable = assessment.usable_text
        if len(usable.split()) < config.min_doc_words:
            return Assessment(pages=verdicts, quarantine_reason=Reason.TOO_SHORT)
        if is_machine_log(usable, config):
            return Assessment(pages=verdicts, quarantine_reason=Reason.MACHINE_LOG)
        if config.sensitive_policy == "quarantine" and any(p.spans for p in verdicts):
            return Assessment(pages=verdicts, quarantine_reason=Reason.SENSITIVE)

    return assessment
