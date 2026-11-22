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
from localmind.ocr import OcrPage
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
    # Stage 2b, OCR:
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"  # A: the engine is not sure enough
    OCR_NO_TEXT = "ocr_no_text"  # blank page (empty back of a duplex scan...)
    OCR_FAILED = "ocr_failed"  # A: the engine crashed on this page


SensitivePolicy = Literal["flag", "quarantine"]


@dataclass(frozen=True)
class QualityConfig:
    # A: a PDF page with fewer visible characters has no real text layer
    # (a scan, maybe with a page number): it needs OCR.
    min_page_chars: int = 20
    # A: a PDF page that draws an image and has fewer visible characters than
    # this is a scan with a thin text layer (watermark, header): it needs OCR.
    scan_page_max_chars: int = 200
    # B: fewer words than this in the whole document is not worth indexing.
    min_doc_words: int = 3
    # B: below this share of "normal" characters, a page is garbage.
    min_normal_char_ratio: float = 0.70
    # B: a document with at least this many lines, most of them log lines.
    log_min_lines: int = 20
    log_line_ratio: float = 0.60
    # A: an OCR-ed page whose (length-weighted) confidence is below this is
    # not trusted. Starting value, to measure at stage 7 (evaluation).
    min_ocr_confidence: float = 0.70
    # Sensitive data: "flag" keeps the document and records where it is.
    sensitive_policy: SensitivePolicy = "flag"


@dataclass(frozen=True)
class PageVerdict:
    page_no: int  # 1-based
    text: str
    reason: Reason | None  # None = usable
    spans: tuple[SensitiveSpan, ...] = ()
    source: str = "text_layer"  # or "ocr"
    ocr_engine: str | None = None
    ocr_confidence: float | None = None

    @property
    def is_blank(self) -> bool:
        """A blank page is stored but is not a problem to fix."""
        return self.reason is Reason.OCR_NO_TEXT


@dataclass(frozen=True)
class Assessment:
    pages: tuple[PageVerdict, ...]
    quarantine_reason: Reason | None

    @property
    def usable_text(self) -> str:
        return "\f".join(p.text for p in self.pages if p.reason is None)

    @property
    def has_set_aside(self) -> bool:
        return any(p.reason is not None and not p.is_blank for p in self.pages)


# --- pure rules -------------------------------------------------------------

# Symbols that belong in real documents: currency (€ = Sc), math (+ = Sm),
# modifiers (^ = Sk), other symbols (° = So). "%" is punctuation (Po).
_NORMAL_SYMBOLS = {"Sc", "Sm", "Sk", "So"}
_LOG_LINE = re.compile(
    # A date alone is not a log line (diaries, expense lists start with dates):
    # a log line starts with date + time, a time, or a level keyword.
    r"^\s*\[?\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{2}:\d{2}:\d{2}|"
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


def _lacks_text_layer(text: str, is_image_page: bool, kind: str, config: QualityConfig) -> bool:
    """Level A: this page's text layer cannot be trusted, OCR must read it."""
    if kind not in ("pdf", "image"):
        return False
    visible = _visible_chars(text)
    if visible < config.min_page_chars:
        return True  # no text layer (a scan, maybe with a page number)
    return is_image_page and visible < config.scan_page_max_chars  # thin layer on a scan


def needs_ocr_pages(extracted: ExtractedText, config: QualityConfig) -> frozenset[int]:
    """The pages OCR should read (1-based)."""
    return frozenset(
        page_no
        for page_no, text in enumerate(extracted.pages, start=1)
        if _lacks_text_layer(text, page_no in extracted.image_pages, extracted.kind, config)
    )


def _ocr_reason(ocr: OcrPage, config: QualityConfig) -> Reason | None:
    if ocr.confidence < 0:
        return Reason.OCR_FAILED
    if _visible_chars(ocr.text) < config.min_page_chars:
        return Reason.OCR_NO_TEXT  # blank: nothing, or a lone page number / specks
    if ocr.confidence < config.min_ocr_confidence:
        return Reason.OCR_LOW_CONFIDENCE
    return None


def _judge_page(
    page_no: int, text: str, extracted: ExtractedText, config: QualityConfig, accepted: bool
) -> PageVerdict:
    is_image_page = page_no in extracted.image_pages
    lacks_text = _lacks_text_layer(text, is_image_page, extracted.kind, config)
    ocr = extracted.ocr.get(page_no)
    if ocr is not None and lacks_text:
        # The OCR text replaces the (missing or thin) text layer: a watermark
        # such as "Scanned with CamScanner" is not content.
        reason = _ocr_reason(ocr, config)
        if reason is None and not accepted and is_garbled(ocr.text, config):
            reason = Reason.GARBLED
        spans = find_sensitive(ocr.text) if reason is None else ()
        return PageVerdict(page_no, ocr.text, reason, spans, "ocr", ocr.engine, ocr.confidence)

    if lacks_text:
        return PageVerdict(page_no, text, Reason.NEEDS_OCR)  # level A
    reason = Reason.GARBLED if not accepted and is_garbled(text, config) else None  # level B
    spans = find_sensitive(text) if reason is None else ()
    return PageVerdict(page_no, text, reason, spans)


def assess(extracted: ExtractedText, config: QualityConfig, *, accepted: bool = False) -> Assessment:
    """Judge each page, then the document as a whole."""
    if extracted.kind == "text" and not extracted.text.strip():
        page = PageVerdict(1, extracted.text, Reason.EMPTY)
        return Assessment(pages=(page,), quarantine_reason=Reason.EMPTY)

    verdicts = tuple(
        _judge_page(page_no, text, extracted, config, accepted)
        for page_no, text in enumerate(extracted.pages, start=1)
    )
    assessment = Assessment(pages=verdicts, quarantine_reason=None)

    if all(p.reason is not None for p in verdicts):
        # Nothing usable. Name the real problem: blank pages only explain it
        # when every page is blank. Counter keeps first-seen order on ties.
        problems = [p.reason for p in verdicts if not p.is_blank] or [Reason.OCR_NO_TEXT]
        most_common = Counter(problems).most_common(1)[0][0]
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
