"""Stage 4: cut a document's pages into chunks (pure, no I/O).

A **chunk** is a passage of a few hundred characters: small enough that its
vector is about *one* thing (a debit day, a notice period), big enough to be
read on its own. Rules:

* a chunk never spans two pages, so every chunk can be cited "file p. N";
* cuts fall on the most natural boundary that fits: a blank line (paragraph),
  then a line break (OCR text and tables rarely have blank lines), then the end
  of a sentence, then a space; never inside a word;
* consecutive chunks of a page share ``overlap_chars`` characters, so a sentence
  cut at the edge of one chunk is still whole in the next;
* a tail shorter than ``min_chars`` joins the previous chunk of its page.

The text given here is already masked (``[IBAN]``...): chunks never hold a
sensitive value.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Natural boundaries, best first. A cut is made at the end of a match, so the
# separator stays with the piece before it.
_BOUNDARIES = (
    re.compile(r"\n[ \t]*\n\s*"),  # blank line: a paragraph
    re.compile(r"\n"),  # line break
    re.compile(r"(?<=[.!?…])\s+"),  # end of a sentence
    re.compile(r"\s+"),  # any space: between two words
)


@dataclass(frozen=True)
class ChunkConfig:
    chunk_chars: int = 600  # stage 5 spike: 10/10 right chunks in the top 3 (800 gave 9/10)
    overlap_chars: int = 120  # ~15 %
    min_chars: int = 200

    @property
    def version(self) -> str:
        """Stored with every chunk: other parameters mean the chunks must be redone."""
        return f"para-v1:{self.chunk_chars}/{self.overlap_chars}/{self.min_chars}"


@dataclass(frozen=True)
class Chunk:
    page_no: int
    seq: int  # position in the whole document, from 0
    text: str


def _units(text: str, start: int, end: int, size: int, level: int = 0) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into contiguous spans of at most ``size``
    characters, using the best boundary that cuts it."""
    if end - start <= size:
        return [(start, end)]
    if level == len(_BOUNDARIES):  # one "word" longer than a chunk: no choice
        return [(start, start + size)] + _units(text, start + size, end, size, level)
    cuts = [m.end() for m in _BOUNDARIES[level].finditer(text, start, end) if start < m.end() < end]
    if not cuts:
        return _units(text, start, end, size, level + 1)
    spans, prev = [], start
    for cut in [*cuts, end]:
        spans.extend(_units(text, prev, cut, size, level + 1))
        prev = cut
    return spans


def _word_start(text: str, pos: int, limit: int) -> int:
    """First word start at or after ``pos`` (never past ``limit``)."""
    if pos > 0 and not text[pos - 1].isspace():
        while pos < limit and not text[pos].isspace():
            pos += 1
    while pos < limit and text[pos].isspace():
        pos += 1
    return pos


def _page_spans(text: str, config: ChunkConfig) -> list[tuple[int, int]]:
    """(start, end) of each chunk of one page, overlap included, before stripping."""
    units = [u for u in _units(text, 0, len(text), config.chunk_chars) if text[u[0] : u[1]].strip()]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        start = units[i][0]
        j = i + 1
        while j < len(units) and units[j][1] - start <= config.chunk_chars:
            j += 1
        end = units[j - 1][1]
        if spans and len(text[start:end].strip()) < config.min_chars:
            spans[-1] = (spans[-1][0], end)  # a small tail joins the previous chunk
        elif spans and config.overlap_chars:
            back = max(spans[-1][1] - config.overlap_chars, spans[-1][0])
            spans.append((_word_start(text, back, start), end))
        else:
            spans.append((start, end))
        i = j
    return spans


def chunk_pages(
    pages: Sequence[tuple[int, str]], config: ChunkConfig = ChunkConfig()
) -> list[Chunk]:
    """Cut ``(page_no, text)`` pages into chunks numbered across the document."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for start, end in _page_spans(text, config):
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(page_no, len(chunks), piece))
    return chunks


def embed_text(file_name: str, chunk: Chunk) -> str:
    """What gets embedded (before the model's task prefix): a header naming the
    file and page, so a chunk like "Total : 84,20 €" still says where it is from."""
    return f"{file_name} p. {chunk.page_no}\n{chunk.text}"
