"""Stage 5: keyword side of the hybrid search (pure, no I/O).

``keyword_query`` turns a question into an FTS5 query: its words, minus French
stop words, each in double quotes (user text is never read as FTS5 syntax),
joined with OR. BM25 then rewards passages holding more of the words, rarer
words more.

``fuse`` merges the meaning ranking and the keyword ranking with reciprocal
rank fusion: a chunk scores ``sum(1 / (constant + rank))`` over the rankings it
appears in. Only ranks count, so a cosine and a BM25 score never need to be
put on the same scale.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# Tuned in the stage 5 spike: grammar words and question words that every
# document or every question holds, so they would only add noise to BM25.
STOP_WORDS = frozenset(
    """le la les l de du des d un une et ou à a au aux en mon ma mes ton ta tes son sa ses
    est sont quel quelle quels quelles qui que quoi qu je j me m suis ai pour par sur dans
    ce cet cette ces combien quand comment""".split()
)

_WORD = re.compile(r"[^\W_]+")  # letters and digits, like the FTS5 unicode61 tokenizer


def keyword_query(question: str) -> str | None:
    """``'"préavis" OR "bail"'`` for "Quel est le préavis de mon bail ?";
    None when no word is left."""
    text = unicodedata.normalize("NFC", question).lower()  # "e" + accent -> "é", as indexed
    words = [w for w in _WORD.findall(text) if w not in STOP_WORDS]
    return " OR ".join(f'"{w}"' for w in words) or None


def fuse(vector_ids: Sequence[int], keyword_ids: Sequence[int], constant: int = 60) -> list[int]:
    """Chunk ids, best fused score first; ties: better vector rank, then smaller id."""
    scores: dict[int, float] = {}
    for ranking in (vector_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids)}
    return sorted(scores, key=lambda c: (-scores[c], vector_rank.get(c, len(vector_ids)), c))
