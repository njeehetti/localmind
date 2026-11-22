"""Vectors and the folder decision (pure functions, no model here).

An *embedding* turns a text into a list of numbers such that texts about the
same thing get close vectors. Normalised to length 1, "closeness" is simply
the dot product of two vectors: the *cosine similarity* (1 = same direction).

A document is compared with the documents already in each folder: the folder
score is the mean of its ``top_k`` best similarities. The decision is "clear"
only when the best folder is high, well ahead of the second, and not a
one-off: otherwise the LLM is asked.
"""

from __future__ import annotations

import math
from array import array
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ClassifyConfig:
    min_score: float = 0.80  # best folder score for a clear case
    min_margin: float = 0.05  # best - second
    min_folder_docs: int = 3  # a folder needs this many documents to decide alone
    top_k: int = 3  # folder score = mean of its k best similarities
    user_weight: int = 2  # a user-placed document counts this many times
    name_merge: float = 0.90  # a proposed name this close to an existing one reuses it
    embed_chars: int = 2000  # characters of text embedded
    llm_chars: int = 3000  # characters of text shown to the LLM
    candidates: int = 3  # candidate folders shown to the LLM
    examples: int = 5  # user corrections shown to the LLM


@dataclass(frozen=True)
class Example:
    folder_id: int
    vector: Sequence[float]
    weight: int = 1


def normalize(vec: Iterable[float]) -> list[float]:
    values = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


def to_bytes(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()  # float32: half the size, plenty of precision


def from_bytes(data: bytes) -> list[float]:
    values = array("f")
    values.frombytes(data)
    return values.tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two normalised vectors."""
    return sum(x * y for x, y in zip(a, b))


def folder_scores(
    vec: Sequence[float], examples: Iterable[Example], top_k: int = 3
) -> dict[int, tuple[float, int]]:
    """Folder id -> (mean of its top_k similarities, number of documents)."""
    sims: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for ex in examples:
        sim = cosine(vec, ex.vector)
        sims[ex.folder_id].extend([sim] * ex.weight)
        counts[ex.folder_id] += 1
    scores = {}
    for folder_id, values in sims.items():
        best = sorted(values, reverse=True)[:top_k]
        scores[folder_id] = (sum(best) / len(best), counts[folder_id])
    return scores


def decide(
    scores: dict[int, tuple[float, int]], config: ClassifyConfig
) -> tuple[Literal["clear", "ask_llm"], list[int]]:
    """Rank folders and say whether the embeddings alone are enough."""
    ranked = sorted(scores, key=lambda folder_id: scores[folder_id][0], reverse=True)
    if not ranked:
        return "ask_llm", []
    best, size = scores[ranked[0]]
    second = scores[ranked[1]][0] if len(ranked) > 1 else 0.0
    clear = (
        best >= config.min_score
        and best - second >= config.min_margin
        and size >= config.min_folder_docs
    )
    return ("clear" if clear else "ask_llm"), ranked
