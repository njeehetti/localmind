"""A scripted stand-in for Ollama, shared by the stage 3 tests.

Vectors: a text containing a known keyword gets that keyword's axis (so all
invoices are identical, cosine 1); any other text gets its own new axis
(cosine 0 with everything else). Answers: a callable receiving the parsed
user payload of the LLM messages, or a list consumed in order (an exception
instance in the list is raised).
"""

from __future__ import annotations

import json

from localmind.ollama import ClassifierUnavailable

DIM = 64
KEYWORDS = ("facture", "paie", "remboursement", "contrat", "ordonnance", "impot")


class FakeClassifier:
    name = "fake-llm+fake-embed"
    embed_model = "fake-embed"

    def __init__(self, answers=None, fail_embed_on=None, unavailable=False):
        self.answers = answers if answers is not None else []
        self.fail_embed_on = fail_embed_on  # substring that makes embed() unavailable
        self.unavailable = unavailable
        self.embedded: list[str] = []
        self.chats: list[list[dict]] = []
        self.releases = 0
        self._axes: dict[str, int] = {}

    def _unavailable(self):
        return ClassifierUnavailable(
            "Ollama not reachable at http://fake: start it (ollama serve), then: localmind classify"
        )

    def _vector(self, text: str) -> list[float]:
        low = text.casefold()
        axis = next((i for i, k in enumerate(KEYWORDS) if k in low), None)
        if axis is None:
            axis = self._axes.setdefault(text, len(KEYWORDS) + len(self._axes))
        vec = [0.0] * DIM
        vec[axis] = 1.0
        return vec

    def embed(self, texts):
        if self.unavailable or (self.fail_embed_on and any(self.fail_embed_on in t for t in texts)):
            raise self._unavailable()
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def choose(self, messages, schema):
        if self.unavailable:
            raise self._unavailable()
        self.chats.append(messages)
        payload = json.loads(messages[-1]["content"])
        answer = self.answers(payload) if callable(self.answers) else self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def release(self):
        self.releases += 1


def answer(folder, new_folder=False, alternative=None, reason="test"):
    return {"folder": folder, "new_folder": new_folder, "alternative": alternative, "reason": reason}
