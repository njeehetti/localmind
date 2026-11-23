"""A tiny client for the local Ollama server (standard library only).

Two models, two jobs:

* an **embedding model** (``nomic-embed-text``) turns a text into a vector;
* an **LLM** (``qwen3:8b``) answers with JSON forced by a schema (Ollama's
  ``format``), so its answer is data we can check, not free text.

Everything stays on ``localhost``. When Ollama is not running, or a model is
not pulled, ``ClassifierUnavailable`` carries a one-line hint for the user.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from localmind.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM,
    DEFAULT_OLLAMA_URL,
    ENV_EMBED_MODEL,
    ENV_LLM,
    ENV_OLLAMA_URL,
)
from localmind.scoring import normalize

# nomic-embed-text is told what each vector is for by a task prefix.
CLASSIFY_PREFIX = "classification: "  # stage 3: sorting into folders
EMBED_PREFIX = CLASSIFY_PREFIX  # stage 3 name, kept
DOCUMENT_PREFIX = "search_document: "  # stage 4: a chunk, to be found
QUERY_PREFIX = "search_query: "  # stage 4: a question, looking for chunks
SEARCH_KEEP_ALIVE = "30m"  # keep the small embedding model loaded between questions
BATCH = 32
CHECK_TIMEOUT = 3  # seconds: is Ollama there at all?
CALL_TIMEOUT = 300  # seconds: a cold model load can take a while
LLM_OPTIONS = {"temperature": 0, "num_ctx": 4096}  # 4k context fits qwen3:8b in 8 GB of VRAM


class ClassifierUnavailable(RuntimeError):
    """Ollama is not reachable, or a model is missing. ``hint`` says what to do:
    ``reason``, then the command to run again once it is fixed."""

    def __init__(self, reason: str, command: str = "localmind classify") -> None:
        self.reason = reason
        self.command = command
        self.hint = f"{reason}, then: {command}"
        super().__init__(self.hint)

    def with_command(self, command: str) -> ClassifierUnavailable:
        """The same problem, pointing at another command (index, search...)."""
        return ClassifierUnavailable(self.reason, command)


class Classifier(Protocol):
    name: str
    embed_model: str

    def embed(
        self, texts: Sequence[str], prefix: str = CLASSIFY_PREFIX, keep_alive: str | None = None
    ) -> list[list[float]]: ...

    def choose(self, messages: list[dict], schema: dict) -> dict: ...

    def release(self) -> None: ...


class OllamaClassifier:
    def __init__(
        self, url: str | None = None, llm: str | None = None, embed_model: str | None = None
    ) -> None:
        self.url = (url or os.environ.get(ENV_OLLAMA_URL) or DEFAULT_OLLAMA_URL).rstrip("/")
        self.llm = llm or os.environ.get(ENV_LLM) or DEFAULT_LLM
        self.embed_model = embed_model or os.environ.get(ENV_EMBED_MODEL) or DEFAULT_EMBED_MODEL
        self.name = f"{self.llm}+{self.embed_model}"
        self._checked = False

    # -- HTTP ------------------------------------------------------------
    def _unreachable(self) -> ClassifierUnavailable:
        return ClassifierUnavailable(f"Ollama not reachable at {self.url}: start it (ollama serve)")

    def _request(self, path: str, body: dict | None, timeout: float) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path, data, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _post(self, path: str, body: dict, model: str) -> Any:
        if not self._checked:
            try:
                self._request("/api/version", None, CHECK_TIMEOUT)
            except (urllib.error.URLError, OSError) as exc:
                raise self._unreachable() from exc
            self._checked = True
        try:
            return self._request(path, body, CALL_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ClassifierUnavailable(f"Model {model} is missing: ollama pull {model}") from exc
            raise self._unreachable() from exc
        except (urllib.error.URLError, OSError) as exc:  # includes timeouts, refused
            raise self._unreachable() from exc

    # -- models ----------------------------------------------------------
    def embed(
        self, texts: Sequence[str], prefix: str = CLASSIFY_PREFIX, keep_alive: str | None = None
    ) -> list[list[float]]:
        """Normalised vectors of ``prefix + text``, in batches of 32. ``keep_alive``
        (e.g. "30m") tells Ollama how long to keep the model loaded afterwards."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = [prefix + t for t in texts[start : start + BATCH]]
            body: dict[str, Any] = {"model": self.embed_model, "input": batch}
            if keep_alive is not None:
                body["keep_alive"] = keep_alive
            answer = self._post("/api/embed", body, self.embed_model)
            got = answer.get("embeddings", [])
            if len(got) != len(batch):
                raise ValueError(f"expected {len(batch)} embeddings, got {len(got)}")
            vectors.extend(normalize(v) for v in got)
        return vectors

    def choose(self, messages: list[dict], schema: dict) -> dict:
        answer = self._post(
            "/api/chat",
            {
                "model": self.llm,
                "messages": messages,
                "format": schema,
                "stream": False,
                "think": False,
                "options": LLM_OPTIONS,
            },
            self.llm,
        )
        content = answer.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("the LLM answer is not JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("the LLM answer is not a JSON object")
        return parsed

    def release(self) -> None:
        """Unload the LLM now, so other models (JARVIS's) get the GPU back."""
        try:
            self._request("/api/generate", {"model": self.llm, "keep_alive": 0}, CHECK_TIMEOUT)
        except (urllib.error.URLError, OSError):
            pass  # already gone: nothing to free
