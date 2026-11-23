"""The Ollama client, tested against a stub HTTP server (no real model)."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localmind.ollama import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    SEARCH_KEEP_ALIVE,
    ClassifierUnavailable,
    OllamaClassifier,
)


class Stub:
    def __init__(self):
        self.replies = {"/api/version": (200, {"version": "0.40.16"})}
        self.requests: list[tuple[str, dict]] = []

    def reply(self, path, body, status=200):
        self.replies[path] = (status, body)

    def last(self, path):
        return [body for p, body in self.requests if p == path][-1]

    def calls(self, path):
        return [body for p, body in self.requests if p == path]


@pytest.fixture
def stub():
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append((self.path, body))
            status, payload = state.replies.get(self.path, (404, {"error": "no route"}))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _answer

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def test_embed_sends_prefix_and_normalises(stub):
    stub.reply("/api/embed", {"embeddings": [[3.0, 4.0]]})
    assert OllamaClassifier(url=stub.url).embed(["x"]) == [pytest.approx([0.6, 0.8])]
    body = stub.last("/api/embed")
    assert body["input"] == ["classification: x"] and body["model"] == "nomic-embed-text"


def test_embed_sends_the_given_prefix_and_keep_alive(stub):
    stub.reply("/api/embed", {"embeddings": [[1.0, 0.0]]})
    OllamaClassifier(url=stub.url).embed(["Total 84 €"], prefix=DOCUMENT_PREFIX, keep_alive=SEARCH_KEEP_ALIVE)
    body = stub.last("/api/embed")
    assert body["input"] == ["search_document: Total 84 €"] and body["keep_alive"] == "30m"
    OllamaClassifier(url=stub.url).embed(["quel jour ?"], prefix=QUERY_PREFIX)
    assert stub.last("/api/embed")["input"] == ["search_query: quel jour ?"]


def test_embed_defaults_to_classification_prefix_without_keep_alive(stub):
    stub.reply("/api/embed", {"embeddings": [[1.0, 0.0]]})
    OllamaClassifier(url=stub.url).embed(["x"])
    body = stub.last("/api/embed")
    assert body["input"] == ["classification: x"] and "keep_alive" not in body


def test_unavailable_hint_names_the_command():
    e = ClassifierUnavailable("Ollama not reachable at http://h: start it (ollama serve)")
    assert e.reason == "Ollama not reachable at http://h: start it (ollama serve)"
    assert e.hint.endswith("then: localmind classify")
    assert e.with_command("localmind index").hint == (
        "Ollama not reachable at http://h: start it (ollama serve), then: localmind index"
    )
    assert str(e.with_command("localmind index")) == e.with_command("localmind index").hint


def test_embed_batches_by_32(stub):
    stub.reply("/api/embed", {"embeddings": [[1.0, 0.0]] * 32})
    client = OllamaClassifier(url=stub.url)
    client.embed(["t"] * 32)
    assert len(stub.calls("/api/embed")) == 1
    stub.reply("/api/embed", {"embeddings": [[1.0, 0.0]]})
    with pytest.raises(ValueError):  # 33 texts → 2 batches; the stub returns 1 vector for 1 → mismatch
        client.embed(["t"] * 33)
    assert len(stub.calls("/api/embed")) == 2


def test_choose_sends_schema_and_options(stub):
    stub.reply("/api/chat", {"message": {"content": '{"folder": "Maison/Factures"}'}})
    client = OllamaClassifier(url=stub.url)
    out = client.choose([{"role": "user", "content": "x"}], {"type": "object"})
    body = stub.last("/api/chat")
    assert out == {"folder": "Maison/Factures"}
    assert body["model"] == "qwen3:8b"
    assert body["think"] is False and body["stream"] is False
    assert body["format"] == {"type": "object"}
    assert body["options"] == {"temperature": 0, "num_ctx": 4096}
    assert body["messages"] == [{"role": "user", "content": "x"}]


@pytest.mark.parametrize("content", ["Maison", "[1, 2]", ""])
def test_choose_rejects_non_json_objects(stub, content):
    stub.reply("/api/chat", {"message": {"content": content}})
    with pytest.raises(ValueError):
        OllamaClassifier(url=stub.url).choose([], {})


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_unreachable_server_raises_with_hint():
    url = f"http://127.0.0.1:{_closed_port()}"
    with pytest.raises(ClassifierUnavailable) as err:
        OllamaClassifier(url=url).embed(["x"])
    assert err.value.hint == (
        f"Ollama not reachable at {url}: start it (ollama serve), then: localmind classify"
    )


def test_missing_model_raises_pull_hint(stub):
    stub.reply("/api/chat", {"error": 'model "qwen3:8b" not found, try pulling it first'}, 404)
    with pytest.raises(ClassifierUnavailable) as err:
        OllamaClassifier(url=stub.url).choose([], {})
    assert err.value.hint == "Model qwen3:8b is missing: ollama pull qwen3:8b, then: localmind classify"


def test_version_is_checked_once(stub):
    stub.reply("/api/embed", {"embeddings": [[1.0]]})
    client = OllamaClassifier(url=stub.url)
    client.embed(["a"])
    client.embed(["b"])
    assert len(stub.calls("/api/version")) == 1


def test_release_unloads_llm(stub):
    stub.reply("/api/generate", {"done": True})
    OllamaClassifier(url=stub.url).release()
    assert stub.last("/api/generate") == {"model": "qwen3:8b", "keep_alive": 0}


def test_release_never_raises_when_ollama_is_gone():
    OllamaClassifier(url=f"http://127.0.0.1:{_closed_port()}").release()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LOCALMIND_LLM", "mistral:7b")
    monkeypatch.setenv("LOCALMIND_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("LOCALMIND_OLLAMA_URL", "http://host:1234/")
    client = OllamaClassifier()
    assert (client.llm, client.embed_model, client.url) == ("mistral:7b", "bge-m3", "http://host:1234")
    assert client.name == "mistral:7b+bge-m3"
