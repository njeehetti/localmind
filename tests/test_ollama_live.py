"""Opt-in: the real local Ollama. Skipped when it is not running."""

import urllib.request

import pytest

from localmind.ollama import OllamaClassifier

try:
    urllib.request.urlopen("http://localhost:11434/api/version", timeout=1).read()
except OSError:
    pytest.skip("Ollama is not running", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    c = OllamaClassifier(url="http://localhost:11434")
    yield c
    c.release()


def test_real_embeddings_are_normalised(client):
    a, b = client.embed(["Facture EDF de mars", "Bulletin de paie de mars"])
    assert len(a) == len(b) > 100
    assert sum(x * x for x in a) == pytest.approx(1.0, abs=1e-4)


def test_real_llm_answers_the_schema(client):
    schema = {"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]}
    out = client.choose(
        [{"role": "user", "content": "Range ce document dans un dossier Domaine/Type. "
          "Document : bulletin de paie de mars, net a payer 1650 euros."}],
        schema,
    )
    assert "/" in out["folder"]


def test_query_and_document_prefixes_bring_the_answer_closer(client):
    from localmind.ollama import DOCUMENT_PREFIX, QUERY_PREFIX, SEARCH_KEEP_ALIVE

    (q,) = client.embed(["quel jour suis-je prélevé ?"], prefix=QUERY_PREFIX, keep_alive=SEARCH_KEEP_ALIVE)
    debit, lease = client.embed(
        [
            "Votre abonnement est réglé par prélèvement le 5 de chaque mois.",
            "Le bail est consenti pour trois ans, loyer mensuel 850 euros.",
        ],
        prefix=DOCUMENT_PREFIX,
        keep_alive=SEARCH_KEEP_ALIVE,
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(q, debit) > dot(q, lease)
