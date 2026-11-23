"""Shared test setup."""

import pytest

from localmind import cli_folders, cli_search
from tests.fake_classifier import FakeClassifier


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch):
    """The CLI classifies and indexes after ingest: tests never reach a real
    model. Tests that need answers patch ``make_classifier`` /
    ``make_embedder`` again."""
    monkeypatch.setattr(cli_folders, "make_classifier", lambda: FakeClassifier(unavailable=True))
    monkeypatch.setattr(cli_search, "make_embedder", lambda: FakeClassifier(unavailable=True))
