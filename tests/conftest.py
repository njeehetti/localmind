"""Shared test setup."""

import pytest

from localmind import cli_folders
from tests.fake_classifier import FakeClassifier


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch):
    """The CLI classifies after ingest: tests never reach a real model.
    Tests that need answers patch ``make_classifier`` again."""
    monkeypatch.setattr(cli_folders, "make_classifier", lambda: FakeClassifier(unavailable=True))
