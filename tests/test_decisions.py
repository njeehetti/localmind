import pytest

from localmind.catalog import Catalog, Decision, FileStatus
from localmind.config import sensitive_policy_from_env
from localmind.ingest import Outcome, apply_decision, ingest, retry
from localmind.quality import QualityConfig, Reason
from tests.pdf_factory import make_pdf

GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."
IBAN = " IBAN FR7630006000011234567890189"


@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "catalog.db") as cat:
        yield cat


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    return folder


def outcomes(report):
    return {r.path.name: r.outcome for r in report.results}


def test_accept_applies_immediately_and_survives_ingest(catalog, inbox):  # Review Focus 4
    f = inbox / "todo.txt"
    f.write_text("appeler", encoding="utf-8")
    ingest(inbox, catalog)

    assert apply_decision(f, Decision.ACCEPT, catalog).outcome is Outcome.NEW
    assert outcomes(ingest(inbox, catalog)) == {"todo.txt": Outcome.UNCHANGED}
    assert catalog.file_by_path(f.resolve()).status is FileStatus.INDEXED


def test_accept_cannot_rescue_a_scan(catalog, inbox):
    f = inbox / "scan.pdf"
    f.write_bytes(make_pdf([""]))
    ingest(inbox, catalog)

    result = apply_decision(f, Decision.ACCEPT, catalog)

    assert result.outcome is Outcome.QUARANTINED and result.reason is Reason.NEEDS_OCR


def test_ignore_removes_and_sticks(catalog, inbox):
    f = inbox / "a.txt"
    f.write_text(GOOD, encoding="utf-8")
    ingest(inbox, catalog)

    assert apply_decision(f, Decision.IGNORE, catalog).outcome is Outcome.IGNORED
    assert catalog.stats().documents == 0

    (inbox / "copy.txt").write_bytes(f.read_bytes())
    assert outcomes(ingest(inbox, catalog))["copy.txt"] is Outcome.IGNORED


def test_ignore_keeps_document_shared_with_another_file(catalog, inbox):
    (inbox / "a.txt").write_text(GOOD, encoding="utf-8")
    (inbox / "b.txt").write_text(GOOD + "\r\n", encoding="utf-8")  # same text, other bytes
    ingest(inbox, catalog)

    apply_decision(inbox / "a.txt", Decision.IGNORE, catalog)

    assert catalog.stats().documents == 1


def test_ignore_applies_to_exact_copies(catalog, inbox):  # final review Important #2
    (inbox / "backup").mkdir()
    (inbox / "note.txt").write_text(GOOD, encoding="utf-8")
    (inbox / "backup" / "note.txt").write_text(GOOD, encoding="utf-8")
    ingest(inbox, catalog)

    apply_decision(inbox / "note.txt", Decision.IGNORE, catalog)

    assert catalog.stats().documents == 0
    assert catalog.stats().ignored == 2


def test_accept_applies_to_exact_copies(catalog, inbox):
    (inbox / "backup").mkdir()
    (inbox / "short.txt").write_text("appeler", encoding="utf-8")
    (inbox / "backup" / "short.txt").write_text("appeler", encoding="utf-8")
    ingest(inbox, catalog)

    apply_decision(inbox / "short.txt", Decision.ACCEPT, catalog)

    assert catalog.quarantined_files() == []


def test_decision_on_unknown_path_raises(catalog, inbox):
    f = inbox / "never.txt"
    f.write_text(GOOD, encoding="utf-8")
    with pytest.raises(KeyError):
        apply_decision(f, Decision.ACCEPT, catalog)


def test_retry_reports_missing_and_keeps_entry(catalog, inbox):
    f = inbox / "s.pdf"
    f.write_bytes(make_pdf([""]))
    ingest(inbox, catalog)
    f.unlink()

    assert [r.outcome for r in retry(catalog).results] == [Outcome.MISSING]
    assert len(catalog.quarantined_files()) == 1


def test_retry_with_new_threshold_rescues_short_file(catalog, inbox):
    (inbox / "todo.txt").write_text("appeler", encoding="utf-8")
    ingest(inbox, catalog)

    report = retry(catalog, QualityConfig(min_doc_words=1))

    assert outcomes(report) == {"todo.txt": Outcome.NEW}


def test_retry_limited_to_folder(catalog, inbox):
    sub = inbox / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_bytes(make_pdf([""]))
    (inbox / "b.pdf").write_bytes(make_pdf(["", ""]))
    ingest(inbox, catalog)

    assert list(outcomes(retry(catalog, under=sub.resolve()))) == ["a.pdf"]


def test_retry_with_quarantine_policy_ignores_accepted_documents(catalog, inbox):
    (inbox / "rib.txt").write_text(GOOD + IBAN, encoding="utf-8")
    ingest(inbox, catalog)

    assert retry(catalog, QualityConfig(sensitive_policy="quarantine")).results == ()


def test_policy_from_environment(monkeypatch):
    monkeypatch.delenv("LOCALMIND_SENSITIVE_POLICY", raising=False)
    assert sensitive_policy_from_env() == "flag"
    monkeypatch.setenv("LOCALMIND_SENSITIVE_POLICY", "quarantine")
    assert sensitive_policy_from_env() == "quarantine"
    monkeypatch.setenv("LOCALMIND_SENSITIVE_POLICY", "nope")
    with pytest.raises(ValueError):
        sensitive_policy_from_env()
