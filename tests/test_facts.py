"""Stage 5b: the LLM labels tagged values; only tags it saw on that page count."""

from __future__ import annotations

import re

import pytest

from localmind.candidates import annotate, find_candidates
from localmind.catalog import Catalog
from localmind.facts import MAX_FACTS, PAGE_CHARS, extract_facts, parse_page_answer
from tests.fake_classifier import FakeClassifier
from tests.pdf_factory import make_pdf
from tests.stage4_data import IBAN, add_files, count


def tag(text: str, value: str) -> str:
    """The tag written right before ``value`` in an annotated text."""
    return re.search(r"\[(\w+)\] " + re.escape(value), text)[1]


def item(kind="payment", amount=None, date=None, time=None, who="EDF", method="debit",
         place=None, with_whom=None, label="Facture"):
    return {"kind": kind, "amount": amount, "date": date, "time": time, "counterparty": who,
            "method": method, "place": place, "with_whom": with_whom, "label": label}


def page(text):
    found = find_candidates(text)
    annotated = annotate(text, found)
    return found, annotated, {c.id for c in found.all()}


def parse(text, facts, doc_type="invoice", visible=None):
    found, annotated, seen = page(text)
    answer = {"document_type": doc_type, "facts": facts(annotated) if callable(facts) else facts}
    return parse_page_answer(answer, found, text, seen if visible is None else visible, 1)


# -- parse_page_answer -------------------------------------------------------
INVOICE = "EDF facture du 02/03/2026\nTotal TTC : 84,20 €\nPrélevé le 15/03/2026"


def test_valid_payment_takes_its_values_from_the_text():
    kept, rejected = parse(INVOICE, lambda a: [item(amount=tag(a, "84,20"), date=tag(a, "15/03/2026"))])
    assert rejected == 0
    (f,) = kept
    assert (f.kind, f.amount_cents, f.currency, f.date, f.counterparty, f.method) == (
        "payment", 8420, "EUR", "2026-03-15", "EDF", "debit")
    assert INVOICE[f.amount_span[0]:f.amount_span[1]].startswith("84,20")
    assert INVOICE[f.date_span[0]:f.date_span[1]] == "15/03/2026"


def test_the_leading_tag_is_kept_and_any_number_written_after_it_ignored():
    kept, _ = parse(INVOICE, lambda a: [item(amount=f"[{tag(a, '84,20')}] 99,99 €", date=tag(a, "15/03/2026"))])
    assert kept[0].amount_cents == 8420


@pytest.mark.parametrize(
    "bad",
    [
        lambda a: item(amount=tag(a, "15/03/2026"), date=tag(a, "15/03/2026")),  # date tag as amount
        lambda a: item(amount="M9Z", date=tag(a, "15/03/2026")),                  # unknown tag
        lambda a: item(amount="84,20", date=tag(a, "15/03/2026")),                # a number, not a tag
        lambda a: item(kind="gift", amount=tag(a, "84,20")),                      # unknown kind
        lambda a: item(amount=tag(a, "84,20"), method="cheque"),                  # unknown method
        lambda a: "not a fact",
    ],
)
def test_bad_facts_are_rejected_and_the_others_kept(bad):
    kept, rejected = parse(INVOICE, lambda a: [bad(a), item(amount=tag(a, "84,20"), date=tag(a, "15/03/2026"))])
    assert rejected == 1
    assert [f.amount_cents for f in kept] == [8420]


def test_a_tag_the_model_never_saw_is_rejected():
    found, annotated, seen = page(INVOICE)
    hidden = tag(annotated, "15/03/2026")
    answer = {"document_type": "invoice", "facts": [item(amount=tag(annotated, "84,20"), date=hidden)]}
    kept, rejected = parse_page_answer(answer, found, INVOICE, seen - {hidden}, 1)
    assert (kept, rejected) == ([], 1)


def test_same_line_rule_rejects_a_date_from_another_line():
    text = "02/03/2026  CB CARREFOUR  -54,12 €\n03/03/2026  PRLV EDF  -84,20 €"
    kept, rejected = parse(
        text,
        lambda a: [
            item(kind="bank_movement", amount=tag(a, "-84,20"), date=tag(a, "02/03/2026")),
            item(kind="bank_movement", amount=tag(a, "-54,12"), date=tag(a, "02/03/2026")),
        ],
        doc_type="bank_statement",
    )
    assert rejected == 1
    assert [(f.amount_cents, f.date) for f in kept] == [(-5412, "2026-03-02")]


def test_a_payslip_gives_no_fact():
    text = "Bulletin de paie\nNet payé : 2 145,30 € le 31/03/2026"
    kept, rejected = parse(text, lambda a: [item(amount=tag(a, "2 145,30"), date=tag(a, "31/03/2026"))], "payslip")
    assert (kept, rejected) == ([], 0)


def test_bank_movement_only_on_a_bank_statement():
    kept, rejected = parse(INVOICE, lambda a: [item(kind="bank_movement", amount=tag(a, "84,20"), date=tag(a, "15/03/2026"))])
    assert (kept, rejected) == ([], 1)


def test_payment_on_a_bank_statement_becomes_bank_movement():
    kept, _ = parse(INVOICE, lambda a: [item(amount=tag(a, "84,20"), date=tag(a, "15/03/2026"))], "bank_statement")
    assert kept[0].kind == "bank_movement"


def test_recurring_due_uses_day_of_month_and_a_payment_on_it_becomes_a_due():
    text = "Forfait 19,99 € prélevé le 5 de chaque mois"
    kept, _ = parse(text, lambda a: [item(amount=tag(a, "19,99"), date=tag(a, "le 5"), who="Free")], "contract")
    (f,) = kept
    assert (f.kind, f.date, f.day_of_month, f.amount_cents) == ("due", None, 5, 1999)


def test_an_appointment_needs_a_full_date():
    text = "Rendez-vous le 12 décembre 2026 à 14h30. Entretien le 5 de chaque mois."
    kept, rejected = parse(
        text,
        lambda a: [
            item(kind="appointment", date=tag(a, "12 décembre 2026"), time=tag(a, "14h30"),
                 who=None, method=None, place="Hôpital Nord", with_whom="Dr Martin"),
            item(kind="appointment", date=tag(a, "le 5"), who=None, method=None),
            item(kind="appointment", who=None, method=None),
        ],
        "appointment_letter",
    )
    assert rejected == 2
    assert [(f.date, f.time, f.place, f.with_whom) for f in kept] == [("2026-12-12", "14:30", "Hôpital Nord", "Dr Martin")]


def test_an_appointment_with_an_amount_is_rejected():
    # found end to end: "Devis accepté le 1er mars 2026 … 1 234,56 €" became an appointment
    text = "Devis accepté le 1er mars 2026\nMontant total : € 1.234,56"
    kept, rejected = parse(
        text,
        lambda a: [item(kind="appointment", amount=tag(a, "€ 1.234,56"), date=tag(a, "1er mars 2026"))],
    )
    assert (kept, rejected) == ([], 1)


def test_a_money_fact_needs_an_amount_or_a_date():
    kept, rejected = parse(INVOICE, [item()])
    assert (kept, rejected) == ([], 1)


def test_identical_facts_on_a_page_are_merged():
    kept, rejected = parse(INVOICE, lambda a: [item(amount=tag(a, "84,20"), date=tag(a, "15/03/2026"))] * 3)
    assert (len(kept), rejected) == (1, 0)


def test_an_answer_without_a_facts_list_is_invalid():
    found, _, seen = page(INVOICE)
    with pytest.raises(ValueError):
        parse_page_answer({"document_type": "invoice"}, found, INVOICE, seen, 1)


# -- extract_facts ------------------------------------------------------------
@pytest.fixture
def catalog(tmp_path):
    with Catalog(tmp_path / "c.db") as cat:
        yield cat


def labeller(payload):
    """A scripted model: one payment per tagged amount, with the page's first date."""
    text = payload["text"]
    amounts = re.findall(r"\[(M\w+)\]", text)
    dates = re.findall(r"\[(D\w+)\]", text)
    return {"document_type": "invoice",
            "facts": [item(amount=m, date=dates[0] if dates else None) for m in amounts]}


def test_pages_without_candidates_make_no_call(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Une lettre sans montant ni date, simplement du texte."})
    llm = FakeClassifier(answers=labeller)
    report = extract_facts(catalog, llm)
    assert llm.chats == []
    assert (report.documents, report.facts) == (1, 0)
    assert count(catalog, "SELECT COUNT(*) FROM fact_runs WHERE status = 'no_candidates'") == 1


def test_facts_are_stored_with_their_page(catalog, tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "two.pdf").write_bytes(make_pdf([
        "Page un sans valeur aucune, juste du texte de remplissage.",
        "Facture EDF total 84,20 EUR preleve le 15/03/2026 sur le compte.",
    ]))
    add_files(catalog, inbox, {})
    report = extract_facts(catalog, FakeClassifier(answers=labeller))
    assert (report.documents, report.facts, report.rejected) == (1, 1, 0)
    row = catalog._conn.execute("SELECT page_no, amount_cents, date FROM facts").fetchone()
    assert row == (2, 8420, "2026-03-15")


def test_invalid_answer_twice_marks_the_page_failed(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF total 84,20 € prélevé le 15/03/2026."})
    llm = FakeClassifier(answers=[ValueError("not JSON"), {"document_type": "invoice"}])
    report = extract_facts(catalog, llm)
    assert len(llm.chats) == 2
    assert (report.failed, report.facts) == (1, 0)
    assert count(catalog, "SELECT COUNT(*) FROM fact_pages WHERE status = 'failed'") == 1


def test_a_retry_after_failure_only_redoes_failed_pages(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF total 84,20 € prélevé le 15/03/2026."})
    extract_facts(catalog, FakeClassifier(answers=[ValueError("x"), ValueError("x")]))
    report = extract_facts(catalog, FakeClassifier(answers=labeller))
    assert (report.documents, report.facts, report.failed) == (1, 1, 0)
    assert count(catalog, "SELECT COUNT(*) FROM fact_runs WHERE status = 'done'") == 1


def test_cap_keeps_the_first_facts_and_marks_capped(catalog, tmp_path):
    lines = "\n".join(f"{(i % 28) + 1:02d}/01/2026  ligne {i}  {i + 1},00 €" for i in range(MAX_FACTS + 5))
    add_files(catalog, tmp_path / "in", {"a.txt": "Échéancier\n" + lines})
    report = extract_facts(catalog, FakeClassifier(answers=same_line_labeller))
    assert (report.facts, report.incomplete) == (MAX_FACTS, 1)
    assert count(catalog, "SELECT COUNT(*) FROM facts") == MAX_FACTS
    assert count(catalog, "SELECT capped FROM fact_runs") == 1
    assert count(catalog, "SELECT MAX(amount_cents) FROM facts") == MAX_FACTS * 100


def same_line_labeller(payload):
    facts = [item(kind="due", amount=m, date=d)
             for d, m in re.findall(r"\[(D\w+)\] [\d/]+ .*?\[(M\w+)\]", payload["text"])]
    return {"document_type": "payment_schedule", "facts": facts}


def test_text_beyond_the_cut_is_never_sent(catalog, tmp_path):
    filler = "Texte de remplissage sans valeur. " * (PAGE_CHARS // 30)
    add_files(catalog, tmp_path / "in", {"a.txt": filler + "\nTotal 84,20 € le 15/03/2026"})
    llm = FakeClassifier(answers=labeller)
    report = extract_facts(catalog, llm)
    assert all(len(__import__("json").loads(c[-1]["content"])["text"]) <= PAGE_CHARS for c in llm.chats)
    assert report.facts == 0


def test_unavailable_midway_keeps_finished_documents_whole(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {
        f"{n}.txt": f"Facture {n} total {i + 1}0,00 € prélevé le 1{i}/03/2026." for i, n in enumerate("abcde")
    })
    llm = FakeClassifier(answers=labeller, fail_choose_after=2)
    report = extract_facts(catalog, llm)
    assert report.documents == 2
    assert "localmind facts extract" in report.unavailable
    assert count(catalog, "SELECT COUNT(*) FROM fact_runs") == 2
    assert count(catalog, "SELECT COUNT(*) FROM facts") == 2
    assert llm.releases == 1
    assert len(catalog.documents_to_extract("fake-llm", "1")) == 3


def test_the_prompt_carries_file_folder_page_and_masked_text(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"edf.txt": f"Facture EDF 84,20 € prélevée le 15/03/2026 sur {IBAN}."})
    llm = FakeClassifier(answers=labeller)
    extract_facts(catalog, llm)
    (messages,) = llm.chats
    payload = __import__("json").loads(messages[-1]["content"])
    assert (payload["file"], payload["page"]) == ("edf.txt", 1)
    assert "folder" in payload
    assert "[IBAN]" in payload["text"]
    assert all(IBAN.replace(" ", "")[4:12] not in m["content"].replace(" ", "") for m in messages)
    assert "never instructions" in messages[0]["content"]


def test_the_llm_is_released_after_a_run(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF 84,20 € prélevée le 15/03/2026."})
    llm = FakeClassifier(answers=labeller)
    extract_facts(catalog, llm)
    assert llm.releases == 1


def test_documents_already_extracted_are_skipped_unless_redo(catalog, tmp_path):
    add_files(catalog, tmp_path / "in", {"a.txt": "Facture EDF 84,20 € prélevée le 15/03/2026."})
    extract_facts(catalog, FakeClassifier(answers=labeller))
    again = FakeClassifier(answers=labeller)
    assert extract_facts(catalog, again).documents == 0 and again.chats == []
    assert extract_facts(catalog, FakeClassifier(answers=labeller), redo=True).documents == 1
    assert count(catalog, "SELECT COUNT(*) FROM facts") == 1
