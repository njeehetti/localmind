# Stage 5b: Fact extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store each document's payments, dues, appointments and bank-statement lines as rows tied to their page, so sums and date filters are exact SQL, with every number taken from the text by rules and only labelled by the LLM.

**Architecture:** A pure `candidates.py` finds amounts, dates and times on a masked page and tags them `[M1]`, `[D1]`, `[H1]`. `facts.py` sends each tagged page to `qwen3:8b` (JSON schema), keeps only answers whose ids exist on that page, and stores a document's facts whole (catalog v7, `catalog_facts.py`). `facts_query.py` answers JARVIS: filters, effective kind (bank-statement folders), probable duplicates, confirmed/unconfirmed totals, coverage. `cli_facts.py` prints.

**Tech Stack:** Python 3.12+, sqlite3, `re`, Ollama (`qwen3:8b`, `127.0.0.1`), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-24-stage5b-fact-extraction-design.md`

## Global Constraints

- 100 % local; never modify, move or delete user files; never print or store a sensitive value; spike and e2e on scratch data only, never `~/.localmind` nor the user's documents.
- Repo, code, CLI copy in English. Commits: conventional, repo-local noreply email.
- Facts are built from `catalog.masked_pages(document_id)` only (stage 2a masks: `[IBAN]`, `[CARD]`, `[NIR]`, `[PASSWORD]`).
- LLM: existing `Classifier.choose(messages, schema)` (temperature 0, `num_ctx` 4096); `release()` after a run. Model key stored in runs: `llm.name.split("+")[0]`.
- Kinds: `payment`, `due`, `appointment`, `bank_movement`. Methods: `debit`, `transfer`, `card`, `other`, null.
- `PROMPT_VERSION = "1"`, `MAX_FACTS = 50` per document, `PAGE_CHARS = 8000` (annotated page text sent, cut there).
- Amounts: integer cents, signed, currency `"EUR"` for `€`/`EUR`/`euros`; a number without a currency mark is not an amount. Dates: day first; 2-digit year `yy` → 20yy if yy < 70 else 19yy; impossible dates dropped.
- Totals: euro only; `confirmed_cents` = `payment`; `unconfirmed_cents` = `due` dated ≤ today; never `bank_movement`; future dues and `day_of_month`-only dues listed, never summed.
- Duplicate: same non-empty `counterparty_key`, same `amount_cents`, dates ≤ 5 days apart; a `payment` wins over a `due`, otherwise the lower fact id wins; the loser keeps its row with `duplicate_of` set and leaves the totals.
- Bank-statement folder: `fold(folder.type).startswith("relev")`, placed or to confirm; decided at query time.
- `SCHEMA_VERSION = 7`. Version `0.5.0`.
- CLI copy (exact):
  - fact line: `f"{date}  {kind:<13} {amount:>12}  {who}  {file} p.{page}"` + `"  (duplicate)"` when `duplicate_of`; `date` is `15/03/2026`, `le 5 du mois` for a day-of-month due, `--` if none; `amount` like `1 078,00 €` (space thousands, comma decimals), `--` if none; `who` = counterparty, else place, else label, else `--`.
  - totals (with `--sum`): `Total: {total} (confirmed {c}, unconfirmed {u})`
  - coverage (always, when non-zero): `  {n} document(s) not analysed yet, {m} incomplete: localmind facts extract`
  - extract report: `Analysed {d} document(s): {f} fact(s)` then, when non-zero, `  {r} answer(s) rejected`, `  {i} document(s) incomplete`.

## Review Focus

1. A candidate beyond the `PAGE_CHARS` cut was never seen by the model: an answer citing its id must be rejected (Task 3 test).
2. An id of the wrong type (`"amount": "D1"`), an unknown id, or an unknown kind/method: that fact rejected and counted, the page's other facts kept (Task 3).
3. Masked IBAN/card text next to amounts (`[IBAN] 84,20 €`), payslip tables, phone numbers `01 23 45 67 89` and references `N° 2026-03-15-A`: never an amount (Task 2).
4. Ollama dies on document 3 of 5: documents 1-2 stored whole, 3-5 untouched and still counted in coverage; no half-stored document (Task 3).
5. A document re-ingested after an edit, pruned, or moved into `Banque/Relevés`: its old facts vanish / its kind becomes `bank_movement` with no re-extraction (Tasks 3, 4, 6).

---

### Task 1: Spike gate — rules + LLM labelling on scratch documents

**Files:**
- Create (scratchpad only, not committed): `spike5b.py`, 12 generated French documents (TXT is enough, plus one two-page PDF via `tests/pdf_factory.py`).
- Modify: spec, new section 7.1 "Spike results".

- [ ] **Step 1:** Generate scratch documents (invented names, no real data): invoice paid by debit ("prélevé le 15/03/2026", 84,20 €, EDF); unpaid invoice ("à payer avant le 30/04/2026", 1 152,00 €); EDF schedule of 12 monthly lines (98,00 €, `05/01/2026`…`05/12/2026`); EDF annual adjustment ("Total 1 274,00 €, déjà réglé 1 176,00 €, reste à payer 98,00 €"); phone contract "prélevé le 5 de chaque mois", 19,99 €; hospital summons (12 décembre 2026 à 14h30, Hôpital Nord, Dr Martin); bank statement with 15 lines including `PRLV SEPA EDF 84,20`; payslip (gross, net, contributions); one document with odd formats (`€ 1.234,56`, `1er mars 2026`, `15.03.26`, `14 h`).
- [ ] **Step 2:** In `spike5b.py`, prototype candidate regexes and the inline tags, call the real Ollama with the spec 5.3 schema, one call per page; print per document: candidates found vs expected, facts right / wrong / missing, seconds per page.
- [ ] **Step 3:** Iterate the prompt (at most 3 versions). Gate: **no `payment` from the payslip; ≥ 10/12 schedule lines; the adjustment gives a `due` of 98,00 €; questions 1, 2, 3 answered right by hand-written SQL over the spike's facts; ≤ 5 s per 1–2 page document.** If the gate fails, **stop and report the table to the user before Task 2.**
- [ ] **Step 4:** Write the results table, the chosen prompt text and whether the "Montant column" rule is kept into spec 7.1; commit `docs: stage 5b spike results`.

### Task 2: Candidates — amounts, dates, times, inline tags (pure)

**Files:**
- Create: `src/localmind/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Amount(id: str, cents: int, currency: str, start: int, end: int)`
  - `@dataclass(frozen=True) DateCandidate(id: str, iso: str | None, day_of_month: int | None, month_only: bool, start: int, end: int)` (exactly one of `iso`, `day_of_month` set; `month_only` → `iso` is the 1st of the month)
  - `@dataclass(frozen=True) TimeCandidate(id: str, hhmm: str, start: int, end: int)`
  - `@dataclass(frozen=True) PageCandidates(amounts, dates, times)` (tuples, text order) with `get(id: str) -> Amount | DateCandidate | TimeCandidate | None`
  - `find_candidates(text: str) -> PageCandidates` — ids `M1…`, `D1…`, `H1…` numbered in text order.
  - `annotate(text: str, found: PageCandidates) -> str` — inserts `"[M1] "` etc. right before each span.

- [ ] **Step 1: Write the failing tests** (parametrised):

```python
@pytest.mark.parametrize("text, cents", [
    ("84,20 €", 8420), ("84,20 EUR", 8420), ("1 234,56 €", 123456), ("1 234,56 €", 123456),
    ("1.234,56 €", 123456), ("€ 84,20", 8420), ("-84,20 €", -8420), ("84.20 €", 8420),
    ("1.234 €", 123400), ("98 euros", 9800), ("Montant : 98,00 €", 9800), ("[IBAN] 84,20 €", 8420),
])
def test_amounts(text, cents): assert [a.cents for a in find_candidates(text).amounts] == [cents]

@pytest.mark.parametrize("text", ["01 23 45 67 89", "N° 2026-03-15-A", "[IBAN] ", "Quantité 3", "12345"])
def test_not_amounts(text): assert find_candidates(text).amounts == ()

@pytest.mark.parametrize("text, iso", [
    ("15/03/2026", "2026-03-15"), ("15/03/26", "2026-03-15"), ("15-03-2026", "2026-03-15"),
    ("15.03.2026", "2026-03-15"), ("2026-03-15", "2026-03-15"), ("15 mars 2026", "2026-03-15"),
    ("1er mars 2026", "2026-03-01"), ("12 décembre 2026", "2026-12-12"),
])
def test_dates(text, iso): ...
def test_month_only(): "mars 2026" → iso "2026-03-01", month_only True
def test_recurring_day(): "prélevé le 5 de chaque mois" and "le 5 du mois" → day_of_month 5, iso None
def test_impossible_dates_dropped(): "31/02/2026", "00/01/2026" → ()
def test_times(): "14h30" "14:30" "14 h" → "14:30", "14:30", "14:00"; "84,20 €" has no time
def test_ids_in_text_order_and_spans_point_at_the_text(): text[a.start:a.end] contains the digits
def test_annotate(): annotate("Total TTC : 84,20 € le 15/03/2026", ...) == "Total TTC : [M1] 84,20 € le [D1] 15/03/2026"
```

- [ ] **Step 2:** Run `uv run pytest tests/test_candidates.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement. Month names: French, accents optional (`fevrier`, `aout`, `decembre`). An amount span inside a date span is not an amount. If spike 7.1 kept the "Montant column" rule, add it with its own test; otherwise omit.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: amount, date and time candidates with inline ids`.

### Task 3: Catalog v7 and extraction

**Files:**
- Create: `src/localmind/catalog_facts.py`, `src/localmind/facts.py`
- Modify: `src/localmind/catalog.py` (`SCHEMA_VERSION = 7`; `Catalog(FolderStore, ChunkStore, FactStore)`; v6 → v7 adds tables only; `"facts"`, `"fact_pages"`, `"fact_runs"` first in `_REBUILT_TABLES`).
- Test: `tests/test_catalog_v7.py`, `tests/test_facts.py`; bump `== 6` version expectations in older migration tests.

**Interfaces:**
- Consumes: `find_candidates`, `annotate`, `PageCandidates.get` (Task 2); `Catalog.masked_pages`, `Catalog.document_folders()`, `Classifier.choose/release`, `ClassifierUnavailable`; `tests/fake_classifier.py` (`answers`, `fail_choose_after`, `chats`, `releases`).
- Produces (`catalog_facts.py`):
  - `counterparty_key(name: str | None) -> str` — `fold()`, punctuation to spaces, trailing legal forms `sa sas sarl sasu eurl se` dropped; `None` → `""`.
  - `@dataclass(frozen=True) NewFact(page_no, kind, amount_cents, currency, date, day_of_month, time, counterparty, method, place, with_whom, label, amount_span: tuple[int, int] | None, date_span: tuple[int, int] | None)`
  - `@dataclass(frozen=True) PageResult(page_no: int, status: Literal["done", "failed", "no_candidates"], facts: tuple[NewFact, ...])`
  - `FactStore.documents_to_extract(model: str, prompt_version: str, redo: bool = False) -> list[int]` — documents with usable pages and no run for (model, prompt_version), plus `incomplete` ones with failed pages and `capped = 0`; all of them if `redo`.
  - `FactStore.failed_pages(document_id: int) -> list[int]`
  - `FactStore.store_extraction(document_id, model, prompt_version, pages: Sequence[PageResult], rejected: int, capped: bool) -> None` — one transaction: replaces facts and `fact_pages` rows **of the given pages only**, then recomputes the run: `failed` if every page failed, `no_candidates` if every page has none, `incomplete` if any page failed or `capped`, else `done`.
- Produces (`facts.py`): `PROMPT_VERSION`, `MAX_FACTS`, `PAGE_CHARS`, `ANSWER_SCHEMA` (spec 5.3), `SYSTEM_PROMPT` (from spike 7.1);
  - `parse_page_answer(answer: dict, found: PageCandidates, seen_chars: int, page_no: int) -> tuple[list[NewFact], int]` — pure; kept facts and rejected count. A candidate whose tag starts at or after `seen_chars` in the annotated text counts as unknown.
  - `@dataclass(frozen=True) ExtractReport(documents: int, facts: int, rejected: int, incomplete: int, failed: int, unavailable: str | None)`
  - `extract_facts(catalog, llm: Classifier, redo: bool = False) -> ExtractReport`

- [ ] **Step 1: Write the failing tests:**

```python
# test_catalog_v7.py
def test_v6_catalog_migrates_to_v7_keeping_chunks_and_folders(tmp_path)
def test_facts_cascade_when_the_document_goes(...)   # edited file re-ingested; prune
def test_store_extraction_replaces_only_given_pages(...)  # page 2 retried, page 1 facts kept
def test_run_status_rules(...)                        # failed / no_candidates / incomplete / capped / done
def test_documents_to_extract(...)                    # new doc; other prompt_version; incomplete with failed page; capped not retried; redo
@pytest.mark.parametrize("name, key", [("EDF SA", "edf"), ("Électricité de France", "electricite de france"), (None, "")])
def test_counterparty_key(name, key)

# test_facts.py — parse_page_answer
def test_valid_payment_takes_values_from_the_text()   # cents, iso, spans from candidates
def test_wrong_type_unknown_id_unknown_kind_are_rejected_others_kept()
def test_id_beyond_the_cut_is_rejected()
def test_recurring_due_uses_day_of_month()
# test_facts.py — extract_facts with FakeClassifier
def test_pages_without_candidates_make_no_call()
def test_invalid_json_twice_marks_the_page_failed_and_the_document_incomplete()
def test_cap_keeps_first_50_and_marks_capped()
def test_unavailable_midway_keeps_finished_documents_whole()   # fail_choose_after; report.unavailable set
def test_prompt_carries_file_folder_page_and_masked_text()     # no IBAN digits in any chat message
def test_llm_released_after_run()
```

- [ ] **Step 2:** Run `uv run pytest tests/test_catalog_v7.py tests/test_facts.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Schema per spec 5.4 (`facts`, `fact_runs`, `fact_pages`, each `document_id … ON DELETE CASCADE`; indexes on `facts(kind, date)` and `facts(counterparty_key)`). `extract_facts`: per document, every page (or only `failed_pages` for an incomplete one, unless `redo`) → candidates → annotate → cut to `PAGE_CHARS` → `choose` with one retry on `ValueError/KeyError/TypeError` → `parse_page_answer`; keep the first `MAX_FACTS` facts in page order; `store_extraction` once per document. `ClassifierUnavailable` → stop, nothing stored for the current document, `unavailable = exc.with_command("localmind facts extract").hint`. `release()` in `finally`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: catalog v7 and fact extraction`.

### Task 4: Querying facts — filters, kinds, duplicates, totals, coverage

**Files:**
- Create: `src/localmind/facts_query.py`
- Modify: `src/localmind/catalog_facts.py` (`fact_rows`, `coverage_counts`)
- Test: `tests/test_facts_query.py`

**Interfaces:**
- Consumes: `FactStore` (Task 3), `Catalog.document_folders(only)`, `parse_folder_filter`, `fold`, `UnknownFolder`, `CatalogBusy` (wrap `sqlite3.OperationalError` containing "locked", as `search.py:142`).
- Produces:
  - `@dataclass(frozen=True) Fact(id, kind, amount_cents, currency, date: date | None, day_of_month, time, counterparty, method, place, with_whom, label, file: str, page_no: int, folder: str | None, duplicate_of: int | None)` — `kind` is the effective kind.
  - `@dataclass(frozen=True) Coverage(missing: int, incomplete: int, failed: int)`
  - `@dataclass(frozen=True) FactsResult(facts: tuple[Fact, ...], confirmed_cents: int, unconfirmed_cents: int, coverage: Coverage)` with property `total_cents`.
  - `facts(catalog, kind: str | Collection[str] | None = None, counterparty: str | None = None, folder: str | None = None, date_from: date | None = None, date_to: date | None = None, upcoming: bool = False, today: date | None = None) -> FactsResult` — sorted by date (undated last), then id. Date filters and `upcoming` drop undated facts. `kind` filters on the effective kind. `counterparty` = substring of `counterparty_key`. Coverage over the folder's documents if `folder`, else all documents with usable text.

- [ ] **Step 1: Write the failing tests** (facts inserted with `store_extraction`):

```python
def test_question_1_electricity_2026():          # 2 payments + 1 past due + 1 future due, kind=("payment","due"), counterparty="edf"
    assert (r.confirmed_cents, r.unconfirmed_cents) == (16840, 9800)   # future due listed, not summed
def test_payment_and_due_same_amount_4_days_apart_counted_once()      # due.duplicate_of == payment.id
def test_6_days_apart_or_other_amount_not_duplicates()
def test_statement_folder_makes_bank_movement_and_leaves_totals()    # move doc to "Banque/Relevés" → kind changes, totals drop
def test_upcoming_appointments_sorted_from_today()                    # today injected; past ones excluded
def test_recurring_due_found_without_dates_and_excluded_by_upcoming()
def test_non_euro_listed_not_summed()
def test_coverage_counts_unread_and_incomplete_in_folder_scope()
def test_unknown_folder_raises()
```

- [ ] **Step 2:** Run `uv run pytest tests/test_facts_query.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement; duplicates decided in Python over the filtered `payment`/`due` rows (sort by date, compare within 5 days).
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: fact queries with totals, duplicates and coverage`.

### Task 5: CLI, ingest hook, public API

**Files:**
- Create: `src/localmind/cli_facts.py`
- Modify: `src/localmind/cli.py` (register `facts extract` / `facts list`; `ingest --no-facts`; after indexing, when Ollama was reachable and not `--no-facts`, `cli_facts.facts_after_ingest(catalog)`; `stats` rows `facts` and `documents analysed`), `src/localmind/__init__.py` (export `extract_facts`, `ExtractReport`, `facts`, `Fact`, `FactsResult`, `Coverage`; `__version__ = "0.5.0"`), `pyproject.toml` (`0.5.0`).
- Test: `tests/test_cli_facts.py`

**Interfaces:**
- Consumes: `extract_facts`, `facts` (Tasks 3-4); the `make_embedder()` pattern of `cli_search.py` (`OllamaClassifier()`).
- Produces: `localmind facts extract [--all]` (exit 2 + hint on stderr when Ollama is down); `localmind facts list [--kind K]... [--who TEXT] [--folder F] [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--upcoming] [--sum]` (never needs Ollama; bad date → `error: …`, exit 2; unknown folder → `error: no folder named '…'`, exit 2).

- [ ] **Step 1: Write the failing tests:** exact fact line (`15/03/2026  payment            84,20 €  EDF  facture.txt p.1`), `le 5 du mois`, `  (duplicate)`, `--sum` totals line with `1 078,00 €`, coverage line, extract report lines, `ingest --no-facts` makes no LLM call, ingest without the flag analyses new documents, unknown folder and bad date exit 2, `--upcoming` uses today.
- [ ] **Step 2:** Run `uv run pytest tests/test_cli_facts.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement (printing only; French amount helper `euros(cents: int) -> str` in `cli_facts.py`).
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: facts commands and analysis after ingest`.

### Task 6: End to end on scratch data, docs

**Files:**
- Create: `docs/05b-fact-extraction.md`; learning notebook `apprentissage/05b-extraction-faits.md` + its index (not in the repo).
- Modify: `README.md` (roadmap row `| 5b | [Fact extraction](docs/05b-fact-extraction.md) | Rules find, a local LLM labels; provenance spans, exact sums, duplicates | ✅ |`, section "Facts: sums and dates"), spec status line.

- [ ] **Step 1:** With the real Ollama and a scratch `--db` in the scratchpad, ingest the Task 1 documents, then run: `facts list --kind payment --kind due --who edf --from 2026-01-01 --to 2026-12-31 --sum`, `facts list --upcoming --kind appointment`, `facts list --kind due --who <phone supplier>`. Expected: question 1 total matches the hand count with the bank line excluded; the summons listed; `le 5 du mois`. Note seconds per document. Any bug → failing test first, then fix, commit `fix: …`.
- [ ] **Step 2:** `localmind move <statement file> "Banque/Relevés"` and re-list: its lines show as `bank_movement` with no re-extraction.
- [ ] **Step 3:** Write the docs (what a fact is, why rules find and the LLM labels, provenance spans, duplicates, coverage, measured numbers) and the notebook entry; run `uv run pytest -q` (all pass); commit `docs: stage 5b fact extraction`.
