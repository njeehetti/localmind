# Stage 2a: Quality Filtering & Quarantine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assess every ingested file page by page, store usable pages, quarantine or set aside the rest with a reason, flag sensitive data, and give the user commands to inspect, retry, accept and ignore.

**Architecture:** Two new pure modules (`sensitive.py`, `quality.py`) decide; `catalog.py` (schema v2) stores; `ingest.py` orchestrates; `cli.py` stays a thin facade. Extractors now return pages.

**Tech Stack:** Python ≥ 3.11, SQLite (stdlib), `pypdf[crypto]`, pytest. Run everything from the repository root with `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-22-stage2a-quarantine-design.md`

## Global Constraints

- Never delete, move or modify an original file.
- Never print or log a sensitive value; CLI shows `kind @ page N`.
- Every set-aside page and quarantined file carries a `Reason`.
- No new heavy dependency; only `pypdf` → `pypdf[crypto]`.
- Frozen dataclasses for all values crossing module boundaries.
- Defaults: `MIN_PAGE_CHARS = 20`, `MIN_DOC_WORDS = 3`, `MIN_NORMAL_CHAR_RATIO = 0.70`, `LOG_MIN_LINES = 20`, `LOG_LINE_RATIO = 0.60`, `sensitive_policy = "flag"`.
- Image suffixes: `.png .jpg .jpeg .tif .tiff .bmp .webp`.
- Catalog schema version 2 via `PRAGMA user_version`.

## Review Focus

1. A PDF page holding only a page number or a short header ("12", "Confidential") among scanned pages must be `needs_ocr`, not usable text. → test in Task 3.
2. A cp1252 text file full of `€`, `°`, `%` must not be `garbled`. → test in Task 3.
3. Digit strings that fail checksums (16-digit order numbers, random 15-digit IDs) must not be flagged; IBANs written with spaces or lowercase must be. → tests in Task 2.
4. `accept` on a file whose bytes did not change must take effect right away, and survive a later `ingest`. → test in Task 5.
5. Opening a stage 1 (v1) catalog must not crash; it is rebuilt and the CLI prints a notice. → tests in Tasks 4 and 6.

---

### Task 1: Extractors return pages; encrypted PDFs

**Files:**
- Modify: `pyproject.toml` (dependency `pypdf[crypto]>=5.0`)
- Modify: `src/localmind/extractors.py`
- Test: `tests/test_extractors.py`

**Interfaces:**
- Produces:
  - `ExtractedText(pages: tuple[str, ...], kind: str, encoding: str | None = None)` with properties `text -> str` (`"\f".join(pages)`) and `page_count -> int | None` (`len(pages)` for `"pdf"`, `None` for `"text"`).
  - `class EncryptedPdf(ExtractionError)`.
  - `IMAGE_SUFFIXES: frozenset[str]`.

- [ ] **Step 1: Update tests.** Existing tests keep passing through `.text` / `.page_count`. Add:

```python
def test_text_file_is_one_page(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("x y z", encoding="utf-8")
    assert extract(f).pages == ("x y z",)

def test_pdf_pages_are_separate(tmp_path):
    f = tmp_path / "d.pdf"; f.write_bytes(make_pdf(["One", "Two"]))
    r = extract(f)
    assert len(r.pages) == 2 and "One" in r.pages[0] and "Two" in r.pages[1]

def test_encrypted_pdf_raises_encrypted(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter(); w.add_blank_page(100, 100); w.encrypt(user_password="secret", algorithm="AES-256")
    f = tmp_path / "locked.pdf"
    with f.open("wb") as s: w.write(s)
    with pytest.raises(EncryptedPdf):
        extract(f)

def test_pdf_encrypted_with_empty_password_is_readable(tmp_path):
    from pypdf import PdfReader, PdfWriter
    src = tmp_path / "src.pdf"; src.write_bytes(make_pdf(["Hello world text"]))
    w = PdfWriter(clone_from=PdfReader(src)); w.encrypt(user_password="", owner_password="o", algorithm="AES-256")
    f = tmp_path / "open.pdf"
    with f.open("wb") as s: w.write(s)
    assert "Hello world text" in extract(f).text
```

- [ ] **Step 2: Run** `uv run pytest tests/test_extractors.py -v` → new tests FAIL (`pages` / `EncryptedPdf` missing).
- [ ] **Step 3: Implement.** `uv add "pypdf[crypto]>=5.0"`. In `_extract_pdf`: if `reader.is_encrypted`, try `reader.decrypt("")`; if it returns `PasswordType.NOT_DECRYPTED` raise `EncryptedPdf`. Text files return `pages=(text,)`.
- [ ] **Step 4: Run** `uv run pytest -v` → all PASS (stage 1 tests included).
- [ ] **Step 5: Commit** `feat: extractors return pages and detect encrypted PDFs` (include `uv.lock`).

---

### Task 2: Sensitive data detectors

**Files:**
- Create: `src/localmind/sensitive.py`
- Test: `tests/test_sensitive.py`

**Interfaces:**
- Produces:
  - `SensitiveSpan(kind: str, start: int, end: int)` (frozen); kinds `"iban" | "card" | "fr_nir" | "password"`.
  - `find_sensitive(text: str) -> tuple[SensitiveSpan, ...]` sorted by `start`.
  - `iban_is_valid(candidate: str) -> bool`, `luhn_is_valid(digits: str) -> bool`, `nir_is_valid(digits: str) -> bool`.

- [ ] **Step 1: Write failing tests** (values verified with the algorithms below).

```python
from localmind.sensitive import find_sensitive, iban_is_valid, luhn_is_valid, nir_is_valid

def kinds(text): return [s.kind for s in find_sensitive(text)]

def test_iban_checksum():
    assert iban_is_valid("FR7630006000011234567890189")
    assert not iban_is_valid("FR7630006000011234567890188")

def test_iban_with_spaces_and_lowercase_is_found():
    assert kinds("IBAN : fr76 3000 6000 0112 3456 7890 189") == ["iban"]

def test_luhn():
    assert luhn_is_valid("4539578763621486")
    assert not luhn_is_valid("4539578763621487")

def test_card_found_with_spaces_or_dashes():
    assert kinds("Carte 4539 5787 6362 1486") == ["card"]
    assert kinds("Carte 4539-5787-6362-1486") == ["card"]

def test_order_number_failing_luhn_is_not_flagged():
    assert kinds("Commande n° 4539578763621487") == []

def test_nir_key():
    assert nir_is_valid("185057800608491")      # 1850578006084, key = 97 - (n mod 97) = 91
    assert not nir_is_valid("185057800608492")

def test_random_15_digits_not_flagged():
    assert kinds("Ref 123456789012345") == []

def test_password_lines():
    assert kinds("login: bob\nmot de passe : hunter2") == ["password"]
    assert kinds("password=abc") == ["password"]
    assert kinds("Mot de passe oublié ? Cliquez ici") == []

def test_span_points_at_the_value():
    text = "IBAN FR7630006000011234567890189 fin"
    (s,) = find_sensitive(text)
    assert text[s.start:s.end].replace(" ", "") == "FR7630006000011234567890189"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_sensitive.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `sensitive.py`. Regex finds candidates, checksum confirms (the pedagogical point; say so in the module docstring):
  - IBAN mod-97: move first 4 chars to the end, letters → numbers (A=10…Z=35), integer mod 97 == 1. Candidate: 2 letters, 2 digits, then 11–30 alphanumerics with optional single spaces.
  - Luhn: from the right, double every second digit, subtract 9 if > 9, sum % 10 == 0. Candidates 13–19 digits with optional single spaces/dashes, not adjacent to other digits.
  - NIR: 15 digits, first digit 1 or 2, `97 - (int(first13) % 97) == int(last2)`. (Corsica `2A/2B` out of scope, comment it.)
  - Password: line-anchored, case-insensitive `(password|passwd|pwd|mot de passe|mdp)\s*[:=]\s*\S+`; span covers the value only.
  - A span already claimed by IBAN is not re-reported as card/NIR.
- [ ] **Step 4: Run** `uv run pytest tests/test_sensitive.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: detect sensitive data with checksum validation`.

---

### Task 3: Quality rules and page-level assessment

**Files:**
- Create: `src/localmind/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: `ExtractedText` (Task 1), `SensitiveSpan`, `find_sensitive` (Task 2).
- Produces:
  - `class Reason(StrEnum)`: `UNSUPPORTED_FORMAT="unsupported_format"`, `NEEDS_OCR="needs_ocr"`, `CORRUPTED="corrupted"`, `ENCRYPTED="encrypted"`, `EMPTY="empty"`, `TOO_SHORT="too_short"`, `GARBLED="garbled"`, `MACHINE_LOG="machine_log"`, `SENSITIVE="sensitive"`.
  - `QualityConfig(min_page_chars=20, min_doc_words=3, min_normal_char_ratio=0.70, log_min_lines=20, log_line_ratio=0.60, sensitive_policy: Literal["flag","quarantine"]="flag")` (frozen).
  - `PageVerdict(page_no: int, text: str, reason: Reason | None, spans: tuple[SensitiveSpan, ...])`; `reason is None` means usable; spans only on usable pages.
  - `Assessment(pages: tuple[PageVerdict, ...], quarantine_reason: Reason | None)` with properties `usable_text -> str` (usable pages joined by `"\f"`) and `has_set_aside -> bool`.
  - `assess(extracted: ExtractedText, config: QualityConfig, *, accepted: bool = False) -> Assessment`.
  - Pure rules: `is_garbled(text: str, config: QualityConfig) -> bool`, `is_machine_log(text: str, config: QualityConfig) -> bool`.

- [ ] **Step 1: Write failing tests** (build `ExtractedText` directly; no files needed).

```python
def pdf(*pages): return ExtractedText(pages=pages, kind="pdf")
def txt(s): return ExtractedText(pages=(s,), kind="text")
GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."

def test_clean_document_accepted():
    a = assess(pdf(GOOD, GOOD), QualityConfig())
    assert a.quarantine_reason is None and not a.has_set_aside

def test_scanned_page_set_aside_others_kept():
    a = assess(pdf(GOOD, "", GOOD), QualityConfig())
    assert [p.reason for p in a.pages] == [None, Reason.NEEDS_OCR, None]
    assert a.quarantine_reason is None and a.has_set_aside

def test_page_with_only_page_number_needs_ocr():          # Review Focus 1
    assert assess(pdf(GOOD, "12"), QualityConfig()).pages[1].reason is Reason.NEEDS_OCR

def test_fully_scanned_pdf_quarantined_needs_ocr():
    assert assess(pdf("", " "), QualityConfig()).quarantine_reason is Reason.NEEDS_OCR

def test_whitespace_text_file_is_empty():
    assert assess(txt(" \n\t "), QualityConfig()).quarantine_reason is Reason.EMPTY

def test_too_short():
    assert assess(txt("appeler"), QualityConfig()).quarantine_reason is Reason.TOO_SHORT

def test_garbled_page():
    junk = "\ufffd\ufffd\x07\x08\ufffd\ufffd\x01\x02\x03\ufffd\ufffd\x04\x05\x06\ufffd\ufffd\x0e\x0f\x10\x11 ab"
    assert assess(pdf(GOOD, junk), QualityConfig()).pages[1].reason is Reason.GARBLED

def test_symbols_are_not_garbled():                      # Review Focus 2
    assert not is_garbled("Total : 30 € — TVA 20 % — 25 °C — n° 12 (réf. A/B)", QualityConfig())

def test_machine_log():
    log = "\n".join(f"2024-03-01 12:00:{i:02d} INFO worker started job {i}" for i in range(25))
    assert assess(txt(log), QualityConfig()).quarantine_reason is Reason.MACHINE_LOG

def test_accept_skips_quality_rules_but_not_technical():
    assert assess(txt("appeler"), QualityConfig(), accepted=True).quarantine_reason is None
    assert assess(pdf(""), QualityConfig(), accepted=True).quarantine_reason is Reason.NEEDS_OCR

def test_sensitive_flagged_by_default():
    a = assess(txt(GOOD + " IBAN FR7630006000011234567890189"), QualityConfig())
    assert a.quarantine_reason is None and a.pages[0].spans[0].kind == "iban"

def test_sensitive_quarantine_policy_and_accept_override():
    doc = txt(GOOD + " IBAN FR7630006000011234567890189")
    cfg = QualityConfig(sensitive_policy="quarantine")
    assert assess(doc, cfg).quarantine_reason is Reason.SENSITIVE
    assert assess(doc, cfg, accepted=True).quarantine_reason is None
```

- [ ] **Step 2: Run** `uv run pytest tests/test_quality.py -v` → FAIL.
- [ ] **Step 3: Implement.** Order inside `assess`:
  1. `kind == "text"` and text is whitespace-only → `EMPTY`.
  2. Per page: pdf page with fewer than `min_page_chars` non-whitespace chars → `NEEDS_OCR`. Else, if not accepted and `is_garbled` → `GARBLED`.
  3. All pages set aside → quarantine with the most common page reason (tie → first page's reason).
  4. If not accepted: fewer than `min_doc_words` words in `usable_text` → `TOO_SHORT`; `is_machine_log(usable_text)` → `MACHINE_LOG`.
  5. Spans via `find_sensitive` on usable pages; any span, policy `"quarantine"`, not accepted → `SENSITIVE`.

  "Normal" character: `unicodedata.category(c)` starts with `L`, `N`, `Z` or `P`, or is `Sc`/`Sm`/`So`/`Sk` (covers €, %, °), or `c in "\n\t\r\f"`. Everything else (U+FFFD is `So`, so test it explicitly as abnormal; `Cc`/`Co`/`Cn`) is abnormal. Log line: starts (after optional `[`) with `\d{4}-\d{2}-\d{2}`, `\d{2}:\d{2}:\d{2}`, or a level word `DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|TRACE`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: page-level quality assessment with named thresholds`.

---

### Task 4: Catalog schema v2

**Files:**
- Modify: `src/localmind/catalog.py`
- Test: `tests/test_catalog.py` (new)

**Interfaces:**
- Consumes: `Reason`, `PageVerdict` (Task 3).
- Produces (replacing stage 1 signatures):
  - `FileStatus(StrEnum)`: `INDEXED="indexed"`, `QUARANTINED="quarantined"`, `IGNORED="ignored"`. `Decision(StrEnum)`: `ACCEPT="accept"`, `IGNORE="ignore"`.
  - `FileRecord(path: Path, file_hash: str, size_bytes: int, status: FileStatus, reason: Reason | None, document_id: int | None)`.
  - `Document(id: int, content_key: str, kind: str, page_count: int | None, status: Literal["accepted","partial"], text: str)`; `text` = usable pages joined by `"\f"`.
  - `SetAsidePage(path: Path, page_no: int, reason: Reason)`, `SensitiveFlag(path: Path, page_no: int, kind: str)`.
  - `Stats(documents, files, duplicate_files, accepted, partial, quarantined, ignored, sensitive_flags)`, all `int`.
  - `Catalog(db_path)` with attribute `rebuilt_from_v1: bool`.
  - Reads: `file_by_path(path) -> FileRecord | None`; `file_by_hash(file_hash) -> FileRecord | None` (any status, prefers indexed); `document_id_by_content_key(key) -> int | None`; `document_for_path(path) -> Document | None`; `decision_for(file_hash) -> Decision | None`; `quarantined_files() -> list[FileRecord]`; `set_aside_pages() -> list[SetAsidePage]`; `sensitive_flags() -> list[SensitiveFlag]`; `retry_candidates(under: Path | None = None) -> list[FileRecord]` (quarantined files + indexed files of partial documents, filtered by path prefix); `stats() -> Stats`; `duplicate_groups()` (indexed files only).
  - Writes: `add_document(content_key: str, kind: str, page_count: int | None, pages: Sequence[PageVerdict]) -> int`; `link_file(path, file_hash, size_bytes, document_id, encoding=None)` (status indexed); `record_quarantine(path, file_hash, size_bytes, reason: Reason)`; `record_ignored(path, file_hash, size_bytes)`; `record_decision(file_hash, decision: Decision)`. Each write that changes a file's `document_id` runs the orphan cleanup; pages and spans go with their document through `ON DELETE CASCADE`.

- [ ] **Step 1: Write failing tests**: `add_document` + `link_file` with one set-aside page → `document_for_path(p).status == "partial"` and `set_aside_pages() == [SetAsidePage(p, 2, Reason.NEEDS_OCR)]`; `record_quarantine` → listed by `quarantined_files()`; quarantining a previously indexed path deletes its now-orphan document; `record_decision` / `decision_for` round-trip; `sensitive_flags()` returns `kind` and `page_no` only; `retry_candidates(under=sub)` keeps only paths under `sub`; `stats()` counts each status. Plus:

```python
def test_v1_catalog_is_rebuilt(tmp_path):              # Review Focus 5
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE documents (id INTEGER PRIMARY KEY, content_hash TEXT, text TEXT);"
                      "INSERT INTO documents VALUES (1, 'x', 'old');")
    con.commit(); con.close()
    with Catalog(db) as cat:
        assert cat.rebuilt_from_v1
        assert cat.stats().documents == 0
    with Catalog(db) as cat:
        assert not cat.rebuilt_from_v1
```

- [ ] **Step 2: Run** `uv run pytest tests/test_catalog.py -v` → FAIL.
- [ ] **Step 3: Implement** the schema from the spec (`documents.content_key UNIQUE`, `pages`, `sensitive_spans`, `decisions`, `files.status` / `reason`, nullable `document_id`), with `PRAGMA foreign_keys = ON`. On open: `user_version == 0` and a `documents` table exists → drop all tables, recreate, `rebuilt_from_v1 = True`; always end with `PRAGMA user_version = 2`.
- [ ] **Step 4: Run** `uv run pytest tests/test_catalog.py -v` → PASS (`test_ingest.py` may fail until Task 5; expected).
- [ ] **Step 5: Commit** `feat: catalog schema v2 with pages, quarantine and decisions`.

---

### Task 5: Ingestion pipeline, retry and decisions

**Files:**
- Modify: `src/localmind/ingest.py`, `src/localmind/config.py`, `src/localmind/__init__.py`
- Test: `tests/test_ingest.py` (update stage 1 tests to the new outcomes), `tests/test_decisions.py` (new)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `Outcome(StrEnum)`: `NEW`, `PARTIAL`, `UPDATED`, `DUPLICATE`, `UNCHANGED`, `QUARANTINED`, `IGNORED`, `MISSING` (values lower-case).
  - `FileResult(path: Path, outcome: Outcome, detail: str = "", reason: Reason | None = None)`.
  - `ingest(root: Path, catalog: Catalog, config: QualityConfig = QualityConfig()) -> IngestReport`.
  - `retry(catalog: Catalog, config: QualityConfig = QualityConfig(), under: Path | None = None) -> IngestReport`.
  - `apply_decision(path: Path, decision: Decision, catalog: Catalog, config: QualityConfig = QualityConfig()) -> FileResult`; raises `KeyError` if the path is not in the catalog.
  - `config.sensitive_policy_from_env() -> Literal["flag","quarantine"]` (reads `LOCALMIND_SENSITIVE_POLICY`, default `"flag"`, other value → `ValueError`).
  - `localmind/__init__.py` also exports `QualityConfig`, `Reason`, `Decision`, `retry`, `apply_decision`.

- [ ] **Step 1: Write failing tests.** Update stage 1 expectations: `photo.jpg` → `QUARANTINED`, `reason == Reason.NEEDS_OCR`; `broken.pdf` → `QUARANTINED` / `CORRUPTED`; add a `.zip` → `QUARANTINED` / `UNSUPPORTED_FORMAT`; every stage 1 fixture text with fewer than 3 words ("Alpha", "Same", "Version 1", "Version 2", …) gets ≥ 3 words, or it would now be `too_short`. Add:

```python
GOOD = "Facture EDF de mars 2024, montant total 84,20 euros TTC."

def test_two_different_scans_are_not_duplicates(catalog, inbox):    # stage 1 bug
    (inbox / "a.pdf").write_bytes(make_pdf(["", ""]))
    (inbox / "b.pdf").write_bytes(make_pdf([""]))
    assert set(outcomes(ingest(inbox, catalog)).values()) == {Outcome.QUARANTINED}

def test_partial_pdf(catalog, inbox):
    (inbox / "c.pdf").write_bytes(make_pdf([GOOD, ""]))
    assert outcomes(ingest(inbox, catalog)) == {"c.pdf": Outcome.PARTIAL}

def test_same_good_page_different_scans_not_merged(catalog, inbox):
    (inbox / "a.pdf").write_bytes(make_pdf([GOOD, ""]))
    (inbox / "b.pdf").write_bytes(make_pdf([GOOD, "", ""]))
    assert sorted(outcomes(ingest(inbox, catalog)).values()) == [Outcome.PARTIAL, Outcome.PARTIAL]

def test_copy_of_quarantined_file_is_quarantined_too(catalog, inbox):
    (inbox / "x.jpg").write_bytes(b"\xff\xd8\xff"); (inbox / "x (1).jpg").write_bytes(b"\xff\xd8\xff")
    ingest(inbox, catalog)
    assert len(catalog.quarantined_files()) == 2

def test_accept_applies_immediately_and_survives_ingest(catalog, inbox):   # Review Focus 4
    f = inbox / "todo.txt"; f.write_text("appeler", encoding="utf-8")
    ingest(inbox, catalog)
    assert apply_decision(f, Decision.ACCEPT, catalog).outcome is Outcome.NEW
    assert outcomes(ingest(inbox, catalog)) == {"todo.txt": Outcome.UNCHANGED}
    assert catalog.file_by_path(f.resolve()).status is FileStatus.INDEXED

def test_ignore_removes_and_sticks(catalog, inbox):
    f = inbox / "a.txt"; f.write_text(GOOD, encoding="utf-8"); ingest(inbox, catalog)
    apply_decision(f, Decision.IGNORE, catalog)
    assert catalog.stats().documents == 0
    (inbox / "copy.txt").write_bytes(f.read_bytes())
    assert outcomes(ingest(inbox, catalog))["copy.txt"] is Outcome.IGNORED

def test_retry_reports_missing_and_keeps_entry(catalog, inbox):
    f = inbox / "s.pdf"; f.write_bytes(make_pdf([""])); ingest(inbox, catalog); f.unlink()
    assert [r.outcome for r in retry(catalog).results] == [Outcome.MISSING]
    assert len(catalog.quarantined_files()) == 1

def test_retry_with_quarantine_policy_ignores_accepted_documents(catalog, inbox):
    (inbox / "rib.txt").write_text(GOOD + " IBAN FR7630006000011234567890189", encoding="utf-8")
    ingest(inbox, catalog)
    assert retry(catalog, QualityConfig(sensitive_policy="quarantine")).results == ()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ingest.py tests/test_decisions.py -v` → FAIL.
- [ ] **Step 3: Implement** the spec's "Ingestion flow" (steps 1–8) in a private `_process(path: Path, catalog: Catalog, config: QualityConfig, *, force: bool) -> FileResult`. `ingest` → `force=False`; `retry` → `force=True` on `retry_candidates(under)`, a missing file → `MISSING` with catalog untouched; `apply_decision` → `record_decision`, then `IGNORE` → `record_ignored` (+ orphan cleanup) and `ACCEPT` → `_process(force=True)`. 0-byte file → `EMPTY` before extraction. Image suffix → `NEEDS_OCR`; other `UnsupportedFormat` → `UNSUPPORTED_FORMAT`; `EncryptedPdf` → `ENCRYPTED`; other `ExtractionError` → `CORRUPTED`. Content key = `content_hash(assessment.usable_text)`, plus `":" + file_hash` when `assessment.has_set_aside`. Outcome when stored: `UPDATED` only if the path was previously **indexed** with other bytes; otherwise `NEW` (a quarantined file that becomes indexed, e.g. after `accept`, is `NEW`); either becomes `PARTIAL` when the document has set-aside pages.
- [ ] **Step 4: Run** `uv run pytest -v` → all PASS.
- [ ] **Step 5: Commit** `feat: quarantine-aware ingestion with retry and user decisions`.

> `test_retry_with_quarantine_policy_ignores_accepted_documents` pins a real limit of the spec: retry only covers quarantined and partial files. Keep it and state the limit in `docs/02-quality-filtering.md`.

---

### Task 6: CLI

**Files:**
- Modify: `src/localmind/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 5 API.
- Produces commands: `ingest PATH... [--sensitive {flag,quarantine}]`; `quarantine` (list); `quarantine retry [PATH] [--sensitive ...]`; `quarantine accept PATH`; `quarantine ignore PATH`; `stats` with the new counters. `--sensitive` default comes from `sensitive_policy_from_env()`.

- [ ] **Step 1: Write failing tests**: `quarantine` lists a scanned PDF under `needs_ocr` and a partial PDF with `page 2`; a file containing the IBAN shows `iban @ page 1` and `"30006000011234567890189" not in` the output; `quarantine accept todo.txt` returns 0 and `stats` then contains `accepted: 1`; ingesting into a v1 catalog writes a line containing `rebuilt` to stderr; `quarantine accept` on a never-ingested path returns 2 with `not in catalog` on stderr.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v` → FAIL.
- [ ] **Step 3: Implement** formatting and argument mapping only.
- [ ] **Step 4: Run** `uv run pytest --cov=localmind --cov-report=term-missing` → all PASS, coverage ≥ 80 %.
- [ ] **Step 5: Commit** `feat: quarantine commands in the CLI`.

---

### Task 7: Documentation and end-to-end check

**Files:**
- Create: `docs/02-quality-filtering.md` (English, repo), `apprentissage/02-filtrage.md` (French, local, gitignored)
- Modify: `README.md` (roadmap row 2 → ✅ 2a), `apprentissage/README.md` (index rows)

- [ ] **Step 1: Write `docs/02-quality-filtering.md`**: page-level decision and why; rule table with thresholds and how to tune them; "regex finds, checksum confirms" with Luhn, mod-97 and NIR worked examples; the dedup key fix; decisions keyed by hash; the retry scope limit.
- [ ] **Step 2: Write `apprentissage/02-filtrage.md`** in the notebook format (one sentence / why / where in code / self-check questions) and add the index rows.
- [ ] **Step 3: End-to-end run** on a scratch inbox (never the user's real files) holding: clean txt, cp1252 txt with €, 0-byte file, image, zip, corrupted pdf, fully scanned pdf, partial pdf, `todo.txt` = "appeler", a 25-line log, a RIB with the test IBAN. Run `ingest`, `quarantine`, `quarantine accept todo.txt`, `stats`, `ingest`. Expected: each file gets the outcome this plan defines, no IBAN digits in any output, the last `ingest` shows only `unchanged`.
- [ ] **Step 4: Commit** `docs: stage 2a quality filtering` (the notebook is gitignored and stays local).
