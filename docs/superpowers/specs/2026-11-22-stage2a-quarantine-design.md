# Stage 2a: Quality filtering & quarantine (design)

Status: draft v2 (after builder check), 2026-11-22. Builds on stage 1 (commit `ac744b9`).

## Intent

localmind accepts "anything" dropped in. Stage 2a decides **what is usable**, sets aside
what is not **with a reason**, and never loses or deletes anything. It is also the
foundation for stage 2b (neural OCR): pages set aside as "needs OCR" are exactly what
OCR will later recover with a retry.

It also fixes a stage 1 bug: every text-less file (scanned PDF, empty `.txt`) has the
same normalized text (empty string), hence the same `content_hash`, so two unrelated
scans are reported as duplicates of each other.

## Goals

- Decide at **page level**, not document level. A 30-page PDF with 2 unreadable pages
  keeps its 28 good pages; the 2 others are set aside with page number and reason.
- Three document outcomes: **accepted** (all pages usable), **partial** (some pages set
  aside), **quarantined** (nothing usable, or file-level problem).
- Rule levels:
  - **A, technical** (file or page is unusable): unsupported format, corrupted,
    encrypted, empty file, page without text layer.
  - **B, quality** (text exists but is worthless): document too short, garbled page,
    machine log file.
- **Sensitive data detection** (IBAN, payment card, French social security number,
  password lines). Default policy: **flag** the span, keep the document. Optional
  policy: quarantine the document.
- Persist everything in the catalog: quarantined files, set-aside pages, reasons,
  sensitive spans, user decisions.
- CLI to inspect and act: list, retry, accept, ignore.
- Pedagogical: each rule and threshold is explained in `docs/02-quality-filtering.md`,
  and thresholds are exposed as named, documented constants.

## Non-goals

- OCR itself (stage 2b). 2a only *marks* pages and images as `needs_ocr`.
- Semantic refusal "this content is irrelevant to me" (level C): stage 3, same LLM call
  as classification.
- Near-duplicate detection (stage 4, embeddings).
- Redacting sensitive values from answers (stage 6). 2a only records where they are.

## Anti-goals (failure even if it "works")

- Deleting, moving or modifying an original file.
- Refusing a whole document because of a small bad part.
- Printing or logging sensitive values (CLI shows kind + page, never the value).
- Silent refusals: every set-aside item has a machine-readable reason.
- User decisions (accept / ignore) lost on the next ingestion.

## Constraints

- Python ≥ 3.11, no new heavy dependency (no AI in 2a). `pypdf` stays the PDF reader,
  installed as `pypdf[crypto]` (adds the small `cryptography` package) so AES-encrypted
  PDFs with an empty password are readable rather than misclassified.
- Fully local. Immutable dataclasses, small modules, tests first (project rules).
- Catalog is SQLite at `~/.localmind/catalog.db`.

## Approaches considered

| | Option | Trade-off | Decision |
|---|---|---|---|
| Granularity | Document-level accept/refuse | Simple, but loses good pages of mixed documents | ✗ |
| | **Page-level assessment** | Slightly richer schema; matches how PDFs and OCR work; reuses page breaks kept in stage 1 | ✓ chosen by user |
| Rules | ML quality classifier | Needs labelled data we don't have; opaque | ✗ |
| | **Pure-function rules with named thresholds** | Transparent, testable, teachable; thresholds need tuning | ✓ |
| Key for user decisions | File path | Breaks when the file is moved or renamed | ✗ |
| | **File hash (content addressing)** | Survives moves; a copy of an ignored file is also ignored, which is the intent | ✓ |
| Where quarantine lives | Separate `quarantine` table | Two places describing one file | ✗ |
| | **Status + reason on `files`, page status on `pages`** | One source of truth per level | ✓ |

## Design

### Reasons (machine-readable codes)

`unsupported_format`, `needs_ocr`, `corrupted`, `encrypted`, `empty`, `too_short`,
`garbled`, `machine_log`, `sensitive` (only when policy = quarantine).

Images (`.png .jpg .jpeg .tif .tiff .bmp .webp`) are quarantined with `needs_ocr`, not
`unsupported_format`, so stage 2b picks them up. Other unknown types are
`unsupported_format`.

### Rules

| Level | Rule | Scope | Default threshold |
|---|---|---|---|
| A | Unknown extension | file | - |
| A | Extraction fails | file → `corrupted` | - |
| A | PDF encrypted and not openable with an empty password | file → `encrypted` | - |
| A | File is 0 bytes, or a **text file** whose content is only whitespace | file → `empty` | - |
| A | Page has fewer than `MIN_PAGE_CHARS` non-whitespace characters | page → `needs_ocr` (PDF only) | 20 |
| B | Whole usable text has fewer than `MIN_DOC_WORDS` words | document → `too_short` | 3 |
| B | Share of "normal" characters (letters any script, digits, whitespace, common punctuation and symbols such as €, %, °) below `MIN_NORMAL_CHAR_RATIO`, or presence of U+FFFD replacement chars above that share | page → `garbled` | 0.70 |
| B | At least `LOG_MIN_LINES` non-empty lines and at least `LOG_LINE_RATIO` of them look like log lines (timestamp and/or level such as INFO/WARN/ERROR/DEBUG at line start) | document → `machine_log` | 20 lines, 0.60 |

PDFs never get `empty` from extraction: a text-less PDF goes through the page rule, so
every page is `needs_ocr` and stage 2b can recover it. U+FFFD replacement characters
count as "abnormal" characters in the garbled ratio (no separate threshold).

A text file is one page. If every page of a document is set aside, the file is
quarantined with the most common page reason. Rules are pure functions
`(text) -> Reason | None` in `quality.py`, thresholds in a frozen `QualityConfig`
dataclass with the defaults above.

### Sensitive data (`sensitive.py`)

Detectors return spans `(kind, start, end)` within a page. Each uses a checksum where
one exists, which is the pedagogical point (regex finds candidates, checksum removes
false positives):

- `iban`: regex + ISO 7064 mod-97 check.
- `card`: 13–19 digits with optional spaces/dashes + Luhn check.
- `fr_nir` (French social security number): 13 digits + 2-digit key, key = 97 − (n mod 97).
- `password`: line with `password|passwd|pwd|mot de passe|mdp` followed by `:` or `=`
  and a non-empty value.

Policy `sensitive_policy = "flag"` (default) or `"quarantine"`, set by
`QualityConfig.sensitive_policy`, the `--sensitive {flag,quarantine}` option of `ingest`
and `quarantine retry`, or the `LOCALMIND_SENSITIVE_POLICY` environment variable (option
wins over variable). A user `accept` overrides a `sensitive` quarantine. A policy change
applies to already-ingested files only through `quarantine retry`. Values are never
stored separately, logged or printed.

### Data model (schema version 2)

```
files     id, path UNIQUE, file_hash, size_bytes, encoding,
          status  ('indexed' | 'quarantined' | 'ignored'),
          reason  (NULL unless quarantined/ignored),
          document_id NULL when not indexed,
          ingested_at
documents id, content_hash UNIQUE, kind, char_count, page_count, created_at
pages     id, document_id, page_no (1-based), text,
          status ('ok' | 'set_aside'), reason
sensitive_spans  id, page_id, kind, start, end
decisions file_hash PRIMARY KEY, decision ('accept' | 'ignore'), decided_at
```

- `documents.text` moves to `pages.text`. Document status is derived: `partial` if any
  page is `set_aside`, else `accepted`.
- **Dedup key fix**: `content_hash` is computed on usable pages only. If a document has
  **any set-aside page (whatever the reason)**, its dedup key also includes its
  `file_hash`. Consequence: page status stays on the document, and a document is only
  ever shared by files with identical set-aside pages (in practice, exact byte copies).
  Without this, two PDFs with the same good page but different scanned or garbled pages
  would wrongly be duplicates and the second one's set-aside pages would be lost.
  Quarantined files have no document, so text-less files are never "duplicates" of each
  other (exact byte copies still are, via `file_hash`).
- **Migration**: `PRAGMA user_version`. The catalog is a rebuildable index (originals are
  untouched and no user decisions existed before v2), so a v1 catalog is dropped and
  recreated, with a printed notice asking to re-run `ingest`.

### Ingestion flow (changes to `ingest.py`)

1. Hash. Path known with same hash → `unchanged` (unless retrying).
2. `decisions` has `ignore` for this hash → `ignored`.
3. Known bytes → `duplicate` (same as today). An exact copy of a quarantined file gets
   its own `files` row with the same `quarantined` status and reason.
4. Extract (extractor now returns `pages: tuple[str, ...]`).
5. File-level rules A → `quarantined`.
6. Page rules (A `needs_ocr`, B `garbled`), then document rules (B `too_short`,
   `machine_log`). A file with `accept` decision skips B rules (A cannot be overridden:
   there is no text to accept).
7. Sensitive scan on usable pages → spans; policy decides flag vs quarantine.
8. Dedup on content key → `duplicate`, else store → `new` / `partial` / `updated`.

`Outcome` becomes: `new`, `partial`, `updated`, `duplicate`, `unchanged`,
`quarantined`, `ignored`. `unsupported` and `failed` disappear (now `quarantined` with a
reason). Stage 1 tests are updated accordingly.

### CLI

```
localmind quarantine                 # quarantined files + set-aside pages, grouped by reason
localmind quarantine retry [PATH]    # re-assess quarantined files and partial documents
localmind quarantine accept PATH     # override B rules for this content (by file hash)
localmind quarantine ignore PATH     # never propose this content again (by file hash)
localmind stats                      # adds: accepted / partial / quarantined / ignored, sensitive flags
```

Sensitive flags are shown as `kind @ page N`, never with the value.

Command semantics:

- `accept` and `ignore` record the decision (by file hash) **and apply it immediately**
  by re-assessing that file; they do not wait for the next `ingest` (which would skip
  the file as `unchanged`). Decisions are also checked on every later ingestion.
- `ignore` on an indexed or partial file sets `files.status = 'ignored'` and unlinks it
  from its document; the document (pages, spans) is deleted only if no other file
  points to it, through the existing orphan cleanup.
- `retry [PATH]` re-extracts and re-assesses, with the current config and decisions and
  bypassing `unchanged`, every quarantined file and every file of a partial document.
  `PATH` (file or folder) restricts it to files under that path. Each file is
  re-assessed on its own. A file that is missing on disk is reported as `missing` and
  left untouched in the catalog; a file whose bytes changed is processed like a normal
  update. In 2a, retry is useful after a threshold, policy or decision change; in 2b it
  is how OCR recovers pages.

### Public API additions

`QualityConfig`, `Reason`, and `Catalog` methods backing the CLI (`quarantined()`,
`set_aside_pages()`, `record_decision()`), exported from `localmind`.

## Left to the builder

- Exact regex patterns and the "normal character" set, as long as the tests in the plan
  pass and the doc explains them.
- Output formatting of the CLI.
- Module split beyond `quality.py` and `sensitive.py`.
