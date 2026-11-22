# Stage 2a: Quality filtering & quarantine

> Goal: decide what is usable, set the rest aside **with a reason**, never lose
> anything, and let the user have the last word.

## 1. Decide per page, not per document

A 30-page PDF with 2 scanned pages should not be thrown away: its 28 good pages are
indexed, the 2 others are **set aside** with their page number and reason. Every file
ends up in one of three states:

| State | Meaning | Example |
|-------|---------|---------|
| accepted | every page is usable | a clean invoice |
| partial | some pages set aside | a contract with 2 scanned annex pages |
| quarantined | nothing usable, or the file itself is the problem | a corrupted PDF, a `.zip`, a fully scanned PDF |

Nothing is deleted. `localmind quarantine` lists everything that was set aside.

## 2. Two levels of rules

**Level A, technical**: the file or page cannot be used at all. A user `accept`
cannot override these: there is no text to accept.

| Rule | Scope | Reason |
|------|-------|--------|
| unknown extension | file | `unsupported_format` |
| image (`.png`, `.jpg`...) | file | `needs_ocr` (stage 2b will read it) |
| extraction fails | file | `corrupted` |
| PDF locked by a real password | file | `encrypted` |
| 0 bytes, or text file with only whitespace | file | `empty` |
| PDF page with < 20 visible characters | page | `needs_ocr` |
| PDF page that draws an image and has < 200 visible characters (a scan with a thin text layer such as "Scanned with CamScanner") | page | `needs_ocr` |

**Level B, quality**: there is text, but it would pollute search. `accept` overrides them.

| Rule | Scope | Default |
|------|-------|---------|
| fewer than 3 words in the whole document | document | `too_short` |
| less than 70 % "normal" characters | page | `garbled` |
| ≥ 20 lines, ≥ 60 % of them start with date + time, a time, or a log level (a date alone is not enough: diaries and expense lists start with dates) | document | `machine_log` |

"Normal" characters are letters (any script), digits, spaces, punctuation and symbols
such as `€`, `°`, `+`. Control characters and `U+FFFD` (the replacement character left
behind by a failed decoding) are not.

### Thresholds are trade-offs

Every threshold lives in `QualityConfig` (`src/localmind/quality.py`). None of them is
"correct": each one trades false positives against false negatives. Example from the
end-to-end test: a contract's last page reads only *"Signature des parties"* (19 visible
characters) and is set aside as `needs_ocr` because the limit is 20. Lower the limit and
pages holding only a page number or a scanner header slip through as "text". The way to
choose is to label a small sample of real files and measure, which is exactly what
stage 7 (evaluation) is about.

## 3. Sensitive data: regex finds, checksum confirms

IBANs, payment cards and French social security numbers (NIR) carry a **check value**
computed from their other digits. A regex alone ("16 digits") would match every order
number; the checksum removes most of those false positives.

| Kind | Check | A random number passes by chance |
|------|-------|----------------------------------|
| IBAN | ISO 7064 mod 97: move the first 4 chars to the end, letters → numbers (A=10…Z=35), result mod 97 = 1 | ~1 in 97 |
| Card | Luhn: from the right, double every 2nd digit (minus 9 if > 9), total mod 10 = 0 | 1 in 10 |
| NIR | key = 97 − (first 13 digits mod 97) | ~1 in 97 |

Luhn on `4539 5787 6362 1486`: the weighted sum is 80, a multiple of 10, so it is valid.
Change the last digit to 7 and the sum becomes 81: rejected. That is why a long order
number is almost never mistaken for a card, and why numbers are never "trimmed" to find
a valid prefix (each shorter prefix would be one more 1-in-10 chance of a false alarm).

Card candidates must also look like real cards: contiguous digits, `4-4-4-4(-3)` or
`4-6-5` with one consistent separator. Allowing a space anywhere made about 1 in 10
pairs of dates on a bank statement (`2024-03-15 2024-03-17`) pass as a card.

Password lines (`password: ...`, `mot de passe : ...`, `mdp=...`) have no checksum; they
are matched only at the start of a line followed by `:` or `=`, so "Mot de passe oublié ?"
is not flagged.

Default policy: **flag**. The document is kept and the catalog records *where* the data
is (kind and page, never the value). `--sensitive quarantine` (or
`LOCALMIND_SENSITIVE_POLICY=quarantine`) sets such documents aside instead. The CLI never
prints a sensitive value.

## 4. The duplicate bug this stage fixes

In stage 1, every text-less file had the same normalized text (the empty string), so two
unrelated scans were reported as duplicates. Now:

- a file with no usable text is quarantined and gets no document, so it is never a
  "duplicate" of another empty file;
- a document with set-aside pages has a dedup key that also includes its file hash.
  Otherwise two PDFs sharing a cover page but with different scanned pages would be
  merged and the second one's set-aside pages lost.

## 5. User decisions, keyed by content

| Command | Effect |
|---------|--------|
| `localmind quarantine accept FILE` | keep this content despite level B rules (and the sensitive policy) |
| `localmind quarantine ignore FILE` | remove it and never propose these bytes again |
| `localmind quarantine retry [PATH]` | re-judge quarantined and partial files with the current settings |

Decisions are stored by **file hash**, not path: they survive renames and moves, and a
copy of an ignored file is ignored too, including copies already in the catalog. They apply immediately (a decision that waited
for the next `ingest` would never apply: the file would be skipped as unchanged).

Known limit: `retry` only revisits quarantined and partial files. Switching the
sensitive policy to `quarantine` does not retroactively quarantine documents that were
already accepted.

## 6. Schema v2

The catalog gained `pages`, `sensitive_spans` and `decisions` tables. Pages and spans are
deleted with their document (`ON DELETE CASCADE`). A stage 1 catalog is detected through
`PRAGMA user_version` and rebuilt: it is only an index of files that are still on disk.

## 7. Try it

```bash
uv run localmind ingest path/to/folder
uv run localmind quarantine
uv run localmind quarantine accept path/to/todo.txt
uv run localmind stats
```
