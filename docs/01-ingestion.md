# Stage 1: Ingestion

> Goal: turn "a pile of files" into "clean text with an identity card", and notice
> when two files hold the same information.

Ingestion does **not** understand anything yet. It is plumbing, but it is where most
RAG systems silently fail: if the extracted text is garbage, every later stage
(search, answers) is garbage too.

## 1. Files vs documents: storing pointers, not copies

localmind never moves or copies your files. The catalog (SQLite) stores **pointers**:

```
files                                  documents
path (unique)                          content_hash (unique)
file_hash   (fingerprint of bytes)     text, kind, page_count
document_id ─────────────────────────► id
```

- A **document** is a piece of information (a unique normalized text).
- A **file** is a place on disk where it lives.
- Several files can point to one document. That *is* duplicate detection.

Like a pointer in C, a stored path can become **dangling** if you move the file. The
fix is the same idea Git uses: **content addressing**. A file is identified by the hash
of its content, so it can be found again wherever it is.

## 2. Two levels of duplicates

| Level | Example | Detected by |
|-------|---------|-------------|
| Exact copy | `invoice.pdf` / `invoice (1).pdf` | SHA-256 of the **bytes** (`file_sha256`) |
| Same content, different wrapping | same note in UTF-8 and Windows-1252, CRLF vs LF, extra spaces | SHA-256 of the **normalized text** (`content_hash`) |
| Near-duplicate | v1 / v2 of a contract | *later, with embeddings* |

Normalization (`normalize_text`) removes differences that change bytes but not meaning:

- **Unicode NFC**: `é` can be one code point (`U+00E9`) or `e` + combining accent
  (`U+0065 U+0301`). Both look the same and must hash the same.
- **Whitespace**: line endings, tabs, repeated spaces.
- **Case is kept**: "Apple" the company is not "apple" the fruit.

Checks run cheapest first: same path and same bytes → skip. Known bytes → link without
even extracting. Only then extract and compare the normalized text.

## 3. Text files: the encoding trap

A `.txt` file is just bytes. The encoding tells how to read them, and it is **not stored
in the file**. Read Windows-1252 bytes as UTF-8 and `Reçu` becomes `ReÃ§u`.

Strategy (`extractors._decode`):

1. **BOM** (byte order mark) at the start? It tells the encoding (UTF-8-sig, UTF-16).
2. Try **UTF-8**. It is strict: random bytes are almost never valid UTF-8, so if
   decoding succeeds it is very likely right.
3. Then **cp1252** (old Windows default), which accepts nearly anything, so it must
   come *after* UTF-8.
4. **latin-1** as the last resort: it maps every byte, so it never fails.

## 4. PDFs: not text, but drawing instructions

A PDF page does not contain sentences. It contains operators like:

```
BT /F1 12 Tf 72 720 Td (Invoice 2024) Tj ET
```

"Begin text, font F1 at 12pt, move to (72, 720), draw these glyphs, end." The extractor
has to rebuild words, lines and reading order from positions, which is why multi-column
layouts and tables often come out scrambled. See [`tests/pdf_factory.py`](../tests/pdf_factory.py),
which writes a valid PDF by hand.

Consequences handled here:

- **Scanned PDFs** have pages but no text layer: extraction returns empty text
  (stage 2 will quarantine them; OCR could come later).
- **Page breaks** are kept as form feeds (`\f`) so later stages can cite page numbers.
- **Library choice**: `pypdf` (BSD license, pure Python). PyMuPDF extracts better
  but is AGPL, a real constraint for a project you publish.

## 5. Try it

```bash
uv run localmind ingest path/to/folder
uv run localmind ingest path/to/folder   # 2nd run: everything "unchanged"
uv run localmind duplicates
```
