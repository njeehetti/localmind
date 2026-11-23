# Stage 4: Chunking, embeddings and vector store (design)

Status: draft for approval, 2026-11-23. Builds on stage 3 (merged, schema v4).

## 1. Intent

localmind must answer questions whose answer sits in **one passage** of one
document: "which day is my phone subscription debited?", "what is the notice
period in my lease?". Stage 3 has one vector per document, made from its first
2000 characters: good for sorting, useless for finding a sentence on page 3.

Stage 4 cuts every accepted document into **chunks** (passages of a few
hundred characters, each tied to its page), embeds each chunk, stores the
vectors in the catalog, and finds the chunks closest to a question. Stage 5
builds the real search on it (folder filter, keywords/BM25, hybrid); stage 6
has a local LLM answer from the chunks with citations ("facture.pdf p.2").

Who uses it: the user from the CLI, and later **JARVIS** (voice assistant),
read-only, through the Python API. JARVIS asks while the user waits for a
spoken answer, so a search must stay well under a second.

Learning goal: the user is learning RAG. Stage 4 shows what a vector database
does (normalised vectors, dot product, top-k) by building a small one, rather
than hiding it behind a library.

## 2. Goals, non-goals, anti-goals

Goals
- Chunk every accepted document page-aware, so each chunk knows its page.
- Embed chunks with `nomic-embed-text` (task prefixes `search_document: ` /
  `search_query: `), store vectors in the catalog, per model.
- `search(question, k)` in the library: the k closest chunks with file, page,
  score, text. Fast enough for JARVIS.
- Chunking runs automatically at ingest (after classification), skippable,
  and can be re-run (`localmind index`).
- A minimal `localmind search "question"` to see it work (raw top-k chunks).

Non-goals (later stages)
- Folder filters, keyword/BM25 or hybrid search, reranking: stage 5.
- Generating an answer with an LLM: stage 6.
- **Fact extraction** (amounts, dates, suppliers, appointments) for "how much
  did I pay for electricity in 2026?" or "when are my next appointments?":
  a separate new stage after this one (agreed with the user, 2026-11-23).
  Chunk retrieval alone cannot sum over all invoices reliably.
- An approximate index (HNSW, IVF): not needed below ~500k chunks.

Anti-goals (failure even if it "works")
- A second store that can drift from the catalog (orphan vectors citing a
  deleted file, documents silently missing from search).
- A sensitive value (IBAN, card, NIR, password) sent to a model, printed, or
  stored in chunk text.
- Anything leaving the machine (Ollama on localhost only, no telemetry).
- Breaking `ingest` for users without Ollama: it must still work and print
  one hint line, as in stage 3.

## 3. Constraints (carried from earlier stages)

- Original files are never moved, modified or deleted.
- Sensitive spans (stage 2a, table `sensitive_spans`) are masked
  (`[IBAN]`, `[CARD]`, `[NIR]`, `[PASSWORD]`) before any text reaches a model.
- Only pages with status `ok` are chunked (set-aside and blank pages are not).
- The core stays an importable library; the CLI only prints.
- Tests never reach the real Ollama (autouse fixture in `tests/conftest.py`);
  end-to-end runs use generated scratch documents only.
- Windows: UTF-8 output and the real-console check from stage 3 apply.
- Hardware: RTX 4070 Laptop 8 GB, shared with JARVIS; the embedding model is
  small (~0.3 GB) and needs no unloading.

## 4. Approaches considered

Where vectors live (the user chose **A**, 2026-11-23, after a side-by-side of A and C):

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Vectors as float32 BLOBs in the SQLite catalog; brute-force cosine with **numpy** | One source of truth: deleting a document cascades to its chunks and vectors in one transaction; folder filtering is a SQL join (stage 5); SQLite FTS5 gives BM25 in the same file (stage 5). ~5 ms per query for 10k chunks. Adds numpy. |
| B | `sqlite-vec` extension | Same file, KNN in SQL; young extension, loading can fail on some Windows Python builds, hides the computation. |
| C | Chroma or LanceDB | ANN index, popular API; a second store to keep in sync with every stage 3 case (copies, edits, rename, move, orphans), heavy dependencies, Chroma's telemetry must be disabled. |

Pure-Python brute force was rejected: 2-4 s per query on 10k chunks, too slow
for a voice answer.

The search sits behind a small `VectorIndex` interface (add, remove, top-k), so a
LanceDB backend can be tried later as a bonus and compared at stage 7.

Chunking: page-aware, paragraph-aware chunks of ~800 characters with ~15 %
overlap (proposed in conversation). Fixed-size cuts that ignore paragraphs were
rejected (they split sentences). Whole pages were rejected (an invoice page
holds many facts; one vector per page blurs them).

## 5. Design

### 5.1 Chunking (pure function, no I/O)

Input: the masked text of each `ok` page of a document, with its page number.
Output: an ordered list of chunks `(page_no, seq, text)`.

- Target size `chunk_chars = 800`, overlap `overlap_chars = 120`, minimum
  `min_chars = 200` (a smaller tail merges into the previous chunk of the same
  page). Starting values; the spike (5.7) and stage 7 measure them.
- Split order: blank lines (paragraphs), then single line breaks (OCR text and
  tables often have no blank lines), then sentence ends, then a hard cut on a
  space as a last resort. Never cut inside a word.
- A chunk never spans two pages. A page shorter than `min_chars` is one chunk.
- Overlap is taken from the end of the previous chunk of the same page,
  starting on a word boundary.
- Masking happens **before** chunking (the stage 3 per-page masking), so chunk
  text never holds a sensitive value and offsets never need remapping.
  Consequence, accepted: a question about an IBAN cannot be answered by
  localmind, by design.

### 5.2 What gets embedded: a header per chunk

Embedded text = `"search_document: " + header + "\n" + chunk text`, with
header = the document's file name (first indexed file, as stage 3) and
`p. N`. A chunk "Montant : 84,20 €" then carries "facture_edf_mars.pdf".

The folder name is **not** in the header (change from the conversation, my
call): folders change through review, `rename` and `move`, which would make
every vector stale; stage 5 filters by folder with a SQL join instead.

Questions are embedded as `"search_query: " + question`. The embedding client
takes the prefix as a parameter; stage 3 keeps `"classification: "`.

### 5.3 Storage (schema v5)

New tables, added by a v4 → v5 migration that only creates tables (like v3 → v4):

```
chunks         id, document_id -> documents (ON DELETE CASCADE), page_no, seq,
               text (masked), chunker (version string of the chunking parameters)
               UNIQUE (document_id, seq)
chunk_vectors  chunk_id -> chunks (ON DELETE CASCADE), model, vector (float32 BLOB,
               L2-normalised), PRIMARY KEY (chunk_id, model)
```

- A document whose text changes becomes a new document (stage 1 `content_key`);
  the old one is deleted as an orphan and its chunks go with it: nothing to sync.
- Copies share a document, so they share chunks: no duplicate hits.
- `chunks` and `chunk_vectors` are rebuildable: they join the tables a v1
  rebuild may drop. Never `decisions`, `ocr_cache`, `folder_decisions`.
- Changing the embedding model adds vectors under the new model name (never a
  mix in one search). Changing the chunking parameters (different `chunker`
  string) means re-chunking: `localmind index --all`.

### 5.4 Indexing run

`index(catalog, embedder, under=None, redo=False, on_document=None) -> IndexReport`:

1. Documents to index: accepted documents with no chunks for the current
   `chunker`, or chunks without a vector for the current model (`redo` = all).
2. Chunk them all, then embed **all chunks of the run in batches of 32** in one
   pass (one model, no LLM switching).
3. Write each document's chunks and vectors in one transaction: a crash leaves
   a document either fully indexed or not at all.
4. Ollama unreachable or model missing: stop, keep what is done, return the hint
   (same `ClassifierUnavailable` hint style; the hint says `localmind index`).

CLI: `ingest` runs it after classification (`--no-index` to skip);
`localmind index [path] [--all]`; `localmind stats` shows chunk counts.

### 5.5 Search

`search(catalog, question, k=5, embedder=None) -> list[Hit]`,
`Hit(file_path, page_no, score, text, document_id, chunk_id)`.

- Embed the question (`search_query: `), dot product with the matrix of all
  vectors of the current model (numpy, float32), `argpartition` for top-k,
  sorted by score.
- `VectorIndex` keeps the matrix in memory between searches (JARVIS asks many
  questions in one process) and reloads when the catalog changed (the builder
  picks the change signal, e.g. `PRAGMA data_version` or a count/max-id check).
- Masked text only: hits never show a sensitive value.
- Exported from `localmind/__init__.py` (`search`, `index`, `Hit`).
- CLI `localmind search "question" [-k 5]`: one line per hit,
  `0.83  facture_edf_mars.pdf p.1  "…excerpt…"`. Without Ollama: the hint line,
  exit code 2.

### 5.6 Errors

- A page with no text after masking: no chunk, no error.
- A document with no `ok` page: nothing to index, counted in the report.
- An embedding answer with the wrong number of vectors: the run stops with the
  hint, as in stage 3.

### 5.7 Spike gate, before building (like stage 3)

On generated French scratch documents (invoices, phone and energy contracts
with a debit day, payslips, a lease, an appointment letter): ask ~10 questions
of the kind the user gave ("quel jour suis-je prélevé pour mon abonnement
téléphonique ?") and measure whether the right chunk is in the top 3, for
chunk sizes 500 / 800 / 1200 and with and without the header. Keep 800 + header
unless the numbers say otherwise; record the table in the stage doc.

### 5.8 Calls made after the builder check (author's calls, open to correction)

- **Which documents the automatic index covers**: every accepted document still
  missing chunks or vectors, not only those under the ingested paths. Documents
  created by `quarantine retry`, `decide accept` or `run_ocr` are therefore
  picked up at the next `ingest` or `index` instead of staying invisible.
- **One hint line**: if classification already found Ollama unreachable, the
  index step is skipped and the single hint names both commands
  (`localmind classify` and `localmind index`). With `--no-classify` the index
  still runs (and prints its own hint if Ollama is down).
- **Cold embedding model**: search and index calls pass `keep_alive: "30m"`
  (nomic-embed-text holds ~0.3 GB); `VectorIndex` offers `warm()` (one tiny
  embed) so JARVIS can load the model at start-up. The "well under a second"
  budget applies with the model loaded; a cold first question may take 1-3 s.
- **Spike rule (blocking gate)**: the chosen setting must put the right chunk in
  the top 3 for at least 8 of 10 questions. Another size replaces 800 only if
  it wins by 2 or more questions. If every setting falls below 8/10, building
  stops and the numbers go back to the user.

### 5.9 The user's answers (2026-11-23)

- **Searching while ingest writes**: not supported in stage 4; the user accepts
  that a search may wait for an ingest. No WAL, no read-only open path now: it
  belongs to the JARVIS integration. Search relies on SQLite's busy timeout
  (Python's default 5 s) and never crashes on a lock: it reports "the catalog is
  busy, try again". (The user's condition, JARVIS keeping its conversations as
  Markdown files, is a JARVIS feature, noted for that project; such `.md` files
  could later be ingested by localmind like any other document.)
- **Deleted files never appear in results**: "if I delete it, it is for a good
  reason". At search time each hit's files are checked on disk; the hit names an
  existing copy, and a document with no existing file is skipped (more
  candidates are taken so that k hits are still returned when possible).
- **Forgetting their content**: a new `localmind prune [path]` command removes
  from the catalog the files that no longer exist, and the documents left with
  no file (their pages, chunks, vectors and classification go with them by
  cascade). Explicit and user-triggered, so a file on an unplugged drive is never
  forgotten by accident: `prune` lists what it will forget and asks `[y/N]`
  (non-interactive: `--yes` required). It never touches files on disk. Kept:
  `decisions`, `ocr_cache`, `folder_decisions` (the user's choices; a file put
  back later finds them again).

## 6. Deliverables

- Code: chunker module, catalog v5 tables and methods, indexing run, vector
  index + search (skipping deleted files), embedding prefix parameter, CLI
  (`index`, `search`, `prune`, `--no-index`, stats), public API exports, numpy dependency.
- Tests: unit (chunker edge cases, migration v4 → v5, cascade on orphan
  deletion, model separation, top-k order, masking in chunks), CLI with the
  fake embedder, one live test marked like `test_ollama_live`.
- Docs: `docs/04-chunking-vectors.md` (English, same style as stage 3, with the
  spike table and an end-to-end example), README roadmap row 4.

## 7. Left to the builder

- Module and class names beyond `index`, `search`, `Hit`, `VectorIndex`.
- The change signal for reloading the in-memory matrix.
- The exact sentence-end regex and the overlap word-boundary rule.
- Whether `IndexReport` reuses stage 3's summary style (it should look alike).
