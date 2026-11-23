# Stage 4 Chunking, Embeddings and Vector Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut every accepted document into page-aware chunks, embed them with `nomic-embed-text`, store the vectors in the SQLite catalog, find the chunks closest to a question with numpy, and let the user forget deleted files (`prune`).

**Architecture:** A pure chunker (`chunking.py`); a `ChunkStore` mixin (`catalog_chunks.py`, schema v5) holding `chunks` and `chunk_vectors`; an indexing run (`indexing.py`) that batches all chunk embeddings of a run; an in-memory `VectorIndex` (`search.py`, numpy brute force) that reloads when the catalog changes and skips files deleted from disk; `prune.py` for forgetting missing files; CLI wiring in a new `cli_search.py`. `ingest()` itself stays unchanged: the CLI (and later JARVIS) call `index()` after it, as with `classify()`.

**Tech Stack:** Python ≥ 3.11 (dev 3.13.5), SQLite, numpy ≥ 2.0 (new), Ollama on `http://localhost:11434` (`nomic-embed-text`), pytest, uv. Run everything from the repository root with `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-23-stage4-chunking-vectors-design.md` (sections 5.8-5.9 hold the post-check calls and the user's answers).

## Global Constraints

- One new dependency: `numpy>=2.0`, imported only in `search.py`.
- Originals are never moved, modified or deleted; `prune` only deletes catalog rows.
- Chunk text is cut from **masked** page text (`[IBAN]`, `[CARD]`, `[NIR]`, `[PASSWORD]`); no sensitive value is stored in `chunks`, sent to a model, or printed.
- Only pages with status `ok` are chunked.
- `ChunkConfig` defaults: `chunk_chars=800`, `overlap_chars=120`, `min_chars=200`; version string `f"para-v1:{chunk_chars}/{overlap_chars}/{min_chars}"` (values may change only through the Task 1 gate).
- Embedded document text: `"search_document: " + f"{file_name} p. {page_no}" + "\n" + chunk text`; question: `"search_query: " + question`; stage 3 keeps `"classification: "`. Vectors L2-normalised, float32 bytes, stored with the model name. Embed calls from index and search pass `keep_alive: "30m"`. Batches of 32.
- A document's chunks and vectors are written in one transaction.
- `chunks` and `chunk_vectors` join the tables a v1 rebuild may drop; never `decisions`, `ocr_cache`, `folder_decisions`.
- Tests never reach the real Ollama (autouse fixture); end-to-end runs use generated scratch documents only.
- CLI copy (exact):
  - index summary: `Indexed {n} document(s): {c} chunk(s)`; if any: `  {e} document(s) with no usable text`
  - index hints: `Ollama not reachable at {url}: start it (ollama serve), then: localmind index` / `Model {model} is missing: ollama pull {model}, then: localmind index`
  - ingest when classification found Ollama unavailable: indexing is skipped and the one hint line ends with `then: localmind classify && localmind index`
  - search hit: `{score:.2f}  {file name} p.{page}  "{excerpt}"` (excerpt = chunk text, whitespace collapsed, first 100 characters, `…` appended when cut)
  - search with nothing indexed: `Nothing indexed yet: localmind index`; busy catalog: `error: the catalog is busy, try again` (stderr, exit 2); Ollama unavailable: the hint with `then: localmind search` (exit 2)
  - prune: `{n} file(s) no longer exist:` then `  {path}` per file, prompt `Forget them? [y/N] `, result `Forgot {f} file(s) and {d} document(s).`; nothing: `Nothing to prune.`; non-interactive without `--yes`: `{n} file(s) no longer exist: localmind prune --yes`
  - stats rows: `chunks` and `indexed docs`

## Review Focus

1. Ollama stopping in the middle of an index run must leave each document either fully indexed or untouched, never half — pinned in Task 4 (`test_ollama_dropping_mid_run_keeps_whole_documents`).
2. An IBAN in a page must appear in no chunk text, no embedded text and no search output — pinned in Task 4 (`test_chunks_never_hold_sensitive_values`) and Task 7 (`test_search_output_masks_sensitive_values`).
3. A file deleted from disk after indexing must never be cited; if an exact copy still exists, the copy is named — pinned in Task 5 (`test_deleted_file_is_skipped_and_copy_is_named`).
4. A long-lived `VectorIndex` (JARVIS) must see documents indexed or pruned after it was built — pinned in Task 5 (`test_index_reloads_after_catalog_change`).
5. A file edited in place must lose its old chunks: no hit may quote the old text — pinned in Task 4 (`test_edited_file_replaces_its_chunks`).

---

### Task 1: Spike gate — chunk size and header, measured on French documents

**Files:**
- Create (scratch, not committed): `<scratchpad>/spike4.py`
- Modify: `docs/superpowers/specs/2026-11-23-stage4-chunking-vectors-design.md` (results table under 5.7)

- [ ] **Step 1: Write `spike4.py`.** It generates ~10 French scratch documents of 1-3 pages (2 electricity invoices, a phone subscription contract stating the debit day, an internet contract with a different debit day, 2 payslips, a lease with a notice period, a medical appointment letter, a car insurance certificate, a health refund) and 10 questions, each with the document and a phrase that must appear in the right chunk. Include: "quel jour suis-je prélevé pour mon abonnement téléphonique ?", "quand est mon prochain rendez-vous chez le médecin ?", "quel est le préavis de mon bail ?", "combien m'a coûté l'électricité en mars ?". A throwaway chunker inside the script is fine.
- [ ] **Step 2: Run it against the real Ollama** for chunk sizes 500 / 800 / 1200 (overlap 15 %), each with and without the `"{file} p. N"` header, using the prefixes of Global Constraints.
Expected: a 6-row table of top-3 hits out of 10.
- [ ] **Step 3: Apply the gate (spec 5.8).** The kept setting needs ≥ 8/10. Another size replaces 800 only if it wins by ≥ 2 questions; the header stays unless dropping it wins by ≥ 2. If every setting is below 8/10: **stop and report the table to the user**.
- [ ] **Step 4: Record the table and the decision** in spec 5.7; update the `ChunkConfig` defaults in this plan's Global Constraints if the gate changed them.
- [ ] **Step 5: Commit** — `git add docs/superpowers && git commit -m "docs: stage 4 spike results"`

### Task 2: Chunker (pure)

**Files:**
- Create: `src/localmind/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `ChunkConfig(chunk_chars=800, overlap_chars=120, min_chars=200)` frozen dataclass with property `version -> str`; `Chunk(page_no: int, seq: int, text: str)` frozen; `chunk_pages(pages: Sequence[tuple[int, str]], config: ChunkConfig = ChunkConfig()) -> list[Chunk]` (`seq` counts from 0 across the whole document); `embed_text(file_name: str, chunk: Chunk) -> str` → `f"{file_name} p. {chunk.page_no}\n{chunk.text}"` (no task prefix).

- [ ] **Step 1: Write the failing tests**

```python
def test_short_page_is_one_chunk():
    assert chunk_pages([(1, "Facture EDF. Montant : 84,20 €")]) == [Chunk(1, 0, "Facture EDF. Montant : 84,20 €")]

def test_chunks_never_span_pages():
    chunks = chunk_pages([(1, "a " * 600), (2, "b " * 600)])
    assert {c.page_no for c in chunks if "a" in c.text} == {1}
    assert [c.seq for c in chunks] == list(range(len(chunks)))

def test_cuts_on_paragraphs_then_lines_then_sentences():
    para = "Premier paragraphe. " * 25          # ~500 chars
    chunks = chunk_pages([(1, para + "\n\n" + para)], ChunkConfig(chunk_chars=600, overlap_chars=0))
    assert chunks[0].text.strip() == para.strip()
    ocr = "\n".join(f"Ligne {i} du relevé de compte" for i in range(60))   # no blank line
    assert all(c.text.split("\n")[-1].startswith("Ligne") for c in chunk_pages([(1, ocr)]))

def test_never_cuts_inside_a_word():
    text = " ".join(["anticonstitutionnellement"] * 100)
    for c in chunk_pages([(1, text)]):
        assert set(c.text.split()) == {"anticonstitutionnellement"}

def test_size_and_overlap():
    text = ". ".join(f"Phrase numéro {i} du contrat" for i in range(200))
    chunks = chunk_pages([(1, text)])
    assert all(len(c.text) <= 800 + 120 for c in chunks)
    assert chunks[1].text[:40] in chunks[0].text          # overlap from the previous chunk

def test_small_tail_merges_into_previous_chunk():
    chunks = chunk_pages([(1, "x " * 450)], ChunkConfig(chunk_chars=800, overlap_chars=0, min_chars=200))
    assert len(chunks) == 1 or len(chunks[-1].text) >= 200

def test_blank_page_text_gives_no_chunk():
    assert chunk_pages([(1, "   \n\n ")]) == []

def test_version_names_the_parameters():
    assert ChunkConfig().version == "para-v1:800/120/200"

def test_embed_text_has_file_and_page_header():
    assert embed_text("facture.pdf", Chunk(2, 0, "Total")) == "facture.pdf p. 2\nTotal"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_chunking.py -q` — Expected: FAIL (`ModuleNotFoundError: localmind.chunking`).
- [ ] **Step 3: Implement `chunking.py`.** Recursive split: blank lines, then `\n`, then sentence ends (`(?<=[.!?])\s+`), then the last space before the limit; pack pieces greedily up to `chunk_chars`; overlap = the last `overlap_chars` of the previous chunk of the same page, advanced to the next word start; tail under `min_chars` merges into the previous chunk of that page.
- [ ] **Step 4: Run** `uv run pytest tests/test_chunking.py -q` — Expected: all pass.
- [ ] **Step 5: Commit** — `feat: page-aware paragraph chunker`

### Task 3: Embedding prefix and keep-alive in the Ollama client

**Files:**
- Modify: `src/localmind/ollama.py`, `tests/fake_classifier.py`
- Test: `tests/test_ollama.py`, `tests/test_ollama_live.py`

**Interfaces:**
- Produces: constants `CLASSIFY_PREFIX = "classification: "` (rename of `EMBED_PREFIX`, keep `EMBED_PREFIX` as an alias), `DOCUMENT_PREFIX = "search_document: "`, `QUERY_PREFIX = "search_query: "`, `SEARCH_KEEP_ALIVE = "30m"`; `Classifier.embed(texts, prefix: str = CLASSIFY_PREFIX, keep_alive: str | None = None) -> list[list[float]]` (protocol and `OllamaClassifier`; `keep_alive` is sent in the `/api/embed` body only when not None); `ClassifierUnavailable(reason: str, command: str = "localmind classify")` with attributes `reason` and `hint == f"{reason}, then: {command}"`, plus `with_command(command: str) -> ClassifierUnavailable`.
- `FakeClassifier.embed(texts, prefix=..., keep_alive=None)` records `self.prefixes: list[str]` and `self.keep_alives: list` per call; vectors still come from the unprefixed text. Add `"electricite"`, `"telephone"`, `"bail"`, `"rendez-vous"` to `KEYWORDS` (append at the end so existing axes keep their index).

- [ ] **Step 1: Write the failing tests** (fake HTTP server as in the existing `test_ollama.py`)

```python
def test_embed_sends_the_given_prefix_and_keep_alive(server):
    client.embed(["Total 84 €"], prefix=DOCUMENT_PREFIX, keep_alive="30m")
    body = server.last_body("/api/embed")
    assert body["input"] == ["search_document: Total 84 €"] and body["keep_alive"] == "30m"

def test_embed_defaults_to_classification_prefix_without_keep_alive(server):
    client.embed(["x"])
    body = server.last_body("/api/embed")
    assert body["input"] == ["classification: x"] and "keep_alive" not in body

def test_unavailable_hint_names_the_command():
    e = ClassifierUnavailable("Ollama not reachable at http://h: start it (ollama serve)")
    assert e.hint.endswith("then: localmind classify")
    assert e.with_command("localmind index").hint == "Ollama not reachable at http://h: start it (ollama serve), then: localmind index"
```

Live (`test_ollama_live.py`): `test_query_and_document_prefixes_bring_the_answer_closer` — the `search_query:` vector of "quel jour suis-je prélevé ?" is closer to a `search_document:` text mentioning "prélèvement le 5 de chaque mois" than to one about a lease.
- [ ] **Step 2: Run** `uv run pytest tests/test_ollama.py -q` — Expected: FAIL (unexpected keyword `prefix`).
- [ ] **Step 3: Implement**; existing hints keep their exact stage 3 text.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass (stage 3 tests unchanged).
- [ ] **Step 5: Commit** — `feat: embedding task prefixes and keep-alive`

### Task 4: Catalog v5 and the indexing run

**Files:**
- Create: `src/localmind/catalog_chunks.py`, `src/localmind/indexing.py`
- Modify: `src/localmind/catalog.py` (mixin, `SCHEMA_VERSION = 5`, v3/v4 branches only add tables, `_REBUILT_TABLES` starts with `"chunk_vectors", "chunks"`, `Stats.chunks`, `Stats.indexed_documents`), `src/localmind/catalog_folders.py` (per-page masking helper reused)
- Test: `tests/test_catalog_v5.py`, `tests/test_indexing.py`

**Interfaces:**
- Consumes: `ChunkConfig`, `Chunk`, `chunk_pages`, `embed_text` (Task 2); `Classifier.embed(prefix, keep_alive)`, `DOCUMENT_PREFIX`, `SEARCH_KEEP_ALIVE`, `ClassifierUnavailable.with_command` (Task 3).
- Produces (`ChunkStore`, on `Catalog`): `masked_pages(document_id) -> list[tuple[int, str]]` (ok pages only, masked); `documents_to_index(chunker: str, model: str, under: Path | None = None, redo: bool = False) -> list[int]`; `store_chunks(document_id, chunker, model, chunks: list[Chunk], vectors: list[list[float]]) -> None` (replaces the document's chunks, one transaction); `vector_matrix_rows(model) -> list[tuple[int, bytes]]` (chunk id, vector); `vector_marker(model) -> tuple[int, int]` (count, max rowid of `chunk_vectors` for the model); `chunk_rows(chunk_ids: Sequence[int]) -> dict[int, ChunkRow]` with `ChunkRow(chunk_id, document_id, page_no, text, paths: tuple[Path, ...])` (paths of indexed files, ordered by file id).
- Produces (`indexing.py`): `IndexReport(documents: int, chunks: int, empty: int, unavailable: str | None)`; `index(catalog, embedder, config: ChunkConfig = ChunkConfig(), under: Path | None = None, redo: bool = False, on_document: Callable[[int], None] | None = None) -> IndexReport`.

- [ ] **Step 1: Write the failing tests**

```python
# test_catalog_v5.py
def test_v4_catalog_gains_chunk_tables_and_keeps_data(v4_db): ...       # user_version 5, folders/decisions intact
def test_orphan_document_takes_its_chunks_and_vectors(catalog): ...     # relink the only file → old chunks gone
def test_vectors_are_kept_per_model(catalog): ...                       # store under "m1" and "m2": vector_matrix_rows("m1") only m1
def test_v1_rebuild_drops_chunks_but_never_decisions(v1_db): ...

# test_indexing.py (FakeClassifier, scratch files, real ingest)
def test_index_chunks_and_embeds_every_accepted_document(tmp_path):
    report = index(catalog, fake)
    assert report.documents == 2 and report.chunks == catalog.stats().chunks
    assert set(fake.prefixes) == {"search_document: "} and set(fake.keep_alives) == {"30m"}
    assert any(t.startswith("contrat_tel.txt p. 1\n") for t in fake.embedded)

def test_all_chunks_of_a_run_go_in_batches_of_32(tmp_path): ...         # 40 chunks → batches [32, 8]
def test_second_run_does_nothing_and_redo_does_all(tmp_path): ...
def test_new_chunker_version_reindexes(tmp_path): ...                   # ChunkConfig(chunk_chars=500)
def test_index_covers_documents_outside_under_when_under_is_none(tmp_path): ...
def test_chunks_never_hold_sensitive_values(tmp_path):
    # page contains a valid French IBAN; after index():
    assert all("FR76" not in r.text for r in catalog.chunk_rows(all_ids).values())
    assert all("FR76" not in t for t in fake.embedded) and any("[IBAN]" in t for t in fake.embedded)
def test_document_with_no_ok_page_counts_as_empty(tmp_path): ...
def test_ollama_dropping_mid_run_keeps_whole_documents(tmp_path):
    fake = FakeClassifier(fail_embed_on="doc3")          # 3 docs, ~20 chunks each
    report = index(catalog, fake)
    assert report.unavailable.endswith("then: localmind index")
    for doc_id in all_doc_ids:                           # each document: all chunks have a vector, or none exist
        assert chunk_count(doc_id) == 0 or vector_count(doc_id) == chunk_count(doc_id)
def test_edited_file_replaces_its_chunks(tmp_path):
    # index, rewrite the file with new text, ingest + index again:
    assert "old sentence" not in " ".join(r.text for r in catalog.chunk_rows(all_ids).values())
```

- [ ] **Step 2: Run** `uv run pytest tests/test_catalog_v5.py tests/test_indexing.py -q` — Expected: FAIL (`ModuleNotFoundError: localmind.indexing`).
- [ ] **Step 3: Implement.** Schema as spec 5.3 (`chunks.chunker` stored; `UNIQUE (document_id, seq)`). `documents_to_index`: documents with an indexed file (under `under` if given) whose chunks are missing, have another `chunker`, or lack a vector for `model`. `index`: collect chunks of all documents first, embed in batches of 32 with `DOCUMENT_PREFIX` + `SEARCH_KEEP_ALIVE`, and call `store_chunks` for each document as soon as all its vectors are in; on `ClassifierUnavailable` return what is done with `exc.with_command("localmind index").hint`.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** — `feat: catalog v5 chunks and the indexing run`

### Task 5: Vector search

**Files:**
- Create: `src/localmind/search.py`
- Modify: `pyproject.toml` (`numpy>=2.0`), `uv.lock` (`uv add "numpy>=2.0"`)
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `vector_matrix_rows`, `vector_marker`, `chunk_rows` (Task 4); `QUERY_PREFIX`, `SEARCH_KEEP_ALIVE` (Task 3).
- Produces: `Hit(path: Path, page_no: int, score: float, text: str, document_id: int, chunk_id: int)` frozen; `CatalogBusy(RuntimeError)` (message `the catalog is busy, try again`); `VectorIndex(catalog, embedder, model: str | None = None)` with `warm() -> None` and `search(question: str, k: int = 5) -> list[Hit]`; module function `search(catalog, question, k=5, embedder=None) -> list[Hit]` (`embedder=None` → `OllamaClassifier()`). Raises `ClassifierUnavailable` (command `localmind search`) and `ValueError` for an empty question or `k < 1`.

- [ ] **Step 1: Write the failing tests**

```python
def test_best_chunk_first_with_file_and_page(indexed):          # 3 docs: telephone, bail, electricite
    hits = search(catalog, "jour du prélèvement telephone", k=2, embedder=fake)
    assert hits[0].path.name == "contrat_tel.txt" and hits[0].page_no == 1
    assert hits[0].score == pytest.approx(1.0) and hits[0].score >= hits[1].score
    assert fake.prefixes[-1] == "search_query: "

def test_k_larger_than_the_catalog_returns_everything(indexed): ...
def test_nothing_indexed_returns_empty_list(tmp_path): ...
def test_empty_question_is_rejected(indexed): ...               # ValueError
def test_only_vectors_of_the_current_model_are_searched(indexed): ...
def test_deleted_file_is_skipped_and_copy_is_named(indexed):
    # bail.txt has an exact copy bail_copie.txt; delete bail.txt, then telephone file entirely:
    hits = search(catalog, "bail", k=5, embedder=fake)
    assert all(h.path.exists() for h in hits) and any(h.path.name == "bail_copie.txt" for h in hits)
    assert not any(h.path.name == "contrat_tel.txt" for h in hits)
def test_skipped_hits_are_replaced_to_keep_k(indexed): ...      # k=2, the best doc deleted → still 2 hits
def test_index_reloads_after_catalog_change(indexed):
    vi = VectorIndex(catalog, fake); vi.search("bail")
    # ingest + index a new "rendez-vous" document through the same catalog
    assert vi.search("rendez-vous", k=1)[0].path.name == "rdv_medecin.txt"
def test_locked_catalog_raises_catalog_busy(indexed, tmp_path): ...   # second connection holds BEGIN EXCLUSIVE; timeout patched low
def test_warm_embeds_once_with_keep_alive(indexed): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/test_search.py -q` — Expected: FAIL (`ModuleNotFoundError: localmind.search`).
- [ ] **Step 3: Implement.** Matrix `float32` of shape (n, dim) built from `vector_matrix_rows`, rebuilt when `vector_marker` differs from the one it was built with. Scores = matrix @ query; take candidates with `argpartition` (start with `4 * k`, widen to all when existence filtering leaves fewer than `k`), sort by score; for each candidate keep the first path of its `ChunkRow.paths` that `is_file()`, skip the chunk when none exists. `sqlite3.OperationalError` containing "locked" → `CatalogBusy`.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** — `feat: numpy vector search over chunks`

### Task 6: Prune — forget files deleted from disk

**Files:**
- Create: `src/localmind/prune.py`
- Modify: `src/localmind/catalog_chunks.py`: `forget_files(paths: Sequence[Path]) -> int` (deletes the file rows, then orphan documents; returns documents removed)
- Test: `tests/test_prune.py`

**Interfaces:**
- Produces: `missing_files(catalog, under: Path | None = None) -> list[Path]` (every catalog file, any status, whose path is not a file; sorted); `PruneReport(files: int, documents: int)`; `prune(catalog, paths: Sequence[Path]) -> PruneReport`.

- [ ] **Step 1: Write the failing tests**

```python
def test_missing_files_lists_only_vanished_paths(tmp_path): ...       # also quarantined ones; `under` filters
def test_prune_forgets_document_with_its_pages_chunks_vectors_and_folder(tmp_path):
    report = prune(catalog, missing_files(catalog))
    assert report == PruneReport(files=1, documents=1)
    assert table_count("pages") == table_count("chunks") == table_count("chunk_vectors") == table_count("classifications") == 0
def test_prune_keeps_a_document_that_still_has_a_copy(tmp_path): ... # PruneReport(files=1, documents=0)
def test_prune_keeps_user_choices(tmp_path): ...                     # decisions, folder_decisions, ocr_cache rows unchanged
def test_prune_never_touches_disk(tmp_path): ...                     # other files' bytes and mtimes unchanged
```

- [ ] **Step 2: Run** `uv run pytest tests/test_prune.py -q` — Expected: FAIL (`ModuleNotFoundError: localmind.prune`).
- [ ] **Step 3: Implement** (`forget_files` in one transaction; orphan deletion reuses the stage 1 rule).
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** — `feat: prune forgets files deleted from disk`

### Task 7: CLI and public API

**Files:**
- Create: `src/localmind/cli_search.py` (`index`, `search`, `prune` commands)
- Modify: `src/localmind/cli.py` (`--no-index`, register, stats rows), `src/localmind/cli_folders.py` (`classify_and_review` returns whether Ollama was reachable and takes the hint's command), `src/localmind/__init__.py` (export `index`, `IndexReport`, `search`, `Hit`, `VectorIndex`, `CatalogBusy`, `prune`, `missing_files`, `ChunkConfig`; `__version__ = "0.3.0"`), `tests/conftest.py` (also patch `cli_search.make_embedder`)
- Test: `tests/test_cli_search.py`

**Interfaces:**
- Consumes: `index`, `IndexReport` (Task 4); `search`, `CatalogBusy` (Task 5); `missing_files`, `prune` (Task 6).
- Produces: `cli_search.make_embedder() -> Classifier` (patched by tests), `cli_search.register(sub)`.

- [ ] **Step 1: Write the failing tests** (exact copy from Global Constraints)

```python
def test_ingest_classifies_then_indexes(tmp_path, capsys): ...          # output ends with "Indexed 2 document(s): {c} chunk(s)"
def test_ingest_no_index_skips_indexing(tmp_path, capsys): ...
def test_ingest_with_ollama_down_prints_one_hint(tmp_path, capsys):
    out = capsys.readouterr().out
    assert out.count("Ollama not reachable") == 1 and "then: localmind classify && localmind index" in out
def test_index_command_and_all_flag(tmp_path, capsys): ...
def test_search_prints_score_file_page_excerpt(tmp_path, capsys):
    assert re.match(r'^1\.00  contrat_tel\.txt p\.1  ".+"$', capsys.readouterr().out.splitlines()[0])
def test_search_output_masks_sensitive_values(tmp_path, capsys): ...    # "[IBAN]" shown, digits never
def test_search_with_nothing_indexed(tmp_path, capsys): ...             # "Nothing indexed yet: localmind index"
def test_search_busy_catalog_exits_2(tmp_path, capsys, monkeypatch): ...
def test_prune_asks_then_forgets(tmp_path, capsys, monkeypatch): ...    # interactive, answer "y"
def test_prune_non_interactive_needs_yes(tmp_path, capsys): ...         # nothing forgotten without --yes
def test_stats_shows_chunks_and_indexed_docs(tmp_path, capsys): ...
def test_public_api_exports_stage4_names(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_search.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** `_cmd_ingest`: after classification (unless `--no-classify`), if Ollama was reachable or classification was skipped, run `index` with `under=None` and print its summary; if classification found Ollama unreachable, print only the combined hint. Interactivity uses the existing `cli._is_interactive`.
- [ ] **Step 4: Run** `uv run pytest -q` and `uv run pytest --cov=localmind -q` — Expected: all pass, coverage ≥ 95 %.
- [ ] **Step 5: Commit** — `feat: index, search and prune commands`

### Task 8: Documentation, notebook and end-to-end check

**Files:**
- Create: `docs/04-chunking-vectors.md`; `apprentissage/04-chunking-vecteurs.md` (gitignored, French)
- Modify: `README.md` (roadmap row 4 ✅, a "Searching your documents" section), `apprentissage/README.md` index rows, `SUIVI.md`

- [ ] **Step 1: End-to-end run on scratch data only** (`<scratchpad>/e2e4/`, a fresh `--db` in the scratchpad), real Ollama: `ingest` the spike documents, then `search` the 10 spike questions, `stats`, delete one file, `search` again, `prune --yes`.
Expected: indexing time and chunk count noted; the deleted file never appears; `prune` reports 1 file and 1 document; no IBAN digits in any output.
- [ ] **Step 2: Write `docs/04-chunking-vectors.md`** in the style of `docs/03-classification.md`: chunking rules, header, prefixes, storage and cascade, brute-force search and why (A vs C table), deleted files and `prune`, spike table, the end-to-end transcript, known limits (no folder filter or BM25 until stage 5; facts like sums need the fact-extraction stage).
- [ ] **Step 3: Write the notebook fiche** (French, one section per concept: chunk, overlap, header, asymmetric prefixes, brute force vs ANN, one source of truth, forgetting) and the index rows.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all pass.
- [ ] **Step 5: Commit** — `docs: stage 4 chunking and vector search` (README + docs only; `apprentissage/` and `SUIVI.md` are gitignored).
