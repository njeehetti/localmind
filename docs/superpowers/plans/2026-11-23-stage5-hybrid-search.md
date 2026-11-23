# Stage 5: Hybrid search and folder filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add keyword search (SQLite FTS5, BM25) next to stage 4's vector search, fuse both rankings with reciprocal rank fusion, and let a search be limited to a folder, with unreviewed documents marked.

**Architecture:** An external-content FTS5 table over `chunks.text`, kept in step by triggers (schema v6). A pure `keyword.py` turns a question into a safe FTS query and fuses two rankings. `catalog_folders` gains the document → folder lookup used for the filter and for each hit's folder. `VectorIndex.search` runs both sides on the filtered documents, fuses, then applies stage 4's deleted/edited-file check; without Ollama it falls back to keywords only.

**Tech Stack:** Python 3.12+, sqlite3 (bundled SQLite 3.45.3, FTS5), numpy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-23-stage5-hybrid-search-design.md`

## Global Constraints

- 100 % local; never modify, move or delete user files; never print a sensitive value; e2e and spike on scratch data only, never `~/.localmind` nor the user's documents.
- Repo, code, CLI copy in English. Commits: conventional, repo-local noreply email.
- FTS table: `chunks_fts`, `fts5(text, content='chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2')`. `SCHEMA_VERSION = 6`.
- Fusion: `score = Σ 1 / (60 + rank)` (rank from 1) over the sides a chunk appears in; ties: better vector rank, then chunk id. Candidates per side: `C = max(30, 4 * k)`.
- Folder filter: `"Domain"` or `"Domain/Type"`, normalised like `parse_folder`; matches a classification's `folder_id` whatever its status (`placed` or `to_confirm`, hesitations included); never `alternative_folder_id`; unclassified documents never match. Known = any row of `folders`.
- CLI copy (exact):
  - hit line: `f'{score}  {name} p.{page}  {folder}  "{excerpt}"'`, where `score` is `f"{h.score:.2f}"` or `"  --"` when None, `folder` is `Maison/Factures`, `Maison/Factures (to confirm)`, and the folder part plus its two spaces is omitted when the document is unclassified.
  - fallback (stderr, once): `meaning search unavailable ({reason}): keyword results only`
  - unknown folder (stderr, exit 2): `error: no folder named '{text}'`
- `Hit.score` stays the cosine (None when the question could not be embedded or the chunk has no vector for the model). Version `0.4.0`.

## Review Focus

1. A question with FTS5 syntax or punctuation (`l'électricité`, `"`, `*`, `-`, `NEAR`, `AND`, a lone `(`) must never raise; it is searched as plain words.
2. A chunk found **only by keywords** whose file was deleted or edited since ingest must never be cited (stage 4's check must cover keyword hits).
3. A long-lived `VectorIndex` must see keyword changes made by another connection (ingest, prune) without being rebuilt.
4. `--folder maison/factures` (case) and `--folder "Maison / Factures"` (spaces) match `Maison/Factures`; a known folder holding no indexed document returns no hit, not an error.
5. Ollama down **with** a folder filter returns keyword hits from that folder only, the warning printed once, exit 0.

---

### Task 1: Spike gate — does hybrid fix the misses?

**Files:**
- Create (scratchpad only, not committed): `spike5.py`, generated documents.
- Modify: `docs/superpowers/specs/2026-11-23-stage5-hybrid-search-design.md` (new section 5.10 "Spike results").

- [ ] **Step 1:** Regenerate the stage 4 spike documents and 10 questions from the stage 4 plan's Task 1 description (French invoices, contracts, payslips, lease, appointment letter, phone contract saying "forfait mobile"), into the scratchpad. Add 5 exact-word questions: a contract number, a supplier name, a person's name, a word present in one document only, `electricite` without accent.
- [ ] **Step 2:** In `spike5.py`, chunk (stage 4 `chunk_pages`, default config), embed with the real Ollama (`nomic-embed-text`, `127.0.0.1`), build an in-memory FTS5 table with the spec's tokenizer, and score each question three ways: vector only, keyword only, RRF (60, C=30). Right chunk in top 3.
- [ ] **Step 3:** Try the stop-word list on/off and RRF constant 60 vs 20; keep the simplest setting that passes.
Expected (gate): phone question in hybrid top 3; every question that vector-only passes still passes; exact-word questions ≥ 4/5. **If the gate fails, stop and report to the user before Task 2.**
- [ ] **Step 4:** Write the results table and the chosen stop-word list into spec section 5.10; commit `docs: stage 5 spike results`.

### Task 2: Schema v6 — keyword index kept in step

**Files:**
- Modify: `src/localmind/catalog_chunks.py` (FTS schema, `keyword_rows`), `src/localmind/catalog.py` (`SCHEMA_VERSION = 6`, migration, `_REBUILT_TABLES`)
- Test: `tests/test_catalog_v6.py`; update version expectations in `tests/test_catalog_v5.py` and older migration tests (`== 6`).

**Interfaces:**
- Produces: `ChunkStore.keyword_rows(match: str, limit: int, document_ids: Collection[int] | None = None) -> list[int]` — chunk ids, best BM25 first (`ORDER BY bm25(chunks_fts)`), restricted to `document_ids` when given (empty collection → `[]`).
- Produces: `CHUNKS_SCHEMA` now also creates `chunks_fts` and triggers `chunks_ai`, `chunks_ad`, `chunks_au` (FTS5 external-content `'delete'` command for old rows).

- [ ] **Step 1: Write the failing tests** in `tests/test_catalog_v6.py`:

```python
def test_keyword_rows_find_accentless_words(catalog, tmp_path):
    # a document indexed with "Relevé électricité" is found by '"electricite"'
    assert keyword_ids(catalog, '"electricite"') == [the_chunk_id]

def test_replacing_chunks_updates_the_keyword_index(...):   # store_chunks twice, old word gone
def test_orphan_cascade_and_prune_remove_keywords(...):     # re-ingest edited file; prune deleted file
def test_masked_iban_is_not_findable(...):                  # IBAN digits ("30006000011234567890189") → []; "IBAN" finds the masked chunk
def test_keyword_rows_respect_document_ids(...):            # same word in 2 docs, filter to one; empty set → []
def test_v5_catalog_is_migrated_with_its_chunks_searchable(tmp_path):
    # build a v5 file (user_version 5, chunks present, no chunks_fts), open with Catalog
    # → user_version 6, keyword_rows finds the old chunks, vectors untouched
def test_v1_rebuild_drops_keyword_index_too(tmp_path)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_catalog_v6.py -q`. Expected: FAIL (`no such table: chunks_fts` / attribute missing).
- [ ] **Step 3:** Implement. Migration: versions 3, 4 and 5 add tables only; after `_SCHEMA` runs, a catalog that was v5 gets `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` (same transaction as the version bump). `"chunks_fts"` goes first in `_REBUILT_TABLES`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: catalog v6 keyword index over chunks`.

### Task 3: Keyword query and rank fusion (pure)

**Files:**
- Create: `src/localmind/keyword.py`
- Test: `tests/test_keyword.py`

**Interfaces:**
- Produces: `STOP_WORDS: frozenset[str]` (from Task 1), `keyword_query(question: str) -> str | None` — words (`\w+`, lowercased) minus stop words, each double-quoted, joined with `" OR "`; `None` when nothing is left.
- Produces: `fuse(vector_ids: Sequence[int], keyword_ids: Sequence[int], constant: int = 60) -> list[int]` — chunk ids, best fused first; ties by vector rank then id.

- [ ] **Step 1: Write the failing tests:**

```python
def test_query_quotes_words_and_drops_stop_words():
    assert keyword_query("Quel est le préavis de mon bail ?") == '"préavis" OR "bail"'

def test_fts_syntax_is_neutralised():                  # Review Focus 1
    for q in ["l'électricité", 'a "b', "x*", "-y", "NEAR(a b)", "a AND b", "("]:
        query = keyword_query(q)
        assert query is None or all(part.startswith('"') for part in query.split(" OR "))

def test_only_stop_words_gives_none():
    assert keyword_query("le de la ?") is None

def test_fuse_rewards_both_sides():
    assert fuse([1, 2, 3], [3, 4])[0] == 3

def test_fuse_ties_prefer_vector_rank():
    assert fuse([1, 2], [2, 1]) == [1, 2]

def test_fuse_keeps_one_sided_hits():
    assert set(fuse([1], [2])) == {1, 2}
```

And in `tests/test_catalog_v6.py`, every Review Focus 1 question passed through `keyword_query` then `keyword_rows` raises nothing.
- [ ] **Step 2:** Run `uv run pytest tests/test_keyword.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: keyword query and reciprocal rank fusion`.

### Task 4: Folder of each document, and the folder filter

**Files:**
- Modify: `src/localmind/folders.py` (`FolderFilter`, `parse_folder_filter`), `src/localmind/catalog_folders.py` (`document_folders`)
- Test: `tests/test_folder_filter.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) FolderFilter(domain: str, type: str | None)` with `__str__`; `parse_folder_filter(text: str) -> FolderFilter` (same normalisation as `parse_folder`, a missing `/Type` is allowed; raises `InvalidFolderName`).
- Produces: `class UnknownFolder(LookupError)` in `folders.py`, message `no folder named '{text}'`.
- Produces: `@dataclass(frozen=True) DocFolder(name: str, to_confirm: bool)`; `FolderStore.document_folders(only: FolderFilter | None = None) -> dict[int, DocFolder]` — document id → folder of its classification (`folder_id` NOT NULL); with `only`, restricted to it, raising `UnknownFolder` when no `folders` row matches the domain (and type).

- [ ] **Step 1: Write the failing tests:**

```python
def test_placed_and_to_confirm_are_listed_marked()
def test_hesitation_matches_its_chosen_folder_not_the_alternative()   # to_confirm, confirmed folder, alternative set
def test_unclassified_and_invalid_llm_answer_never_match()           # no row; folder_id NULL
def test_domain_filter_covers_its_folders()                           # "Maison" → Maison/Factures + Maison/Contrats
def test_filter_is_case_and_space_insensitive()                       # Review Focus 4: "maison/factures", "Maison / Factures"
def test_unknown_folder_raises_naming_it()                            # str(exc) == "no folder named 'Maison/Factur'"
def test_known_folder_without_documents_is_empty()                    # Review Focus 4
def test_llm_proposed_folder_is_known()                               # confirmed = 0
```
- [ ] **Step 2:** Run `uv run pytest tests/test_folder_filter.py -q`. Expected: FAIL (names missing).
- [ ] **Step 3:** Implement (one SQL join `classifications` × `folders`; `to_confirm = status == 'to_confirm'`).
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: folder of each document and folder filter`.

### Task 5: Hybrid search

**Files:**
- Modify: `src/localmind/search.py`
- Test: `tests/test_search.py` (existing tests keep passing), `tests/test_hybrid_search.py`

**Interfaces:**
- Consumes: `keyword_rows` (Task 2), `keyword_query`, `fuse` (Task 3), `parse_folder_filter`, `document_folders`, `UnknownFolder` (Task 4).
- Produces: `Hit(path, page_no, score: float | None, text, document_id, chunk_id, folder: str | None = None, to_confirm: bool = False, matched: frozenset[str] = frozenset())`, `matched` ⊆ `{"meaning", "keywords"}`.
- Produces: `VectorIndex.search(question: str, k: int = 5, folder: str | None = None) -> list[Hit]`; `VectorIndex.last_warning: str | None` — `None`, or the `ClassifierUnavailable.reason` when the last search fell back to keywords only. Module `search(catalog, question, k=5, embedder=None, folder=None) -> list[Hit]`. `UnknownFolder` and `InvalidFolderName` propagate.

- [ ] **Step 1: Write the failing tests** in `tests/test_hybrid_search.py` (fake embedder; build a case where the fake vector ranks the wrong chunk first but an exact word is in the right one):

```python
def test_exact_word_found_even_when_meaning_misses()
def test_both_sides_agreeing_rank_first()
def test_hit_carries_folder_mark_and_matched()          # folder "Maison/Factures", to_confirm True, matched == {"meaning", "keywords"}
def test_folder_filter_applies_before_ranking()         # 6 docs outside the folder rank higher globally; k=2 still returns 2 hits from the folder
def test_keyword_only_hit_of_deleted_or_edited_file_is_never_cited()   # Review Focus 2
def test_long_lived_index_sees_keywords_from_another_connection()     # Review Focus 3: second Catalog adds/prunes
def test_ollama_down_falls_back_to_keywords(): # hits all matched == {"keywords"}, score None, last_warning == reason
def test_ollama_down_with_folder_keeps_the_filter()     # Review Focus 5
def test_unknown_folder_raises()
def test_keyword_hit_without_vector_has_no_score()      # chunk embedded by another model only
```
- [ ] **Step 2:** Run `uv run pytest tests/test_hybrid_search.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement: resolve the filter to allowed document ids (and the folder map for all hits) first; vector side on the masked rows of allowed documents; keyword side via `keyword_rows(..., document_ids)`; `fuse`; walk the fused order with stage 4's `_unchanged` check; widen both sides to all candidates when fewer than `k` survive. On `ClassifierUnavailable` from the question embedding: keyword side only, `last_warning = exc.reason`. Update the module docstring (hybrid now). Keep `CatalogBusy` mapping.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass, including every stage 4 search test.
- [ ] **Step 5:** Commit `feat: hybrid keyword and vector search with folder filter`.

### Task 6: CLI and public API

**Files:**
- Modify: `src/localmind/cli_search.py`, `src/localmind/__init__.py` (export `DocFolder`, `FolderFilter`, `UnknownFolder`; `__version__ = "0.4.0"`), `pyproject.toml` (`0.4.0`)
- Test: `tests/test_cli_search.py`

**Interfaces:**
- Consumes: Task 5 `VectorIndex` (CLI builds one to read `last_warning`), Task 4 errors.

- [ ] **Step 1: Write the failing tests** (exact copy from Global Constraints):

```python
def test_search_prints_folder_and_mark(...)     # line matches '0.xx  facture.txt p.1  Maison/Factures (to confirm)  "..."'
def test_unclassified_hit_has_no_folder_column(...)
def test_folder_option_limits_results(...)
def test_unknown_folder_exits_2(...)            # stderr == "error: no folder named 'Maison/Factur'\n"
def test_invalid_folder_exits_2(...)            # "error: invalid folder name ..."
def test_ollama_down_prints_warning_once_and_keyword_hits(...)  # exit 0, score column "  --"
```
Replace stage 4's `test_search_with_ollama_down_*` expectation (exit 2) with the fallback, and say so in the ledger.
- [ ] **Step 2:** Run `uv run pytest tests/test_cli_search.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `--folder` on `search`, the line format, the warning, the errors.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit `feat: search --folder and keyword fallback in the CLI`.

### Task 7: Documentation, notebook and end-to-end check

**Files:**
- Create: `docs/05-hybrid-search.md`, `apprentissage/05-recherche-hybride.md` (French, gitignored) + rows in `apprentissage/README.md`
- Modify: `README.md` (roadmap row 5 → link ✅, search section shows `--folder`), `SUIVI.md` (gitignored)

- [ ] **Step 1:** E2E on scratch data with the real Ollama (Task 1 documents, fresh scratch catalog): `ingest`, `folders review` non-interactive state left as is (to-confirm marks visible), `search` for the 15 spike questions, `--folder` on a domain and a folder, Ollama stopped (point `LOCALMIND_OLLAMA_URL` to a closed port) → keyword fallback. Measure in-process search time.
Expected: same scores as the spike gate; < 100 ms in-process; IBAN never printed.
- [ ] **Step 2:** Write `docs/05-hybrid-search.md` (why keywords, BM25 in one paragraph, FTS5 external content + triggers, RRF with a worked example, folder filter and marks, fallback, spike table, e2e transcript, known limits) and the README changes.
- [ ] **Step 3:** French fiche: BM25 (tf, idf, length), index inversé, FTS5 external content, triggers, RRF, filtrer avant de classer, mode dégradé.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass. Commit `docs: stage 5 hybrid search` (README + docs only).
