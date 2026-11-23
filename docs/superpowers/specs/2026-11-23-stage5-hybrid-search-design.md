# Stage 5: Hybrid search and folder filter (design)

Status: approaches chosen with the user on 2026-11-23 (FTS5 + rank fusion; documents
still to confirm are included in a folder filter, marked). Awaiting approval of this
document.

## 1. Intent

Stage 4 finds passages by **meaning** only. It misses when the question and the
document use different words for the same thing, or when the answer hangs on one
exact token: the stage 4 spike missed "quel jour suis-je prélevé pour mon abonnement
**téléphonique** ?" against a contract that says "forfait **mobile**". Exact tokens
(a contract number, a supplier name, "électricité") are what keyword search is good at.

Stage 5 adds keyword search (BM25) next to the vector search, fuses both rankings, and
lets a search be limited to a folder ("Maison", "Maison/Factures"). It prepares stage 6
(answers written by a local LLM), whose quality depends on the passages it is given.

Users: the person at the command line, and later JARVIS through the Python API
(read-only, long-lived process, many questions).

## 2. Goals, non-goals, anti-goals

Goals:
- Keyword search over the same masked chunks as stage 4, accents ignored
  ("electricite" finds "électricité").
- One ranked list from both searches (reciprocal rank fusion).
- A folder filter: a whole domain (`Maison`) or one folder (`Maison/Factures`).
  Documents the LLM placed there but the user has not reviewed yet are **included and
  marked** ("to confirm"). User's choice, 2026-11-23.
- Measured before building: the stage 4 spike questions plus exact-word questions.

Non-goals:
- Answers written by an LLM (stage 6), reranking with a cross-encoder, query rewriting.
- Facts, sums and date filters (the later "fact extraction" stage).
- Filters other than folder (date, file type).

Anti-goals (failure even if it "works"):
- A keyword index that can disagree with the catalog: a pruned, deleted or edited
  file's text found by keywords. One source of truth stays the rule.
- A sensitive value findable by keyword (`IBAN` text must never be indexed unmasked).
- Search slower than a voice answer can bear (target: < 100 ms in-process for the
  expected catalog size, a few thousand chunks).
- A folder filter that silently hides a document the LLM placed in that folder.

## 3. Constraints (carried)

- 100 % local; nothing leaves the machine. Python library first, thin CLI.
- Never modify user files. Never print a sensitive value. Repo and CLI copy in English.
- Catalog = SQLite at `~/.localmind/catalog.db`; schema migrations must keep user data.
- SQLite bundled with Python here: 3.45.3, FTS5 available, `unicode61
  remove_diacritics 2` tokenizer verified.

## 4. Approaches considered

| | 1. FTS5 + rank fusion (chosen) | 2. Hand-made BM25 + rank fusion | 3. FTS5 + weighted score mix |
|---|---|---|---|
| keyword engine | SQLite FTS5 `bm25()` | own inverted index in numpy | SQLite FTS5 |
| sync with catalog | triggers, same transaction | rebuilt in memory on change | triggers |
| French tokens/accents | tokenizer option | ours to write | tokenizer option |
| fusion | ranks only, no calibration | ranks only | `a·cos + (1-a)·bm25`, `a` to tune |
| code | small | large | small, plus tuning |

Chosen: **1** (user, 2026-11-23). 2 teaches the formula by coding it but adds code and
fragility for the same result; the stage doc explains BM25 instead. 3 needs a weight
tuned on 10 questions: tuning by chance.

## 5. Design

### 5.1 Keyword index (schema v6)

An FTS5 table over `chunks.text`, **external content** (the text is stored once, in
`chunks`):

```
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
```

Triggers on `chunks` (after insert, after delete, after update of text) keep it in step.
Chunks are deleted by `store_chunks` (replace) and by cascades from documents (orphan
delete on re-ingest, `prune`); the builder verifies with a test that a cascade delete
fires the trigger (SQLite fires triggers for foreign-key actions). Masked text only:
the index holds `[IBAN]`, never the value.

Migration v5 → v6: create the table and triggers, then
`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`. Older versions follow the
existing path to v6. No vector is recomputed.

### 5.2 From a question to a keyword query

Words of the question (FTS5 tokenizer rules: letters and digits), each quoted, joined
with `OR`: BM25 rewards passages holding more of the words, rarer words more. Quoting
means user text is never parsed as FTS5 syntax (no injection, no syntax error on `"`,
`-`, `*`). A small French stop-word list (le, la, les, de, du, des, un, une, et, ou,
mon, ma, mes, est, quel, quelle...) is dropped first; if nothing is left, the keyword
side returns nothing. The exact list is the builder's, fixed by the spike.

### 5.3 Fusion (reciprocal rank fusion)

Each side returns its `C = max(30, 4·k)` best chunks (after the folder filter).
A chunk's fused score is `Σ 1 / (60 + rank)` over the sides where it appears (rank from
1). Ties: better vector rank first, then chunk id. The deleted/edited-file check of
stage 4 then walks the fused order; if fewer than `k` hits survive, both sides widen to
all chunks (as stage 4 does).

`Hit.score` stays the **cosine** with the question (computed for keyword-only hits too:
every vector is in the matrix), because it means something to a reader; the order is
the fused order. New fields:

```
Hit.folder: str | None     # "Maison/Factures", None when unclassified
Hit.to_confirm: bool       # classification not reviewed yet
Hit.matched: frozenset[str]  # {"meaning"}, {"keywords"} or both
```

### 5.4 Folder filter

`folder="Maison"` (a domain: every folder of that domain) or `"Maison/Factures"`.
Parsed with the existing `FolderName` rules (case and spacing normalised the same way).
A document matches through its classification's `folder_id`, whatever its status:
`placed` (reviewed) or `to_confirm` (not reviewed yet, marked). This **includes
hesitations**: `to_confirm` in a confirmed folder with an `alternative_folder_id`
(classify.py ~245-258). It is *not* the rule of `catalog_folders.examples()` (line
~356), which leaves hesitations out on purpose and must not be reused here: reusing it
would silently hide documents the LLM put in the folder. The alternative folder never
matches (the LLM's second guess; a document has one folder, `Hit.folder` stays one
string). Unclassified documents (no classification, or `folder_id` NULL after an
invalid LLM answer) never match a filter. A known folder or domain is any row of
`folders`, confirmed or only proposed by the LLM. An unknown folder or domain is an
error naming it (`no folder named 'Maison/Factur'`), not an empty result.

Filtering happens **before** ranking on both sides (vector rows masked by document id,
FTS query joined to the allowed documents), so `k` hits come from the folder, not the
global top `k` filtered down.

### 5.5 Without Ollama

Keyword search needs no model. If the question cannot be embedded (Ollama down, model
missing), search returns keyword-only hits (`score` = None, `matched = {"keywords"}`) and
the result says so: `SearchResult`-level flag or a warning list (builder's choice of
shape, the CLI prints one stderr line: `meaning search unavailable (<reason>): keyword
results only`). Exit code 0. Rationale: JARVIS still answers something useful. (Author's
call; stage 4 exited 2 here.)

### 5.6 CLI

```
localmind search "question" [-k 5] [--folder Maison/Factures]
0.70  facture_edf_mars.txt p.1  Maison/Factures  "Montant total TTC : 112,47 €..."
0.52  facture_sfr_mai.txt p.1  Maison/Factures (to confirm)  "..."
```

Folder column omitted for unclassified documents. No new command.

### 5.7 Python API

`search(catalog, question, k=5, embedder=None, folder=None)` and
`VectorIndex.search(question, k=5, folder=None)` (name kept: JARVIS code may already
import it; its docstring says it is hybrid now). Exported names unchanged plus any new
result type.

### 5.8 Spike gate, before building

On the stage 4 scratch data (never real files): the 10 spike questions plus about 5
exact-word questions (a contract number, a supplier, a person's name, a word only in one
document, "electricite" without accent). Compare vector-only vs hybrid, right chunk in
top 3. Gate: the phone question in the top 3, no question that passed in stage 4 now
failing, exact-word questions ≥ 4/5. Fixes C, the 60 constant and the stop words. If the
gate fails, come back to the user before building.

### 5.9 Calls made after the builder check (author's calls, open to correction)

- Hesitations match the folder the LLM chose (`folder_id`), marked to confirm; never
  their alternative folder (5.4).
- Proposed (unconfirmed) folders count as known for the unknown-folder error (5.4).
- A keyword hit whose chunk has no vector for the current model gets `score = None`
  and still appears, ranked by its keyword rank only.
- The stage 4 spike script lived in a scratchpad: the spike data is regenerated from
  the stage 4 plan's description (same 10 documents and questions), scratch data only.

### 5.10 Spike results (2026-11-24)

Stage 4 scratch documents (11 text files), real `nomic-embed-text`, in-memory FTS5;
right chunk in the top 3 for 10 questions plus 5 exact-word questions:

| chunk size | meaning only | keywords only | hybrid (RRF) | phone question, hybrid rank |
|---|---|---|---|---|
| 800 | 9/10 | 9/10 | 9/10 | 6 (gate failed) |
| **600** | **10/10** | 8/10 | **10/10** | **1** |
| 500 | 10/10 | 9/10 | 10/10 | 1 |

Exact-word questions: 5/5 for every method (this corpus is too small to separate them).
Stop words and the constant (60 vs 20) changed little: 60 kept. Cause of the 800 miss:
the debit day sits in a chunk that is mostly legal boilerplate (meaning rank 18); at 600
it has its own chunk. **Decision (user, 2026-11-24): chunk size 600** (`para-v1:600/90/150`: overlap and minimum scale with the size, 15 % and a quarter, as in the spike; 600/120/200 was checked end to end and gave the phone question rank 6 again);
existing catalogs are re-cut by `localmind index --all`. Caveat: 10 questions, sizes
chosen on the same questions; stage 7 measures on real documents.

## 6. Deliverables

- `catalog` schema v6 (FTS table, triggers, migration), keyword query + fusion + folder
  filter in `search`, CLI `--folder` and folder column, keyword-only fallback.
- Tests for every point above, including: trigger sync on replace / cascade / prune;
  IBAN not findable by keyword; FTS syntax characters in a question; unknown folder;
  to_confirm marked; filter before ranking.
- `docs/05-hybrid-search.md`, README roadmap row 5, `apprentissage/05-…` (French,
  gitignored), version 0.4.0.

## 7. Left to the builder

Module split (e.g. `keyword.py` for the FTS side), the stop-word list, the exact shape
of the fallback warning, test data.
