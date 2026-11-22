# Stage 3: Classification into virtual folders — design

Status: reviewed (builder check done), awaiting user approval · Branch: `feat/stage3-classification` · Builds on stages 1, 2a, 2b (catalog schema v3).

## 1. Intent

localmind should sort every indexed document into a **virtual folder** on its own, the way
the user would, and stay honest about it: it always says where it put each document, it asks
before doing anything the user could disagree with (a new folder, a hesitation), and the user
can always move a document. Original files are never touched: a folder is a row in the catalog.

Why: the user drops piles of personal documents (invoices, contracts, payslips, medical
papers, scans and photos) and wants to find them by area of life later, and stage 5 will let
search (and JARVIS) be filtered by folder. It is also a learning project: this stage
introduces **embeddings** (reused in stage 4) and the use of an LLM as a *judge* whose output
is constrained and checked, not trusted blindly.

## 2. What the user decided

- **Two levels: area of life / document type**, e.g. `Maison/Factures`, `Santé/Remboursements`,
  `Travail/Bulletins de paie`. Never a third level.
- **The LLM creates folders freely** (both levels), but it **always tells** where it placed
  each document, it **may consult** the user, and the user can **always correct** it.
- **When it consults (option A):** it places a document on its own when it is sure; it asks
  only when it wants to **create a new folder** or **hesitates between two**. Everything else
  appears in the end-of-ingest summary and can be corrected afterwards.
- **How "sure" is decided (approach 2, hybrid):** embeddings first, LLM second (section 5).

## 3. Goals, non-goals, anti-goals

Goals
- Every indexed document ends up in exactly one folder, or in the review queue with a reason.
- The confidence that separates "placed" from "to confirm" is **measured** (similarities), not
  the LLM's self-assessment.
- User corrections are permanent, survive re-ingest and rebuilds, follow exact copies, and
  improve later placements (they act as examples).
- Fully local: Ollama on `localhost`, nothing leaves the machine.
- Works when nobody can answer (script, JARVIS): nothing is asked, nothing blocks.

Non-goals (this stage)
- Several folders per document, tags, a third level.
- Moving or linking real files on disk (the "Explorer view" with hard links is a later bonus).
- Chunking and search (stages 4-5). Embeddings here are one vector per document.
- Fine-tuning or retraining a model.
- The LLM renaming or merging existing folders on its own.

Anti-goals (a failure even if it "works")
- A folder created without the user having seen it.
- A user correction undone by a later run.
- Folder sprawl: `Facture`, `Factures`, `Factures EDF`, `Électricité` side by side.
- Ingest failing or hanging because Ollama is not running.
- A sensitive value (IBAN, card number…) printed in a reason or a summary.

## 4. Approaches considered

1. **LLM only**: one LLM call per document returns a folder and a self-reported confidence.
   Simple, but LLM confidence is poorly calibrated (it says 0.9 when wrong), so the split
   between "placed" and "to confirm" would be close to arbitrary; 1-3 s per document.
2. **Hybrid, embeddings then LLM — chosen.** A document vector is compared with the documents
   already in each folder. Clear cases are placed without the LLM, with a measured confidence;
   ambiguous or novel cases go to the LLM, constrained to candidates or to proposing a new
   folder. Two independent judges (geometry and LLM) must agree for a placement to be silent.
   More parts; at cold start (no folders) the LLM does everything.
3. **Clustering first** (group all documents by similarity, the LLM names each group).
   Good for a first bulk sort, poor for documents added over time and for user corrections.
   Dropped.

## 5. How a document is classified

Input of a document: its **usable text** (pages with status `ok`, the same text stage 4 will
index) and its file name. Sensitive spans found in stage 2a are **masked** (`[IBAN]`,
`[CARTE]`…) in everything sent to the models, so no reason or summary can echo them.

### 5.1 Embedding

- Model `nomic-embed-text` through Ollama (`/api/embed`), input =
  `"classification: " + file name + "\n" + first 2,000 characters of the masked usable text`.
- **Gate before freezing the plan:** a short spike on generated French documents (invoices,
  payslips, medical refunds, contracts…) checks that same-type documents are clearly closer
  than different ones. If `nomic-embed-text` (mostly English) separates French poorly, the
  builder switches the default to a **multilingual** embedding model available in Ollama
  (e.g. `bge-m3`), with its own input convention and thresholds, and records the measurements.
  The chosen model is the one stage 4 reuses.
- One vector per document, stored with the model name; a different model name means
  re-embedding (the old vectors are ignored, never compared with new ones).
- Vectors are L2-normalised so cosine similarity is a dot product.

### 5.2 Folder scores

For each folder, the score of a document is the **mean of its 3 highest similarities** with
the documents of that folder (fewer if the folder has fewer documents). Documents placed or
confirmed **by the user count twice** (they are duplicated in the comparison set). Documents
waiting for confirmation do count, in their proposed folder (so a folder proposed for the
first invoice attracts the next ones in the same run).

### 5.3 Decision

Let `best` and `second` be the two highest folder scores.

| Case | Condition (starting values) | Result |
|---|---|---|
| clear | `best ≥ 0.80`, `best − second ≥ 0.05`, the best folder holds ≥ 3 documents | **placed** by embeddings, no LLM call |
| … but the best folder is only proposed | same, folder not confirmed yet | **joins that proposal** (to confirm), no LLM call |
| otherwise | | **ask the LLM** (5.4) |

`0.80`, `0.05` and `3` live in a config object (like `QualityConfig`) and are starting
values; stage 7 will measure them. They are cosine values for `nomic-embed-text`; a builder
spike on synthetic French documents may adjust them before the plan freezes them (record it).

### 5.4 LLM step

- Model `qwen3:8b` through Ollama `/api/chat`, `temperature 0`, thinking off, output forced
  with Ollama's `format` = JSON schema.
- The prompt contains: the file name and the first 3,000 characters of masked usable text;
  the **3 best candidate folders** from 5.2 with 2 example file names each; the **full list of
  existing folders** (names only); up to **5 user corrections** nearest to this document
  (file name → folder) as examples; the naming rules (5.5).
- Output schema: `{"folder": "Domaine/Type", "new_folder": bool, "alternative": "Domaine/Type" | null, "reason": "one short sentence"}`.
- **Two judges:**
  - the LLM picks the embeddings' top candidate, no alternative → **placed** (method `llm`),
    or joins the proposal if that folder is not confirmed yet;
  - the LLM picks another existing folder, or gives an alternative → **to confirm,
    hesitation** between the two;
  - the LLM proposes a new folder → name checks (5.5) → **to confirm, new folder**.
- Invalid output (not JSON, schema violated, a name breaking the rules after one retry) →
  **to confirm**, reason `llm_invalid`, the document stays in "À trier".
- Document text is data, not instructions: the prompt says so, and whatever the text says,
  the output can only be a folder name that passes 5.5, and a new folder is never created
  without the user.

### 5.5 Folder names and sprawl

- Format `Domaine/Type`: exactly two non-empty parts, each ≤ 40 characters, letters, digits,
  spaces, `-`, `'`, `&` only; trimmed, collapsed spaces, NFC, first letter upper-case.
- A proposed name **equal ignoring case and accents** to an existing one → the existing one.
- Domains and types are compared **separately**, never as one `Domaine/Type` string (so
  `Maison/Factures` and `Santé/Factures` are never merged):
  - a proposed **domain** whose name embedding has cosine **≥ 0.90** with an existing domain
    → the existing domain;
  - then a proposed **type** whose name embedding has cosine ≥ 0.90 with a type already used
    **in that domain** → that type; otherwise, if a near-identical type exists in **another**
    domain, its spelling is used (`Contrats` everywhere, never `Contrat` here and `Contrats`
    there).
  - The summary says "proposed X, used existing Y". `0.90` is a config value like the others.
- At most one new folder proposal per document. Folders proposed and then refused by the user
  are remembered for the run and not proposed again for the same documents.

### 5.6 Proposed (unconfirmed) folders

A document is **placed** only in a **confirmed** folder (confirmed by the user, or created by
the user with `move`/`rename`). Anything that lands in a proposed folder joins that proposal's
group with status `to_confirm`, so the single "confirm the folder" question covers it. If the
user renames the proposal, its documents follow the new name; if the user sends it to another
folder, all its documents go there; `later` keeps the group. Unconfirmed folders are shown as
"(proposed)" in `folders` and are not offered as a search filter in stage 5.

### 5.7 Cold start

With no folders, every document goes to the LLM and proposes folders: all are "to confirm".
Documents are processed in order and a proposed folder immediately counts as a candidate
(5.2), so the first ingest produces a handful of **proposed folders with their documents**,
and the user confirms **a folder once for all its documents** (section 6), not 300 questions.

## 6. What the user sees

### 6.1 End of `ingest`

Classification runs automatically on documents that are new or whose text changed, after the
OCR offer. Each placed document line shows its folder:

```
 + new         facture_edf_mars.pdf            → Maison/Factures
 + new         decompte_ameli.pdf              → Santé/Remboursements

42 document(s) classified:
  38 placed               Maison/Factures (12), Santé/Remboursements (9), ...
   3 to confirm           new folder proposed: Loisirs/Billets (3 documents)
   1 hesitation           contrat_mutuelle.pdf → Santé/Contrats or Travail/Avantages?
Confirm the 4 now? [y/N]
```

(User-facing CLI copy is English like the rest of the CLI; the exact strings are fixed in the
plan.) `--no-classify` skips the step.

### 6.2 The review (interactive)

One question per **proposed folder** (covering all its documents), then one per hesitation:

```
New folder Loisirs/Billets for 3 document(s): billet_train.pdf, concert.pdf, expo.jpg
  [y] yes  [r] rename  [o] other existing folder  [l] later
contrat_mutuelle.pdf
  [1] Santé/Contrats  [2] Travail/Avantages  [o] other  [l] later
```

- `rename` / `other` take a `Domaine/Type` (checked by 5.5; an existing folder can be typed).
- `later` (and Ctrl+C, and EOF) leaves the rest in the queue; nothing is lost.
- Every answer is a **user decision** (section 7).

### 6.3 Non-interactive (stdin not a terminal)

No question. Placed documents are placed; the others wait in the review queue, and one line
says: `4 classification(s) to confirm: localmind folders review`.

### 6.4 Commands

- `localmind folders` — the tree with document counts, plus the review queue size.
- `localmind folders review [PATH]` — the interactive review of 6.2.
- `localmind folders show "Maison/Factures"` — the documents of a folder.
- `localmind move FILE "Domaine/Type"` — correction; creates the folder if needed (the user
  creating it counts as confirmed).
- `localmind folders rename "Old/Name" "New/Name"` — renames (or merges into an existing one).
- `localmind classify [PATH] [--all]` — classifies documents not classified yet (e.g. after
  Ollama was down); `--all` re-runs everything **except user decisions**.

### 6.5 Ollama absent or a model missing

Ingest finishes normally; documents stay unclassified ("À trier" with reason
`classifier_unavailable`) and one line says how to fix it (`ollama serve`, or
`ollama pull <model>`), then `localmind classify`. Never a traceback, never a hang
(short connection timeout; generous read timeout for the LLM).

## 7. Data model (schema v4, migrated in place from v3)

- `folders(id, domain, type, created_by 'user'|'llm', confirmed bool, created_at)`,
  `UNIQUE(domain, type)` on the normalised names.
- `classifications(document_id PK → documents ON DELETE CASCADE, folder_id → folders,
  status 'placed'|'to_confirm', method 'embedding'|'llm'|'user', score, alternative_folder_id,
  reason, model, classified_at)`. A document in the queue has `status = 'to_confirm'`.
- `embeddings(document_id, model, vector BLOB, PK(document_id, model))`.
- `folder_decisions(file_hash PK, domain, type, decided_at)` — the user's choice, keyed by
  **file hash** like 2a's decisions (survives re-ingest, rebuilds and document versions;
  applies to every exact copy). Never dropped by a migration, like `decisions` and `ocr_cache`.
- `folder_decisions` also stores the **path** of the file the user corrected (indexed).
  **User's choice (A): a correction follows the file's path.** When a file at that path is
  edited (new hash), the new version keeps the user's folder; the decision row gets the new
  hash, and the summary shows `→ Domaine/Type (your choice)`. A file replaced by unrelated
  content under the same name therefore stays in the old folder until the user moves it
  (accepted risk: always visible in the summary).
- A document's folder is: a user decision matching one of its files by hash, else by path
  (the most recent wins), else the classifier's result. A new version of a document (text
  changed) not covered by a decision is re-classified.
- A folder with no document left and `created_by = 'llm'` disappears; user-created folders
  stay until renamed away.
- The v3 → v4 migration only adds tables (no data rewritten); `user_version` is set inside
  the migration transaction.

## 8. Library API (JARVIS uses this, not the CLI)

- `classify(catalog, classifier, config=…, under=None, redo=False) -> ClassifyReport`
- `move(catalog, path, folder: str) -> None`, `rename_folder(catalog, old, new) -> None`
- `Classifier` protocol: `embed(texts) -> list[vector]`, `choose(prompt_data) -> dict`,
  plus `name` (models). `OllamaClassifier` is the real one; tests use a fake with scripted
  vectors and answers. `import localmind` imports nothing heavy.
- The Ollama client uses the Python standard library (HTTP + JSON); no new dependency.

## 9. Constraints

- Originals never moved, modified or deleted.
- Sensitive values never printed or logged, never sent to the models (masked).
- Models: defaults `qwen3:8b` and `nomic-embed-text`, overridable by `LOCALMIND_LLM` and
  `LOCALMIND_EMBED_MODEL`; Ollama URL `LOCALMIND_OLLAMA_URL` (default `http://localhost:11434`).
- 8 GB of VRAM: Ollama loads one model at a time as needed; classification never runs at the
  same time as OCR (it runs after it). The LLM runs with `num_ctx 4096` (measured: 5.6 GB, 100 %
  GPU; Ollama's default 16k context spills 20 % onto the CPU), and the last call of a run asks
  Ollama to unload it (`keep_alive: 0`), so JARVIS's own model (`qwen3.5:4b`, 3.1 GB) gets the
  GPU back. Classification happens at ingest time, never while JARVIS answers: JARVIS only
  reads folders from the catalog (no model call).
- Tests never call a real model except one opt-in test skipped when Ollama is not reachable.
- End-to-end checks only on generated scratch documents, never the user's files.

## 10. Left to the builder

- Prompt wording (within 5.4), the exact CLI strings, and the number formatting.
- Batching embeddings, timeouts, and the storage format of vectors (e.g. float32 bytes).
- Whether "À trier" is a real `folders` row or the absence of a classification.
- How rename/merge handles user decisions that point at the old name (they must follow).

## 11. Calls made after the builder check

- A document is placed only in a confirmed folder; documents landing in a proposed folder join
  its group (5.6). (designer call, follows the user's "new folder → consult")
- The embedding model is gated by a French spike and may switch to a multilingual model (5.1).
  (designer call)
- Domain and type names are compared separately for the anti-sprawl rule; 0.90 is
  configurable (5.5). (designer call)
- Corrections follow the file path when a file is edited in place (7). (user's choice: A)

## 12. Spike results (2026-11-22, generated French documents)

- `nomic-embed-text` on 35 generated documents of 7 types (EDF and internet invoices,
  payslips, health refunds, leases, prescriptions, tax notices): same-type cosine 0.970 on
  average (min 0.910), different-type 0.766 (max 0.848); leave-one-out top-1 35/35, all 35
  "clear" under 0.80 / 0.05 (min best 0.93, min margin 0.13). **Gate passed: the model stays.**
  Generated documents are more uniform than real ones; stage 7 measures on real data.
- `qwen3:8b`, 4k context: ~10 s for the first call (model load), then ~2.5 s per document.
- `qwen3.5:4b` (JARVIS's model) was tried as the classifier: not faster (3.5-5 s) and worse
  (put a payslip, a health refund and a lease all in `Maison`, ignored the two-level format).
  `qwen3:8b` stays the default.
