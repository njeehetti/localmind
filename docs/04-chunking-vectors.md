# Stage 4: Chunking, embeddings and a vector store

> Goal: find the **passage** that answers a question ("which day is my phone
> subscription debited?"), in your own documents, on your own machine, in a few
> milliseconds.

Stage 3 has one vector per document, made from its first 2000 characters: good for
sorting, useless for finding a sentence on page 3. Stage 4 cuts every document into
**chunks**, gives each chunk its own vector, and searches them.

## 1. Chunks: one vector per idea

A vector is an average of everything in its text. A whole invoice page (supplier,
amount, consumption, legal small print) gives a blurred vector; a passage of a few
hundred characters stays about *one* thing. localmind cuts each page into chunks of
about **600 characters** (800 until stage 5, see section 7):

- cuts fall on the most natural boundary that fits: a blank line (a paragraph), then a
  line break (OCR text and tables rarely have blank lines), then the end of a
  sentence, then a space; **never inside a word**;
- two consecutive chunks share **120 characters** (overlap), so a sentence cut at the
  edge of one chunk is whole in the next;
- a tail shorter than 200 characters joins the previous chunk;
- a chunk **never spans two pages**, so every result can be cited `bail.pdf p.3`.

The parameters are stored with each chunk (`para-v1:600/120/200`): change them and
`localmind index --all` cuts everything again.

## 2. A header, and two kinds of prefix

A chunk that reads `Montant total TTC : 112,47 €` does not say what it is about. So
what gets embedded starts with a **header**, the file name and page:

```
search_document: facture_edf_mars.txt p. 1
Montant total TTC : 112,47 € ...
```

`nomic-embed-text` is trained with **task prefixes**: `search_document: ` for the
passages, `search_query: ` for the question. The two are deliberately different
(**asymmetric** retrieval): a question ("which day?") does not look like its answer
("debited on the 5th"), and the prefixes tell the model to bring them together.
Stage 3 keeps its own `classification: ` prefix.

The folder name is *not* in the header: folders change (review, `rename`, `move`), and
every vector would have to be redone. Stage 5 filters by folder instead.

## 3. Nothing sensitive in a chunk

Chunks are cut from the **masked** text (`[IBAN]`, `[CARD]`, `[NIR]`, `[PASSWORD]`, found
in stage 2a). A chunk never holds a sensitive value, so no model, no search result and
no answer can leak one. The price, accepted: localmind will never answer "what is my
IBAN?".

## 4. Where the vectors live: the catalog itself

Two new tables (schema v5):

```
chunks          document_id, page_no, seq, text (masked), chunker
chunk_vectors   chunk_id, model, vector (768 float32, length 1)
```

Before building, two ways were compared:

| | A: SQLite + numpy (chosen) | C: Chroma / LanceDB |
|---|---|---|
| stores | one: the catalog | two, to keep in sync |
| a file edited, copied, pruned | its chunks follow, in the same transaction | sync code for every case |
| filter by folder (stage 5) | a SQL join | copy the folder into each vector's metadata |
| keyword search (stage 5) | SQLite FTS5, same file | LanceDB yes, Chroma less so |
| search speed | ~5 ms for 10 000 chunks | < 1 ms, even for millions |
| dependencies | numpy | dozens of packages |

Speed does not decide: an answer from the LLM takes seconds, and personal papers will
not reach the hundreds of thousands of chunks where an approximate index (HNSW, IVF)
starts to matter. **One source of truth** decides.

## 5. Searching: a matrix product

All vectors of the current model are loaded into one numpy matrix `M` (one row per
chunk). Every vector has length 1, so the cosine with the question `q` is a dot
product, and all of them at once are one product:

```
scores = M @ q                       # one cosine per chunk
best   = argpartition(-scores, k)    # the k best, without sorting everything
```

This is **brute force** (exact nearest neighbours): every chunk is compared, no
approximation. `VectorIndex` keeps the matrix in memory between questions (JARVIS asks
many in one process) and rebuilds it when the catalog changed.

## 6. Files you deleted

The catalog keeps a file it cannot find (it may be on an unplugged drive), but search
never cites one, nor a file edited since the last `ingest` (its stored text is outdated):
each result is checked on disk (exists, same size), an unchanged copy is named instead,
or the passage is skipped and the next one taken. To forget them for good:

```
$ localmind prune
1 file(s) no longer exist:
  ...\inbox\facture_edf_mars.txt
Forget them? [y/N] y
Forgot 1 file(s) and 1 document(s).
```

Their pages, chunks, vectors and folder go with them. Your decisions (accept, folder,
OCR already done) stay, keyed by content: a file put back finds them again. Nothing on
disk is ever touched.

## 7. Measured before building (spike)

10 generated French documents (invoices, contracts, payslips, a lease, an appointment
letter...), 10 questions, right chunk in the top 3:

| chunk size | with header | without header |
|---|---|---|
| 500 | 10/10 | 8/10 |
| **800** | **9/10** | 8/10 |
| 1200 | 9/10 | 8/10 |

The header always helps. 800 was kept at first (500 wins by only one question); the
stage 5 spike moved it to **600** (10/10 meaning-only and hybrid, see
`05-hybrid-search.md`). The one miss is
telling: "quel jour suis-je prélevé pour mon abonnement **téléphonique** ?" against a
contract that says "forfait **mobile**" and buries the debit day among legal small
print. Meaning-based search does not save every wording; stage 5 adds keyword search
(BM25) and stage 7 measures on real documents.

## 8. A slow-localhost trap (found end to end)

The first end-to-end run took **5 s per search**. Profiling: 4 s in the embedding call
alone, model already loaded. Cause: on Windows, Python resolves `localhost` to IPv6
`::1` first; Ollama listens on IPv4, and every HTTP request paid about 2 s before
falling back. With `http://127.0.0.1:11434` (the new default) the same call takes
50 ms. It also cut stage 3's classification of 11 documents from 78 s to 30 s.

## 9. Using it

```bash
uv run localmind ingest ~/Documents/inbox     # reads, classifies, then indexes
uv run localmind search "quel est le préavis de mon bail ?"
uv run localmind index                         # index what is missing (e.g. after OCR)
uv run localmind index --all                   # cut and embed everything again
uv run localmind prune                         # forget files deleted from disk
uv run localmind ingest ... --no-index         # skip the step
```

```python
from localmind import Catalog, OllamaClassifier, VectorIndex, default_db_path

with Catalog(default_db_path()) as catalog:
    vi = VectorIndex(catalog, OllamaClassifier())
    vi.warm()                                  # load the embedding model now
    for hit in vi.search("quand expire mon assurance auto ?", k=3):
        print(f"{hit.score:.2f}", hit.path.name, hit.page_no, hit.text[:80])
```

Without Ollama, `ingest` still works and prints one line saying what to run later.

## 10. What an end-to-end run looks like

The spike's 10 documents plus a note holding an IBAN, real Ollama, 8 GB laptop GPU:

```
$ localmind ingest inbox < /dev/null
...
11 classification(s) to confirm: localmind folders review
Indexed 11 document(s): 29 chunk(s)

$ localmind search "quand expire mon assurance auto ?" -k 1
0.70  attestation_assurance_auto.txt p.1  "de validité : du 1er avril 2026 au 31 mars 2027. Le contrat se renouvelle..."

$ localmind search "coordonnées bancaires pour le loyer" -k 1
0.73  note_rib.txt p.1  "Pour le loyer du bail, voici mes coordonnées bancaires : IBAN [IBAN]. Merci."
```

- ingest + index without classification: **2 s** for 11 documents; with
  classification: 30 s (the LLM);
- one search: **~70 ms** in a running process (`VectorIndex`), ~1.2 s from the command
  line (mostly starting Python);
- the passage holding the answer is in the top 3 for **9 of the 10** spike questions;
  the miss is again the phone question;
- after deleting one invoice, search no longer cites it, and `prune --yes` forgot
  1 file and 1 document.

**Known limits**: no folder filter and no keyword search until stage 5. Sums and dates
across documents ("how much did I pay for electricity in 2026?", "when are my next
appointments?") need facts extracted from each document: that is a later stage, not
something nearest-passage search can do reliably.
