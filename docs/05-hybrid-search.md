# Stage 5: Hybrid search and a folder filter

> Goal: find the passage even when the question and the document do not use the same
> words, or when the answer hangs on one exact token (an invoice number, a supplier,
> a plate number), and let a search stay inside one folder.

Stage 4 searches by **meaning** only. It is good at paraphrase ("électricité" finds
"consommation relevée") and weak at exact tokens: a reference like `2026-04-5120` or a
name like `Durand SARL` carries almost no meaning a vector can hold. Keyword search is
the opposite. Stage 5 runs both and fuses their rankings.

## 1. Keyword search: BM25 in one paragraph

BM25 scores a passage for a set of words. A word counts more when it appears **often in
the passage** (term frequency, with diminishing returns: the tenth "facture" adds little),
when it is **rare in the collection** (inverse document frequency: "AB-123-CD" is in one
passage, "contrat" is in half of them), and a long passage is slightly penalised so it
does not win just by holding more words. No model, no training: counting.

localmind does not code BM25 by hand: SQLite has it. **FTS5** is SQLite's full-text
index (an *inverted index*: for every word, the list of passages that hold it) and
`bm25()` is one of its functions.

## 2. One source of truth: an external-content index

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
```

- **External content** (`content='chunks'`): the text is stored once, in `chunks`; the
  FTS table only holds the inverted index. Three triggers (after insert, delete, update
  of `chunks.text`) keep it in step, in the same transaction. A chunk deleted by a
  cascade (an edited file re-ingested, `prune`) fires the delete trigger too: tested.
- **Masked text only**: chunks are cut from masked pages (stage 4), so the index holds
  `[IBAN]`, never the digits. Tested: the digits of an IBAN find nothing.
- **Accents ignored** (`remove_diacritics 2`): `electricite` finds `électricité`.
- Schema **v6**. An older catalog is migrated in place: the table and triggers are
  created, then `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` indexes the
  chunks already there. No vector is recomputed.

## 3. From a question to a keyword query

`keyword.py`: the question's words (letters and digits), lowercased, minus a short list
of French stop words (`le`, `de`, `mon`, `quel`, `combien`...), each **in double quotes**,
joined with `OR`:

```
"Quel est le préavis de mon bail ?"  ->  "préavis" OR "bail"
```

Quoting every word means the user's text is never read as FTS5 syntax: `l'électricité`,
`"`, `*`, `-`, `NEAR(`, `AND`, a lone `(` are searched as plain words and never raise.
A question made only of stop words gives no keyword query (the keyword side is empty).

## 4. Fusing two rankings: reciprocal rank fusion (RRF)

A cosine (0.62) and a BM25 score (-7.3) are not on the same scale, and weighting them
(`a·cos + (1-a)·bm25`) needs an `a` tuned on data we do not have. RRF ignores the scores
and only uses **ranks**:

```
fused(chunk) = Σ over the two rankings that hold it of 1 / (60 + rank)
```

A worked example, real numbers from the end-to-end run (with the chunking that was later
replaced, see section 7), question "quel jour suis-je prélevé pour mon abonnement
téléphonique ?":

| chunk | meaning rank | keyword rank | fused |
|---|---|---|---|
| box internet, "prélèvement automatique le 18" | 3 | 5 | 1/63 + 1/65 = 0.0313 |
| mobile contract, "...prélevé... le 5 de chaque mois" | 28 | 2 | 1/88 + 1/62 = 0.0275 |
| a payslip's legal small print | 1 | (none) | 1/61 = 0.0164 |

Agreement wins: a chunk both sides rank well beats a chunk one side ranks first. The
constant 60 (the value from the RRF paper) flattens the difference between rank 1 and
rank 3, so one side alone cannot dominate. Ties go to the better meaning rank, then the
smaller chunk id. Each side contributes its `max(30, 4·k)` best chunks.

`Hit.score` stays the **cosine** (it means something to a reader); the order is the
fused order. `Hit.matched` says which side found it: `{"meaning"}`, `{"keywords"}` or
both.

## 5. A folder filter, applied before ranking

```bash
localmind search "prélèvement" --folder Maison            # a whole domain
localmind search "électricité" --folder maison/factures   # one folder (case, spaces, accents ignored)
```

- A document matches through the folder its classification **chose**, reviewed or not.
  Documents the LLM placed but you have not reviewed yet are **included and marked**
  `(to confirm)`: a filter must never silently hide a document the LLM put there.
  An LLM hesitation counts for its chosen folder, never for its second guess.
- Unclassified documents never match a filter.
- An unknown folder is an error naming it (`error: no folder named 'Maison/Factur'`,
  exit 2), not an empty result. A known folder holding nothing gives no hit.
- The filter is applied **before** ranking, on both sides (vector rows masked by
  document, FTS query restricted to the allowed documents): `-k 2` gives two hits from
  the folder, not the global top 2 filtered down to zero.

## 6. Without Ollama: keywords still answer

Keyword search needs no model. If the question cannot be embedded (Ollama stopped,
model missing), search returns keyword hits only, with no score (`  --`), and says so
once on stderr; exit code 0 (stage 4 exited 2):

```
meaning search unavailable (Ollama not reachable at http://127.0.0.1:9: start it (ollama serve)): keyword results only
  --  bulletin_paie_janvier.txt p.1  Travail/Bulletins de paie (to confirm)  "BULLETIN DE PAIE Période : janvier 2026 Employeur : Société Durand SARL…"
```

In Python, `VectorIndex.last_warning` holds the reason (None otherwise): JARVIS can still
answer something useful. A chunk with no vector for the current model (never indexed
with it) is still found by keywords, with `score = None`.

## 7. Measured before building, and once more end to end

The stage 4 scratch documents (11 French text files), real `nomic-embed-text`, right
chunk in the top 3, 10 questions plus 5 exact-word questions:

| chunking | meaning only | keywords only | hybrid | phone question, hybrid rank |
|---|---|---|---|---|
| 800/120/200 (stage 4) | 9/10 | 9/10 | 9/10 | 6 |
| **600/90/150** | **10/10** | 8/10 | **10/10** | **1** |
| 500/75/125 | 10/10 | 9/10 | 10/10 | 1 |

Exact-word questions were 5/5 for every method: on 11 documents they do not separate the
methods. The phone question failed at 800 because its answer sat in a chunk that is
mostly legal small print (meaning rank 18); smaller chunks give it its own.

The end-to-end check caught a mistake: the spike scaled the overlap (15 %) and the
minimum tail (a quarter) with the size, but the first change only set the size, giving
600/120/200, and the phone question was 6th again. A 200-character minimum glued the
answer back onto the small print. Chunking is now **600/90/150**; end to end on the real
pipeline: 10/10, exact words 5/5. Changing it re-cuts everything on the next
`localmind index --all` (the parameters are stored with every chunk).

Honest limit: 10 questions, and the size was chosen on those same questions. Whether a
cut lands before or after one sentence decided one question. Stage 7 measures on real
documents.

**Speed** (in-process, 44 chunks, question embedding included): median 44 ms, max
61 ms; the keyword side alone takes 0.5 ms. Well under the 100 ms target.

## 8. What an end-to-end run looks like

Scratch documents, scratch catalog, nothing of yours:

```
$ localmind search "quel jour suis-je prélevé pour mon abonnement téléphonique ?" -k 3
0.67  contrat_mobile_sfr.txt p.1  Maison/Contrats (to confirm)  "exercer auprès de notre délégué à la protection des données. MODALITÉS DE PAIEMENT Le montant de vot…"
0.65  contrat_mobile_sfr.txt p.1  Maison/Contrats (to confirm)  "CONTRAT D'ABONNEMENT MOBILE Forfait 5G 100 Go Titulaire : M. Martin Le forfait comprend les appels, …"
0.65  contrat_box_internet.txt p.1  Maison/Contrats (to confirm)  "Votre abonnement box est réglé par prélèvement automatique le 18 de chaque mois. Les informations fi…"
$ localmind search "combien m'a coûté l'électricité ?" --folder maison/factures -k 2
0.68  facture_edf_avril.txt p.1  Maison/Factures (to confirm)  "FACTURE D'ÉLECTRICITÉ Période de consommation : avril 2026 Numéro de facture 2026-04-5120 Les inform…"
0.66  facture_edf_mars.txt p.1  Maison/Factures (to confirm)  "FACTURE D'ÉLECTRICITÉ Période de consommation : mars 2026 Numéro de facture 2026-03-4471 Consommatio…"
$ localmind search "salaire" --folder Maison/Factur
error: no folder named 'Maison/Factur'
$ localmind search "IBAN RIB coordonnées bancaires" -k 1
0.79  note_rib.txt p.1  Maison/Contrats (to confirm)  "Pour le loyer du bail, voici mes coordonnées bancaires : IBAN [IBAN]. Merci."
```

(The LLM put the appointment letter and the RIB note in `Maison/Contrats`: classification
is stage 3's job, and `localmind folders review` is where you fix it.)

## 9. Known limits

- Keywords match **words**, not meanings or forms: "prélevé" does not find
  "prélèvement" (no stemming). The meaning side covers that.
- Stop words are a short hand-made French list; an English question keeps all its words.
- RRF needs both sides to be reasonable: if one side ranks noise high, fusion averages
  it in. A reranker (a model reading question and passage together) would be the next
  step if measurements ask for it.
- Folders filter only; dates and amounts ("my March invoices over 100 €") need facts
  extracted from documents: a later stage.
