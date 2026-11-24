# Stage 5b: Fact extraction

> Goal: answer "how much did I pay for electricity in 2026?", "what are my next
> appointments?" and "on which day am I debited?" with exact sums and dates, each
> number traceable to the page it came from.

Search (stages 4-5) finds **passages**. It cannot add up every electricity invoice of a
year, nor filter on dates in the future: a top-5 of passages is not "all of them", and a
passage is text, not a number. Stage 5b reads each document once and stores its
**facts** as rows. Sums and date filters then become plain SQL.

## 1. What a fact is

| kind | what | example |
|------|------|---------|
| `payment` | money you paid, or that will be debited on a stated date | 84,20 €, 15/03/2026, EDF, debit |
| `due` | money you must pay, not said to be paid (a due date, a schedule line, "le 5 de chaque mois") | 1 152,00 €, 30/04/2026, Trésor Public |
| `appointment` | a meeting fixed at a date: date, time, place, with whom (never an amount) | 12/12/2026 14:30, Dr Martin |
| `bank_movement` | one line of a bank statement, signed | −84,20 €, 15/03/2026, "PRLV SEPA EDF" |

Every fact keeps its document, its page and the **span** (start, end) of its amount and
date in the page text: provenance, so any number can be shown in context.

## 2. Rules find, the LLM labels

A language model is good at *understanding* a page ("this is the amount to pay, that is
the VAT, this company is the one being paid") and bad at *copying numbers*: it turns
`1 234,56` into `1234.5`, or picks the date of the line above. So the work is split:

1. **Rules** (`candidates.py`, regular expressions, no model) find every amount, date and
   time on the page: `84,20 €`, `€ 1.234,56`, `15/03/2026`, `15.03.26`, `1er mars 2026`,
   `le 5 de chaque mois`, `14h30`… and write a **tag** before each one:

   ```
   Total TTC : [M2A] 84,20 €
   Ce montant sera prélevé le [D3P] 15/03/2026
   ```

2. **The LLM** (`qwen3:8b`, JSON schema) answers with tags only:
   `{"kind": "payment", "amount": "M2A", "date": "D3P", "counterparty": "EDF", …}`.

3. **The code** checks each fact and takes the values from the rules, never from the
   model. A fact is **rejected** (and counted) when a tag does not exist on that page,
   is of the wrong type (a date tag as an amount), was cut from the text the model saw,
   or breaks one of the rules below. A wrong guess costs a missing fact, never a wrong
   number.

### What the spike taught (spec 7.1)

Five versions on 12 scratch documents before writing the real code:

- Tags numbered `M1, D1, M2, D2…` made the model **count instead of copy**: on a bank
  statement it paired each amount with the date of the line above (the period dates
  were D1-D2). Every amount was right, every date wrong, and the check could not see it.
  Two fixes: one numbering for all tags in text order, written as **non-sequential codes**
  (`M2A`, `D3P`), so a counted guess lands on a tag that does not exist or has the wrong
  type; and the **same-line rule**: if the amount's line holds dates, the fact's date
  must be one of them.
- The model is asked the **document type first**. A payslip gives no fact (a salary is
  money you receive; the prompt alone did not stop "Net payé" from becoming a payment),
  a `bank_movement` only comes from a bank statement.
- Smaller rules: a payment on "le 5 de chaque mois" is a `due`; an appointment needs a
  full date and no amount; month-only dates ("mars 2027") are not candidates; identical
  facts on a page are merged.

## 3. Totals you can say out loud

`facts()` (`facts_query.py`) returns the rows, two totals and a coverage:

- **confirmed** = payments dated today or before (a "will be debited on 15/12" waits
  for its day); **unconfirmed** = dues dated today or before. Future dues and
  day-of-month dues are listed, never summed. Bank lines are never summed with invoices
  (the same money would count twice). Euro only.
- **Probable duplicates**: same counterparty, same amount, dates at most 5 days apart
  count once (a payment confirms a due). The other row stays in the list, marked.
- **Effective kind** at query time: a document in a folder whose type starts with
  "Relev" (`Banque/Relevés`) gives bank lines. Move it, and its facts follow, with no
  new reading.
- **Coverage**: documents not read yet, or read with failures. A voice assistant can then
  say "1 162,20 €, of which 1 078,00 € not confirmed; 2 documents not analysed yet".

## 4. Storage (catalog v7)

`facts`, `fact_runs` (one per document: model, prompt version, status, counts) and
`fact_pages` (status per page), all deleted with their document (`ON DELETE CASCADE`).
A document is read again only when the prompt version or the model changes, or to retry
its failed pages. At most 50 facts per document; beyond that the document is marked
incomplete.

## 5. Commands

```bash
uv run localmind ingest ~/Documents/inbox        # now also reads facts (--no-facts to skip)
uv run localmind facts extract [--all]           # read documents not read yet (or all)
uv run localmind facts list --kind payment --kind due --who edf --from 2026-01-01 --to 2026-12-31 --sum
uv run localmind facts list --upcoming --kind appointment
uv run localmind facts list --kind due --who free
```

## 6. Measured, end to end

Scratch documents, scratch catalog, real Ollama (RTX 4070 laptop):

```
$ localmind ingest docs/            # 12 documents: classify + index + facts in 2 min 27 s
Analysed 12 document(s): 37 fact(s)
  1 answer(s) rejected
$ localmind facts list --kind payment --kind due --who edf --from 2026-01-01 --to 2026-12-31 --sum
05/01/2026  due                98,00 €  EDF  echeancier_edf_2026.txt p.1
…
15/03/2026  payment            84,20 €  EDF  facture_edf_mars.txt p.1
…
05/12/2026  due                98,00 €  EDF  echeancier_edf_2026.txt p.1
Total: 1 162,20 € (confirmed 84,20 €, unconfirmed 1 078,00 €)
$ localmind facts list --upcoming --kind appointment
12/12/2026  appointment             --  bâtiment B, 2e étage  convocation_hopital.txt p.1
$ localmind facts list --kind due --who free
le 5 du mois  due                19,99 €  FREE MOBILE  contrat_free_mobile.txt p.1
```

The bank line `PRLV SEPA EDF −84,20 €` is listed as a `bank_movement` and stays out of
the total; the payslip gives nothing. Time: 3-5 s per ordinary page, 24-30 s for a page
with 12-15 lines (the model writes every fact). The end-to-end run found one bug, now
fixed with a test: "Devis accepté le 1er mars 2026 … 1 234,56 €" had become an appointment
with an amount.

## 7. Known limits

- A number just before an amount ("Mensualité 3 150,00 €", "n° 4 250,00 €") is read as a
  counter, not a thousands group: 150,00 €. A real "3 150,00 €" after such a word is read
  short (a smaller fact beats an inflated one).
- French formats only; an amount without `€`/`EUR`/`euros` is not seen.
- Long tables are slow (output-bound) and capped at 50 facts per document.
- The model decides what a value *is*: a wrong role (an amount that was a deposit, not a
  payment) is possible; the number itself is always the one written in the document.
- Matching a bank line to its invoice (reconciliation) is not done: they are kept apart.
- Turning a spoken question into filters (`--who edf --from …`) is the caller's job
  (JARVIS, or stage 6).
