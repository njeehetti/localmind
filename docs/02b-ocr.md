# Stage 2b: OCR for scans and photos

> Goal: read the pages stage 2a set aside as `needs_ocr` (scanned PDFs, photos of
> documents), on your own machine, and keep track of how sure the reading is.

## 1. What OCR is, and why it is a guess

A scanned page is a **picture** of text: the PDF contains pixels, not letters.
OCR (Optical Character Recognition) looks at the pixels and *guesses* the letters.
Like any guess, it can be wrong, so localmind stores, for every OCR-ed page:

- the text,
- its **source** (`ocr`, as opposed to `text_layer` for real digital text),
- the **engine** that read it (`doctr-1.1.0/db_resnet50+crnn_vgg16_bn`),
- its **confidence** (0 to 1).

Stage 6 will be able to cite "page 2 (OCR, 94 %)" instead of presenting a guess as a fact.

## 2. docTR: two neural networks in a row

localmind uses [docTR](https://github.com/mindee/doctr), running on PyTorch:

1. **Detection** (`db_resnet50`): a segmentation network looks at the whole page and
   draws a box around every word.
2. **Recognition** (`crnn_vgg16_bn`): each box is cropped, and a second network reads
   it character by character, with a confidence per word.

Both models handle French and English (accents included) and are downloaded once, the
first time they are used (about 100 MB, cached in `~/.cache/doctr`). After that,
nothing leaves the machine.

A GPU does the many small matrix multiplications of these networks in parallel:
on an RTX 4070 a page takes well under a second; on a CPU, several seconds.

## 3. Page confidence

docTR gives a confidence per word. localmind averages them **weighted by word
length**, so a misread "a" counts less than a misread "Facture":

```
page confidence = sum(len(word) * confidence(word)) / sum(len(word))
```

| OCR result on a page | Verdict |
|---|---|
| fewer than 20 visible characters (nothing, a lone page number, specks) | `blank`: stored, **not** a problem (the empty back of a duplex scan) |
| confidence < 0.70 (`QualityConfig.min_ocr_confidence`) | set aside: `ocr_low_confidence` |
| the engine crashed on that page | set aside: `ocr_failed` (retried next time) |
| otherwise | the OCR text replaces the missing (or thin) text layer, then the usual rules apply (garbled, too short, sensitive data...) |

A watermark layer such as "Scanned with CamScanner" is **replaced** by the OCR text: it
is not content. An IBAN on a scanned bank detail sheet is flagged like any other.

0.70 is a starting value: stage 7 (evaluation) will measure it on real documents.

## 4. Never read the same page twice

OCR is slow, so every result is saved in the `ocr_cache` table, keyed by
**(file hash, page number, engine)**, the moment it is read:

- an exact copy of a scan (same bytes) reuses the result;
- `quarantine retry`, `accept` and `ignore` reuse it even without the OCR extra;
- Ctrl+C in the middle of a long run keeps the pages already read.

Changing the engine or its models changes the engine name, so stale results are never
reused by mistake.

## 5. Asked, not imposed

`ingest` never starts OCR on its own. When it leaves scanned pages behind, it asks:

```
2 scanned page(s) in 2 file(s) can be read by OCR (~1 s on GPU).
Run OCR now? [y/N]
```

- Only the paths you just ingested are counted; scans waiting elsewhere get a one-line
  reminder (`quarantine retry` reads them).
- `--ocr` / `--no-ocr` answer in advance.
- When nobody can answer (a script, a voice assistant: stdin is not a terminal), nothing
  is asked and nothing slow starts; one line says how to run it.
- Without the `ocr` extra, one line says how to install it.

## 6. Install, size and privacy

```bash
uv sync --extra ocr     # python-doctr + PyTorch with CUDA (~2 GB download)
uv cache prune          # optional: reclaim the download cache
```

`import localmind` never imports PyTorch: the base install stays small, and the engine
is only loaded when OCR actually runs. Photos and PDF pages are turned into images with
Pillow and pypdfium2, one page at a time (a 300 dpi A4 page is about 25 MB in memory).
Phone photos taken sideways are turned upright using their EXIF orientation.

## 7. What an end-to-end run looks like

On a scratch folder (a scanned lease, its copy, an invoice whose second page is
scanned, a sideways phone photo of a receipt, a blank page):

```
  OCR page 1/5: copie_scan.pdf p.1
  OCR page 2/5: facture_partielle.pdf p.2
  OCR page 3/5: page_blanche.png p.1
  OCR page 4/5: ticket_photo.jpg p.1
 + new         copie_scan.pdf
 ^ updated     facture_partielle.pdf
 ! quarantined page_blanche.png  (ocr_no_text)
 = duplicate   scan_contrat.pdf  (same text as a known file)
 + new         ticket_photo.jpg
```

The copy of the lease was not read twice (4 readings for 5 pages), accents came out
right ("relevé", "début"), and page confidences were between 0.89 and 0.94.
