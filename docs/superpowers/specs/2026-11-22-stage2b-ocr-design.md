# Stage 2b: OCR for scanned pages and photos (design)

Status: approved · Date: 2026-11-22 · Builds on stage 2a
(`2026-11-22-stage2a-quarantine-design.md`).

## Intent

Stage 2a sets aside every page that has no usable text layer (`needs_ocr`):
fully scanned PDFs, scanned pages inside otherwise digital PDFs, scans with a
thin watermark layer ("Scanned with CamScanner"), and photos (`.jpg`, `.png`…).
For a person who feeds localmind their real paperwork (contracts, invoices,
receipts, letters, in French and English), that is often most of the
interesting content. Stage 2b reads those pages with a neural OCR engine
running **on the user's own machine** (GPU when available) and brings them back
into the catalog, page by page, without ever claiming more certainty than the
OCR actually has.

It is also a learning project: the user wants to understand PyTorch, the GPU
and what an OCR model returns, so the code stays readable and the docs explain
each piece.

## Decisions taken with the user

1. **OCR runs on demand, not inside every ingestion** (option A). `ingest`
   stays fast and never needs the GPU.
2. **But the user is asked, not expected to remember a command.** At the end
   of an interactive `ingest` that leaves OCR-able pages behind, the CLI asks
   (CLI text in English like the rest of the CLI):

   ```
   14 scanned pages in 6 files can be read by OCR (~1 min on GPU).
   Run OCR now? [y/N]
   ```
   - `y` runs it immediately; `N` / Enter does nothing and the question comes
     back at the next ingestion; `quarantine retry` still works.
   - **Non-interactive** (stdin not a TTY: scripts, JARVIS): never ask, never
     block, never run OCR implicitly; print one info line instead.
   - `ingest --ocr` / `--no-ocr` answer in advance.
   - The future web UI will show the same offer as a button (out of scope).
3. **Engine: docTR on PyTorch**, installed as an optional extra
   (`uv sync --extra ocr`), with a CUDA build of PyTorch so the RTX 4070 is
   used. Rejected: OnnxTR (lighter, same models, but GPU setup on Windows is
   fiddly and it teaches less), Tesseract (worse on photos; may come later as a
   second engine).
4. **Several engines possible** through one interface; only docTR is
   implemented now.
5. From stage 2a: decisions are page-level; partial documents are fine.

## Goals

- Read only what needs it: pages set aside as `needs_ocr` and image files.
  Pages that already have a good text layer are never OCR-ed.
- An image file (`.png .jpg .jpeg .tif .tiff .bmp .webp`) becomes a one-page
  document once read.
- Every OCR-ed page goes through the same stage 2a page rules (garbled,
  sensitive data…) as any other text. An IBAN on a scanned RIB is flagged.
- Confidence is kept and used: a page whose OCR confidence is below a
  threshold stays set aside with reason `ocr_low_confidence`; a page where OCR
  finds no text stays set aside with reason `ocr_no_text`.
- Traceability: each stored page records where its text came from
  (`text_layer` or `ocr`), and for OCR the engine name and confidence, so
  stage 6 can cite "page 2 (OCR, 96 %)".
- OCR work is never redone needlessly: results are cached by
  (file hash, page number, engine). A second `retry`, or an exact copy of the
  scan elsewhere, reuses them.
- Without the extra installed, everything works as in 2a; the CLI says how to
  install OCR instead of asking to run it.

## Non-goals

- Handwriting, tables/layout reconstruction, reading order of multi-column
  pages beyond what docTR gives.
- Language detection or per-language settings (docTR's default recognition
  model reads French and English).
- A second engine (Tesseract…): only the interface is designed for it.
- Web UI button (bonus stage).
- Re-OCR of pages whose text layer is already accepted.

## Anti-goals (failure even if it "works")

- Any image, page or text leaving the machine. The only network access
  allowed is docTR downloading its model weights once (then cached locally).
- The base install getting heavier: `uv sync` without the extra must not pull
  PyTorch, docTR or OpenCV. Importing `localmind` must not import torch.
- The CLI blocking on a question when nobody can answer it.
- OCR text presented as if it were certain (no source / confidence kept).
- Losing user decisions (accept/ignore) when the catalog schema changes.
- Modifying, moving or deleting original files (unchanged rule).
- Printing sensitive values found in OCR text (unchanged rule: `kind @ page N`).

## Constraints

- Machine: Windows 11, Python 3.13.5, RTX 4070 Laptop 8 GB, NVIDIA driver
  610.91, 16 GB RAM. C: has ~37 GB free: tell the user the install size and
  how to clear the uv cache afterwards.
- docTR `python-doctr` 1.1.0 (requires Python ≥ 3.11; pulls torch,
  torchvision, opencv-python, pypdfium2…). `pypdfium2` renders PDF pages to
  images (pypdf cannot).
- Public repository: other users may have no NVIDIA GPU. The engine must fall
  back to CPU (with a one-line notice), and the install instructions must say
  how to get a CPU-only build.
- The test suite must run without the extra (CI-friendly): OCR logic is tested
  with a fake engine; a real-docTR test exists but is skipped when docTR is not
  installed.
- Stage 2a rules stay as they are; stage 2a deferred minor #1 (a schema bump
  would erase decisions) must be fixed here since this stage bumps the schema.

## Approaches considered

| | docTR + PyTorch (chosen) | OnnxTR | Tesseract |
|---|---|---|---|
| Quality | docTR models | same models | weaker on photos |
| Size | ~3 GB + cache | ~0.5 GB | small, but system install |
| GPU on Windows | works out of the box with CUDA wheels | needs manual CUDA/cuDNN DLLs | CPU only |
| Learning | PyTorch, tensors, GPU | ONNX export/runtime | classic OCR |
| Reuse | torch reusable for embeddings (stage 4) | OCR only | OCR only |

The user chose docTR + PyTorch.

## Design

### Flow

```
ingest (unchanged)  ──►  pages set aside "needs_ocr" / image files quarantined
        │
        └─ interactive and OCR-able pages exist ──► "Run OCR now? [y/N]"
                                                         │ y
quarantine retry ───────────────────────────────────────┤
                                                         ▼
       for each candidate file: extract → for each needs_ocr page:
           cache hit? use it : render page → engine.read(image) → cache
       → replace that page's text, source=ocr, confidence
       → assess (2a rules + OCR confidence rule) → store (new / partial / …)
```

- OCR is used when it is **available and requested**: `retry` with OCR
  installed, the `y` answer, or `ingest --ocr`. `ingest` alone never loads the
  engine.
- Which pages: those the 2a rules mark `needs_ocr` (no text layer, or image
  page with a thin text layer). For those pages the OCR text **replaces** the
  thin text layer (a watermark is not content).
- Image files are OCR candidates instead of being refused at file level when
  OCR is used; without OCR they stay quarantined `needs_ocr` exactly as today.

### Calls made after the builder check (designer's calls, reviewable)

- **Cached OCR always applies.** Using a cached OCR result is not "running
  OCR": every re-assessment (`accept`, `ignore`, `retry`, `retry --no-ocr`,
  `retry` without the extra) reuses cached OCR text for `needs_ocr` pages.
  `--no-ocr` means "do not run the engine", never "forget OCR results".
  Otherwise an `accept` on one page would silently drop every page OCR
  recovered.
- **Blank pages are not problems.** A page where OCR finds no words, or fewer
  than `min_page_chars` visible characters (a lone page number, specks), gets
  a new page status `blank` (reason `ocr_no_text`): stored, not indexed, but
  it does **not** make the document partial, does not enter the duplicate key,
  is not a retry candidate and is not listed under `quarantine` (only counted
  in `stats`). Duplex scans have many empty backs. A file whose pages are all
  blank is quarantined `ocr_no_text`.
- **Multi-page images** (multi-frame TIFF and similar) are read frame by
  frame, one page per frame; other images are one page.
- **Rules for OCR text:** the two PDF text-layer rules (min visible chars,
  thin layer on an image page) do not apply to OCR pages; instead: no words /
  too short → `blank`; confidence below threshold → `ocr_low_confidence`;
  then the normal page rules (garbled) and document rules (too short, machine
  log, sensitive) as for any text.
- **Engine name includes its version and models** (e.g.
  `doctr-1.1.0/db_resnet50+crnn_vgg16_bn`), so changing models never reuses a
  stale cache entry. A page that makes the engine fail is set aside with
  reason `ocr_failed` (with the error class, never page content) and the run
  continues.

### Engine interface

- `OcrEngine` (protocol): a `name` (e.g. `"doctr"`) and a method that takes one
  page image and returns an `OcrResult`: the page text (lines joined in
  docTR's reading order) and a page confidence in [0, 1].
- Page confidence = average of word confidences weighted by word length (long
  words count more than "a"). No words → `ocr_no_text`.
- `load_engine()` returns the docTR engine or raises a clear "OCR not
  installed" error. torch/docTR are imported only inside it (lazy import).
- The docTR engine uses CUDA when `torch.cuda.is_available()`, otherwise CPU
  with a notice. The model is loaded once per run, not per page.
- PDF pages are rendered with pypdfium2 at a resolution suited to docTR
  (builder's call, ~200-300 dpi); image files are opened with Pillow (EXIF
  rotation applied).

### Thresholds (`QualityConfig`)

- `min_ocr_confidence = 0.70`: below it, the page is set aside as
  `ocr_low_confidence`. Like other level A reasons, `accept` cannot force it
  (there is no trustworthy text); a better engine or scan can.
  This is a starting value, to revisit at stage 7 (evaluation) like the other
  thresholds.

### Catalog (schema v3)

- `pages` gains `source` (`text_layer` | `ocr`), `ocr_engine` (nullable),
  `ocr_confidence` (nullable).
- New table `ocr_cache(file_hash, page_no, engine, text, confidence,
  created_at)`, primary key (file_hash, page_no, engine). Independent of
  documents, so rebuilding a document never loses OCR work.
- Migration v2 → v3 is **in place** (add columns with default `text_layer`,
  create table); it must not drop `decisions` or any data. v1 catalogs are
  still rebuilt as in 2a, but `decisions` is never part of a rebuild from now
  on. (Fixes 2a deferred minor #1.)
- New reasons in `Reason`: `ocr_low_confidence`, `ocr_no_text`, `ocr_failed`.
- `pages.status` accepts `blank` in addition to `ok` and `set_aside`.

### CLI

- `ingest` end-of-run offer as described above; `--ocr` / `--no-ocr`.
  **Scope (user's decision): only the paths just ingested.** Count = OCR-able
  pages (set-aside `needs_ocr` pages + quarantined image / scan files) under
  those paths, not already cached; `y` runs OCR on those paths only. If other
  OCR-able pages wait elsewhere in the catalog, add one reminder line:
  `(300 more scanned pages are waiting: localmind quarantine retry)`.
  Time estimate from a per-page constant (GPU vs CPU, builder's call,
  labelled "~").
- Without the extra: no question; one line
  `N scanned pages could be read with OCR: uv sync --extra ocr`.
- `quarantine retry [PATH]`: uses OCR when installed; shows progress
  (`page 3/14…`) since it can take minutes; `--no-ocr` keeps the 2a behaviour.
- `quarantine` listing shows OCR reasons with confidence
  (`page 2 [ocr_low_confidence 58 %]`).
- `stats` adds `ocr pages: N`.

### Install

- `[project.optional-dependencies] ocr = [python-doctr, torch, torchvision]`
  (pypdfium2 comes with docTR), with uv configured so torch comes from the
  PyTorch CUDA wheel index on Windows/Linux. Exact CUDA variant: the newest one
  the installed torch release publishes that driver 610 supports (builder's
  call, recorded in the docs).
- README: `uv sync --extra ocr`, expected size, `uv cache prune` to reclaim
  space, CPU-only alternative for machines without NVIDIA.

## Testing

- Fake engine returning scripted text/confidence: covers which pages get OCR-ed
  (only `needs_ocr`), images becoming 1-page documents, confidence threshold,
  `ocr_no_text`, cache reuse (engine called once for two copies / two retries),
  source/confidence stored, sensitive data found in OCR text, partial → full
  document, OCR text replacing a watermark layer.
- CLI: interactive prompt yes / no (simulated TTY input), non-interactive never
  prompts and never blocks, `--ocr` / `--no-ocr`, message when OCR is not
  installed.
- Migration: a v2 catalog with decisions opens as v3 with decisions intact.
- Import hygiene: importing `localmind` does not import `torch`.
- Real docTR test (skipped without the extra): render a short French sentence
  to an image with Pillow, read it, check the words and a high confidence.
- Manual end-to-end on scratch files only (scanned PDF, partial PDF, photo of
  text, blank image), never the user's real documents.

## Left to the builder

Module layout (e.g. `ocr.py` for interface + docTR engine), rendering dpi,
per-page time constants, progress display, exact uv index configuration,
how the CLI detects interactivity (`sys.stdin.isatty()` is fine) and how tests
simulate it.
