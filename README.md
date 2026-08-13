# OCR → Structured DB Pipeline

Extracts text, tables and key fields from documents (starting with electricity bill and financial
quarterly reports) using PaddleOCR / PP-StructureV3, verifies the extraction and loads into a database schema that's ready to be leveraged for any future projects

Entire pipeline is split into five independent stages:
```
Ingestion  →  OCR + Layout  →  Structuring  →  Verification  →  DB
(PDF/img)     (PP-StructureV3)  (doc-type       (confidence +
              + PaddleOCR)       schema)          checksum)
```
- doc-type-specific logic (financial report vs. electricity bill vs. handwritten
  notes etc) is isolated mostly to `structuring.py`
- OCR is run once per document — raw output is persisted along with structured data

## Project layout

```
ocr-pipeline/
├── src/
│   ├── ingestion.py      # PDF/image -> normalized page images
│   ├── ocr_engine.py     # PP-StructureV3 wrapper (layout detection + OCR extraction)
│   ├── structuring.py    # raw OCR output -> doc-type schema (electricity bill and financial report specific)
│   ├── verification.py   # confidence gating for both + arithmetic cross-checks currently for financial report
│   ├── db.py             # Relational DB schema + read/write helpers
│   └── pipeline.py        # orchestrates the process end-to-end
├── data/
│   └── processed/pages    # Split the PDF into page images (OCR)
│   └── processed/ocr_raw  # Intermediate JSON files post OCR step
│   └── processed          # Intermediate files post structuring step 
├── requirements.txt
└── README.md
└── ocr_pipeline.db        # Final output
```
This will:
1. Split the PDF into page images 
2. Run layout detection + OCR on each page
3. Parse tables/fields into the document based schema
4. Run confidence gating + line-item checksum verification
5. Write everything to `ocr_pipeline.db`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.pipeline --input filename.pdf --doc-type financial_report
python -m src.pipeline --input filename.pdf --doc-type electricity_bill
```
## Status

* Electricity bills run end-to-end - structuring, verification and DB writing all work, verified field-by-field against a synthetic test bill 
* Financial reports run end-to-end on a subset of pages (~6 out of 35 page Swiggy’s Q1’27 quarterly report) but table extraction accuracy is a known open issue on certain table layouts — detailed below

## Known limitations

* Nested/indented tables don't always extract correctly - On the financial report, tables with blank separator rows and indented sub-labels (e.g. a summary row followed by "Growth - QoQ" / "Growth - YoY" sub-rows) sometimes come out of PP-StructureV3's table model with cells merged or misaligned across rows. Flat tables (one label, N value columns, no nesting) extract accurately - this showed up consistently across multiple tables.
* Small text is less reliable at lower DPI - Rasterizing at DPI 100 (needed to stay within memory limits of a laptop) degrades recognition on small font sizes - footnotes and disclaimers sometimes come out garbled while body text and table content stay accurate.
* Checksum verification is disabled - It's implemented for financial reports (component rows summing to a stated total) but currently commented out pending further testing and isn't implemented at all for electricity bills.
* DB schema limitations - For electricity bills, normalised row level data is currently not stored in DB, however raw HTML is persisted. For financial reports, both raw HTML and row level data is stored in DB
