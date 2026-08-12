# OCR → Structured DB Pipeline

Extracts text, tables and key fields from documents (starting with electricity bill and financial
quarterly reports) using PaddleOCR / PP-StructureV3, verifies the extraction and loads into a database schema that's ready to be chunked/embedded for future projects (RAG)

## Why this shape

The pipeline is split into five independent stages so that:
```
Ingestion  →  OCR + Layout  →  Structuring  →  Verification  →  DB
(PDF/img)     (PP-StructureV3)  (doc-type       (confidence +
              + PaddleOCR)       schema)          checksum)
```
- doc-type-specific logic (financial report vs. electricity bill vs. handwritten
  notes etc) is isolated mostly to `structuring.py`
- OCR is only ever run once per document — raw output is persisted along with structured data

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

First run of PaddleOCR/PP-StructureV3 downloads model weights (~few hundred MB)
from Baidu's model hub — do this once on a network with unrestricted access,
not inside a sandboxed environment.

## Run

```bash
python -m src.pipeline --input filename.pdf --doc-type report_type
```
