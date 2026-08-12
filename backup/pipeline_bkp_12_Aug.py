"""
End-to-end orchestration: ingestion -> OCR -> structuring -> verification -> DB.

Usage:
    python -m src.pipeline --input data/raw/swiggy_q1_fy27.pdf --doc-type financial_report
    -m tells Python to run a file as part of its package (using import-style resolution) rather than as a standalone script — so internal imports like from src.structuring resolve correctly
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

from src.ingestion import ingest
from src.ocr_engine import PPStructureEngine, save_ocr_result
from src.structuring import STRUCTURERS
from src.verification import verify_document
from src.db import get_connection, save_document


def run_pipeline(
    input_path: str,
    doc_type: str,
    output_dir: str = "data/processed",
    db_path: str = "ocr_pipeline.db",
) -> None:
    output_dir = Path(output_dir)
    doc_id = Path(input_path).stem

    if doc_type not in STRUCTURERS:
        raise ValueError(f"Unknown doc_type '{doc_type}'. Available: {list(STRUCTURERS)}")

    # Stage 1: ingestion
    print(f"[1/4] Ingesting {input_path} ...")
    pages = ingest(input_path, output_dir / "pages")
    print(f"      -> {len(pages)} page(s)")

    # Stage 2: OCR + layout (the slow, expensive step — cached to disk)
    print("[2/4] Running OCR + layout detection ...")
    engine = PPStructureEngine()
    ocr_results = []
    for page in pages:
        result = engine.run(page)
        save_ocr_result(result, output_dir / "ocr_raw")
        ocr_results.append(result)
        print(f"      page {page.page_number}: {len(result.regions)} region(s) detected")
        del result
        gc.collect()

    # Stage 3: structuring
    print(f"[3/4] Structuring as '{doc_type}' ...")
    structurer = STRUCTURERS[doc_type]
    doc = structurer(ocr_results, doc_id)
    print(f"      -> {len(doc.tables)} table(s), {len(doc.narrative_text)} narrative block(s)")

    structured_out = output_dir / f"{doc_id}_structured.json"
    structured_out.write_text(json.dumps(asdict(doc), indent=2))
'''
    structured_out.write_text(
        json.dumps(
            {
                "doc_id": doc.doc_id,
                "doc_type": doc.doc_type,
                "tables": [
                    {
                        "page_number": t.page_number,
                        "table_index": t.table_index,
                        "section_title": t.section_title,
                        "line_items": [vars(li) for li in t.line_items],
                    }
                    for t in doc.tables
                ],
                "narrative_text": doc.narrative_text,
            },
            indent=2,
        )
    )
'''
    # Stage 4: verification
    if doc_type == "financial_report":
        print("[4/4] Verifying ...")
        flags = verify_document(doc)
        if flags:
            print(f"      {len(flags)} flag(s) raised:")
            for f in flags:
                print(f"        - [{f.reason}] p{f.table_page} t{f.table_index} '{f.label}': {f.detail}")
        else:
            print("      no flags — all checks passed")
    else:
        print("[4/4] Verifying ... skipped (not yet implemented for this doc type)")

    # Write to DB
    if doc_type == "financial_report":
        conn = get_connection(db_path)
        save_document(conn, doc, flags, filename=str(input_path))
        conn.close()
        print(f"\nDone. Structured output: {structured_out}")
        print(f"DB: {db_path}")
    else:
        print("[4/4] Writing to DB ... skipped (not yet implemented for this doc type)")


def main():
    parser = argparse.ArgumentParser(description="OCR -> structured DB pipeline")
    parser.add_argument("--input", required=True, help="Path to input PDF or image") #required argument with specific input path
    parser.add_argument( #required argument with specific doc type as choices
        "--doc-type",
        required=True,
        choices=list(STRUCTURERS.keys()),
        help="Document type, determines structuring schema",
    )
    parser.add_argument("--output-dir", default="data/processed") #optional argument with default value
    parser.add_argument("--db-path", default="ocr_pipeline.db") #optional argument with default value
    args = parser.parse_args()

    run_pipeline(args.input, args.doc_type, args.output_dir, args.db_path)


if __name__ == "__main__":
    main()
