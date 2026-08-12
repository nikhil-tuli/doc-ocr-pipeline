"""
DB layer — SQLite for now, schema is designed to move to Postgres
without changes when this feeds a RAG pipeline later.

Raw OCR output and structured output are both kept, so the RAG layer
(chunking/embedding) can be built later without ever re-running OCR.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.structuring import StructuredDocument
from src.verification import VerificationFlag

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    doc_type TEXT NOT NULL,
    filename TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'processed'  -- processed | needs_review | verified
);

CREATE TABLE IF NOT EXISTS extracted_tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    section_title TEXT,
    raw_html TEXT
);

CREATE TABLE IF NOT EXISTS table_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL REFERENCES extracted_tables(id),
    label TEXT NOT NULL,
    value REAL,
    raw_value TEXT,
    period TEXT,
    confidence REAL,
    verified INTEGER DEFAULT 0  -- 0 = unverified/flagged, 1 = passed all checks
);

CREATE TABLE IF NOT EXISTS narrative_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    page_number INTEGER NOT NULL,
    text TEXT,
    confidence REAL,
    section_title TEXT
);

CREATE TABLE IF NOT EXISTS verification_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    reason TEXT NOT NULL,
    table_page INTEGER,
    table_index INTEGER,
    label TEXT,
    detail TEXT
);
"""


def get_connection(db_path: str | Path = "ocr_pipeline.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def save_document(
    conn: sqlite3.Connection,
    doc: StructuredDocument,
    flags: list[VerificationFlag],
    filename: str = "",
) -> None:
    flagged_labels = {(f.table_page, f.table_index, f.label) for f in flags}
    status = "needs_review" if flags else "verified"

    conn.execute(
        "INSERT OR REPLACE INTO documents (doc_id, doc_type, filename, status) VALUES (?, ?, ?, ?)",
        (doc.doc_id, doc.doc_type, filename, status),
    )

    for table in doc.tables:
        cursor = conn.execute(
            """INSERT INTO extracted_tables (doc_id, page_number, table_index, section_title, raw_html)
               VALUES (?, ?, ?, ?, ?)""",
            (doc.doc_id, table.page_number, table.table_index, table.section_title, table.raw_html),
        )
        table_id = cursor.lastrowid

        # table_rows is shaped for financial_report's LineItem (label, value,
        # raw_value, period). electricity_bill's key_value_pairs/rows don't
        # map onto that shape cleanly — skipping for now, raw_html above
        # still preserves the full table data for this doc type.
        if hasattr(table, "line_items"):
            for item in table.line_items:
                key = (table.page_number, table.table_index, item.label)
                verified = 0 if key in flagged_labels else 1
                conn.execute(
                    """INSERT INTO table_rows (table_id, label, value, raw_value, period, confidence, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (table_id, item.label, item.value, item.raw_value, item.period, item.confidence, verified),
                )

    for block in doc.narrative_text:
        conn.execute(
            """INSERT INTO narrative_text (doc_id, page_number, text, confidence, section_title)
               VALUES (?, ?, ?, ?, ?)""",
            (doc.doc_id, block["page_number"], block["text"], block["confidence"], block.get("section_title")),
        )

    for flag in flags:
        conn.execute(
            """INSERT INTO verification_flags (doc_id, reason, table_page, table_index, label, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (doc.doc_id, flag.reason, flag.table_page, flag.table_index, flag.label, flag.detail),
        )

    conn.commit()
