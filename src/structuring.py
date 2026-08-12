"""
Stage 3: Structuring

Converts raw OCR regions (from ocr_engine.py) into a doc-type-specific
schema. This is the only stage that should need new code when you add a
new document type (e.g. electricity_bill) — ingestion, OCR, verification,
and DB writing all stay generic.

financial_report is the first doc type implemented, matching the Swiggy
quarterly report use case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from src.ocr_engine import PageOCRResult, OCRRegion

@dataclass
class LineItem: #one entry per cell in the table
    label: str # row headers
    value: float | None # cleaned-up value eg -1234.5
    raw_value: str # raw value from OCR eg (1,234.50)
    period: str | None = None  # e.g. "Q1 FY27" if column headers are detected
    confidence: float = 0.0


@dataclass 
class StructuredTable: #one entry per table
    page_number: int
    table_index: int #unique identifier for each table on the page
    line_items: list[LineItem] = field(default_factory=list) #every time someone creates a new StructuredTable without specifying line_items, call list() fresh, to build a brand new empty list just for this one instance
    section_title: str | None = None  # nearest preceding "title" region, helps associate the table with a section of the document
    raw_html: str = ""


@dataclass
class StructuredDocument:
    doc_id: str
    doc_type: str
    tables: list[StructuredTable] = field(default_factory=list)
    narrative_text: list[dict[str, Any]] = field(default_factory=list)  # non-table text per page, a list where each element is a dictionary with string keys and values of any type


class _SimpleTableHTMLParser(HTMLParser): #HTMLParser works on event driven model invoking methods whenever it encounters an opening HTML tag, a closing tag or text data
    """
    Minimal HTML table parser — avoids pulling in a full HTML/DOM library
    just to walk PP-StructureV3's <table><tr><td> output.
    """

    def __init__(self):
        super().__init__() #calls HTMLParser's own constructor first
        self.rows: list[list[str]] = []
        self._current_row: list[str] = [] 
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs): #fires when an opening tag is seen eg <td> or <th> or <tr>
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag): #fires when a closing tag is seen eg </td> or </th> or </tr>
        if tag in ("td", "th"):
            self._current_row.append("".join(self._current_cell).strip()) #merges all the text pieces collected for this cell into one string
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data): #fires for raw text between tags
        if self._in_cell:
            self._current_cell.append(data)


_NUMERIC_RE = re.compile(r"[-+]?[\d,]+\.?\d*") #find a numeric substring anywhere within a string: eg Rs. 1,200.5 
_PURE_NUMBER_RE = re.compile(r"^[-+(]?[\d,]+\.?\d*\)?$") #check if the entire string is only a number eg (1,234.5) or -500

def _looks_numeric(raw: str) -> bool:
    cleaned = raw.strip().replace("₹", "").replace("Rs.", "").replace("INR", "").strip()
    return bool(cleaned) and bool(_PURE_NUMBER_RE.match(cleaned))


def _parse_numeric(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    is_negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.strip("()").replace(",", "").replace("₹", "").replace("Rs.", "").replace("INR", "").strip()
    match = _NUMERIC_RE.search(cleaned)
    if not match:
        return None
    try:
        value = float(match.group()) #returns the actual substring of text that matched the pattern eg 1,200.5 for Rs. 1,200.5
    except ValueError:
        return None
    return -value if is_negative else value

def structure_financial_report(ocr_results: list[PageOCRResult], doc_id: str) -> StructuredDocument: #Takes the raw OCR output (a list of PageOCRResult, one per page) and a doc_id, returns one fully assembled StructuredDocument
    """
    Parses OCR output from a financial/quarterly report into tables of
    LineItems plus narrative text blocks.

    Assumption: each table row is [label, value1, value2, ...] where the
    first cell is the line-item label (e.g. "Revenue from operations") and
    subsequent cells are numeric values (current period, prior period, etc).
    If a table has period headers in row 0, those are used as `period`;
    otherwise period is left None and you'll want to set it manually or
    infer it from the section title / page context.
    """
    doc = StructuredDocument(doc_id=doc_id, doc_type="financial_report")

    for page_result in ocr_results:
        last_title: str | None = None
        table_idx_on_page = 0

        for region in page_result.regions:
            if region.region_type == "title" and region.text:
                last_title = region.text.strip()

            elif region.region_type == "table" and region.table_html:
                table_idx_on_page += 1
                rows = _SimpleTableHTMLParser() # rows.rows == []  (empty, nothing parsed yet)
                rows.feed(region.table_html) #triggers callbacks, which build up rows.rows

                if not rows.rows: #if no rows were parsed, skip to the next region
                    continue

                header_row = rows.rows[0] 
                has_header = any(
                    not _looks_numeric(cell) and cell.strip() for cell in header_row[1:] #skips the first cell and checks if any of the subsequent cells in the header row are not numeric and not empty
                )
                data_rows = rows.rows[1:] if has_header else rows.rows 
                periods = header_row[1:] if has_header else []

                structured_table = StructuredTable(
                    page_number=page_result.page_number,
                    table_index=table_idx_on_page, #unique identifier for each table on the page
                    section_title=last_title, #nearest preceding "title" region, helps associate the table with a section of the document
                    raw_html=region.table_html,
                )

                for row in data_rows:
                    if not row:
                        continue
                    label, *values = row #label is the first cell, values is a list of the remaining cells
                    if not label.strip():
                        continue

                    for i, raw_value in enumerate(values):
                        period = periods[i] if i < len(periods) else None
                        structured_table.line_items.append(
                            LineItem(
                                label=label.strip(), # row header
                                value=_parse_numeric(raw_value),
                                raw_value=raw_value,
                                period=period, #column header
                                confidence=region.confidence, #confidence score for entire table
                            )
                        )

                doc.tables.append(structured_table)

            elif region.region_type == "text" and region.text:
                doc.narrative_text.append(
                    {
                        "page_number": page_result.page_number,
                        "text": region.text,
                        "confidence": region.confidence,
                        "section_title": last_title,
                    }
                )

    return doc

@dataclass
class KeyValuePair:
    """One label/value pair from a key-value style table (e.g. consumer
    details, bill summary) — as opposed to a header+rows tabular table."""
    label: str
    raw_value: str
    value: float | None  # parsed numeric, or None if the value isn't numeric
    confidence: float = 0.0


@dataclass
class TabularRow:
    """One data row from a header+rows table (e.g. meter readings, tariff
    slabs). `label` is the row's own first-cell label; raw_cells/values are
    keyed by the table's column headers."""
    label: str
    raw_cells: dict[str, str] = field(default_factory=dict)
    values: dict[str, float | None] = field(default_factory=dict)
    is_section_header: bool = False  # e.g. "Energy Charges" divider row, no data
    confidence: float = 0.0


@dataclass
class ElectricityBillTable:
    page_number: int
    table_index: int
    section_title: str | None
    table_type: str  # "key_value" | "tabular"
    key_value_pairs: list[KeyValuePair] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)   # only for "tabular"
    rows: list[TabularRow] = field(default_factory=list)  # only for "tabular"
    raw_html: str = ""


@dataclass
class StructuredElectricityBillDocument:
    doc_id: str
    doc_type: str = "electricity_bill"
    tables: list[ElectricityBillTable] = field(default_factory=list)
    narrative_text: list[dict[str, Any]] = field(default_factory=list)

    def find(self, label: str) -> str | None:
        """Convenience lookup, e.g. doc.find('Total Amount Due')."""
        for table in self.tables:
            for kv in table.key_value_pairs:
                if kv.label.strip().lower() == label.strip().lower():
                    return kv.raw_value
        return None


def _is_label_like(cell: str) -> bool:
    cleaned = cell.strip()
    return bool(cleaned) and not _looks_numeric(cleaned)


def _is_key_value_table(rows: list[list[str]]) -> bool:
    """
    True if EVERY row is alternating (label, value, label, value, ...) —
    e.g. ["Consumer No.", "042...", "Meter No.", "MTR..."] — rather than a
    header row followed by data rows sharing common columns.
    """
    if not rows:
        return False
    for row in rows:
        if len(row) < 2 or len(row) % 2 != 0:
            return False
        for i in range(0, len(row), 2):
            if not _is_label_like(row[i]):
                return False
    return True


def _parse_key_value_table(rows: list[list[str]], confidence: float) -> list[KeyValuePair]:
    pairs = []
    for row in rows:
        for i in range(0, len(row), 2):
            label, raw_value = row[i].strip(), row[i + 1].strip()
            if not label:
                continue
            pairs.append(
                KeyValuePair(
                    label=label,
                    raw_value=raw_value,
                    value=_parse_numeric(raw_value) if _looks_numeric(raw_value) else None,
                    confidence=confidence,
                )
            )
    return pairs


def _parse_tabular_table(rows: list[list[str]], confidence: float) -> tuple[list[str], list[TabularRow]]:
    header_row = rows[0]
    headers = header_row[1:]
    data_rows = []

    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        cells = row[1:]
        is_section_header = all(not c.strip() for c in cells)

        raw_cells: dict[str, str] = {}
        values: dict[str, float | None] = {}
        for i, cell in enumerate(cells):
            header = headers[i] if i < len(headers) else f"col_{i}"
            raw_cells[header] = cell.strip()
            values[header] = _parse_numeric(cell) if _looks_numeric(cell) else None

        data_rows.append(
            TabularRow(label=label, raw_cells=raw_cells, values=values,
                       is_section_header=is_section_header, confidence=confidence)
        )

    return headers, data_rows


def structure_electricity_bill(ocr_results: list[PageOCRResult], doc_id: str) -> StructuredElectricityBillDocument:
    """
    Electricity bills mix several table shapes on one page — key-value
    pairs (consumer details, bill summary) and header+rows tables (meter
    readings, tariff slabs) — unlike financial_report's single fixed shape.
    Each table is classified independently via _is_key_value_table().
    """
    doc = StructuredElectricityBillDocument(doc_id=doc_id)

    for page_result in ocr_results:
        last_title: str | None = None
        table_idx_on_page = 0

        for region in page_result.regions:
            if region.region_type == "title" and region.text:
                last_title = region.text.strip()

            elif region.region_type == "table" and region.table_html:
                table_idx_on_page += 1
                parser = _SimpleTableHTMLParser()
                parser.feed(region.table_html)
                if not parser.rows:
                    continue

                if _is_key_value_table(parser.rows):
                    table = ElectricityBillTable(
                        page_number=page_result.page_number,
                        table_index=table_idx_on_page,
                        section_title=last_title,
                        table_type="key_value",
                        key_value_pairs=_parse_key_value_table(parser.rows, region.confidence),
                        raw_html=region.table_html,
                    )
                else:
                    headers, data_rows = _parse_tabular_table(parser.rows, region.confidence)
                    table = ElectricityBillTable(
                        page_number=page_result.page_number,
                        table_index=table_idx_on_page,
                        section_title=last_title,
                        table_type="tabular",
                        headers=headers,
                        rows=data_rows,
                        raw_html=region.table_html,
                    )
                doc.tables.append(table)

            elif region.region_type == "text" and region.text:
                doc.narrative_text.append({
                    "page_number": page_result.page_number,
                    "text": region.text,
                    "confidence": region.confidence,
                    "section_title": last_title,
                })

    return doc

# Registry so pipeline.py can dispatch by --doc-type without an if/elif chain
STRUCTURERS = {
    "financial_report": structure_financial_report,
    "electricity_bill": structure_electricity_bill,
}
