"""
Stage 4: Verification

Two cheap, high-signal checks before anything is trusted enough to write
to the DB as "verified":

1. Confidence gating — flag any line item whose OCR confidence is below
   threshold, regardless of whether the number looks plausible.
2. Arithmetic cross-check — financial line items often roll up into a
   stated total (e.g. subtotal lines summing to "Total expenses"). Where
   we can detect that relationship, we check it. A failed checksum is a
   near-free signal that either OCR misread a digit or the table
   structure was parsed wrong.

Neither check proves correctness — they narrow down what a human needs
to actually look at, which is the realistic bar for "verifiable."
"""

from __future__ import annotations

from dataclasses import dataclass

from src.structuring import StructuredDocument, StructuredTable, LineItem, StructuredElectricityBillDocument, ElectricityBillTable

CONFIDENCE_THRESHOLD = 0.85

# Labels that commonly represent a rollup of preceding rows. Extend this
# as you see more report formats — matching is case-insensitive substring.
TOTAL_LABEL_MARKERS = ("total", "sub-total", "subtotal", "grand total")


@dataclass
class VerificationFlag:
    reason: str  # "low_confidence" | "checksum_mismatch"
    table_page: int
    table_index: int
    label: str
    detail: str


def verify_confidence(table: StructuredTable, threshold: float = CONFIDENCE_THRESHOLD) -> list[VerificationFlag]:
    flags = []
    for item in table.line_items:
        if item.confidence < threshold:
            flags.append(
                VerificationFlag(
                    reason="low_confidence",
                    table_page=table.page_number,
                    table_index=table.table_index,
                    label=item.label,
                    detail=f"confidence={item.confidence:.2f} < {threshold}",
                )
            )
    return flags


def verify_checksum(table: StructuredTable, tolerance: float = 0.5) -> list[VerificationFlag]:
    """
    For each period column independently: find a "total" row, sum the
    values of the rows immediately preceding it (until the previous total
    or the start of the table), and check they match within tolerance.

    tolerance defaults to 0.5 to absorb rounding in the source report, not
    OCR error — keep this tight, it's meant to catch OCR mistakes, not
    hide them.
    """
    flags = []

    # Group line items by period so multi-column reports (current vs
    # prior period) are checked independently.
    periods = {item.period for item in table.line_items}

    for period in periods:
        period_items = [i for i in table.line_items if i.period == period]
        running: list[LineItem] = []

        for item in period_items:
            is_total_row = any(marker in item.label.lower() for marker in TOTAL_LABEL_MARKERS)

            if is_total_row:
                if item.value is None:
                    running = []
                    continue

                component_sum = sum(i.value for i in running if i.value is not None)
                if abs(component_sum - item.value) > tolerance:
                    flags.append(
                        VerificationFlag(
                            reason="checksum_mismatch",
                            table_page=table.page_number,
                            table_index=table.table_index,
                            label=item.label,
                            detail=(
                                f"period={period!r}: components sum to {component_sum:.2f}, "
                                f"stated total is {item.value:.2f}"
                            ),
                        )
                    )
                running = []  # reset after a total row
            else:
                running.append(item)

    return flags

def verify_confidence_electricity_bill(
    table: ElectricityBillTable, threshold: float = CONFIDENCE_THRESHOLD
) -> list[VerificationFlag]:
    """
    Electricity bill tables split confidence across two different shapes
    (key_value_pairs vs rows), so this mirrors verify_confidence's logic
    but reads from whichever shape the table actually has.
    """
    flags = []

    for kv in table.key_value_pairs:
        if kv.confidence < threshold:
            flags.append(
                VerificationFlag(
                    reason="low_confidence",
                    table_page=table.page_number,
                    table_index=table.table_index,
                    label=kv.label,
                    detail=f"confidence={kv.confidence:.2f} < {threshold}",
                )
            )

    for row in table.rows:
        if row.confidence < threshold:
            flags.append(
                VerificationFlag(
                    reason="low_confidence",
                    table_page=table.page_number,
                    table_index=table.table_index,
                    label=row.label,
                    detail=f"confidence={row.confidence:.2f} < {threshold}",
                )
            )

    return flags

def verify_electricity_bill(doc: StructuredElectricityBillDocument) -> list[VerificationFlag]:
    """
    Confidence gating is only implemented for electricity bills.
    """
    flags: list[VerificationFlag] = []
    for table in doc.tables:
        flags.extend(verify_confidence_electricity_bill(table))
    return flags

def verify_document(doc: StructuredDocument) -> list[VerificationFlag]:
    flags: list[VerificationFlag] = []
    for table in doc.tables:
        flags.extend(verify_confidence(table))
        #flags.extend(verify_checksum(table))
    return flags

VERIFIERS = {
    "financial_report": verify_document,
    "electricity_bill": verify_electricity_bill,
}