"""
Stage 2: OCR + Layout Detection

Wraps PP-StructureV3 to turn a page image into layout-aware regions:
text blocks, titles, and tables (tables come back as structured cell
grids, not flattened text — this matters for the financial report use
case where row/column alignment is the whole point).

Raw output is persisted to disk as-is (see pipeline.py) so re-running
downstream stages never requires re-running OCR — OCR is the expensive,
slow step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.ingestion import Page


@dataclass
class OCRRegion:
    region_type: str  # "title" | "text" | "table" | "figure"
    bbox: list[float]  # [x1, y1, x2, y2]
    text: str | None = None
    table_html: str | None = None  # present only when region_type == "table"
    confidence: float = 0.0


@dataclass
class PageOCRResult:
    doc_id: str
    page_number: int
    regions: list[OCRRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "regions": [asdict(r) for r in self.regions],
        }


class PPStructureEngine:
    """
    Thin wrapper around PaddleOCR's PPStructure pipeline. Instantiate once
    and reuse across pages — model loading is the slow part.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        # Import is deferred so the rest of the pipeline (and its tests) can
        # run without paddleocr installed/models downloaded.
        from paddleocr import PPStructure

        self.engine = PPStructure(
            lang=lang,
            layout=True,
            table=True,
            ocr=True,
            show_log=False,
            use_gpu=use_gpu,
        )

    def run(self, page: Page) -> PageOCRResult:
        import cv2

        img = cv2.imread(str(page.image_path))
        raw_result = self.engine(img)

        regions = []
        for item in raw_result:
            region_type = item.get("type", "text")
            bbox = item.get("bbox", [0, 0, 0, 0])

            if region_type == "table":
                table_html = item.get("res", {}).get("html", "")
                regions.append(
                    OCRRegion(
                        region_type="table",
                        bbox=bbox,
                        table_html=table_html,
                        confidence=item.get("score", 0.0),
                    )
                )
            else:
                # text / title / figure regions: res is a list of
                # {"text": ..., "confidence": ...} line-level dicts
                lines = item.get("res", [])
                text = " ".join(l.get("text", "") for l in lines if isinstance(l, dict))
                confidences = [l.get("confidence", 0.0) for l in lines if isinstance(l, dict)]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

                regions.append(
                    OCRRegion(
                        region_type=region_type,
                        bbox=bbox,
                        text=text,
                        confidence=avg_conf,
                    )
                )

        return PageOCRResult(doc_id=page.doc_id, page_number=page.page_number, regions=regions)


def save_ocr_result(result: PageOCRResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.doc_id}_page{result.page_number:03d}_ocr.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    return out_path
