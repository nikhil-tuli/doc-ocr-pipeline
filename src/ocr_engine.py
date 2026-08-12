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
    regions: list[OCRRegion] = field(default_factory=list) #defaults to a new empty list for each instance

    def to_dict(self) -> dict[str, Any]: #Output key is string, value any type
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "regions": [asdict(r) for r in self.regions], #conversion step that makes output of PageOCRResult.to_dict() JSON-serializable (ready to write in json file)
        }


class PPStructureEngine:
    """
    Thin wrapper around PaddleOCR's PPStructure pipeline. Instantiate once
    and reuse across pages — model loading is the slow part.
    """
    '''
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        # Import is deferred so the rest of the pipeline (and its tests) can
        # run without paddleocr installed/models downloaded.
        from paddleocr import PPStructureV3

        self.engine = PPStructureV3(
            lang=lang,
            layout=True,
            table=True,
            ocr=True,
            show_log=False,
            use_gpu=use_gpu,
        )
    '''

    def __init__(self, lang: str = "en", device: str = "cpu"):
        # Import is deferred so the rest of the pipeline (and its tests) can
        # run without paddleocr installed/models downloaded.
        from paddleocr import PPStructureV3

        self.engine = PPStructureV3(
            lang=lang,
            device=device,
            use_table_recognition=True,
            use_region_detection=True,
            use_formula_recognition=False,   
            use_chart_recognition=False,     
            use_seal_recognition=False,      
        )

    def run(self, page: Page) -> PageOCRResult:
        import cv2
        import gc

        img = cv2.imread(str(page.image_path))
        raw_result = self.engine.predict(img)

        del img
        gc.collect()

        page_result = raw_result[0]
        parsing_items = page_result.get("parsing_res_list", [])

        LABEL_MAP = {
            "table": "table",
            "figure": "figure",
            "chart": "figure",
            "title": "title",
            "figure_title": "title",
            "paragraph_title": "title",
            "text": "text",
            "abstract": "text",
            "content": "text",
            "reference": "text",
            "footnote": "text",
            "header": "text",
            "footer": "text",
            "number": "text",
        }

        regions = []
        for item in parsing_items:
            raw_label = item.label
            region_type = LABEL_MAP.get(raw_label, "text")
            bbox = item.bbox if item.bbox is not None else [0, 0, 0, 0]
            content = item.content or ""

            if region_type == "table":
                regions.append(
                    OCRRegion(
                        region_type="table",
                        bbox=list(bbox),
                        table_html=content,
                        confidence=1.0,
                    )
                )
            else:
                regions.append(
                    OCRRegion(
                        region_type=region_type,
                        bbox=list(bbox),
                        text=content,
                        confidence=1.0,
                    )
                )

        return PageOCRResult(doc_id=page.doc_id, page_number=page.page_number, regions=regions)

def save_ocr_result(result: PageOCRResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.doc_id}_page{result.page_number:03d}_ocr.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2)) #Converts PageOCRResult into a plain dict via to_dict(), serializes it to a JSON string and writes that string to the file at out_path
    return out_path
