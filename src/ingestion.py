"""
Stage 1: Ingestion

Takes a PDF or image file and normalizes it into a list of page images
(as file paths) that the OCR stage can consume uniformly, regardless of
whether the source was a scanned image, a native-text PDF, or a photo.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    doc_id: str
    page_number: int  # 1-indexed
    image_path: Path
    has_text_layer: bool = False  # True if source PDF page had extractable text


def ingest(source_path: str | Path, output_dir: str | Path, dpi: int = 100) -> list[Page]: # source_path and output_dir could be string or Path object
    """
    Normalize a document into per-page images.

    Args:
        source_path: path to a .pdf, .png, .jpg, or .jpeg file
        output_dir: where to write the normalized page images
        dpi: rasterization resolution for PDFs (300 is a good OCR default;
             go to 400+ only if small print/table text is getting missed)

    Returns:
        List of Page objects, one per page, in order.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # create the output directory if it doesn't exist, parents=True means create parent folders if they don't exist, exist_ok=True don't raise error if folder already exists

    doc_id = source_path.stem
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        return _ingest_pdf(source_path, output_dir, doc_id, dpi) #dpi is additional parameter for PDFs since they are vector/paged content, not pixels — DPI controls how many pixels get generated when rasterizing the page for OCR
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return _ingest_image(source_path, output_dir, doc_id)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _ingest_pdf(source_path: Path, output_dir: Path, doc_id: str, dpi: int) -> list[Page]:
    import fitz  # PyMuPDF

    pages: list[Page] = []
    doc = fitz.open(source_path)

    for i, pdf_page in enumerate(doc, start=1):
        # Render page to image regardless of whether it has a text layer —
        # PP-StructureV3 works off the rendered image, and keeping the
        # pipeline image-based means scanned and native PDFs are handled
        # identically downstream.
        zoom = dpi / 72  # PDF default is 72 dpi
        matrix = fitz.Matrix(zoom, zoom) # builds a 2D scaling transformation matrix (same zoom factor for x and y) and not default 72 dpi
        pix = pdf_page.get_pixmap(matrix=matrix) # rasterizes pdf page into an in-memory Pixmap object

        image_path = output_dir / f"{doc_id}_page{i:03d}.png"
        pix.save(str(image_path)) #stores rasterized pdf on disk

        has_text_layer = len(pdf_page.get_text().strip()) > 20 #get_text is method specific to PyMuPDF's Page object

        pages.append(
            Page(
                doc_id=doc_id,
                page_number=i,
                image_path=image_path,
                has_text_layer=has_text_layer,
            )
        )

    doc.close()
    return pages #returns rasterised pdf pages for ocr extraction


def _ingest_image(source_path: Path, output_dir: Path, doc_id: str) -> list[Page]:
    dest = output_dir / f"{doc_id}_page001{source_path.suffix.lower()}"
    shutil.copy(source_path, dest) #stores image as such for ocr extraction
    return [Page(doc_id=doc_id, page_number=1, image_path=dest, has_text_layer=False)]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion <path_to_pdf_or_image>")
        sys.exit(1)

    pages = ingest(sys.argv[1], "data/processed/pages")
    for p in pages:
        print(f"page {p.page_number}: {p.image_path} (text_layer={p.has_text_layer})")
