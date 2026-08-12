from PIL import Image
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from paddleocr import PaddleOCR
from transformers import LayoutLMv3ForTokenClassification

LAYOUTREADER_ROOT = Path(__file__).resolve().parents[1] / "layoutreader" 
sys.path.insert(0, str(LAYOUTREADER_ROOT))
from v3.helpers import prepare_inputs, boxes2inputs, parse_logits

# Initialize PaddleOCR 
ocr = PaddleOCR(lang='en')

# Load image
image_path = "report_original.png"
#Image.open(image_path).show()
#img = Image.open(image_path)
#plt.imshow(img)
#plt.axis('off')
#plt.show()

# Run OCR 
result = ocr.predict(image_path)
page = result[0]
texts = page['rec_texts']      # recognized text strings
scores = page['rec_scores']    # confidence scores
boxes = page['rec_polys']      # bounding box coordinates

print(f"Extracted {len(texts)} text regions")
print("\nFirst 10 regions:")
for text, score, box in list(zip(texts, scores, boxes))[:10]:
    coords = box.astype(int).tolist()
    print(f"{text:40} | {score:.3f} | {coords}")

# Visualize OCR bounding boxes
processed_img = page['doc_preprocessor_res']['output_img']
img_plot = processed_img.copy()
show_text= False

for text, box in zip(texts, boxes):
    pts = np.array(box, dtype=int)
    cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)
    x, y = pts[0]
    if show_text:
        cv2.putText(img_plot, text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Aligned Bounding Boxes (Processed Image)")
plt.show()

# Store OCR results in a structured format
@dataclass
class OCRRegion:
    text: str
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    
    @property
    def bbox_xyxy(self):
        """Return bbox as [x1, y1, x2, y2] format."""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

ocr_regions: List[OCRRegion] = []
for text, score, box in zip(texts, scores, boxes):
    ocr_regions.append(OCRRegion(
        text=text, 
        bbox=box.astype(int).tolist(), 
        confidence=score
    ))

print(f"Stored {len(ocr_regions)} OCR regions")

# Load LayoutReader model
print("Loading LayoutReader model...")
model_slug = "hantian/layoutreader"
layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)
print("Model loaded successfully!")

# LayoutLM Reading Order
def get_reading_order(ocr_regions):
    """
    Use LayoutReader to determine reading order of OCR regions.
    Returns list of reading order positions for each region index.
    """
    # 1. Calculate image dimensions from bounding boxes (with padding)
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1   # Add 10% padding
    image_height = max_y * 1.1

    # 2. Convert bboxes to LayoutReader format (normalized to 0-1000)
    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        # Normalize to 0-1000 range
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    # 3. Prepare inputs
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    
    # 4. Run inference
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    
    # 5. Parse the model's outputs to get reading order
    reading_order = parse_logits(logits, len(boxes))

    return reading_order

# Get reading order
reading_order = get_reading_order(ocr_regions)

print(f"Reading order determined for {len(reading_order)} regions")
print(f"First 20 positions: {reading_order[:20]}")

#Visualizing the reading order
import matplotlib.patches as patches

def visualize_reading_order(ocr_regions, image_array, reading_order, title="Reading Order"):
    """
    Visualize OCR regions with their reading order numbers using matplotlib.
    """
    
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    
    # Create order mapping: index -> reading order position
    order_map = {i: order for i, order in enumerate(reading_order)}
    
    for i, region in enumerate(ocr_regions):
        bbox = region.bbox
        if bbox and len(bbox) >= 4:
            # Draw polygon
            ax.add_patch(patches.Polygon(bbox, linewidth=2, 
                                         edgecolor='blue',
                                         facecolor='none', alpha=0.7))
            # Add reading order number at center
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            ax.text(sum(xs)/len(xs), sum(ys)/len(ys), 
                    str(order_map.get(i, i)),
                    fontsize=13, color='red', 
                    ha='center', va='center', fontweight='bold')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_reading_order(ocr_regions, processed_img, reading_order, "LayoutLM Reading Order")

# Create ordered text content
def get_ordered_text(ocr_regions, reading_order):
    """
    Return OCR regions sorted by reading order
    with their text and confidence.
    """
    # 1. Create (reading_position, index, region) tuples and sort
    indexed_regions = [(reading_order[i], 
                        i, 
                        ocr_regions[i]) for i in range(len(ocr_regions))]
    
    # 2. Sort by reading position
    indexed_regions.sort(key=lambda x: x[0])  
    
    # 3. Extract ordered text info
    ordered_text = []
    for position, original_idx, region in indexed_regions:
        ordered_text.append({
            "position": position,
            "text": region.text,
            "confidence": region.confidence,
            "bbox": region.bbox_xyxy
        })
    
    return ordered_text

ordered_text = get_ordered_text(ocr_regions, reading_order)

print("Text in reading order:")
print("=" * 70)
ordered_text[:5]

from paddleocr import LayoutDetection

# Initialize layout detection 
layout_engine = LayoutDetection()

# Process document layout 
def process_document(image_path):
    """Get layout regions from document."""
    layout_result = layout_engine.predict(image_path)
    
    regions = []
    for box in layout_result[0]['boxes']:
        regions.append({
            'label': box['label'],
            'score': box['score'],
            'bbox': box['coordinate'],  # [x1, y1, x2, y2]
        })
    
    # Sort by confidence
    regions = sorted(regions, key=lambda x: x['score'], reverse=True)
    return regions

layout_results = process_document(image_path)

print(f"Detected {len(layout_results)} layout regions:")
for r in layout_results:
    print(f"  {r['label']:20} score: {r['score']:.3f}  bbox: {[int(x) for x in r['bbox']]}")

layout_results[0:5]

#Structuring layout results
@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    
# Store layout regions in structured format
layout_regions: List[LayoutRegion] = []
for i, r in enumerate(layout_results):
    layout_regions.append(LayoutRegion(
        region_id=i,
        region_type=r['label'],
        bbox=[int(x) for x in r['bbox']],
        confidence=r['score']
    ))

print(f"Stored {len(layout_regions)} layout regions")

# Visualize layout detection 
from matplotlib import colormaps

def visualize_layout(image_path, layout_regions, min_confidence=0.5, 
                     title="Layout Detection"):
    """
    Visualize layout detection results using cv2 (same pattern as L2).
    """
    img = cv2.imread(image_path)
    img_plot = img.copy()
    
    # Get unique labels and generate colors
    labels = list(set(r.region_type for r in layout_regions))
    cmap = colormaps.get_cmap('tab20')
    color_map = {}
    for i, label in enumerate(labels):
        rgba = cmap(i % 20)
        color_map[label] = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
    
    for region in layout_regions:
        if region.confidence < min_confidence:
            continue
            
        color = color_map[region.region_type]
        x1, y1, x2, y2 = region.bbox
        
        # Draw rectangle
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
        cv2.polylines(img_plot, [pts], True, color, 2)
        
        # Add label
        text = f"{region.region_id}: {region.region_type} ({region.confidence:.2f})"
        cv2.putText(img_plot, text, (x1, y1-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    
    return img_plot

visualize_layout(image_path, layout_regions, 
                 min_confidence=0.5, title="PaddleOCR Layout Detection");

import base64
from io import BytesIO

# Crop and save layout regions for agent tools
def crop_region(image, bbox, padding=10):
    """Crop a region from image with optional padding."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))

def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Load image for cropping
pil_image = Image.open(image_path)

# Store cropped regions in dictionary
region_images = {}
for region in layout_regions:
    cropped = crop_region(pil_image, region.bbox)
    region_images[region.region_id] = {
        'image': cropped,
        'base64': image_to_base64(cropped),
        'type': region.region_type,
        'bbox': region.bbox
    }

print(f"Cropped {len(region_images)} regions")

# Also store full image
full_image_base64 = image_to_base64(pil_image)

# Show cropped regions
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (region_id, data) in enumerate(list(region_images.items())[:14]):
    axes[i].imshow(data['image'])
    axes[i].set_title(f"Region {region_id}: {data['type']}")
    axes[i].axis('off')

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()