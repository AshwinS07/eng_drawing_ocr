from typing import List, Dict
from utils.geometry import quad_to_xywh, iou

def assign_to_cells(ocr_items: List[Dict], cells_xywh: List[List[int]]):
    # Convert any quad -> bbox_xywh
    for it in ocr_items:
        if "bbox_xywh" not in it and "bbox_quad" in it:
            it["bbox_xywh"] = quad_to_xywh(it["bbox_quad"])

    rows = []
    for cell in cells_xywh:
        contents = [it for it in ocr_items if iou(it["bbox_xywh"], cell) > 0.05]
        # Sort by y then x for stable concatenation
        contents_sorted = sorted(contents, key=lambda z: (z["bbox_xywh"][1], z["bbox_xywh"][0]))
        text = " ".join([c["text"] for c in contents_sorted]).strip()
        rows.append({"cell_bbox": cell, "text": text, "items": contents_sorted})
    return rows
