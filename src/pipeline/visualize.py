import cv2
import numpy as np

def overlay(img, ocr_items, symbol_items=None, cell_boxes=None):
    """
    Draw bounding boxes for OCR items, symbols, and table cells.

    Args:
        img (np.ndarray): BGR image.
        ocr_items (list): OCR results with bbox_xywh + text.
        symbol_items (list): Optional list of symbol detections with bbox + label.
        cell_boxes (list): Optional list of detected table cell polygons.
    """
    vis = img.copy()

    # Draw OCR boxes (texts, numbers)
    for it in ocr_items:
        if "bbox_xywh" in it:
            x, y, w, h = it["bbox_xywh"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, it.get("text", ""), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw symbol boxes (optional, blue)
    if symbol_items:
        for sym in symbol_items:
            if "bbox" in sym:
                x, y, w, h = sym["bbox"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(vis, sym.get("label", "SYM"), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw table cells (optional, red polygons)
    if cell_boxes:
        for cell in cell_boxes:
            pts = np.array(cell, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 0, 255), 1)

    return vis
