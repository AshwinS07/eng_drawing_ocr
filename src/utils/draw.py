import cv2
import numpy as np

def draw_boxes(img_bgr, items):
    vis = img_bgr.copy()
    for it in items:
        # prefer bbox_xywh, fallback to bbox_quad
        if it.get("bbox_xywh"):
            x, y, w, h = map(int, it["bbox_xywh"])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pt = (x, max(0, y - 6))
        else:
            quad = np.array(it.get("bbox_quad", []), dtype=np.int32)
            if quad.size:
                cv2.polylines(vis, [quad.reshape((-1, 2))], True, (0, 255, 0), 2)
                pt = tuple(quad[0])
            else:
                continue
        text = it.get("text", "")
        cv2.putText(vis, text if len(text) < 40 else text[:40]+"...", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
    return vis
