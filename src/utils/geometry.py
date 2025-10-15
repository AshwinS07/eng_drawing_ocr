from typing import List

def quad_to_xywh(quad: List[List[float]]):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return [float(x), float(y), float(w), float(h)]

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union
