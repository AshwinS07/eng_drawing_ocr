# improved_easyocr_pipeline.py
import cv2
import numpy as np
from pathlib import Path
import easyocr
import math
import re
import json
import csv

# -------------------------
# Helpers
# -------------------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = aw * ah
    area_b = bw * bh
    return inter_area / (area_a + area_b - inter_area + 1e-9)

def bbox_xywh_from_polygon(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))

def map_rotated_point_to_original(xr, yr, angle, orig_w, orig_h):
    if angle == 0:
        return float(xr), float(yr)
    elif angle == 90:
        return float(yr), float(orig_h - 1 - xr)
    else:
        return float(xr), float(yr)  # fallback

def normalize_value_text(txt, label_hint=None):
    s = txt.strip()
    s = s.replace('×', 'x').replace('*', 'x')
    s = s.replace('º', '°')
    s = re.sub(r'\s*[xX]\s*', 'x', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'([+\-]?\d)\s+(\d{1,3})(?!\d)', r'\1.\2', s)
    s = re.sub(r'(\d)\s*[oO]\b', r'\1°', s)
    s = re.sub(r'\bdeg\b', '°', s, flags=re.I)
    if label_hint:
        lh = label_hint.lower()
        if any(k in lh for k in ['id', 'od', 'dia', 'diam', 'ø', 'major dia']):
            s = re.sub(r'^[Oo0]\s*(?=\d)', 'Ø ', s)
    s = s.strip(' ,;:')
    return s

# -------------------------
# Multi-angle OCR (0,90)
# -------------------------
def ocr_multi_angle(rgb, reader, angles=(0,90), conf_threshs=(0.45,0.3)):
    h, w = rgb.shape[:2]
    raw = []
    for angle_idx, angle in enumerate(angles):
        conf_thresh = conf_threshs[angle_idx] if isinstance(conf_threshs, (list,tuple)) else conf_threshs
        if angle == 0:
            rot = rgb
        elif angle == 90:
            rot = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        res = reader.readtext(rot, detail=1)
        for bbox, text, conf in res:
            if conf is None or conf < conf_thresh:
                continue
            mapped = [map_rotated_point_to_original(xr, yr, angle, w, h) for xr, yr in bbox]
            raw.append((mapped, text.strip(), float(conf)))

    # deduplicate by text+IoU
    items = []
    for poly, text, conf in raw:
        xywh = bbox_xywh_from_polygon(poly)
        keep = True
        for it in items:
            if it['text'] == text and iou_xywh(it['bbox_xywh'], xywh) > 0.5:
                if conf > it['conf']:
                    it['bbox_xywh'] = xywh
                    it['conf'] = conf
                keep = False
                break
        if keep:
            items.append({'text': text, 'bbox_xywh': xywh, 'conf': conf})
    return items

# -------------------------
# Detect callout regions
# -------------------------
def detect_callout_regions(rgb, min_area=150):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    H, W = gray.shape
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < 15 or h < 8:
            continue
        pad = 4
        rx = max(0, x-pad)
        ry = max(0, y-pad)
        rw = min(W - rx, w + 2*pad)
        rh = min(H - ry, h + 2*pad)
        regions.append((rx, ry, rw, rh))
    return regions

# -------------------------
# Cluster fragments
# -------------------------
def cluster_value_fragments(value_items, dist_thresh_factor=0.03, img_shape=None):
    used = set()
    clusters = []
    if img_shape is None:
        img_shape = (1000,1000)
    dist_thresh = max(img_shape[:2]) * dist_thresh_factor
    centers = [ (it['bbox_xywh'][0] + it['bbox_xywh'][2]/2,
                 it['bbox_xywh'][1] + it['bbox_xywh'][3]/2) for it in value_items ]
    for i, it in enumerate(value_items):
        if i in used:
            continue
        group = [i]
        used.add(i)
        cx_i, cy_i = centers[i]
        changed = True
        while changed:
            changed = False
            for j, jt in enumerate(value_items):
                if j in used:
                    continue
                cx_j, cy_j = centers[j]
                d = math.hypot(cx_i - cx_j, cy_i - cy_j)
                if d < dist_thresh:
                    group.append(j)
                    used.add(j)
                    xs = [centers[k][0] for k in group]
                    ys = [centers[k][1] for k in group]
                    cx_i, cy_i = sum(xs)/len(xs), sum(ys)/len(ys)
                    changed = True
        clusters.append([value_items[k] for k in group])
    combined = []
    for cl in clusters:
        centers_cl = [ (it['bbox_xywh'][0] + it['bbox_xywh'][2]/2,
                        it['bbox_xywh'][1] + it['bbox_xywh'][3]/2) for it in cl ]
        xs = [c[0] for c in centers_cl]
        ys = [c[1] for c in centers_cl]
        if max(xs)-min(xs) >= max(ys)-min(ys):
            order = sorted(range(len(cl)), key=lambda k: centers_cl[k][0])
        else:
            order = sorted(range(len(cl)), key=lambda k: centers_cl[k][1])
        texts = [normalize_value_text(cl[k]['text']) for k in order]
        joined = " | ".join([t for t in texts if t])
        xs_all = [cl[k]['bbox_xywh'][0] for k in range(len(cl))] + [cl[k]['bbox_xywh'][0]+cl[k]['bbox_xywh'][2] for k in range(len(cl))]
        ys_all = [cl[k]['bbox_xywh'][1] for k in range(len(cl))] + [cl[k]['bbox_xywh'][1]+cl[k]['bbox_xywh'][3] for k in range(len(cl))]
        bbox = (int(min(xs_all)), int(min(ys_all)), int(max(xs_all)-min(xs_all)), int(max(ys_all)-min(ys_all)))
        combined.append({'text': joined, 'bbox_xywh': bbox})
    return combined

# -------------------------
# Pair labels and values
# -------------------------
def pair_labels_and_values(labels, value_clusters, max_distance_factor=6.0):
    results = []
    used_vals = set()
    for lab in labels:
        lx, ly, lw, lh = lab['bbox_xywh']
        l_cent = np.array([lx + lw/2, ly + lh/2])
        best = None
        best_score = float('inf')
        for i, val in enumerate(value_clusters):
            if i in used_vals:
                continue
            vx, vy, vw, vh = val['bbox_xywh']
            v_cent = np.array([vx + vw/2, vy + vh/2])
            d = np.linalg.norm(l_cent - v_cent)
            dx = v_cent[0] - l_cent[0]
            dy = abs(v_cent[1] - l_cent[1])
            score = d - 0.8*max(0, dx) + 0.2*dy
            if score < best_score:
                best_score = score
                best = (i, d)
        diag = math.hypot(lw, lh)
        if best and best[1] < max_distance_factor*diag:
            idx = best[0]
            used_vals.add(idx)
            results.append({
                'Label': lab['text'],
                'Value': value_clusters[idx]['text'],
                'Label_BBox': lab['bbox_xywh'],
                'Value_BBox': value_clusters[idx]['bbox_xywh']
            })
        else:
            results.append({
                'Label': lab['text'],
                'Value': '',
                'Label_BBox': lab['bbox_xywh'],
                'Value_BBox': None
            })
    return results

# -------------------------
# Main run
# -------------------------
def run(image_path, out_dir="outputs", gpu=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Image not found: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # preprocessing: grayscale + CLAHE + denoise
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    rgb_enh = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # OCR
    reader = easyocr.Reader(['en'], gpu=gpu)
    items = ocr_multi_angle(rgb_enh, reader, angles=(0,90), conf_threshs=(0.45,0.3))
    if len(items) == 0:
        items = ocr_multi_angle(rgb_enh, reader, angles=(0,), conf_threshs=(0.35,))

    # save raw OCR
    with open(out_dir / "ocr_items.json", "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)

    # label vs value
    symbol_chars = set("Ø⊥°×xX±⌖Φø⌀")
    labels, values = [], []
    for it in items:
        txt = it['text'].strip()
        if any(ch.isdigit() for ch in txt) or any(ch in symbol_chars for ch in txt):
            values.append(it)
        else:
            labels.append(it)

    # callout detection
    callouts = detect_callout_regions(rgb_enh)
    used_val_indices = set()
    grouped_values = []
    for (rx, ry, rw, rh) in callouts:
        group = []
        for i, v in enumerate(values):
            vx, vy, vw, vh = v['bbox_xywh']
            cx, cy = vx + vw/2, vy + vh/2
            if (rx <= cx <= rx+rw and ry <= cy <= ry+rh):
                group.append(i)
        if group:
            texts = [normalize_value_text(values[i]['text']) for i in sorted(group, key=lambda k: values[k]['bbox_xywh'][0])]
            combined_text = " | ".join(t for t in texts if t)
            xs_all = [values[i]['bbox_xywh'][0] for i in group] + [values[i]['bbox_xywh'][0]+values[i]['bbox_xywh'][2] for i in group]
            ys_all = [values[i]['bbox_xywh'][1] for i in group] + [values[i]['bbox_xywh'][1]+values[i]['bbox_xywh'][3] for i in group]
            bbox = (int(min(xs_all)), int(min(ys_all)), int(max(xs_all)-min(xs_all)), int(max(ys_all)-min(ys_all)))
            grouped_values.append({'text': combined_text, 'bbox_xywh': bbox})
            for i in group:
                used_val_indices.add(i)

    remaining_values = [values[i] for i in range(len(values)) if i not in used_val_indices]
    clustered = cluster_value_fragments(remaining_values, dist_thresh_factor=0.03, img_shape=rgb.shape)
    value_clusters = grouped_values + clustered

    # pairing
    results = pair_labels_and_values(labels, value_clusters)
    for r in results:
        if r['Value']:
            r['Value'] = normalize_value_text(r['Value'], label_hint=r['Label'])

    # save JSON & CSV
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "results.csv", "w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Label","Value","Label_BBox","Value_BBox"])
        for r in results:
            writer.writerow([r['Label'], r['Value'], str(r['Label_BBox']), str(r['Value_BBox'])])

    # visualization
    vis = bgr.copy()
    for r in results:
        lx, ly, lw, lh = r['Label_BBox']
        cv2.rectangle(vis, (lx, ly), (lx+lw, ly+lh), (255,0,0), 2)
        cv2.putText(vis, r['Label'][:30], (lx, max(ly-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1, cv2.LINE_AA)
        if r['Value_BBox'] is not None:
            vx, vy, vw, vh = r['Value_BBox']
            cv2.rectangle(vis, (vx, vy), (vx+vw, vy+vh), (0,255,0), 2)
            cv2.putText(vis, r['Value'][:40], (vx, min(vy+vh+14, vis.shape[0]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

    viz_path = out_dir / "viz_improved.jpg"
    cv2.imwrite(str(viz_path), vis)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return {"visualization": vis_rgb, "results": results}
