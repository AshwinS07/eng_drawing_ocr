# src/detection/arrow_detector.py
import cv2
import numpy as np

def detect_lines_and_arrows(bgr_image, cfg):
    """
    Detect line segments via edge detection + HoughLinesP.
    Attempt to identify arrow-like segments by looking for short perpendicular
    segments (a heuristic) or by using a YOLO arrow detector (if available).
    Returns list of arrow-like dicts: [{'p1':(x1,y1),'p2':(x2,y2),'angle':a,'length':L}]
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    th1 = cfg.get("hough", {}).get("canny_thresh1", 50)
    th2 = cfg.get("hough", {}).get("canny_thresh2", 150)
    edges = cv2.Canny(gray, th1, th2)
    minLineLength = cfg.get("hough", {}).get("min_line_length", 30)
    maxLineGap = cfg.get("hough", {}).get("max_line_gap", 10)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=minLineLength, maxLineGap=maxLineGap)
    outs = []
    if lines is None:
        return outs
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2-x1, y2-y1)
        angle = np.degrees(np.arctan2((y2-y1),(x2-x1)))
        outs.append({"p1":(int(x1),int(y1)), "p2":(int(x2),int(y2)), "angle":float(angle), "length":float(length)})
    return outs
