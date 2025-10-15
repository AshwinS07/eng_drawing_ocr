import cv2
import numpy as np

def detect_table_cells(bin_img, min_area=2000):
    # Expect binary image (0 or 255)
    # Detect horizontal and vertical lines using morphology to find table structure
    img = 255 - bin_img  # invert: text/lines -> white
    # scale kernels by image width
    scale = max(1, bin_img.shape[1] // 800)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(40*scale), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(40*scale)))

    h_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    v_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    table_mask = cv2.add(h_open, v_open)
    # dilate to merge
    table_mask = cv2.dilate(table_mask, np.ones((3,3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < min_area:
            continue
        boxes.append([int(x), int(y), int(w), int(h)])
    # sort top->bottom
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes
