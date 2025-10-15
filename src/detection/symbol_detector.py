# src/detection/symbol_detector.py
from pathlib import Path
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class SymbolDetector:
    """
    Wrapper for YOLOv8-based symbol detection.
    Expects a model that outputs classes like 'diameter','weld','arrow','perp','plus_minus', etc.
    """

    def __init__(self, model_path: str = "models/symbols.pt", conf: float = 0.25):
        self.model_path = str(model_path)
        self.conf = conf
        self.model = None
        if YOLO is not None:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # will still initialize but will error when run; user must supply model
                self.model = None

    def run(self, bgr_image):
        """
        Run symbol detection on BGR uint8 image.
        Returns list of dicts: [{'class': 'diameter', 'conf': 0.9, 'bbox_xywh': [x,y,w,h], 'xyxy':[x1,y1,x2,y2]}, ...]
        """
        if self.model is None:
            # No model: return empty list
            return []

        # Ultralytics model expects RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, imgsz=1280, conf=self.conf, verbose=False)
        outs = []
        # results is a list; handle first item
        res = results[0]
        boxes = res.boxes
        # boxes.xyxy, boxes.cls, boxes.conf
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()  # x1,y1,x2,y2
            cls = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            item = {
                "class_id": cls,
                "conf": conf,
                "bbox_xywh": [float(x1), float(y1), float(w), float(h)],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)]
            }
            # If model has names, include them
            try:
                item["class_name"] = res.names[cls]
            except Exception:
                item["class_name"] = str(cls)
            outs.append(item)
        return outs
