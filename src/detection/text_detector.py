from typing import List, Dict
import easyocr
from PIL import Image
import numpy as np

class EasyOCRDetector:
    def __init__(self, langs=["en"], gpu=False):
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def run(self, image_pil: Image.Image, min_conf=0.55) -> List[Dict]:
        img_np = np.array(image_pil)
        result = self.reader.readtext(img_np, detail=1, paragraph=False)
        items = []
        for quad, text, conf in result:
            if conf < min_conf:
                continue
            items.append({
                "text": text,
                "conf": float(conf),
                "bbox_quad": [[float(x), float(y)] for x, y in quad]
            })
        return items
