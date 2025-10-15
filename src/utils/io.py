from pathlib import Path
import json
from PIL import Image

def read_image(path):
    return Image.open(path).convert("RGB")

def save_json(obj, path, indent=2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
