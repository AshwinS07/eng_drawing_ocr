import argparse
from pathlib import Path
from src.pipeline.run_pipeline import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--cfg", default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--out", default="outputs/run1", help="Output directory")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    payload = run(args.image, args.cfg, args.out)
    print(f"Saved -> {args.out}/result.json")
    print(f"Found {len(payload.get('ocr_items', []))} OCR items, {len(payload.get('table_cells', []))} table regions.")

if __name__ == "__main__":
    main()
