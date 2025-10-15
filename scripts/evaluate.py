"""
Simple evaluation helper (stub).
Use this to compare ground-truth text and predicted text for a small dataset.
You can extend to compute Word Error Rate (WER) or character-level metrics.
"""
import json
from pathlib import Path

def load_prediction(path):
    return json.load(open(path, "r", encoding="utf-8"))

def compare(gt_json, pred_json):
    # Very basic: list predicted texts vs ground truth lines
    gt = load_prediction(gt_json)
    pred = load_prediction(pred_json)
    print("GT items:", len(gt))
    print("Pred items:", len(pred))
    # Extend for better metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    args = p.parse_args()
    compare(args.gt, args.pred)
