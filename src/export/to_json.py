# src/export/to_csv.py
import csv

def save_to_csv(results, out_path):
    fieldnames = ["Label", "Value", "Label_BBox", "Value_BBox"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "Label": r["label"],
                "Value": r["value"],
                "Label_BBox": " ".join(map(str, r["label_bbox"])),
                "Value_BBox": " ".join(map(str, r["value_bbox"]))
            })
