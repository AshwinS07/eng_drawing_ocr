# src/export/to_csv.py
import pandas as pd

def export_rows_to_csv(rows, out_path):
    """
    Export detected label-value pairs to CSV.
    rows may be:
      - list of dicts like {"label": ..., "value": ...}
      - list of lists/tuples like [label, value]
    """
    if not rows:
        return

    # Normalize rows
    normalized = []
    for r in rows:
        if isinstance(r, dict):
            normalized.append([r.get("label", ""), r.get("value", "")])
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            normalized.append([r[0], r[1]])
        else:
            continue

    df = pd.DataFrame(normalized, columns=["Label", "Value"])
    df.to_csv(out_path, index=False)
