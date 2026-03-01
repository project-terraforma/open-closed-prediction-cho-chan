# Requires: pip install pandas pyarrow (or use this repo's .venv)
# Usage: python viewData.py [path/to/file.parquet]   (default: overture.parquet in repo or parent folder)
import sys
from pathlib import Path

import pandas as pd

MAX_VAL_LEN = 60

def fmt(val):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (ValueError, TypeError):
        pass  # arrays etc. can't be used with pd.isna
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    s = str(val)
    return s if len(s) <= MAX_VAL_LEN else s[:MAX_VAL_LEN] + "..."

# Script lives in open-closed-prediction-cho-chan/; data may be here or in parent (Terraforma)
repo_dir = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    parquet_path = Path(sys.argv[1])
else:
    parquet_path = repo_dir / "overture.parquet"
    if not parquet_path.exists():
        parquet_path = repo_dir.parent / "overture.parquet"

if not parquet_path.exists():
    print(f"File not found: {parquet_path}")
    print("Usage: python viewData.py [path/to/file.parquet]")
    sys.exit(1)

df = pd.read_parquet(parquet_path, engine="pyarrow")
n_show = 5
cols = list(df.columns)

print(f"File: {parquet_path}")
print(f"Total rows: {df.shape[0]}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
if "open" in df.columns:
    open_counts = df["open"].value_counts(dropna=False).to_dict()
    num_open = int(open_counts.get(1, 0))
    num_not_open = int(df.shape[0] - num_open)
    frac_open = num_open / df.shape[0] if df.shape[0] else 0.0
    print(f"Open vs non-open: {num_open} open, {num_not_open} not-open "
          f"({frac_open:.3%} open)")

print(f"\nColumns: {cols}\n")
print("=" * 50)

for i in range(min(n_show, len(df))):
    print(f"\n--- Row {i} ---")
    for c in cols:
        print(f"  {c}: {fmt(df.iloc[i][c])}")