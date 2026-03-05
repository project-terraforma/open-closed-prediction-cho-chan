# Requires: pip install pandas pyarrow (or use this repo's .venv)
# Usage: python exploreBalanced.py [path/to/file.parquet]
#        defaults to the most recent overture_us_balanced_*.parquet in data/
import json
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
        pass
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    s = str(val)
    return s if len(s) <= MAX_VAL_LEN else s[:MAX_VAL_LEN] + "..."

def section(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print('=' * 50)

# --- resolve file path ---
repo_dir = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    parquet_path = Path(sys.argv[1])
else:
    # pick the most recently modified balanced parquet in data/
    candidates = sorted(
        (repo_dir / "data").glob("overture_us_balanced_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("No overture_us_balanced_*.parquet found in data/")
        print("Usage: python exploreBalanced.py [path/to/file.parquet]")
        sys.exit(1)
    parquet_path = candidates[0]

if not parquet_path.exists():
    print(f"File not found: {parquet_path}")
    sys.exit(1)

df = pd.read_parquet(parquet_path, engine="pyarrow")
n = len(df)

# ── 1. basic overview ────────────────────────────────────────────────────────
section("OVERVIEW")
print(f"File   : {parquet_path}")
print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

if "open" in df.columns:
    num_open     = int((df["open"] == 1).sum())
    num_not_open = int((df["open"] == 0).sum())
    print(f"\nClass balance:")
    print(f"  open     : {num_open:>7,}  ({num_open/n:.1%})")
    print(f"  not open : {num_not_open:>7,}  ({num_not_open/n:.1%})")

if "operating_status" in df.columns:
    print(f"\noperating_status breakdown:")
    for status, cnt in df["operating_status"].value_counts().items():
        print(f"  {status:<25} {cnt:>7,}  ({cnt/n:.1%})")

# ── 2. sample rows ────────────────────────────────────────────────────────────
section("SAMPLE ROWS (first 3 open, first 3 not-open)")
for label, label_name in [(1, "OPEN"), (0, "NOT OPEN")]:
    subset = df[df["open"] == label].head(3)
    for i, (_, row) in enumerate(subset.iterrows()):
        print(f"\n  [{label_name}] Row {i}")
        for c in df.columns:
            print(f"    {c}: {fmt(row[c])}")

# ── 3. field presence rates ───────────────────────────────────────────────────
# these become binary "has_X" features for the model
section("FIELD PRESENCE RATES  →  candidate binary features")

def is_present(val):
    """true if the field has a non-null, non-empty value"""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        parsed = val.strip()
        return parsed not in ("", "null", "[]", "{}")
    return True

presence_cols = ["name", "websites", "socials", "emails", "phones", "brand", "addresses"]
print(f"\n  {'field':<15} {'present':>10}  {'% overall':>10}  {'% open':>10}  {'% not-open':>12}")
print(f"  {'-'*15} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")
for col in presence_cols:
    if col not in df.columns:
        continue
    mask      = df[col].apply(is_present)
    pct_all   = mask.mean()
    pct_open  = mask[df["open"] == 1].mean()
    pct_close = mask[df["open"] == 0].mean()
    print(f"  {col:<15} {mask.sum():>10,}  {pct_all:>10.1%}  {pct_open:>10.1%}  {pct_close:>12.1%}")

# ── 4. confidence score stats ─────────────────────────────────────────────────
section("CONFIDENCE SCORE  →  direct numeric feature")
if "confidence" in df.columns:
    for label, name in [(1, "open"), (0, "not-open")]:
        sub = df[df["open"] == label]["confidence"].dropna()
        print(f"  {name:<10}  mean={sub.mean():.3f}  median={sub.median():.3f}"
              f"  min={sub.min():.3f}  max={sub.max():.3f}")

# ── 5. category breakdown ─────────────────────────────────────────────────────
section("TOP basic_category  →  candidate categorical feature")
if "basic_category" in df.columns:
    top = df["basic_category"].value_counts().head(15)
    print(f"\n  {'category':<35} {'count':>8}  {'%':>6}")
    print(f"  {'-'*35} {'-'*8}  {'-'*6}")
    for cat, cnt in top.items():
        print(f"  {str(cat):<35} {cnt:>8,}  {cnt/n:>6.1%}")

# ── 6. source dataset breakdown ───────────────────────────────────────────────
# num_sources and which datasets contributed are useful features
section("SOURCE DATASETS  →  candidate features (num_sources, source_is_meta, etc.)")
if "sources" in df.columns:
    dataset_counts = {}
    num_sources_list = []
    for raw in df["sources"]:
        try:
            srcs = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            srcs = []
        num_sources_list.append(len(srcs))
        for s in srcs:
            ds = s.get("dataset", "unknown") if isinstance(s, dict) else "unknown"
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

    ns = pd.Series(num_sources_list)
    print(f"\n  num_sources per place:  mean={ns.mean():.2f}  max={int(ns.max())}")
    print(f"\n  contributing datasets:")
    for ds, cnt in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        print(f"    {ds:<20} {cnt:>8,}")

# ── 7. feature engineering ideas ─────────────────────────────────────────────
section("FEATURE ENGINEERING IDEAS (from this dataset)")
ideas = [
    ("has_website",        "1 if websites field is non-empty"),
    ("has_social",         "1 if socials field is non-empty"),
    ("has_phone",          "1 if phones field is non-empty"),
    ("has_email",          "1 if emails field is non-empty"),
    ("has_brand",          "1 if brand field is non-null"),
    ("completeness_score", "count of non-empty optional fields (0–5)"),
    ("confidence",         "overture confidence score directly"),
    ("num_sources",        "number of contributing data sources"),
    ("source_is_meta",     "1 if meta (facebook) is a source"),
    ("source_is_msft",     "1 if Microsoft is a source"),
    ("basic_category_enc", "label-encoded basic_category"),
    ("taxonomy_depth",     "length of taxonomy.hierarchy array"),
    ("has_full_address",   "1 if locality + postcode + region all present"),
]
print()
for feat, desc in ideas:
    print(f"  {feat:<25} — {desc}")
