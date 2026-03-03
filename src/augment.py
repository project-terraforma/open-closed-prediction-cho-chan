"""
augment.py
----------
Pull additional labeled records from the Feb parquet release to supplement
project_c_samples.json during training.

Outputs ONLY the new parquet records (not the original labeled set) to
data/parquet_augment.json.  split.py --augment keeps project_c_samples.json
as the permanent val benchmark and adds these records to train only.

The open-record count is chosen so that the COMBINED dataset
(original + parquet_augment) hits the target closed percentage.

Usage:
    python src/augment.py              # default: 15% closed combined
    python src/augment.py --pct 20
    python src/augment.py --pct 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import numpy as np

ROOT         = Path(__file__).parent.parent
PARQUET_GLOB = str(ROOT / "data" / "parquet" / "2026-02-18.0" / "*.parquet")
SAMPLES_PATH = ROOT / "data" / "project_c_samples.json"
OUTPUT_PATH  = ROOT / "data" / "parquet_augment.json"

SEED = 42

# Columns to pull from parquet (everything feature_engineering.py uses)
SELECT_COLS = """
    id,
    sources,
    names,
    categories,
    confidence,
    websites,
    socials,
    emails,
    phones,
    brand,
    addresses,
    bbox,
    operating_status
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(val: Any) -> Any:
    """Recursively convert pandas/numpy types to JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return [_clean(v) for v in val.tolist()]
    if isinstance(val, (np.floating, float)):
        return None if np.isnan(val) else float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, dict):
        return {k: _clean(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean(v) for v in val]
    return val


def _list(val: Any) -> list:
    """Clean val and guarantee a list (never None)."""
    c = _clean(val)
    return c if isinstance(c, list) else []


def _dict(val: Any) -> dict:
    """Clean val and guarantee a dict (never None)."""
    c = _clean(val)
    return c if isinstance(c, dict) else {}


def row_to_record(row: dict, label: int) -> dict:
    """Convert a parquet row dict to project_c_samples.json compatible format."""
    return {
        "id":         row.get("id"),
        "sources":    _list(row.get("sources")),
        "names":      _clean(row.get("names")),
        "categories": _dict(row.get("categories")),
        "confidence": float(_clean(row.get("confidence")) or 0.0),
        "websites":   _clean(row.get("websites")),
        "socials":    _clean(row.get("socials")),
        "emails":     _clean(row.get("emails")),
        "phones":     _clean(row.get("phones")),
        "brand":      _clean(row.get("brand")),
        "addresses":  _list(row.get("addresses")),
        "bbox":       _clean(row.get("bbox")),
        "open":       label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(closed_pct: int = 15) -> None:
    print(f"Target ratio: {closed_pct}% closed / {100 - closed_pct}% open\n")

    con = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    # 1. Read existing IDs and label counts (for dedup + open sizing)
    # ------------------------------------------------------------------
    print(f"Reading {SAMPLES_PATH.name} for dedup and sizing ...")
    existing_ids: set[str] = set()
    n_existing_open = 0
    n_existing_closed = 0
    with open(SAMPLES_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            existing_ids.add(rec["id"])
            if rec["open"] == 1:
                n_existing_open += 1
            else:
                n_existing_closed += 1
    n_existing = n_existing_open + n_existing_closed
    print(f"  {n_existing:,} records  ({n_existing_open:,} open, {n_existing_closed:,} closed)")

    # ------------------------------------------------------------------
    # 2. Load all closed parquet records (excluding existing IDs)
    # ------------------------------------------------------------------
    print(f"\nLoading closed records from parquet ...")
    t0 = time.perf_counter()

    # Put existing IDs in a temp table for efficient anti-join
    con.execute(
        "CREATE TEMP TABLE existing_ids AS "
        f"SELECT unnest({list(existing_ids)!r}) AS id"
    )

    closed_df: pd.DataFrame = con.execute(f"""
        SELECT {SELECT_COLS}
        FROM read_parquet('{PARQUET_GLOB}') p
        ANTI JOIN existing_ids e ON p.id = e.id
        WHERE p.operating_status = 'closed'
    """).df()

    elapsed = (time.perf_counter() - t0) * 1000
    n_new_closed = len(closed_df)
    print(f"  {n_new_closed:,} new closed records  ({elapsed:.0f} ms)")

    # ------------------------------------------------------------------
    # 3. Determine how many open records to sample
    # ------------------------------------------------------------------
    total_closed = n_existing_closed + n_new_closed
    # total = total_closed / (closed_pct/100)  →  open = total - total_closed
    total_target = round(total_closed / (closed_pct / 100))
    # Already have existing open; need the remainder from parquet
    n_open_needed = total_target - total_closed - n_existing_open
    n_open_needed = max(0, n_open_needed)

    total_open_combined = n_existing_open + n_open_needed
    total_combined      = total_closed + total_open_combined
    print(f"\nDataset plan (combined original + parquet_augment):")
    print(f"  Original records        : {n_existing:,}  ({n_existing_closed:,} closed)")
    print(f"  New closed from parquet : {n_new_closed:,}")
    print(f"  New open from parquet   : {n_open_needed:,}")
    print(f"  Combined total          : {total_combined:,}  "
          f"({total_closed/total_combined*100:.1f}% closed)")

    # ------------------------------------------------------------------
    # 4. Sample open records from parquet
    # ------------------------------------------------------------------
    print(f"\nSampling {n_open_needed:,} open records from parquet ...")
    t0 = time.perf_counter()

    open_df: pd.DataFrame = con.execute(f"""
        SELECT {SELECT_COLS}
        FROM read_parquet('{PARQUET_GLOB}') p
        ANTI JOIN existing_ids e ON p.id = e.id
        WHERE p.operating_status = 'open'
        USING SAMPLE {n_open_needed} ROWS (RESERVOIR, {SEED})
    """).df()

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {len(open_df):,} open records sampled  ({elapsed:.0f} ms)")

    # ------------------------------------------------------------------
    # 5. Write output JSONL
    # ------------------------------------------------------------------
    print(f"\nWriting {OUTPUT_PATH.name} ...")
    t0 = time.perf_counter()

    # Use JSON round-trip to convert all numpy/pyarrow types to plain Python
    closed_records = json.loads(closed_df.to_json(orient="records"))
    open_records   = json.loads(open_df.to_json(orient="records"))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for row in closed_records:
            out.write(json.dumps(row_to_record(row, label=0)) + "\n")
        for row in open_records:
            out.write(json.dumps(row_to_record(row, label=1)) + "\n")

    elapsed = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    n_out = n_new_closed + len(open_df)
    print(f"  {elapsed:.0f} ms")
    print(f"\nParquet augment written to {OUTPUT_PATH}")
    print(f"  Records : {n_out:,}  ({n_new_closed:,} closed, {len(open_df):,} open)")
    print(f"\nNext:")
    print(f"  python src/split.py data/project_c_samples.json --augment data/parquet_augment.json")
    print(f"  python src/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct", type=int, default=15,
                        help="Target closed percentage (default: 15)")
    args = parser.parse_args()
    main(closed_pct=args.pct)
