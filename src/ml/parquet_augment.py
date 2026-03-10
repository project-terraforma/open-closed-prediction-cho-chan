"""
parquet_augment.py
------------------
Extract labeled records from an Overture parquet release for training
augmentation.

Strategy:
  - Closed: all 'closed' / 'permanently_closed' records from top-N regions
  - Open:   fetch a random pool from the same regions, then stratify by
            category in Python to match the labeled training set's distribution

Outputs (written to out_dir):
  parquet_aug_5050.json  — 1:1 closed/open  (50% closed)
  parquet_aug_9010.json  — 9:1 open/closed  (~10% closed, matches real-world rate)

Run:
  python src/ml/parquet_augment.py
  python src/ml/parquet_augment.py \\
      --parquet data/parquet/2026-02-18.0 \\
      --labeled data/project_c_samples.json \\
      --out     data \\
      --top-n   10

Then augment training (use 9010 to keep class balance stable):
  python src/ml/split.py data/project_c_samples.json \\
      --augment data/parquet_aug_9010.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import extract_features, load_dataset

RANDOM_SEED = 42

# Operating-status values that indicate a closed place.
# Overture parquet uses 'permanently_closed'; the pre-built .ddb normalises to 'closed'.
_CLOSED_VALUES = ("closed", "permanently_closed")


# ---------------------------------------------------------------------------
# DataFrame → plain-Python-dict conversion
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of plain Python dicts.

    Handles Overture-specific types that pandas doesn't serialise natively:
    - bytes / bytearray (binary id, geometry) → hex strings
    - numpy scalars → Python int / float
    Uses a JSON round-trip so nested structs and lists also become plain types.
    """
    df = df.copy()
    for col in df.columns:
        # Detect binary columns by inspecting the first non-null value
        sample = next(
            (v for v in df[col] if v is not None and not (isinstance(v, float) and np.isnan(v))),
            None,
        )
        if isinstance(sample, (bytes, bytearray)):
            df[col] = df[col].apply(
                lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else None
            )
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def _glob(parquet_dir: Path) -> str:
    return str(parquet_dir / "*.parquet")


def _top_regions(con: duckdb.DuckDBPyConnection, glob: str, n: int) -> list[str]:
    """Return the top-N region codes ordered by closed-record count."""
    closed_filter = " OR ".join(f"operating_status = '{v}'" for v in _CLOSED_VALUES)
    rows = con.execute(f"""
        SELECT addresses[1].region AS region, COUNT(*) AS cnt
        FROM read_parquet('{glob}')
        WHERE ({closed_filter})
          AND addresses[1].region IS NOT NULL
        GROUP BY region
        ORDER BY cnt DESC
        LIMIT {n}
    """).fetchall()
    print(f"Top-{n} regions by closed count:")
    for region, cnt in rows:
        print(f"  {region:6s}  {cnt}")
    return [r[0] for r in rows]


def _fetch_closed(
    con: duckdb.DuckDBPyConnection, glob: str, regions: list[str]
) -> list[dict]:
    """Fetch all closed records from the given regions."""
    region_list   = ", ".join(f"'{r}'" for r in regions)
    closed_filter = " OR ".join(f"operating_status = '{v}'" for v in _CLOSED_VALUES)
    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{glob}')
        WHERE ({closed_filter})
          AND addresses[1].region IN ({region_list})
    """).df()
    return _df_to_records(df)


def _category_distribution(labeled_path: Path) -> dict[str, float]:
    """Fractional category distribution from the labeled dataset (train proxy)."""
    X, _ = load_dataset(labeled_path)
    return X["primary_category"].value_counts(normalize=True).to_dict()


def _fetch_open_stratified(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    regions: list[str],
    n_open: int,
    cat_dist: dict[str, float],
) -> list[dict]:
    """
    Fetch open records and stratify by category in Python.

    Pulls a random pool (5× target, min 20k) then allocates:
      80% of slots → known categories (proportional to labeled-set distribution)
      20% of slots → any other category (broader coverage)
    """
    region_list = ", ".join(f"'{r}'" for r in regions)
    open_filter = "operating_status IS NULL OR operating_status = 'open'"

    # Fetch a random pool large enough for stratification
    pool_size = max(n_open * 5, 20_000)
    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{glob}')
        WHERE ({open_filter})
          AND addresses[1].region IN ({region_list})
        USING SAMPLE {pool_size} ROWS (reservoir, {RANDOM_SEED})
    """).df()

    if len(df) == 0:
        print("  [WARN] No open records found — check operating_status values in parquet")
        return []

    print(f"  Pool: {len(df):,} open records fetched for stratification")

    # Extract primary category from the nested categories struct
    def _primary(x) -> str:
        if isinstance(x, dict):
            return x.get("primary") or "unknown"
        return "unknown"

    df = df.copy()
    df["_primary_cat"] = df["categories"].apply(_primary)

    known_cats = set(cat_dist.keys())
    n_known = int(n_open * 0.8)
    n_other = n_open - n_known
    rng = np.random.default_rng(RANDOM_SEED)
    frames: list[pd.DataFrame] = []

    # Known categories — proportional allocation
    known_df = df[df["_primary_cat"].isin(known_cats)]
    for cat, frac in cat_dist.items():
        cat_df = known_df[known_df["_primary_cat"] == cat]
        if len(cat_df) == 0:
            continue
        n_want = max(1, round(frac * n_known))
        n_take = min(n_want, len(cat_df))
        frames.append(cat_df.sample(n=n_take, random_state=int(rng.integers(1_000_000))))

    # Other / unknown categories — fill remaining slots
    other_df = df[~df["_primary_cat"].isin(known_cats)]
    if len(other_df) > 0 and n_other > 0:
        n_take = min(n_other, len(other_df))
        frames.append(other_df.sample(n=n_take, random_state=int(rng.integers(1_000_000))))

    if not frames:
        return []

    result = (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["_primary_cat"])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    return _df_to_records(result)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")


def _write_jsonl(records: list[dict], labels: list[int], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row, label in zip(records, labels):
            row = dict(row)
            row["open"] = label
            f.write(json.dumps(row, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(records: list[dict], label: int, n: int = 5) -> bool:
    """Spot-check extract_features() on the first n rows."""
    ok = 0
    for row in records[:n]:
        try:
            extract_features(row)
            ok += 1
        except Exception as e:
            print(f"  [WARN] extract_features failed (label={label}): {e}")
    if ok:
        print(f"  Validation OK — {ok}/{min(n, len(records))} rows parsed (label={label})")
    return ok > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_augment(
    parquet_dir: Path,
    labeled_path: Path,
    out_dir: Path,
    top_n: int = 10,
    ratios: dict[str, int] | None = None,
) -> None:
    """
    Build augmentation JSONL files from the Overture parquet release.

    Args:
        parquet_dir:  Directory containing *.parquet files.
        labeled_path: Labeled JSONL (project_c_samples.json) for category dist.
        out_dir:      Output directory.
        top_n:        Number of top regions (by closed count) to pull from.
        ratios:       {file_suffix: open_multiplier}.
                      Default: {"5050": 1, "9010": 9}
    """
    if ratios is None:
        ratios = {"5050": 1, "9010": 9}

    glob = _glob(parquet_dir)
    con  = duckdb.connect()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parquet : {glob}")
    print(f"Labeled : {labeled_path}\n")

    # --- Top regions ---
    regions = _top_regions(con, glob, top_n)

    # --- Closed records ---
    print(f"\nFetching closed records ...")
    closed_rows = _fetch_closed(con, glob, regions)
    n_closed = len(closed_rows)
    print(f"  {n_closed:,} closed records fetched")
    _validate(closed_rows, label=0)

    # --- Category distribution ---
    print(f"\nComputing category distribution from labeled set ...")
    cat_dist = _category_distribution(labeled_path)
    print(f"  {len(cat_dist)} unique categories")

    # --- Build one file per ratio ---
    for suffix, multiplier in ratios.items():
        n_open = n_closed * multiplier
        print(f"\n=== {suffix}  ({n_closed} closed + {n_open} open target) ===")

        open_rows = _fetch_open_stratified(con, glob, regions, n_open, cat_dist)
        actual_open = len(open_rows)
        print(f"  {actual_open:,} open records sampled")
        if actual_open:
            _validate(open_rows, label=1)

        all_rows   = closed_rows + open_rows
        all_labels = [0] * n_closed + [1] * actual_open

        out_path = out_dir / f"parquet_aug_{suffix}.json"
        _write_jsonl(all_rows, all_labels, out_path)
        closed_pct = 100 * n_closed / len(all_rows)
        print(f"  Saved → {out_path}  ({len(all_rows):,} records, {closed_pct:.1f}% closed)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build augmentation JSONL from Overture parquet release"
    )
    parser.add_argument(
        "--parquet", type=Path,
        default=Path("data/parquet/2026-02-18.0"),
        help="Directory of *.parquet files  (default: data/parquet/2026-02-18.0)",
    )
    parser.add_argument(
        "--labeled", type=Path,
        default=Path("data/project_c_samples.json"),
        help="Labeled JSONL for category distribution  (default: data/project_c_samples.json)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data"),
        help="Output directory  (default: data)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top regions by closed count  (default: 10)",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        sys.exit(f"Parquet directory not found: {args.parquet}")
    if not args.labeled.exists():
        sys.exit(f"Labeled file not found: {args.labeled}")

    build_augment(
        parquet_dir  = args.parquet,
        labeled_path = args.labeled,
        out_dir      = args.out,
        top_n        = args.top_n,
    )
