"""
parquet_probe.py
----------------
Quick diagnostic before augmenting the training set from the Feb parquet release.

Checks:
  1. Distinct operating_status values and their counts
  2. How many parquet 'closed' IDs already exist in project_c_samples.json
  3. How many net-new closed records are available after dedup
  4. Open record availability for sampling

Run:
    python src/parquet_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

ROOT         = Path(__file__).parent.parent
PARQUET_GLOB = str(ROOT / "data" / "parquet" / "2026-02-18.0" / "*.parquet")
SAMPLES_PATH = ROOT / "data" / "project_c_samples.json"


def main() -> None:
    con = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    # 1. operating_status distribution in Feb parquet
    # ------------------------------------------------------------------
    print("Operating status distribution in Feb parquet ...")
    t0 = time.perf_counter()
    rows = con.execute(f"""
        SELECT
            operating_status,
            COUNT(*) AS cnt
        FROM read_parquet('{PARQUET_GLOB}')
        GROUP BY operating_status
        ORDER BY cnt DESC
    """).fetchall()
    elapsed = (time.perf_counter() - t0) * 1000
    total_parquet = sum(r[1] for r in rows)
    print(f"  ({elapsed:.0f} ms  |  {total_parquet:,} total records)")
    for status, cnt in rows:
        pct = cnt / total_parquet * 100
        print(f"  {str(status):30s}  {cnt:>10,}  ({pct:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Load existing labeled IDs from project_c_samples.json
    # ------------------------------------------------------------------
    print(f"\nLoading existing IDs from {SAMPLES_PATH.name} ...")
    existing_ids: set[str] = set()
    existing_closed_ids: set[str] = set()
    with open(SAMPLES_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            existing_ids.add(rec["id"])
            if rec.get("open") == 0:
                existing_closed_ids.add(rec["id"])
    print(f"  {len(existing_ids):,} total  |  {len(existing_closed_ids):,} closed")

    # ------------------------------------------------------------------
    # 3. Closed records in parquet — overlap with existing
    # ------------------------------------------------------------------
    print(f"\nChecking closed parquet records vs existing labeled set ...")
    t0 = time.perf_counter()
    closed_ids_parquet = set(
        row[0] for row in con.execute(f"""
            SELECT id
            FROM read_parquet('{PARQUET_GLOB}')
            WHERE operating_status = 'closed'
        """).fetchall()
    )
    elapsed = (time.perf_counter() - t0) * 1000

    overlap      = closed_ids_parquet & existing_ids
    net_new      = closed_ids_parquet - existing_ids
    print(f"  ({elapsed:.0f} ms)")
    print(f"  Closed in parquet      : {len(closed_ids_parquet):,}")
    print(f"  Already in labeled set : {len(overlap):,}")
    print(f"  Net-new closed records : {len(net_new):,}")

    # ------------------------------------------------------------------
    # 4. Open records available (excluding existing IDs)
    # ------------------------------------------------------------------
    print(f"\nOpen records available for sampling (excluding existing) ...")
    t0 = time.perf_counter()
    open_available = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE operating_status = 'open'
          AND id NOT IN (SELECT unnest({list(existing_ids)!r}))
    """).fetchone()[0]
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {open_available:,} open records available  ({elapsed:.0f} ms)")

    # ------------------------------------------------------------------
    # 5. Recommended dataset sizes
    # ------------------------------------------------------------------
    total_closed = len(existing_closed_ids) + len(net_new)
    print(f"\nProjected augmented dataset sizes:")
    print(f"  Total closed available : {total_closed:,}  "
          f"({len(existing_closed_ids):,} existing + {len(net_new):,} new)")
    for pct in [10, 15, 20]:
        n_open = round(total_closed * (100 - pct) / pct)
        total  = total_closed + n_open
        print(f"  {pct}% closed → pull {n_open:,} open  (total {total:,})")

    print("\nDone.")


if __name__ == "__main__":
    main()
