"""
yelp_probe.py
-------------
Compare Yelp academic dataset against project_c_samples.json.

Phase 1: spatial bbox candidates at 50/100/200m
Phase 2: apply rapidfuzz name similarity on 100m candidates at multiple
         thresholds to find the true match rate and label agreement.

Run:
    python src/yelp_probe.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("WARNING: rapidfuzz not installed. Run: pip install rapidfuzz")
    print("         Skipping name similarity section.\n")

ROOT         = Path(__file__).parent.parent
YELP_PATH    = ROOT / "data" / "yelp-academic-dataset-business.json"
SAMPLES_PATH = ROOT / "data" / "project_c_samples.json"

THRESHOLDS = {
    " 50m": 0.00045,
    "100m": 0.00090,
    "200m": 0.00180,
}
NAME_THRESHOLDS = [60, 70, 75, 80, 85, 90]


def main() -> None:
    con = duckdb.connect(":memory:")

    # ------------------------------------------------------------------
    # 1. Load Yelp
    # ------------------------------------------------------------------
    print("Loading Yelp ...")
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE TEMP TABLE yelp AS
        SELECT
            business_id,
            name,
            latitude,
            longitude,
            CAST(is_open AS INTEGER) AS is_open,
            state
        FROM read_ndjson(
            '{YELP_PATH}',
            columns = {{
                business_id : 'VARCHAR',
                name        : 'VARCHAR',
                latitude    : 'DOUBLE',
                longitude   : 'DOUBLE',
                is_open     : 'INTEGER',
                state       : 'VARCHAR'
            }},
            ignore_errors = true
        )
    """)
    y_total, y_open, y_closed = con.execute(
        "SELECT COUNT(*), SUM(is_open), COUNT(*) - SUM(is_open) FROM yelp"
    ).fetchone()
    print(f"  {(time.perf_counter()-t0)*1000:.0f} ms  |  "
          f"{y_total:,} total  |  {y_open:,} open  |  {y_closed:,} closed")

    # ------------------------------------------------------------------
    # 2. Load labeled Overture dataset (include name)
    # ------------------------------------------------------------------
    print(f"\nLoading {SAMPLES_PATH.name} ...")
    t0 = time.perf_counter()
    con.execute(f"""
        CREATE TEMP TABLE overture AS
        SELECT
            id,
            json_extract_string(names, '$.primary')                     AS o_name,
            CAST(bbox.xmin AS DOUBLE)                                   AS lon,
            CAST(bbox.ymin AS DOUBLE)                                   AS lat,
            CAST(open AS INTEGER)                                       AS label
        FROM read_ndjson(
            '{SAMPLES_PATH}',
            columns = {{
                id    : 'VARCHAR',
                names : 'JSON',
                bbox  : 'STRUCT(xmin DOUBLE, xmax DOUBLE, ymin DOUBLE, ymax DOUBLE)',
                open  : 'INTEGER'
            }},
            ignore_errors = true
        )
    """)
    o_total, o_open, o_closed = con.execute(
        "SELECT COUNT(*), SUM(label), COUNT(*) - SUM(label) FROM overture"
    ).fetchone()
    print(f"  {(time.perf_counter()-t0)*1000:.0f} ms  |  "
          f"{o_total:,} total  |  {o_open:,} open  |  {o_closed:,} closed")

    # ------------------------------------------------------------------
    # 3. Spatial candidate counts at each radius
    # ------------------------------------------------------------------
    print("\nPhase 1 — Spatial candidates (bbox only):")
    print(f"  {'Radius':>6}  {'Matched OV':>12}  {'Matched Yelp':>14}  {'Pairs':>8}  {'Time':>8}")
    print("  " + "-" * 58)

    for label, delta in THRESHOLDS.items():
        t0 = time.perf_counter()
        m_ov, m_yelp, pairs = con.execute(f"""
            SELECT
                COUNT(DISTINCT o.id),
                COUNT(DISTINCT y.business_id),
                COUNT(*)
            FROM overture o
            JOIN yelp y
              ON y.longitude BETWEEN o.lon - {delta} AND o.lon + {delta}
             AND y.latitude  BETWEEN o.lat - {delta} AND o.lat + {delta}
        """).fetchone()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {label:>6}  {m_ov:>8,} ({m_ov/o_total*100:4.1f}%)  "
              f"{m_yelp:>9,} ({m_yelp/y_total*100:4.1f}%)  {pairs:>6,}  {elapsed:>5.0f} ms")

    # ------------------------------------------------------------------
    # 4. Pull 100m candidates into Python for name filtering
    # ------------------------------------------------------------------
    if not HAS_RAPIDFUZZ:
        return

    delta = THRESHOLDS["100m"]
    print(f"\nPhase 2 — Name similarity on ~100m candidates ...")
    t0 = time.perf_counter()
    candidates: pd.DataFrame = con.execute(f"""
        SELECT
            o.id          AS o_id,
            o.o_name      AS o_name,
            o.label       AS o_label,
            y.business_id AS y_id,
            y.name        AS y_name,
            y.is_open     AS y_label
        FROM overture o
        JOIN yelp y
          ON y.longitude BETWEEN o.lon - {delta} AND o.lon + {delta}
         AND y.latitude  BETWEEN o.lat - {delta} AND o.lat + {delta}
    """).df()
    fetch_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(candidates):,} candidate pairs fetched in {fetch_ms:.0f} ms")

    # Compute token_sort_ratio for each pair (handles word-order differences)
    t0 = time.perf_counter()
    candidates["name_sim"] = [
        rfuzz.token_sort_ratio(str(a or ""), str(b or ""))
        for a, b in zip(candidates["o_name"], candidates["y_name"])
    ]
    sim_ms = (time.perf_counter() - t0) * 1000
    print(f"  name similarity computed in {sim_ms:.0f} ms")

    # ------------------------------------------------------------------
    # 5. Match counts at each name similarity threshold
    # ------------------------------------------------------------------
    print(f"\n  {'Sim≥':>5}  {'Pairs':>8}  {'Uniq OV':>10}  {'Uniq Yelp':>11}  "
          f"{'Agree %':>9}")
    print("  " + "-" * 52)

    for thr in NAME_THRESHOLDS:
        df = candidates[candidates["name_sim"] >= thr]
        if df.empty:
            print(f"  {thr:>5}  {'0':>8}  {'0':>10}  {'0':>11}  {'—':>9}")
            continue
        # For each Overture place keep only the best-scoring Yelp match
        best = df.sort_values("name_sim", ascending=False).drop_duplicates("o_id")
        agree = (best["o_label"] == best["y_label"]).sum()
        pct   = agree / len(best) * 100
        print(f"  {thr:>5}  {len(df):>8,}  {df['o_id'].nunique():>10,}  "
              f"{df['y_id'].nunique():>11,}  {pct:>8.1f}%")

    # ------------------------------------------------------------------
    # 6. Label agreement detail at threshold=80 (best match per OV place)
    # ------------------------------------------------------------------
    thr = 80
    df80 = candidates[candidates["name_sim"] >= thr]
    best80 = df80.sort_values("name_sim", ascending=False).drop_duplicates("o_id")

    print(f"\n  Label agreement detail at sim≥{thr} (best match per Overture place):")
    label_map = {1: "open  ", 0: "closed"}
    print(f"  {'Overture':>10}  {'Yelp':>10}  {'Count':>8}  Note")
    print("  " + "-" * 48)
    for (ol, yl), grp in best80.groupby(["o_label", "y_label"]):
        note = "✓ agree" if ol == yl else "✗ conflict"
        print(f"  {label_map.get(ol, ol):>10}  {label_map.get(yl, yl):>10}  "
              f"{len(grp):>8,}  {note}")

    # ------------------------------------------------------------------
    # 7. Sample of high-confidence matches
    # ------------------------------------------------------------------
    print(f"\n  Top 10 highest-similarity pairs (sim≥{thr}):")
    top = df80.sort_values("name_sim", ascending=False).head(10)
    print(f"  {'Overture name':30s}  {'Yelp name':30s}  {'Sim':>4}  {'OV':>4}  {'Yelp':>4}")
    print("  " + "-" * 78)
    for _, row in top.iterrows():
        ov   = str(row["o_name"] or "")[:29]
        yp   = str(row["y_name"] or "")[:29]
        sim  = int(row["name_sim"])
        ol   = "open" if row["o_label"] == 1 else "clos"
        yl   = "open" if row["y_label"] == 1 else "clos"
        print(f"  {ov:30s}  {yp:30s}  {sim:>4}  {ol:>4}  {yl:>4}")

    print("\nDone.")


if __name__ == "__main__":
    main()
