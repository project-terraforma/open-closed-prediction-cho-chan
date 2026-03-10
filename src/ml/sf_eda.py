"""
sf_eda.py
---------
EDA for the SF registered business dataset (GeoJSON FeatureCollection).

Analyses:
  1. Record count and field null rates
  2. Labeling signal: dba_end_date, location_end_date, administratively_closed
  3. Date distributions and business duration
  4. Industry (NAICS) distribution
  5. Geography: city, neighborhood, corridor, supervisor district
  6. Potential labeling strategies and class balance
  7. Cross-tabs between closure signals

Run:
    python src/ml/sf_eda.py data/sf_open_dataset_20260309.geojson

Output is printed to stdout AND saved to data/sf_eda_results.txt.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reference date: data_as_of in the dataset
REFERENCE_DATE = datetime(2026, 3, 9, tzinfo=timezone.utc)


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path: Path):
        self._file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_geojson(path: str | Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    features = data["features"]
    rows = [f["properties"] for f in features]
    df = pd.DataFrame(rows)

    # Add geometry as has_geometry flag
    df["has_geometry"] = [f["geometry"] is not None for f in features]

    print(f"  {len(df):,} records, {len(df.columns)} columns\n")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(s) -> datetime | None:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _age_days(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    return (REFERENCE_DATE - dt).days


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def pct(n: int, total: int) -> str:
    return f"{n:>8,}  ({100*n/total:5.1f}%)"


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def null_rates(df: pd.DataFrame) -> None:
    section("NULL RATES BY FIELD")
    n = len(df)
    rows = []
    for col in df.columns:
        null_count = df[col].isna().sum() + (df[col] == "").sum()
        rows.append((col, null_count, n - null_count))
    rows.sort(key=lambda x: x[1])
    print(f"{'Field':<42} {'Null':>8}  {'Filled':>8}  {'Fill%':>6}")
    print("-" * 72)
    for col, null_c, filled_c in rows:
        fill_pct = 100 * filled_c / n if n else 0
        print(f"  {col:<40} {null_c:>8,}  {filled_c:>8,}  {fill_pct:5.1f}%")


def closure_signals(df: pd.DataFrame) -> None:
    section("CLOSURE SIGNALS")
    n = len(df)

    # Parse dates
    df["_dba_end_dt"]      = df["dba_end_date"].apply(_parse_dt)
    df["_loc_end_dt"]      = df["location_end_date"].apply(_parse_dt)
    df["_dba_start_dt"]    = df["dba_start_date"].apply(_parse_dt)
    df["_loc_start_dt"]    = df["location_start_date"].apply(_parse_dt)

    # Signals
    df["_dba_end_past"]    = df["_dba_end_dt"].apply(lambda d: d is not None and d <= REFERENCE_DATE)
    df["_loc_end_past"]    = df["_loc_end_dt"].apply(lambda d: d is not None and d <= REFERENCE_DATE)
    df["_admin_closed"]    = df["administratively_closed"].notna() & (df["administratively_closed"] != "")
    df["_no_end_date"]     = df["_dba_end_dt"].isna()

    print(f"\n  Total records: {n:,}")
    print(f"\n  -- dba_end_date --")
    print(f"  Has end date (any):             {pct(df['_dba_end_dt'].notna().sum(), n)}")
    print(f"  End date in past (closed):      {pct(df['_dba_end_past'].sum(), n)}")
    print(f"  End date in future (active?):   {pct((df['_dba_end_dt'].notna() & ~df['_dba_end_past']).sum(), n)}")
    print(f"  No end date (presumed open):    {pct(df['_no_end_date'].sum(), n)}")

    print(f"\n  -- location_end_date --")
    print(f"  Has end date:                   {pct(df['_loc_end_dt'].notna().sum(), n)}")
    print(f"  End date in past:               {pct(df['_loc_end_past'].sum(), n)}")

    print(f"\n  -- administratively_closed --")
    admin_vals = df["administratively_closed"].value_counts(dropna=False)
    for val, cnt in admin_vals.items():
        label = "(null)" if pd.isna(val) else repr(val)
        print(f"    {label:<30} {pct(cnt, n)}")

    print(f"\n  -- Labeling strategies --")
    # Strategy A: dba_end_date in past
    closed_A = df["_dba_end_past"]
    # Strategy B: dba_end_date OR administratively_closed
    closed_B = df["_dba_end_past"] | df["_admin_closed"]
    # Strategy C: location_end_date in past (location-level closure)
    closed_C = df["_loc_end_past"]
    # Strategy D: any end signal (union)
    closed_D = df["_dba_end_past"] | df["_loc_end_past"] | df["_admin_closed"]

    for label, mask in [
        ("A: dba_end_date in past", closed_A),
        ("B: A + administratively_closed", closed_B),
        ("C: location_end_date in past", closed_C),
        ("D: any end signal (union)", closed_D),
    ]:
        closed_n = mask.sum()
        open_n   = n - closed_n
        print(f"\n  Strategy {label}")
        print(f"    Closed: {pct(closed_n, n)}")
        print(f"    Open:   {pct(open_n, n)}")

    # Agreement between signals
    print(f"\n  -- Signal agreement --")
    both    = (df["_dba_end_past"] & df["_admin_closed"]).sum()
    dba_only = (df["_dba_end_past"] & ~df["_admin_closed"]).sum()
    adm_only = (~df["_dba_end_past"] & df["_admin_closed"]).sum()
    print(f"  dba_end_past AND admin_closed:  {pct(both, n)}")
    print(f"  dba_end_past only:              {pct(dba_only, n)}")
    print(f"  admin_closed only:              {pct(adm_only, n)}")

    return df  # return with parsed columns attached


def date_distributions(df: pd.DataFrame) -> None:
    section("DATE DISTRIBUTIONS")

    for col, dt_col in [
        ("dba_end_date", "_dba_end_dt"),
        ("dba_start_date", "_dba_start_dt"),
    ]:
        dts = df[dt_col].dropna()
        if len(dts) == 0:
            continue
        years = pd.Series([d.year for d in dts])
        print(f"\n  {col}  (n={len(dts):,})")
        print(f"    Range: {min(dts).date()} → {max(dts).date()}")
        print(f"    Year distribution:")
        for yr, cnt in sorted(years.value_counts().items()):
            bar = "#" * min(int(cnt / max(years.value_counts()) * 40), 40)
            print(f"      {yr}  {cnt:>6,}  {bar}")

    # Business duration for records with both start and end
    mask = df["_dba_start_dt"].notna() & df["_dba_end_dt"].notna()
    if mask.sum() > 0:
        durations = df[mask].apply(
            lambda r: (_parse_dt(r["dba_end_date"]) - _parse_dt(r["dba_start_date"])).days / 365.25
            if _parse_dt(r["dba_end_date"]) and _parse_dt(r["dba_start_date"]) else None,
            axis=1,
        ).dropna()
        print(f"\n  Business duration (years, closed records with both dates, n={len(durations):,})")
        q = durations.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"    p10={q[0.10]:.1f}  p25={q[0.25]:.1f}  median={q[0.50]:.1f}  p75={q[0.75]:.1f}  p90={q[0.90]:.1f} years")
        print(f"    mean={durations.mean():.1f}  max={durations.max():.1f} years")

    # Age of closed businesses (end date → reference date gap)
    if "_dba_end_past" in df.columns:
        past_ends = df[df["_dba_end_past"]]["_dba_end_dt"].dropna()
        ages = pd.Series([_age_days(d) for d in past_ends]).dropna()
        if len(ages) > 0:
            print(f"\n  Staleness of end date (days since dba_end_date, closed records, n={len(ages):,})")
            q = ages.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            print(f"    p10={q[0.10]:.0f}  p25={q[0.25]:.0f}  median={q[0.50]:.0f}  p75={q[0.75]:.0f}  p90={q[0.90]:.0f} days")


def industry_distribution(df: pd.DataFrame) -> None:
    section("INDUSTRY (NAICS) DISTRIBUTION")
    n = len(df)

    # NAICS code description
    filled = df["naic_code_description"].notna() & (df["naic_code_description"] != "")
    print(f"\n  naic_code_description filled: {pct(filled.sum(), n)}")
    if filled.sum() > 0:
        top = df["naic_code_description"].value_counts().head(20)
        print(f"\n  Top 20 NAICS descriptions:")
        for desc, cnt in top.items():
            print(f"    {cnt:>6,}  {str(desc)[:60]}")

    # LIC codes
    filled_lic = df["lic_code_description"].notna() & (df["lic_code_description"] != "")
    print(f"\n  lic_code_description filled: {pct(filled_lic.sum(), n)}")
    if filled_lic.sum() > 0:
        top_lic = df["lic_code_description"].value_counts().head(10)
        print(f"\n  Top 10 LIC descriptions:")
        for desc, cnt in top_lic.items():
            print(f"    {cnt:>6,}  {str(desc)[:60]}")

    # Parking / occupancy tax
    print(f"\n  parking_tax=True:              {pct(df['parking_tax'].sum(), n)}")
    print(f"  transient_occupancy_tax=True:  {pct(df['transient_occupancy_tax'].sum(), n)}")


def geography(df: pd.DataFrame) -> None:
    section("GEOGRAPHY")
    n = len(df)

    print(f"\n  has_geometry (lat/lon):  {pct(df['has_geometry'].sum(), n)}")

    for col in ["city", "state", "supervisor_district", "neighborhoods_analysis_boundaries",
                "business_corridor", "community_benefit_district"]:
        filled = df[col].notna() & (df[col] != "")
        if filled.sum() == 0:
            continue
        top = df[col].value_counts().head(10)
        print(f"\n  {col}  (filled={filled.sum():,}/{n:,})")
        for val, cnt in top.items():
            print(f"    {cnt:>6,}  ({100*cnt/n:4.1f}%)  {str(val)[:50]}")


def closure_vs_industry(df: pd.DataFrame) -> None:
    section("CLOSURE RATE BY INDUSTRY (top NAICS, Strategy A: dba_end_date past)")

    if "_dba_end_past" not in df.columns:
        print("  (run closure_signals first)")
        return

    naics = df["naic_code_description"].fillna("(none)")
    top_cats = naics.value_counts().head(15).index.tolist()
    sub = df[naics.isin(top_cats)].copy()
    sub["_naics"] = naics[sub.index]

    print(f"\n  {'NAICS description':<45} {'Total':>7} {'Closed':>7} {'Closed%':>8}")
    print("-" * 75)
    rates = []
    for cat in top_cats:
        mask = sub["_naics"] == cat
        total = mask.sum()
        closed = (sub[mask]["_dba_end_past"]).sum()
        rates.append((cat, total, closed, 100*closed/total if total else 0))
    for cat, total, closed, rate in sorted(rates, key=lambda x: -x[3]):
        print(f"  {str(cat):<45} {total:>7,} {closed:>7,} {rate:>7.1f}%")


def open_business_signals(df: pd.DataFrame) -> None:
    section("OPEN BUSINESS SIGNALS (records with no end date)")

    if "_no_end_date" not in df.columns:
        return

    open_df = df[df["_no_end_date"]]
    n_open = len(open_df)
    n = len(df)
    print(f"\n  Records with no dba_end_date (presumed open): {pct(n_open, n)}")

    # Among these, how many have location_end_date set?
    if "_loc_end_dt" in df.columns:
        loc_closed = (open_df["_loc_end_dt"].notna()).sum()
        print(f"  ...but with location_end_date set:           {pct(loc_closed, n_open)}")

    # Among these, how many are admin closed?
    if "_admin_closed" in df.columns:
        adm = open_df["_admin_closed"].sum()
        print(f"  ...but administratively_closed set:          {pct(adm, n_open)}")

    # Age of "open" businesses
    ages = open_df["_dba_start_dt"].dropna().apply(lambda d: _age_days(d) / 365.25)
    if len(ages) > 0:
        print(f"\n  Business age (years since dba_start_date, open records, n={len(ages):,})")
        q = ages.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"    p10={q[0.10]:.1f}  p25={q[0.25]:.1f}  median={q[0.50]:.1f}  p75={q[0.75]:.1f}  p90={q[0.90]:.1f} yrs")


def overture_match_feasibility(df: pd.DataFrame) -> None:
    section("OVERTURE MATCH FEASIBILITY")
    n = len(df)

    # What we'd use to match: address + business name
    print(f"\n  Fields available for matching to Overture:")
    for col in ["dba_name", "full_business_address", "city", "state", "business_zip"]:
        filled = df[col].notna() & (df[col] != "")
        print(f"    {col:<30} filled: {pct(filled.sum(), n)}")

    # Geo match feasibility
    print(f"\n  has_geometry (direct lat/lon match): {pct(df['has_geometry'].sum(), n)}")

    # SF-only records (could narrow to Overture US places)
    sf_mask = df["city"].str.strip().str.lower() == "san francisco"
    print(f"\n  Records where city='San Francisco': {pct(sf_mask.sum(), n)}")
    if sf_mask.sum() > 0:
        sf = df[sf_mask]
        if "_dba_end_past" in df.columns:
            closed_sf = sf["_dba_end_past"].sum()
            print(f"  SF closed (dba_end past):           {pct(closed_sf, sf_mask.sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sf_open_dataset_20260309.geojson")

    if not path.exists():
        sys.exit(f"File not found: {path}")

    out_path = path.parent / "sf_eda_results.txt"
    tee = _Tee(out_path)
    sys.stdout = tee

    try:
        df = load_geojson(path)
        df = closure_signals(df)   # returns df with _* parsed columns
        null_rates(df)
        date_distributions(df)
        industry_distribution(df)
        geography(df)
        closure_vs_industry(df)
        open_business_signals(df)
        overture_match_feasibility(df)

        print(f"\n\nDone. Results saved to {out_path}")
    finally:
        sys.stdout = tee._stdout
        tee.close()
