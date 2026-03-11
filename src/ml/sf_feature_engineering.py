"""
sf_feature_engineering.py
--------------------------
Build a training dataset from the SF Registered Businesses DuckDB.

Extracts features directly from SF business columns, appends the 21 spatial
KNN neighborhood features, and returns (X, y) ready for model training.

Usage:
    from sf_feature_engineering import load_dataset

    X, y = load_dataset("data/sf_registered_businesses.ddb")
    # X is a DataFrame of 30 features (9 SF-native + 21 spatial KNN)
    # y is a numpy int64 array (1=open, 0=closed)
    # Uncertain records are excluded automatically.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from spatial_knn_features import SpatialKNNFeatures

_LABEL_SQL = """
    CASE
        WHEN dba_end_date IS NOT NULL AND dba_end_date <= CURRENT_TIMESTAMP THEN 'closed'
        WHEN administratively_closed = 'true' THEN 'closed'
        WHEN dba_end_date IS NULL
             AND location_end_date IS NULL
             AND administratively_closed IS NULL
             AND data_as_of >= CURRENT_TIMESTAMP - INTERVAL '6 months' THEN 'open'
        ELSE NULL
    END
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    db_path: str | Path = "data/sf_registered_businesses.ddb",
    spatial_cache: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load SF registered businesses and return (X, y) for model training.

    Loads only confidently labeled records (open or closed). Uncertain
    records — those with no end date but stale data_as_of — are excluded.

    Args:
        db_path: Path to the DuckDB file produced by fetch-sf-registered-businesses.py.
        spatial_cache: If True (default), save/load spatial KNN features to/from a
            parquet file alongside the DB (<db_path>.spatial_features.parquet).
            On the first run this is computed (takes ~30 min). Every subsequent run
            loads from cache in seconds. Set to False to force recomputation.

    Returns:
        X: DataFrame of 30 features (9 SF-native + 21 spatial KNN). naic_code and
           lic are string columns (encode before passing to a model).
        y: int64 numpy array of labels, shape (N,). 1=open, 0=closed.
    """
    db_path = Path(db_path)
    # Spatial cache lives next to the DB file so it's easy to find and delete.
    # Named after the DB so different DB files get separate caches.
    _cache_path = db_path.with_suffix(".spatial_features.parquet")

    conn = duckdb.connect(str(db_path), read_only=True)

    raw = conn.sql(f"""
        SELECT
            lat,
            lon,
            ({_LABEL_SQL})              AS status,
            naic_code,
            lic,
            parking_tax,
            transient_occupancy_tax,
            business_zip,
            supervisor_district,
            neighborhoods_analysis_boundaries,
            business_corridor,
            community_benefit_district,
            location_start_date,
            dba_start_date
            -- data_as_of removed (feature audit 2026-03-10): data_staleness_days was a
            -- data leak — it is directly used in the open label derivation
            -- (data_as_of >= 6 months ago), so the model would learn the label rule,
            -- not a genuine signal.
        FROM sf_registered_businesses
        WHERE lat IS NOT NULL
          AND lon IS NOT NULL
          AND ({_LABEL_SQL}) IS NOT NULL
    """).df()
    conn.close()

    # --- Direct SF column features ---
    X_sf = _extract_sf_features(raw)

    # --- Spatial KNN features (self-excluded: each business excluded from its own neighborhood) ---
    # CACHING: the BallTree transform loop over ~318k points takes ~30 minutes.
    # We cache the output as a parquet file next to the DB. On subsequent runs the
    # cache is loaded in seconds. A row-count check guards against stale caches
    # (e.g. if the DB was re-fetched with more records, the cache is regenerated).
    X_spatial = _load_or_compute_spatial(raw, _cache_path, spatial_cache)

    X = pd.concat([X_sf, X_spatial], axis=1)
    y = (raw["status"] == "open").astype(np.int64).values

    return X, y


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_or_compute_spatial(
    raw: pd.DataFrame,
    cache_path: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """Return spatial KNN features for all rows in raw, using a disk cache.

    Cache key: the parquet file at cache_path. A row-count check detects stale
    caches (DB re-fetched with more/fewer records) and triggers recomputation.

    Args:
        raw:        The full labeled query result from load_dataset().
        cache_path: Where to read/write the parquet cache.
        use_cache:  If False, always recompute (and overwrite any existing cache).

    Returns:
        DataFrame of 21 spatial KNN features, same index as raw.
    """
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(raw):
            print(f"Loaded spatial features from cache: {cache_path}")
            return cached.reset_index(drop=True)
        # Row count mismatch — DB was updated; fall through to recompute.
        print(
            f"Spatial cache row count mismatch "
            f"({len(cached):,} cached vs {len(raw):,} current) — recomputing ..."
        )

    print(f"Building BallTree over {len(raw):,} reference businesses ...")
    featurizer = SpatialKNNFeatures.from_dataframe(raw)

    query_df = raw[["lat", "lon", "naic_code"]].copy()
    print("Computing spatial neighborhood features (this takes ~30 min on first run) ...")
    X_spatial = featurizer.transform(query_df, exclude_self=True)

    if use_cache:
        X_spatial.to_parquet(cache_path, index=False)
        print(f"Saved spatial features cache → {cache_path}")
        print("  Future runs will load from cache in seconds.")

    return X_spatial


def _extract_sf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 9 direct SF column features from the raw query result.

    NOTE: data_staleness_days was removed (feature audit 2026-03-10). It used
    data_as_of, which is part of the open label derivation
    (data_as_of >= 6 months ago), creating a direct data leak.
    """
    now = datetime.now(timezone.utc)
    now_ns = np.datetime64(now, "ns")

    # Business age: prefer location_start_date, fall back to dba_start_date
    start_dates = df["location_start_date"].combine_first(df["dba_start_date"])
    start_dates_utc = pd.to_datetime(start_dates, utc=True, errors="coerce")
    business_age_days = (
        (now_ns - start_dates_utc.values) / np.timedelta64(1, "D")
    ).astype(float)

    return pd.DataFrame({
        # Business lifecycle
        "business_age_days":        business_age_days,

        # Industry classification (categorical — encode before model)
        "naic_code":                 df["naic_code"].values,

        # License type (categorical — encode before model)
        "lic":                       df["lic"].values,

        # Tax flags (business type proxies)
        "parking_tax":               df["parking_tax"].fillna(False).astype(int).values,
        "transient_occupancy_tax":   df["transient_occupancy_tax"].fillna(False).astype(int).values,

        # Geography (categorical — encode before model)
        "business_zip":              df["business_zip"].values,
        "supervisor_district":       df["supervisor_district"].values,
        "neighborhood":              df["neighborhoods_analysis_boundaries"].values,

        # Presence flags for sparse geographic boundaries
        "in_business_corridor":      df["business_corridor"].notna().astype(int).values,
        "in_community_benefit_district": df["community_benefit_district"].notna().astype(int).values,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/ml/sf_feature_engineering.py [db_path]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sf_registered_businesses.ddb")
    print(f"Loading dataset from {db} ...")
    X, y = load_dataset(db)

    print(f"\nShape: {X.shape}")
    print(f"Labels: {y.sum():,} open, {(y == 0).sum():,} closed")
    print(f"\nFeature dtypes:\n{X.dtypes}")
    print(f"\nNumerical summary:\n{X.select_dtypes('number').describe()}")
    print(f"\nMissing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
