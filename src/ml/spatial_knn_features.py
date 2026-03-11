"""
spatial_knn_features.py
-----------------------
Spatial neighborhood health features derived from the SF Registered Businesses dataset.

For every query point (lat/lon), computes neighborhood features at three radii
(100m, 250m, 500m) using a BallTree over all SF businesses as the reference pool.

BallTree is a spatial index structure that partitions points into nested
hyperspheres ("balls"), allowing fast radius lookups without computing distances
to every point. Here it lets us find all SF businesses within X meters of any
query location in O(log n) time instead of O(n), which matters at 350k+ points.

Produces 21 features total:
  - 6 base features × 3 radii = 18 features
  - 1 category-specific closure rate × 3 radii = 3 features (NaN if no naic_code)

Dropped as low-signal noise (feature audit 2026-03-10):
  admin_closed_rate  — largely a subset of closure_rate; redundant signal
  business_age_std   — std of neighbor ages adds noise without predictive value
  parking_tax_rate   — % of neighbors paying parking tax is weakly predictive
  tot_tax_rate       — hotel/short-term-rental tax rate irrelevant for most businesses

Usage:
    from spatial_knn_features import SpatialKNNFeatures

    # Build once from the SF DuckDB database
    featurizer = SpatialKNNFeatures.from_duckdb("data/sf_registered_businesses.ddb")

    # query_df must have columns: lat, lon
    # Optional columns: naic_code (enables category closure rate)
    #                   uniqueid  (enables self-exclusion when query is in reference pool)
    spatial_df = featurizer.transform(query_df)

    # Join onto your main feature matrix
    X = pd.concat([X, spatial_df], axis=1)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000
RADII_M = [100, 250, 500]

# Age threshold in days for "new business" signal
_NEW_BUSINESS_THRESHOLD_DAYS = 365

# Self-exclusion threshold: neighbors within this distance (meters) are
# considered the same point and excluded (handles query point in reference pool)
_SELF_EXCLUSION_RADIUS_M = 1.0

# Label derivation mirrors fetch-sf-registered-businesses.py exactly
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

class SpatialKNNFeatures:
    """BallTree-backed spatial neighborhood featurizer.

    Build once, call transform() for any set of query points.
    """

    def __init__(self, ref_df: pd.DataFrame) -> None:
        """Construct from a reference DataFrame.

        Args:
            ref_df: DataFrame with columns:
                lat, lon, status, naic_code,
                location_start_date, dba_start_date
                (administratively_closed, parking_tax, transient_occupancy_tax
                 were removed in feature audit 2026-03-10 — no longer accessed)
        """
        ref_df = ref_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

        # BallTree requires radians for haversine metric
        coords_rad = np.deg2rad(ref_df[["lat", "lon"]].values.astype(float))
        self._tree = BallTree(coords_rad, metric="haversine")

        # Pre-compute all reference data as flat numpy arrays, indexed identically
        # to BallTree rows. Doing this once at construction time means transform()
        # never touches the original DataFrame — every neighbor lookup is a fast
        # numpy integer-index into these arrays.
        self._is_closed = (ref_df["status"] == "closed").values.astype(float)
        # NOTE: admin_closed_rate removed (feature audit 2026-03-10) — was largely
        # a subset of closure_rate. self._is_admin_closed no longer computed.

        # PERF: business age is computed with fully vectorized numpy datetime arithmetic
        # instead of a row-by-row Python apply() loop. On 350k rows this is ~100x faster.
        # Prefer location_start_date (when the business opened at this spot);
        # fall back to dba_start_date (when the account was created) if not set.
        now = datetime.now(timezone.utc)
        start_dates = ref_df["location_start_date"].combine_first(ref_df["dba_start_date"])
        start_dates_utc = pd.to_datetime(start_dates, utc=True, errors="coerce")
        now_ns = np.datetime64(now, "ns")
        self._business_age_days = (
            (now_ns - start_dates_utc.values) / np.timedelta64(1, "D")
        ).astype(float)
        self._is_new = (self._business_age_days < _NEW_BUSINESS_THRESHOLD_DAYS).astype(float)

        self._naic_codes = ref_df["naic_code"].values
        # PERF: pre-compute the "NAICS code is known" boolean mask once here so
        # _compute_features() never calls pd.notna() inside the hot loop.
        self._naic_known = pd.notna(ref_df["naic_code"]).values

        # Pre-computed mask: True where status is confidently known (open or closed).
        # Used in _compute_features() to exclude uncertain records from closure_rate,
        # preventing records that quietly stopped operating from diluting the signal
        # toward open (since uncertain records have NULL status, not 'closed').
        self._status_known = ref_df["status"].notna().values

        # NOTE: parking_tax_rate and tot_tax_rate removed (feature audit 2026-03-10) —
        # low-signal features. self._parking_tax and self._tot_tax no longer computed.

        # Pre-compute max radius in radians — BallTree uses radians for haversine,
        # so we convert once here rather than on every transform() call.
        self._max_radius_rad = max(RADII_M) / EARTH_RADIUS_M

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_duckdb(cls, db_path: str | Path) -> "SpatialKNNFeatures":
        """Load the SF reference pool from an existing DuckDB database.

        The database must contain the sf_registered_businesses table
        (created by fetch-sf-registered-businesses.py).
        """
        conn = duckdb.connect(str(db_path), read_only=True)
        conn.sql("INSTALL spatial; LOAD spatial;")
        ref_df = conn.sql(f"""
            SELECT
                ST_Y(geom)                AS lat,
                ST_X(geom)                AS lon,
                ({_LABEL_SQL})            AS status,
                naic_code,
                location_start_date,
                dba_start_date
                -- administratively_closed, parking_tax, transient_occupancy_tax removed
                -- (feature audit 2026-03-10 — those reference arrays no longer built)
            FROM sf_registered_businesses
            WHERE geom IS NOT NULL
        """).df()
        conn.close()
        return cls(ref_df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SpatialKNNFeatures":
        """Build directly from a pre-loaded DataFrame."""
        return cls(df)

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def transform(
        self,
        query_df: pd.DataFrame,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Compute spatial neighborhood features for all query points.

        Args:
            query_df: DataFrame with columns lat, lon.
                      Optional column naic_code enables same-category closure rate.
                      Optional column uniqueid is ignored (self-exclusion is
                      distance-based — any neighbor within 1m is excluded).
            exclude_self: If True, exclude neighbors within 1m of the query point.
                          Set to False only if query points are not in the reference pool.

        Returns:
            DataFrame of spatial features with the same index as query_df.
            All features are NaN for query points with missing lat/lon or 0 neighbors.
        """
        has_naic = "naic_code" in query_df.columns

        query_coords = query_df[["lat", "lon"]].values.astype(float)
        valid_mask = ~np.isnan(query_coords).any(axis=1)
        valid_idxs = np.where(valid_mask)[0]

        feature_rows: list[Optional[dict]] = [None] * len(query_df)

        if valid_idxs.size > 0:
            query_coords_rad = np.deg2rad(query_coords[valid_mask])

            # Query the BallTree once at the largest radius (500m) and reuse the
            # results for the smaller radii (100m, 250m) — avoids 3 separate tree
            # traversals per query point.
            # sort_results=True returns neighbors in ascending distance order, which
            # lets _compute_features() use np.searchsorted (binary search, O(log k))
            # to find the cutoff index for each radius instead of scanning all
            # neighbors with a boolean mask on every radius (O(k) × 3 per point).
            all_idxs, all_dists_rad = self._tree.query_radius(
                query_coords_rad,
                r=self._max_radius_rad,
                return_distance=True,
                sort_results=True,
            )

            for qi, orig_idx in enumerate(valid_idxs):
                idxs = all_idxs[qi]
                dists_m = all_dists_rad[qi] * EARTH_RADIUS_M

                if exclude_self:
                    keep = dists_m >= _SELF_EXCLUSION_RADIUS_M
                    idxs = idxs[keep]
                    dists_m = dists_m[keep]

                query_naic = query_df.iloc[orig_idx]["naic_code"] if has_naic else None
                feature_rows[orig_idx] = self._compute_features(idxs, dists_m, query_naic)

        # Fill invalid rows with all-NaN feature dicts
        nan_row = _nan_row(has_naic)
        rows = [r if r is not None else nan_row for r in feature_rows]

        return pd.DataFrame(rows, index=query_df.index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        all_idxs: np.ndarray,
        all_dists_m: np.ndarray,
        query_naic: Optional[str],
    ) -> dict:
        feat: dict = {}

        for r in RADII_M:
            # PERF: binary search on sorted distances to find how many neighbors
            # fall within radius r. O(log k) vs scanning all k neighbors with a mask.
            n = int(np.searchsorted(all_dists_m, r, side="right"))
            suffix = f"_{r}m"

            if n == 0:
                feat.update(_nan_features_for_radius(r, query_naic is not None))
                continue

            idxs = all_idxs[:n]

            # Subset reference arrays to neighbors
            is_closed    = self._is_closed[idxs]
            status_known = self._status_known[idxs]
            ages         = self._business_age_days[idxs]
            is_new       = self._is_new[idxs]
            naic_codes   = self._naic_codes[idxs]
            naic_known   = self._naic_known[idxs]

            # --- Closure signals ---
            # FIX: only average over neighbors whose status is confidently known.
            # Without this, uncertain records (NULL status) count as "not closed"
            # in the mean, understating the true closure rate in the neighborhood.
            n_known = status_known.sum()
            feat[f"closure_rate{suffix}"] = (
                float(is_closed[status_known].mean()) if n_known > 0 else np.nan
            )
            # admin_closed_rate REMOVED (feature audit 2026-03-10): redundant subset of closure_rate.

            # --- Vitality signals ---
            feat[f"n_businesses{suffix}"]      = float(n)
            feat[f"new_business_rate{suffix}"] = is_new.mean()

            valid_ages = ages[~np.isnan(ages)]
            feat[f"median_business_age{suffix}"] = (
                float(np.median(valid_ages)) if len(valid_ages) > 0 else np.nan
            )
            # business_age_std REMOVED (feature audit 2026-03-10): adds noise without signal.

            # --- Diversity signal ---
            # PERF: uses pre-computed _naic_known boolean mask (set in __init__)
            # instead of calling pd.notna() or checking for None inside this loop.
            feat[f"naics_diversity{suffix}"] = (
                float(len(set(naic_codes[naic_known]))) if naic_known.any() else np.nan
            )

            # parking_tax_rate and tot_tax_rate REMOVED (feature audit 2026-03-10):
            # % of neighbors paying parking/hotel tax is weakly predictive of closure.

            # --- Category-specific closure rate (only if query has a NAICS code) ---
            if query_naic is not None:
                same_cat = (naic_codes == query_naic) & status_known
                feat[f"same_category_closure_rate{suffix}"] = (
                    float(is_closed[same_cat].mean()) if same_cat.any() else np.nan
                )
            else:
                feat[f"same_category_closure_rate{suffix}"] = np.nan

        return feat


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _nan_features_for_radius(r: int, include_category: bool = True) -> dict:
    # 6 base features kept (feature audit 2026-03-10 removed admin_closed_rate,
    # business_age_std, parking_tax_rate, tot_tax_rate as low-signal noise).
    suffix = f"_{r}m"
    d = {
        f"closure_rate{suffix}":               np.nan,
        f"n_businesses{suffix}":               np.nan,
        f"new_business_rate{suffix}":          np.nan,
        f"median_business_age{suffix}":        np.nan,
        f"naics_diversity{suffix}":            np.nan,
        f"same_category_closure_rate{suffix}": np.nan,
    }
    return d


def _nan_row(has_naic: bool = True) -> dict:
    row = {}
    for r in RADII_M:
        row.update(_nan_features_for_radius(r, include_category=has_naic))
    return row


def _to_utc(d) -> datetime:
    """Convert a pandas Timestamp or datetime to UTC-aware datetime."""
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/ml/spatial_knn_features.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sf_registered_businesses.ddb")
    print(f"Building featurizer from {db} ...")
    featurizer = SpatialKNNFeatures.from_duckdb(db)
    print(f"Reference pool size: {featurizer._tree.data.shape[0]:,} points")

    # Sample 5 SF businesses as query points using the same DB
    conn = duckdb.connect(str(db), read_only=True)
    conn.sql("INSTALL spatial; LOAD spatial;")
    sample = conn.sql("""
        SELECT
            ST_Y(geom) AS lat,
            ST_X(geom) AS lon,
            naic_code,
            dba_name
        FROM sf_registered_businesses
        WHERE geom IS NOT NULL
        LIMIT 5
    """).df()
    conn.close()

    print(f"\nQuery sample:\n{sample[['dba_name', 'lat', 'lon', 'naic_code']]}\n")

    result = featurizer.transform(sample, exclude_self=True)
    print(f"Output shape: {result.shape}")
    print(f"\nFeatures:\n{result.T}")
