"""
sf_feature_engineering.py
--------------------------
Extract features from the SF registered business dataset (GeoJSON FeatureCollection).

Label (Strategy A): open=0 if dba_end_date is set and in the past, else open=1.

Exports NUMERIC_FEATURES, CATEGORICAL_FEATURES, CONF_FEATURES so that split.py
can import this module as an alternate feature engineering schema via --schema sf.

Usage:
    python src/ml/split.py data/sf_open_dataset_20260309.geojson --schema sf
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Fixed reference date — same as data_as_of in the dataset.
# Keeps training features reproducible; update for future dataset snapshots.
_REFERENCE_DATE = datetime(2026, 3, 9, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Feature lists — imported by split.py when --schema sf is used
# ---------------------------------------------------------------------------

CONF_FEATURES: list[str] = []   # no source-confidence equivalent in SF registry

NUMERIC_FEATURES = [
    "business_age_days",        # days since dba_start_date (how established)
    "location_age_days",        # days since location_start_date
    # DROPPED: has_location_end_date — location_end_date is set at same time as dba_end_date
    #          (closure event), making it a near-perfect label proxy (leakage)
    # DROPPED: location_end_age_days — same reason
    "has_naic_code",            # NAICS coverage: missing strongly correlated with closed (EDA: 73%)
    "parking_tax",              # industry signal (parking operators)
    "transient_occupancy_tax",  # industry signal (hotels / short-term rentals)
    "has_geometry",             # lat/lon available — completeness proxy
    "city_is_sf",               # San Francisco vs Bay Area / other CA
    # DROPPED: has_mailing_address — registry nulls mailing info on closure;
    #          192,097 null mailing vs 192,068 closed = near-perfect label proxy (leakage)
    "has_lic",                  # regulatory LIC code present
    "has_business_corridor",    # inside a named commercial corridor
    "has_neighborhood",         # SF neighborhood boundary identified (non-SF records: 0)
]

CATEGORICAL_FEATURES = ["naic_code_description"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(s: Any) -> datetime | None:
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _age_days(dt: datetime | None, missing: float = -1.0) -> float:
    """Days from dt to reference date. Returns `missing` if dt is None."""
    if dt is None:
        return missing
    return max(float((_REFERENCE_DATE - dt).days), 0.0)   # clamp future dates to 0


def _is_closed(props: dict) -> bool:
    """Strategy A: dba_end_date set and <= reference date → closed (label=0)."""
    dt = _parse_dt(props.get("dba_end_date"))
    return dt is not None and dt <= _REFERENCE_DATE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(record: dict[str, Any]) -> dict[str, Any]:
    """Return a flat feature dict from one SF business properties dict.

    Compatible with the same interface as feature_engineering.extract_features.

    Args:
        record: Properties dict from a GeoJSON feature, optionally with
                '_lat' / '_lon' injected by load_dataset for has_geometry.

    Returns:
        Dict mapping feature name → scalar (int, float, or str).
    """
    dba_start = _parse_dt(record.get("dba_start_date"))
    loc_start = _parse_dt(record.get("location_start_date"))

    business_age_days   = _age_days(dba_start, missing=0.0)
    location_age_days   = _age_days(loc_start, missing=0.0)

    has_naic_code       = int(bool(record.get("naic_code")))
    naic_desc: str      = record.get("naic_code_description") or "unknown"

    parking_tax         = int(bool(record.get("parking_tax")))
    transient_occupancy = int(bool(record.get("transient_occupancy_tax")))

    # _lat injected by load_dataset from GeoJSON geometry; or has_geometry already set
    has_geometry        = int(
        record.get("_lat") is not None
        or record.get("has_geometry") is True
    )

    city                = (record.get("city") or "").strip().lower()
    city_is_sf          = int(city == "san francisco")

    has_lic             = int(bool(record.get("lic")))
    has_business_corridor = int(bool(record.get("business_corridor")))
    has_neighborhood    = int(bool(record.get("neighborhoods_analysis_boundaries")))

    return {
        "business_age_days":       business_age_days,
        "location_age_days":       location_age_days,
        "has_naic_code":           has_naic_code,
        "parking_tax":             parking_tax,
        "transient_occupancy_tax": transient_occupancy,
        "has_geometry":            has_geometry,
        "city_is_sf":              city_is_sf,
        "has_lic":                 has_lic,
        "has_business_corridor":   has_business_corridor,
        "has_neighborhood":        has_neighborhood,
        # categorical
        "naic_code_description":   naic_desc,
    }


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load SF GeoJSON FeatureCollection and extract features.

    Labels (Strategy A): open=0 if dba_end_date in the past, else open=1.
    Records with no end date are presumed open.

    Args:
        path: Path to the GeoJSON file.

    Returns:
        X: DataFrame of features. naic_code_description is a string column.
        y: int64 numpy array of labels (0=closed, 1=open).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    labels: list[int] = []

    for feature in data["features"]:
        props: dict = dict(feature.get("properties") or {})

        # Inject lat/lon so extract_features can set has_geometry
        geom = feature.get("geometry")
        if geom and geom.get("type") == "Point":
            coords = geom.get("coordinates") or []
            if len(coords) >= 2:
                props["_lon"] = coords[0]
                props["_lat"] = coords[1]

        rows.append(extract_features(props))
        labels.append(0 if _is_closed(props) else 1)

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sf_open_dataset_20260309.geojson")
    print(f"Loading {path} ...")
    X, y = load_dataset(path)

    print(f"\nShape: {X.shape}")
    print(f"Labels: {y.sum():,} open  |  {(y==0).sum():,} closed  ({100*(y==0).mean():.1f}% closed)")
    print(f"\nFeature dtypes:\n{X.dtypes}")
    print(f"\nNumerical summary:\n{X.describe()}")
    print(f"\nTop naic_code_description:\n{X['naic_code_description'].value_counts().head(10)}")
    print(f"\nMissing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
