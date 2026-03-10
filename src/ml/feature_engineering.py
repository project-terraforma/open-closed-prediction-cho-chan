"""
feature_engineering.py
----------------------
Extract a flat feature dict from a single Overture place JSON record.

Usage:
    from feature_engineering import extract_features, load_dataset

    X, y = load_dataset("data/project_c_samples.json")
    # X is a DataFrame; primary_category is a string column (encode before model)
    # y is a numpy int64 array (0=closed, 1=open)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reference date: Feb 2026 Overture release (used for Microsoft update-age)
_REFERENCE_DATE = datetime(2026, 2, 18, tzinfo=timezone.utc)

_META = "meta"
_MICROSOFT = "microsoft"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(record: dict[str, Any]) -> dict[str, Any]:
    """Return a flat feature dict from one Overture place JSON record.

    Categorical features (primary_category) are returned as strings.
    Encode them with LabelEncoder or OrdinalEncoder before passing to a model.

    Args:
        record: Parsed JSON dict for a single place.

    Returns:
        Dict mapping feature name -> scalar (int, float, or str).
    """
    sources: list[dict] = record.get("sources") or []

    # --- Source-level features ---
    src_datasets = [s.get("dataset", "").lower() for s in sources]
    src_confs = [s["confidence"] for s in sources if s.get("confidence") is not None]

    source_count = len(sources)
    has_meta = int(_META in src_datasets)
    has_microsoft = int(_MICROSOFT in src_datasets)

    max_src_conf = max(src_confs) if src_confs else 0.0
    min_src_conf = min(src_confs) if src_confs else 0.0
    mean_src_conf = float(np.mean(src_confs)) if src_confs else 0.0
    confidence_spread = max_src_conf - min_src_conf

    # Microsoft update_time is a real staleness signal (meta is always batch date)
    msft_update_age_days = _msft_update_age(sources)
    # Fresh Microsoft data (updated within the last year before release)
    msft_fresh = int(0.0 <= msft_update_age_days <= 365.0)

    # All-source update staleness (meta excluded — always batch date, not real freshness)
    all_src_ages = _all_source_ages(sources)
    n_sources_with_update_time = len(all_src_ages)
    min_update_age_days = min(all_src_ages) if all_src_ages else -1.0
    max_update_age_days = max(all_src_ages) if all_src_ages else -1.0

    # Interaction: confidence × corroboration (captures "high-conf AND multi-source")
    conf_x_source = mean_src_conf * source_count
    # Single-source flag (regardless of which source — less corroboration)
    has_single_source = int(source_count == 1)

    # --- Completeness features ---
    websites = record.get("websites") or []
    phones = record.get("phones") or []
    socials = record.get("socials") or []
    brand = record.get("brand")
    # has_email dropped: all-zero in labeled dataset (EDA confirmed, d=0.000)

    has_website = int(bool(websites))
    has_phone = int(bool(phones))
    has_socials = int(bool(socials))
    has_brand = int(brand is not None)
    website_count = len(websites)
    phone_count = len(phones)

    optional_flags = [has_website, has_phone, has_socials, has_brand]
    completeness_score = sum(optional_flags) / len(optional_flags)

    # Single-source Meta-only places: lower quality / more likely stale
    has_only_meta = int(source_count == 1 and has_meta == 1)

    # --- Category features ---
    cats = record.get("categories") or {}
    primary_category: str = cats.get("primary") or "unknown"
    alternates: list = cats.get("alternate") or []
    has_alternate_categories = int(bool(alternates))
    alternate_category_count = len(alternates)

    # --- Top-level confidence ---
    confidence = float(record.get("confidence") or 0.0)

    # --- Address features ---
    # has_address is constant (all records have addresses) so only completeness is kept
    addresses = record.get("addresses") or []
    address_completeness = _address_completeness(addresses)

    return {
        # source
        "source_count": source_count,
        "has_meta": has_meta,
        "has_microsoft": has_microsoft,
        "has_only_meta": has_only_meta,
        "max_source_confidence": max_src_conf,
        "min_source_confidence": min_src_conf,
        "mean_source_confidence": mean_src_conf,
        "confidence_spread": confidence_spread,
        "msft_update_age_days": msft_update_age_days,
        "msft_fresh": msft_fresh,
        "n_sources_with_update_time": n_sources_with_update_time,
        "min_update_age_days": min_update_age_days,
        "max_update_age_days": max_update_age_days,
        "conf_x_source": conf_x_source,
        "has_single_source": has_single_source,
        # completeness
        "has_website": has_website,
        "has_phone": has_phone,
        "has_socials": has_socials,
        "has_brand": has_brand,
        "website_count": website_count,
        "phone_count": phone_count,
        "completeness_score": completeness_score,
        # category
        "primary_category": primary_category,
        "has_alternate_categories": has_alternate_categories,
        "alternate_category_count": alternate_category_count,
        # top-level confidence
        "confidence": confidence,
        # address
        "address_completeness": address_completeness,
    }


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a JSONL file of Overture place records and extract features.

    Args:
        path: Path to a JSONL file where each line is one place record
              with an "open" field (1=open, 0=closed).

    Returns:
        X: DataFrame of features (N x 20). primary_category is a string column.
        y: int64 numpy array of labels, shape (N,).
    """
    rows: list[dict] = []
    labels: list[int] = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(extract_features(rec))
            labels.append(int(rec["open"]))

    X = pd.DataFrame(rows)
    y = np.array(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _msft_update_age(sources: list[dict]) -> float:
    """Days between the most recent Microsoft source update_time and the Feb 2026 release.

    Returns -1.0 if no Microsoft source is present (so the model can learn
    'no Microsoft source' as a distinct signal).
    """
    times: list[datetime] = []
    for s in sources:
        if s.get("dataset", "").lower() != _MICROSOFT:
            continue
        raw = s.get("update_time")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            times.append(dt)
        except ValueError:
            pass

    if not times:
        return -1.0

    return float((_REFERENCE_DATE - max(times)).days)


def _all_source_ages(sources: list[dict]) -> list[float]:
    """Ages in days for all non-meta sources that have a real update_time.

    Meta is excluded because its update_time is always the batch date (not a
    real freshness signal).  Returns a list of floats; empty if none qualify.
    """
    ages = []
    for s in sources:
        if s.get("dataset", "").lower() == _META:
            continue
        raw = s.get("update_time")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ages.append(float((_REFERENCE_DATE - dt).days))
        except ValueError:
            pass
    return ages


def _address_completeness(addresses: list[dict]) -> float:
    """Fraction of key address subfields populated in the first address entry."""
    if not addresses:
        return 0.0
    addr = addresses[0]
    subfields = ["freeform", "locality", "postcode", "region"]
    filled = sum(1 for f in subfields if addr.get(f))
    return filled / len(subfields)


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python src/feature_engineering.py [path]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/project_c_samples.json")
    print(f"Loading {path} ...")
    X, y = load_dataset(path)

    print(f"\nShape: {X.shape}")
    print(f"Labels: {y.sum()} open, {(y == 0).sum()} closed")
    print(f"\nFeature dtypes:\n{X.dtypes}")
    print(f"\nNumerical summary:\n{X.describe()}")
    print(f"\nTop primary_category values:\n{X['primary_category'].value_counts().head(10)}")
    print(f"\nMissing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
