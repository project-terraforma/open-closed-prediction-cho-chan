"""
yelp_feature_engineering.py
---------------------------
Convert Yelp Academic Dataset businesses to Overture-compatible training records.

Reads yelp-academic-dataset-business.json and writes data/yelp_features.jsonl
using the same pseudo-Overture schema that feature_engineering.extract_features()
processes.  The output JSONL can be passed directly to split.py --augment.

Feature mapping decisions
--------------------------
confidence / source_confidence  ← log-normalised review_count (see note below)
has_phone / has_website         ← 0  (fields absent from Yelp Academic schema)
has_socials / has_brand         ← 0  (same)
completeness_score              ← 0  for all Yelp records (consequence of above)
msft_update_age_days            ← -1.0  (no Microsoft source)
primary_category                ← first Yelp category, lowercased with spaces→_
                                   (will be OOV in the Overture category encoder)
address_completeness            ← fraction of address/city/postal_code/state
                                   that are non-empty

Note on confidence proxy
--------------------------
review_count is log-normalised:
    conf = min( log1p(review_count) / log1p(500), 1.0 )
This gives:
    0 reviews  → 0.00
    1 review   → 0.14
   10 reviews  → 0.38
   50 reviews  → 0.64
  500 reviews  → 1.00
Directional alignment: more reviews → more established → higher Overture confidence.
The proxy is imperfect; setting completeness_score=0 for all Yelp records is the
main known limitation (the model will see Yelp records as "no optional fields filled").

Run:
    python src/yelp_feature_engineering.py
    python src/yelp_feature_engineering.py --pct 15 --max-closed 10000

Outputs (data/):
    yelp_features.jsonl  — JSONL readable by feature_engineering.load_dataset()
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT        = Path(__file__).parent.parent.parent
YELP_PATH   = ROOT / "data" / "yelp-academic-dataset-business.json"
OUTPUT_PATH = ROOT / "data" / "yelp_features.jsonl"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _confidence_proxy(review_count: int | None) -> float:
    """Log-normalised review_count mapped to [0, 1].

    Args:
        review_count: number of Yelp reviews (None treated as 0).

    Returns:
        Float in [0.0, 1.0].
    """
    rc = max(int(review_count or 0), 0)
    return min(math.log1p(rc) / math.log1p(500), 1.0)


def yelp_to_overture(biz: dict) -> dict:
    """Convert one Yelp business record to a pseudo-Overture JSON record.

    The output schema matches what feature_engineering.extract_features() expects.

    Args:
        biz: Parsed Yelp business JSON dict.

    Returns:
        Pseudo-Overture record dict with an "open" label field.
    """
    conf = _confidence_proxy(biz.get("review_count"))

    # Category: Yelp gives comma-separated string "Restaurants, Italian, Food"
    cats_raw: str = biz.get("categories") or ""
    cat_list = [c.strip().lower().replace(" ", "_") for c in cats_raw.split(",") if c.strip()]
    primary_cat = cat_list[0] if cat_list else "unknown"
    alternates  = cat_list[1:] if len(cat_list) > 1 else []

    # Address: map Yelp flat fields to Overture address subfields
    address_entry = {
        "freeform":  biz.get("address") or "",
        "locality":  biz.get("city") or "",
        "postcode":  biz.get("postal_code") or "",
        "region":    biz.get("state") or "",
    }

    return {
        # Minimal Overture-compatible record
        "id":         f"yelp_{biz.get('business_id', '')}",
        # Single Yelp source — confidence derived from review_count
        "sources": [{"dataset": "yelp", "confidence": conf}],
        "names":      {"primary": biz.get("name") or ""},
        "categories": {"primary": primary_cat, "alternate": alternates},
        "confidence": conf,
        # Contact fields — not present in Yelp Academic schema
        "websites":   [],    # has_website = 0
        "socials":    [],    # has_socials  = 0
        "emails":     [],
        "phones":     [],    # has_phone    = 0
        "brand":      None,  # has_brand    = 0
        "addresses":  [address_entry],
        "bbox":       None,
        # Label: Yelp is_open=1 → open=1 (same encoding as Overture)
        "open": int(biz.get("is_open") or 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(closed_pct: int = 15, max_closed: int | None = None) -> None:
    if not YELP_PATH.exists():
        sys.exit(f"Yelp dataset not found: {YELP_PATH}")

    # ------------------------------------------------------------------
    # 1. Load all records, split into closed / open lists
    # ------------------------------------------------------------------
    print(f"Reading {YELP_PATH.name} ...")
    closed_records: list[dict] = []
    open_records:   list[dict] = []

    with open(YELP_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = yelp_to_overture(biz)
            if rec["open"] == 0:
                closed_records.append(rec)
            else:
                open_records.append(rec)

    n_closed_raw = len(closed_records)
    n_open_raw   = len(open_records)
    print(f"  {n_closed_raw + n_open_raw:,} total  |  {n_closed_raw:,} closed  |  {n_open_raw:,} open")

    # ------------------------------------------------------------------
    # 2. Cap closed records if requested
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    if max_closed is not None and len(closed_records) > max_closed:
        idx = rng.choice(len(closed_records), size=max_closed, replace=False)
        closed_records = [closed_records[i] for i in idx]
        print(f"  Capped closed to {len(closed_records):,}  (--max-closed {max_closed})")

    # ------------------------------------------------------------------
    # 3. Sample open records to hit target closed percentage
    # ------------------------------------------------------------------
    n_closed = len(closed_records)
    # total = n_closed / (pct/100)  →  n_open = total - n_closed
    n_open_target = round(n_closed * (100 - closed_pct) / closed_pct)
    n_open_target = min(n_open_target, n_open_raw)

    idx = rng.choice(len(open_records), size=n_open_target, replace=False)
    open_sample = [open_records[i] for i in idx]

    total = n_closed + n_open_target
    print(f"\nDataset plan:")
    print(f"  Closed  : {n_closed:,}")
    print(f"  Open    : {n_open_target:,}")
    print(f"  Total   : {total:,}  ({n_closed/total*100:.1f}% closed)")

    # ------------------------------------------------------------------
    # 4. Write output JSONL
    # ------------------------------------------------------------------
    print(f"\nWriting {OUTPUT_PATH.name} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for rec in closed_records:
            out.write(json.dumps(rec) + "\n")
        for rec in open_sample:
            out.write(json.dumps(rec) + "\n")

    print(f"  {total:,} records written to {OUTPUT_PATH}")
    print(f"\nNext:")
    print(f"  python src/split.py data/project_c_samples.json --augment data/yelp_features.jsonl")
    print(f"  python src/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pct", type=int, default=15,
        help="Target closed percentage in output (default: 15)",
    )
    parser.add_argument(
        "--max-closed", type=int, default=None, dest="max_closed",
        help="Cap the number of closed records (default: all)",
    )
    args = parser.parse_args()
    main(closed_pct=args.pct, max_closed=args.max_closed)
