"""
eda.py
------
Class-conditional EDA for the labeled Overture places dataset.

Run:
    python src/eda.py data/project_c_samples.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path so we can import feature_engineering
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import load_dataset

NUMERIC_FEATURES = [
    "source_count",
    "has_meta",
    "has_microsoft",
    "max_source_confidence",
    "min_source_confidence",
    "mean_source_confidence",
    "confidence_spread",
    "msft_update_age_days",
    "has_website",
    "has_phone",
    "has_socials",
    "has_brand",
    "website_count",
    "phone_count",
    "completeness_score",
    "has_alternate_categories",
    "alternate_category_count",
    "confidence",
    "address_completeness",
]


def _sep(char="─", width=72) -> str:
    return char * width


def run_eda(path: str | Path) -> None:
    print(f"\nLoading {path} ...")
    X, y = load_dataset(path)

    open_mask = y == 1
    closed_mask = y == 0
    X_open = X[open_mask]
    X_closed = X[closed_mask]

    n_total = len(y)
    n_open = open_mask.sum()
    n_closed = closed_mask.sum()

    # -------------------------------------------------------------------------
    # 1. Class distribution
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("1. CLASS DISTRIBUTION")
    print(_sep())
    print(f"  Total:  {n_total:>6,}")
    print(f"  Open:   {n_open:>6,}  ({100 * n_open / n_total:.1f}%)")
    print(f"  Closed: {n_closed:>6,}  ({100 * n_closed / n_total:.1f}%)")

    # -------------------------------------------------------------------------
    # 2. Class-conditional means — sorted by effect size (Cohen's d approx)
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("2. CLASS-CONDITIONAL FEATURE MEANS  (sorted by |closed - open|/pooled_std)")
    print(_sep())
    print(f"  {'Feature':<30}  {'Open':>10}  {'Closed':>10}  {'Diff':>8}  {'Effect':>8}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    rows = []
    for feat in NUMERIC_FEATURES:
        mu_open = X_open[feat].mean()
        mu_closed = X_closed[feat].mean()
        diff = mu_closed - mu_open
        pooled_std = X[feat].std()
        effect = abs(diff) / pooled_std if pooled_std > 0 else 0.0
        rows.append((feat, mu_open, mu_closed, diff, effect))

    rows.sort(key=lambda r: r[4], reverse=True)

    for feat, mu_o, mu_c, diff, effect in rows:
        flag = " <-- strong" if effect > 0.3 else (" <-- moderate" if effect > 0.1 else "")
        print(f"  {feat:<30}  {mu_o:>10.4f}  {mu_c:>10.4f}  {diff:>+8.4f}  {effect:>8.3f}{flag}")

    # -------------------------------------------------------------------------
    # 3. msft_update_age_days — exclude -1.0 (no Microsoft source)
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("3. msft_update_age_days  (excluding -1.0 = no Microsoft source)")
    print(_sep())
    for label, mask in [("Open", open_mask), ("Closed", closed_mask)]:
        ages = X[mask]["msft_update_age_days"]
        has_msft = ages[ages >= 0]
        pct = 100 * len(has_msft) / mask.sum()
        if len(has_msft) > 0:
            print(f"  {label:<8}  has_microsoft={pct:.1f}%  "
                  f"age_days: mean={has_msft.mean():.0f}  "
                  f"median={has_msft.median():.0f}  "
                  f"min={has_msft.min():.0f}  max={has_msft.max():.0f}")
        else:
            print(f"  {label:<8}  has_microsoft=0%")

    # -------------------------------------------------------------------------
    # 4. Top categories by closed rate
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("4. TOP CATEGORIES BY CLOSED RATE  (min 10 total)")
    print(_sep())
    cat_df = pd.DataFrame({"category": X["primary_category"], "closed": (y == 0).astype(int)})
    cat_stats = (
        cat_df.groupby("category")
        .agg(total=("closed", "count"), closed=("closed", "sum"))
        .query("total >= 10")
        .assign(closed_rate=lambda d: d["closed"] / d["total"])
        .sort_values("closed_rate", ascending=False)
        .head(15)
    )
    print(f"  {'Category':<35}  {'Total':>6}  {'Closed':>6}  {'Rate':>6}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cat, row in cat_stats.iterrows():
        print(f"  {cat:<35}  {int(row.total):>6}  {int(row.closed):>6}  {row.closed_rate:>6.1%}")

    # -------------------------------------------------------------------------
    # 5. Confidence value distribution by class
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("5. CONFIDENCE SCORE DISTRIBUTION BY CLASS")
    print(_sep())
    bins = [0.0, 0.4, 0.6, 0.77, 0.9, 0.95, 1.01]
    labels_bins = ["≤0.4", "0.4-0.6", "0.6-0.77", "0.77-0.9", "0.9-0.95", ">0.95"]
    for label, mask in [("Open", open_mask), ("Closed", closed_mask)]:
        conf = X[mask]["confidence"]
        counts, _ = np.histogram(conf, bins=bins)
        pcts = 100 * counts / counts.sum()
        parts = "  ".join(f"{b}:{p:.1f}%" for b, p in zip(labels_bins, pcts))
        print(f"  {label:<8}  {parts}")

    # -------------------------------------------------------------------------
    # 6. Source count distribution by class
    # -------------------------------------------------------------------------
    print(f"\n{_sep()}")
    print("6. SOURCE COUNT BY CLASS")
    print(_sep())
    for label, mask in [("Open", open_mask), ("Closed", closed_mask)]:
        vc = X[mask]["source_count"].value_counts().sort_index()
        parts = "  ".join(f"{k} src:{100*v/mask.sum():.1f}%" for k, v in vc.items())
        print(f"  {label:<8}  {parts}")

    print(f"\n{_sep()}\n")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/project_c_samples.json")
    run_eda(path)
