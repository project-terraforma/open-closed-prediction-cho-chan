"""
gbm.py
------
Train GBM and XGBoost classifiers on raw features with proper one-hot encoding
for the primary_category field (294 categories + 1 OOV = 295 values).

Treating primary_category as a numeric ordinal in tree models is wrong — a
category integer has no natural order. OHE fixes this so trees can split on
individual categories.

Run:
    python src/gbm.py

Outputs (models/):
    ohe.pkl              — fitted OneHotEncoder (apply to X_val before inference)
    gbm.pkl              — fitted GradientBoostingClassifier
    xgb.pkl              — fitted XGBClassifier
    gbm_importances.json — GBM feature importances (sorted)
    xgb_importances.json — XGBoost feature importances (sorted)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from encoder import load_splits


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_MAX_OHE_CATS = 300   # cap OHE at this many categories; beyond this, OHE adds
                      # mostly-zero columns that hurt GBM without adding signal.
                      # category_closure_rate (numeric) already captures per-category
                      # signal for the full vocabulary.


def _load_n_cats(splits_dir: Path) -> int:
    """Read actual category vocab size from the fitted LabelEncoder, capped at _MAX_OHE_CATS.

    High-cardinality OHE hurts GBM — category_closure_rate (numeric) carries
    the signal for rare categories so we don't need to OHE all of them.
    """
    enc_path = splits_dir / "category_encoder.pkl"
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            enc = pickle.load(f)
        return min(len(enc.classes_) + 1, _MAX_OHE_CATS)  # +1 OOV, capped
    return _MAX_OHE_CATS


def fit_ohe(X_train: np.ndarray, n_cats: int) -> OneHotEncoder:
    """Fit OHE on the category column of X_train."""
    cat_col = X_train.shape[1] - 1  # last column is always primary_category
    ohe = OneHotEncoder(
        categories=[list(range(n_cats))],
        sparse_output=False,
        handle_unknown="ignore",   # unseen category → all-zero row
    )
    ohe.fit(X_train[:, cat_col].astype(int).reshape(-1, 1))
    return ohe


def apply_ohe(X: np.ndarray, ohe: OneHotEncoder) -> np.ndarray:
    """Replace the integer category column with its one-hot encoding."""
    cat_col = X.shape[1] - 1  # last column is always primary_category
    X_num = X[:, :cat_col]
    X_cat = ohe.transform(X[:, cat_col].astype(int).reshape(-1, 1))
    return np.hstack([X_num, X_cat]).astype(np.float32)


def build_feature_names(numeric_names: list[str], n_cats: int) -> list[str]:
    cat_names = [f"cat_{i}" for i in range(n_cats)]
    return numeric_names + cat_names


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _print_importances(name: str, feature_names: list[str], importances: np.ndarray) -> None:
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n{name} — top-10 feature importances:")
    for feat, imp in ranked[:10]:
        bar = "#" * int(imp * 300)
        print(f"  {feat:<32}  {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_gbm(
    splits_dir: str | Path = "splits",
    out_dir: str | Path = "models",
) -> None:
    splits_dir = Path(splits_dir)
    out_dir    = Path(out_dir)

    if not splits_dir.exists():
        sys.exit("splits/ not found — run: python src/split.py data/project_c_samples.json")
    out_dir.mkdir(exist_ok=True)

    X_train, X_val, y_train, y_val = load_splits(splits_dir)
    with open(splits_dir / "feature_names.json") as f:
        numeric_names: list[str] = json.load(f)

    n_cats = _load_n_cats(splits_dir)
    n_numeric = X_train.shape[1] - 1
    print(f"Train: {len(y_train):,}  (closed={(y_train==0).sum()}  open={(y_train==1).sum()})")
    print(f"Val:   {len(y_val):,}  (closed={(y_val==0).sum()}  open={(y_val==1).sum()})")
    print(f"n_numeric: {n_numeric}  |  n_cats (OHE): {n_cats}")

    # --- OHE ---
    print(f"\nOne-hot encoding primary_category ({n_cats} values) ...")
    ohe = fit_ohe(X_train, n_cats)
    X_train_ohe = apply_ohe(X_train, ohe)
    X_val_ohe   = apply_ohe(X_val,   ohe)
    print(f"  Feature shape after OHE: {X_train_ohe.shape[1]} "
          f"({n_numeric} numeric + {n_cats} category dummies)")

    with open(out_dir / "ohe.pkl", "wb") as f:
        pickle.dump(ohe, f)

    feature_names = build_feature_names(numeric_names, n_cats)
    sample_w = compute_sample_weight("balanced", y_train)

    def val_auc(model) -> float:
        p_closed = 1.0 - model.predict_proba(X_val_ohe)[:, 1]
        return roc_auc_score((y_val == 0).astype(int), p_closed)

    # --- GBM ---
    print("\nFitting GBM ...")
    gbm = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbm.fit(X_train_ohe, y_train, sample_weight=sample_w)
    print(f"  Val AUC-ROC (closed): {val_auc(gbm):.4f}")
    _print_importances("GBM", feature_names, gbm.feature_importances_)

    with open(out_dir / "gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    imp_gbm = {n: float(v) for n, v in
               sorted(zip(feature_names, gbm.feature_importances_), key=lambda x: -x[1])}
    with open(out_dir / "gbm_importances.json", "w") as f:
        json.dump(imp_gbm, f, indent=2)

    # --- XGBoost ---
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("\nxgboost not installed — skipping (pip install xgboost)")
        return

    print("\nFitting XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train_ohe, y_train, sample_weight=sample_w)
    print(f"  Val AUC-ROC (closed): {val_auc(xgb):.4f}")
    _print_importances("XGBoost", feature_names, xgb.feature_importances_)

    with open(out_dir / "xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    imp_xgb = {n: float(v) for n, v in
               sorted(zip(feature_names, xgb.feature_importances_), key=lambda x: -x[1])}
    with open(out_dir / "xgb_importances.json", "w") as f:
        json.dump(imp_xgb, f, indent=2)

    print(f"\nSaved to {out_dir}/: ohe.pkl  gbm.pkl  xgb.pkl")


if __name__ == "__main__":
    train_gbm()
