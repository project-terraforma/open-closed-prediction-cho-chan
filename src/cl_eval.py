"""
cl_eval.py
----------
Continual learning evaluation: simulate a month-over-month Overture release update.

The training set is split into two simulated releases:
  Release 0 (70%): initial labeled data — NCM and SLDA are fit here
  Release 1 (30%): new labeled data from the next Overture release — update() only

Three things are demonstrated:
  1. Accuracy improves after update  (more data → better class prototypes)
  2. Incremental == full fit         (mathematical guarantee, mean diff ≈ 0)
  3. Update is orders of magnitude faster than XGBoost full retrain

The MLP encoder is NEVER retrained. Only the classifier state (class means /
covariance) is updated — requiring no access to Release 0 data.

Run:
    python src/train.py    (must have been run first)
    python src/gbm.py      (optional, for retrain timing)
    python src/cl_eval.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from ncm import NearestClassMean
from slda import StreamingLDA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auc_closed(classifier, Z: np.ndarray, y: np.ndarray) -> float:
    """AUC-ROC with closed (label=0) as the positive class."""
    probs = classifier.predict_proba(Z)
    p_closed = probs[:, 0]
    is_closed = (y == 0).astype(int)
    return float(roc_auc_score(is_closed, p_closed))


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    models_dir: str | Path = "models",
    splits_dir: str | Path = "splits",
    release_0_frac: float = 0.70,
    seed: int = 0,
) -> None:
    models_dir = Path(models_dir)
    splits_dir = Path(splits_dir)

    for p in [models_dir / "embeddings_train.npy",
              models_dir / "embeddings_val.npy",
              splits_dir / "y_train.npy",
              splits_dir / "y_val.npy"]:
        if not p.exists():
            sys.exit(f"Missing: {p}  — run train.py first")

    # --- Load pre-extracted embeddings (encoder is frozen, never touched again) ---
    Z_train = np.load(models_dir / "embeddings_train.npy")   # (2740, 32)
    Z_val   = np.load(models_dir / "embeddings_val.npy")     # (685,  32)
    y_train = np.load(splits_dir / "y_train.npy")
    y_val   = np.load(splits_dir / "y_val.npy")

    # --- Stratified split into Release 0 / Release 1 ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - release_0_frac, random_state=seed)
    idx_r0, idx_r1 = next(sss.split(Z_train, y_train))

    Z_0, y_0 = Z_train[idx_r0], y_train[idx_r0]
    Z_1, y_1 = Z_train[idx_r1], y_train[idx_r1]

    section("Simulated Overture Release Setup")
    print(f"  Encoder: FROZEN  (trained once, never retrained)")
    print(f"  Release 0 ({int(release_0_frac*100)}%): {len(y_0):>5} samples  "
          f"(closed={(y_0==0).sum():>3}  open={(y_0==1).sum():>4})")
    print(f"  Release 1 ({int((1-release_0_frac)*100)}%): {len(y_1):>5} samples  "
          f"(closed={(y_1==0).sum():>3}  open={(y_1==1).sum():>4})")
    print(f"  Val set:        {len(y_val):>5} samples  "
          f"(closed={(y_val==0).sum():>3}  open={(y_val==1).sum():>4})")

    # -----------------------------------------------------------------------
    # NCM
    # -----------------------------------------------------------------------
    section("NCM — Nearest Class Mean")

    # Initial fit on Release 0
    ncm_r0 = NearestClassMean()
    ncm_r0.fit(Z_0, y_0)
    auc_ncm_r0 = auc_closed(ncm_r0, Z_val, y_val)

    # Incremental update with Release 1 (no Release 0 data used)
    ncm_upd = NearestClassMean()
    ncm_upd.fit(Z_0, y_0)
    t0 = time.perf_counter()
    ncm_upd.update(Z_1, y_1)
    ncm_update_ms = (time.perf_counter() - t0) * 1000
    auc_ncm_upd = auc_closed(ncm_upd, Z_val, y_val)

    # Full fit on Release 0 + 1 (ground truth — should match incremental)
    ncm_full = NearestClassMean()
    ncm_full.fit(Z_train, y_train)
    auc_ncm_full = auc_closed(ncm_full, Z_val, y_val)

    # Correctness: incremental means == full fit means
    diff_ncm_0 = np.abs(ncm_upd.means_[0] - ncm_full.means_[0]).max()
    diff_ncm_1 = np.abs(ncm_upd.means_[1] - ncm_full.means_[1]).max()

    print(f"\n  {'State':<28}  AUC-ROC   Δ vs R0")
    print(f"  {'-'*50}")
    print(f"  {'R0 only (initial fit)':<28}  {auc_ncm_r0:.4f}")
    print(f"  {'After update (R0+R1)':<28}  {auc_ncm_upd:.4f}   "
          f"{auc_ncm_upd - auc_ncm_r0:+.4f}")
    print(f"  {'Full fit (R0+R1)':<28}  {auc_ncm_full:.4f}   "
          f"{auc_ncm_full - auc_ncm_r0:+.4f}")
    print(f"\n  Correctness (update == full fit):")
    print(f"    mean diff class0: {diff_ncm_0:.8f}  (should be ~0)")
    print(f"    mean diff class1: {diff_ncm_1:.8f}  (should be ~0)")
    print(f"\n  Update time: {ncm_update_ms:.3f} ms  ({len(Z_1)} new samples, 32-dim)")

    # -----------------------------------------------------------------------
    # SLDA
    # -----------------------------------------------------------------------
    section("SLDA — Streaming LDA")

    # Initial fit on Release 0
    slda_r0 = StreamingLDA()
    slda_r0.fit(Z_0, y_0)
    auc_slda_r0 = auc_closed(slda_r0, Z_val, y_val)

    # Incremental update with Release 1
    slda_upd = StreamingLDA()
    slda_upd.fit(Z_0, y_0)
    t0 = time.perf_counter()
    slda_upd.update(Z_1, y_1)
    slda_update_ms = (time.perf_counter() - t0) * 1000
    auc_slda_upd = auc_closed(slda_upd, Z_val, y_val)

    # Full fit on Release 0 + 1
    slda_full = StreamingLDA()
    slda_full.fit(Z_train, y_train)
    auc_slda_full = auc_closed(slda_full, Z_val, y_val)

    # Correctness: scatter matrices match
    diff_slda_mean0 = np.abs(slda_upd.means_[0] - slda_full.means_[0]).max()
    diff_slda_mean1 = np.abs(slda_upd.means_[1] - slda_full.means_[1]).max()
    diff_slda_S0    = np.abs(slda_upd.scatter_[0] - slda_full.scatter_[0]).max()
    diff_slda_S1    = np.abs(slda_upd.scatter_[1] - slda_full.scatter_[1]).max()

    print(f"\n  {'State':<28}  AUC-ROC   Δ vs R0")
    print(f"  {'-'*50}")
    print(f"  {'R0 only (initial fit)':<28}  {auc_slda_r0:.4f}")
    print(f"  {'After update (R0+R1)':<28}  {auc_slda_upd:.4f}   "
          f"{auc_slda_upd - auc_slda_r0:+.4f}")
    print(f"  {'Full fit (R0+R1)':<28}  {auc_slda_full:.4f}   "
          f"{auc_slda_full - auc_slda_r0:+.4f}")
    print(f"\n  Correctness (update == full fit):")
    print(f"    mean diff  class0: {diff_slda_mean0:.8f}  class1: {diff_slda_mean1:.8f}")
    print(f"    scatter diff class0: {diff_slda_S0:.6f}  class1: {diff_slda_S1:.6f}")
    print(f"\n  Update time: {slda_update_ms:.3f} ms  ({len(Z_1)} new samples, 32-dim)")

    # -----------------------------------------------------------------------
    # XGBoost retrain (for timing comparison)
    # -----------------------------------------------------------------------
    section("XGBoost — Full Retrain Timing (no incremental update possible)")

    xgb_path = models_dir / "xgb.pkl"
    ohe_path  = models_dir / "ohe.pkl"
    if xgb_path.exists() and ohe_path.exists():
        from encoder import load_splits
        from gbm import apply_ohe
        from sklearn.utils.class_weight import compute_sample_weight
        import copy

        X_train_raw, _, _, _ = load_splits(splits_dir)
        with open(ohe_path, "rb") as f:
            ohe = pickle.load(f)
        X_train_ohe = apply_ohe(X_train_raw, ohe)

        with open(xgb_path, "rb") as f:
            xgb_template = pickle.load(f)

        # Clone a fresh unfitted XGB with same params and time a full retrain
        from xgboost import XGBClassifier
        xgb_retrain = XGBClassifier(**xgb_template.get_params())
        sample_w = compute_sample_weight("balanced", y_train)

        t0 = time.perf_counter()
        xgb_retrain.fit(X_train_ohe, y_train, sample_weight=sample_w)
        xgb_retrain_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  XGBoost full retrain: {xgb_retrain_ms:,.1f} ms  "
              f"(300 trees, {X_train_ohe.shape[1]} features, {len(y_train)} samples)")
        print(f"\n  Speed comparison (adding {len(Z_1)} new samples):")
        print(f"    NCM   update():   {ncm_update_ms:>8.3f} ms")
        print(f"    SLDA  update():   {slda_update_ms:>8.3f} ms")
        print(f"    XGBoost retrain:  {xgb_retrain_ms:>8.1f} ms  "
              f"({xgb_retrain_ms / max(ncm_update_ms, 0.001):.0f}x slower than NCM)")
    else:
        print(f"\n  XGBoost not found — run gbm.py to include retrain timing")
        print(f"\n  Speed comparison (adding {len(Z_1)} new samples):")
        print(f"    NCM   update():   {ncm_update_ms:>8.3f} ms")
        print(f"    SLDA  update():   {slda_update_ms:>8.3f} ms")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    section("Summary")
    print(f"  {'Model':<30}  {'R0 AUC':>7}  {'Updated AUC':>11}  {'Full-fit AUC':>12}")
    print(f"  {'-'*65}")
    print(f"  {'NCM':<30}  {auc_ncm_r0:>7.4f}  {auc_ncm_upd:>11.4f}  {auc_ncm_full:>12.4f}")
    print(f"  {'SLDA':<30}  {auc_slda_r0:>7.4f}  {auc_slda_upd:>11.4f}  {auc_slda_full:>12.4f}")
    print()
    print("  Key result: incremental update == full fit (no old data needed).")
    print("  Encoder was never retrained.")


if __name__ == "__main__":
    main()
