"""
cost_table.py
-------------
Operational cost comparison across all trained models.

Measures and prints a side-by-side table covering:
  - Accuracy      (AUC-ROC, F1 — from saved eval results or recomputed)
  - Inference     (latency for 685 val samples and per-sample µs)
  - Scale         (estimated minutes to score 100 M places on a single CPU)
  - Model size    (KB on disk for all required artifacts)
  - Update cost   (NCM/SLDA incremental update vs XGBoost full retrain)
  - CL support    (whether monthly releases can be absorbed without retraining)

Run:
    python src/train.py && python src/gbm.py    (must have been run first)
    python src/cost_table.py

Outputs (models/):
    cost_table.json  — all measured values for each model
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from encoder import PlaceDataset, PlaceEncoder, load_splits, N_NUMERIC
from evaluate import best_f1_threshold, evaluate_scores
from ncm import NearestClassMean
from slda import StreamingLDA


# Number of timing repetitions (take median to reduce noise)
N_REPS = 7
SCALE_TARGET = 100_000_000   # places at full Overture scale


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

def artifact_size_kb(paths: list[Path]) -> float:
    """Total size in KB across a list of artifact paths (missing files = 0)."""
    return sum(p.stat().st_size for p in paths if p.exists()) / 1024.0


# ---------------------------------------------------------------------------
# Inference timing helpers
# ---------------------------------------------------------------------------

def time_mlp_inference(
    encoder: PlaceEncoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_reps: int = N_REPS,
) -> float:
    """Return median ms to run the full MLP head over X_val across n_reps trials.

    Args:
        encoder: trained PlaceEncoder in eval mode.
        X_val:   feature matrix (N, 20).
        y_val:   label array (N,).
        n_reps:  number of timing repetitions.

    Returns:
        Median inference time in milliseconds.
    """
    ds = PlaceDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=len(y_val), shuffle=False)
    x_num, x_cat, _ = next(iter(loader))

    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.softmax(encoder(x_num, x_cat), dim=1).numpy()
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times_ms))


def time_embed_then_classify(
    encoder: PlaceEncoder,
    classifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_reps: int = N_REPS,
) -> float:
    """Return median ms for MLP encode + NCM/SLDA classify over X_val.

    Args:
        encoder:    trained PlaceEncoder in eval mode.
        classifier: fitted NearestClassMean or StreamingLDA instance.
        X_val:      feature matrix (N, 20).
        y_val:      label array (N,).
        n_reps:     number of timing repetitions.

    Returns:
        Median inference time in milliseconds.
    """
    ds = PlaceDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=len(y_val), shuffle=False)
    x_num, x_cat, _ = next(iter(loader))

    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            Z = encoder.encode(x_num, x_cat).numpy()
        _ = classifier.predict_proba(Z)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times_ms))


def time_ohe_then_predict(model, ohe, X_val: np.ndarray, n_reps: int = N_REPS) -> float:
    """Return median ms for OHE transformation + GBM/XGBoost predict_proba.

    Args:
        model:  fitted sklearn-compatible classifier.
        ohe:    fitted OneHotEncoder for primary_category column.
        X_val:  feature matrix (N, 20).
        n_reps: number of timing repetitions.

    Returns:
        Median inference time in milliseconds.
    """
    from gbm import apply_ohe
    times_ms = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        X_ohe = apply_ohe(X_val, ohe)
        _ = model.predict_proba(X_ohe)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times_ms))


# ---------------------------------------------------------------------------
# Update timing helpers
# ---------------------------------------------------------------------------

def time_cl_update(
    encoder: PlaceEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    clf_class,
    release_0_frac: float = 0.70,
    n_reps: int = N_REPS,
    seed: int = 0,
) -> tuple[float, float]:
    """Simulate a new-release incremental update and return (update_ms, n_new).

    Splits training data into Release 0 (fit) and Release 1 (update).
    Returns the median update() time in ms across n_reps and the size of Release 1.

    Args:
        encoder:         trained PlaceEncoder (for embedding extraction).
        X_train:         training feature matrix.
        y_train:         training labels.
        clf_class:       NearestClassMean or StreamingLDA.
        release_0_frac:  fraction of training data assigned to Release 0.
        n_reps:          timing repetitions.
        seed:            random seed for the stratified split.

    Returns:
        (median_update_ms, n_release_1_samples)
    """
    # Extract all training embeddings (encoder frozen)
    ds = PlaceDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_Z: list[np.ndarray] = []
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            all_Z.append(encoder.encode(x_num, x_cat).numpy())
    Z_train = np.concatenate(all_Z, axis=0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - release_0_frac,
                                 random_state=seed)
    idx_r0, idx_r1 = next(sss.split(Z_train, y_train))
    Z0, y0 = Z_train[idx_r0], y_train[idx_r0]
    Z1, y1 = Z_train[idx_r1], y_train[idx_r1]

    times_ms = []
    for _ in range(n_reps):
        clf = clf_class()
        clf.fit(Z0, y0)
        t0 = time.perf_counter()
        clf.update(Z1, y1)
        times_ms.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times_ms)), len(Z1)


def time_xgb_retrain(
    xgb_path: Path,
    ohe_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_reps: int = max(N_REPS - 4, 3),
) -> float:
    """Return median ms for a full XGBoost retrain on X_train.

    Args:
        xgb_path:  path to a saved XGBClassifier (used only to copy hyperparams).
        ohe_path:  path to the saved OneHotEncoder.
        X_train:   raw training feature matrix.
        y_train:   training labels.
        n_reps:    timing repetitions (fewer because retraining is slow).

    Returns:
        Median retrain time in milliseconds.
    """
    from gbm import apply_ohe
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return float("nan")

    with open(xgb_path, "rb") as f:
        xgb_template = pickle.load(f)
    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)

    X_train_ohe = apply_ohe(X_train, ohe)
    sample_w = compute_sample_weight("balanced", y_train)
    params = xgb_template.get_params()

    times_ms = []
    for _ in range(n_reps):
        xgb = XGBClassifier(**params)
        t0 = time.perf_counter()
        xgb.fit(X_train_ohe, y_train, sample_weight=sample_w)
        times_ms.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times_ms))


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def auc_from_scores(p_closed: np.ndarray, y_val: np.ndarray) -> float:
    """Compute AUC-ROC with closed (label=0) as positive class."""
    is_closed = (y_val == 0).astype(int)
    return float(roc_auc_score(is_closed, p_closed))


def f1_from_scores(p_closed: np.ndarray, y_val: np.ndarray) -> float:
    """Compute best F1 with closed as positive class."""
    is_closed = (y_val == 0).astype(int)
    f1, _ = best_f1_threshold(is_closed, p_closed)
    return float(f1)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(entries: list[dict]) -> None:
    """Print the formatted cost comparison table.

    Args:
        entries: list of per-model dicts (see main() for required keys).
    """
    col_w = 16
    models = [e["name"] for e in entries]
    header_row = f"  {'Metric':<32}" + "".join(f"  {m:>{col_w}}" for m in models)
    sep = "  " + "-" * (len(header_row) - 2)

    def row(label: str, key: str, fmt: str = ".3f", suffix: str = "") -> str:
        vals = []
        for e in entries:
            v = e.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(f"{'—':>{col_w}}")
            elif isinstance(v, bool):
                vals.append(f"{'Yes' if v else 'No':>{col_w}}")
            elif isinstance(v, str):
                vals.append(f"{v:>{col_w}}")
            else:
                vals.append(f"{v:{fmt}}{suffix}".rjust(col_w))
        return f"  {label:<32}" + "".join(f"  {v}" for v in vals)

    print(f"\n{'='*82}")
    print(f"  Cost and Accuracy Comparison")
    print(f"{'='*82}")
    print(header_row)
    print(sep)
    print("  [Accuracy]")
    print(row("  AUC-ROC",              "auc_roc"))
    print(row("  F1 (closed, optimal)", "f1"))
    print(sep)
    print("  [Inference — val set (685 samples)]")
    print(row("  Latency (ms)",         "infer_ms", ".1f"))
    print(row("  Per-sample (µs)",      "us_per_sample", ".1f"))
    print(row("  100M places (min est)","scale_min", ".1f"))
    print(sep)
    print("  [Artifacts on disk]")
    print(row("  Model size (KB)",      "model_kb", ".1f"))
    print(sep)
    print("  [Monthly update (new Overture release)]")
    print(row("  Update method",        "update_method"))
    print(row("  Update time (ms)",     "update_ms", ".1f"))
    print(row("  Needs old data",       "needs_old_data"))
    print(row("  Incremental update",   "incremental"))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    splits_dir: str | Path = "splits",
    models_dir: str | Path = "models",
) -> None:
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)

    for p in [splits_dir, models_dir / "encoder.pt"]:
        if not Path(p).exists():
            sys.exit(f"Missing: {p}  — run split.py then train.py first")

    # --- Load splits ---
    X_train, X_val, y_train, y_val = load_splits(splits_dir)
    n_val = len(y_val)
    print(f"Val set: {n_val} samples  |  closed={(y_val==0).sum()}  open={(y_val==1).sum()}")
    print(f"Running {N_REPS} timing reps per model ...\n")

    # --- Load encoder ---
    with open(models_dir / "encoder_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cpu")
    encoder = PlaceEncoder(cat_vocab_size=cfg["cat_vocab_size"]).to(device)
    encoder.load_state_dict(torch.load(models_dir / "encoder.pt", map_location=device,
                                       weights_only=True))
    encoder.eval()

    # Pre-extract val embeddings once (shared by NCM + SLDA)
    Z_val = np.load(models_dir / "embeddings_val.npy")

    entries: list[dict] = []

    # -----------------------------------------------------------------------
    # GBM
    # -----------------------------------------------------------------------
    gbm_path  = models_dir / "gbm.pkl"
    ohe_path  = models_dir / "ohe.pkl"
    if gbm_path.exists() and ohe_path.exists():
        print("Timing GBM ...")
        with open(gbm_path, "rb") as f:
            gbm = pickle.load(f)
        with open(ohe_path, "rb") as f:
            ohe = pickle.load(f)
        from gbm import apply_ohe
        X_val_ohe = apply_ohe(X_val, ohe)
        p_closed_gbm = 1.0 - gbm.predict_proba(X_val_ohe)[:, 1]

        infer_ms  = time_ohe_then_predict(gbm, ohe, X_val)
        model_kb  = artifact_size_kb([gbm_path, ohe_path])

        entries.append({
            "name":           "GBM",
            "auc_roc":        auc_from_scores(p_closed_gbm, y_val),
            "f1":             f1_from_scores(p_closed_gbm, y_val),
            "infer_ms":       infer_ms,
            "us_per_sample":  infer_ms * 1000 / n_val,
            "scale_min":      (infer_ms / 1000 / n_val * SCALE_TARGET) / 60,
            "model_kb":       model_kb,
            "update_method":  "full retrain",
            "update_ms":      float("nan"),
            "needs_old_data": True,
            "incremental":    False,
        })
    else:
        print("GBM not found — skipping (run python src/gbm.py)")

    # -----------------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------------
    xgb_path = models_dir / "xgb.pkl"
    if xgb_path.exists() and ohe_path.exists():
        print("Timing XGBoost inference + retrain ...")
        with open(xgb_path, "rb") as f:
            xgb = pickle.load(f)
        with open(ohe_path, "rb") as f:
            ohe_xgb = pickle.load(f)
        from gbm import apply_ohe
        X_val_ohe_xgb = apply_ohe(X_val, ohe_xgb)
        p_closed_xgb = 1.0 - xgb.predict_proba(X_val_ohe_xgb)[:, 1]

        infer_ms     = time_ohe_then_predict(xgb, ohe_xgb, X_val)
        retrain_ms   = time_xgb_retrain(xgb_path, ohe_path, X_train, y_train)
        model_kb     = artifact_size_kb([xgb_path, ohe_path])

        entries.append({
            "name":           "XGBoost",
            "auc_roc":        auc_from_scores(p_closed_xgb, y_val),
            "f1":             f1_from_scores(p_closed_xgb, y_val),
            "infer_ms":       infer_ms,
            "us_per_sample":  infer_ms * 1000 / n_val,
            "scale_min":      (infer_ms / 1000 / n_val * SCALE_TARGET) / 60,
            "model_kb":       model_kb,
            "update_method":  "full retrain",
            "update_ms":      retrain_ms,
            "needs_old_data": True,
            "incremental":    False,
        })
    else:
        print("XGBoost not found — skipping (run python src/gbm.py)")

    # -----------------------------------------------------------------------
    # MLP head
    # -----------------------------------------------------------------------
    print("Timing MLP head ...")
    ds_val = PlaceDataset(X_val, y_val)
    loader = DataLoader(ds_val, batch_size=len(y_val), shuffle=False)
    x_num, x_cat, _ = next(iter(loader))
    with torch.no_grad():
        probs_mlp = torch.softmax(encoder(x_num, x_cat), dim=1).numpy()
    p_closed_mlp = probs_mlp[:, 0]

    infer_ms = time_mlp_inference(encoder, X_val, y_val)
    model_kb = artifact_size_kb([models_dir / "encoder.pt",
                                  models_dir / "encoder_config.json"])

    entries.append({
        "name":           "MLP head",
        "auc_roc":        auc_from_scores(p_closed_mlp, y_val),
        "f1":             f1_from_scores(p_closed_mlp, y_val),
        "infer_ms":       infer_ms,
        "us_per_sample":  infer_ms * 1000 / n_val,
        "scale_min":      (infer_ms / 1000 / n_val * SCALE_TARGET) / 60,
        "model_kb":       model_kb,
        "update_method":  "full retrain",
        "update_ms":      float("nan"),
        "needs_old_data": True,
        "incremental":    False,
    })

    # -----------------------------------------------------------------------
    # MLP + NCM
    # -----------------------------------------------------------------------
    ncm_path = models_dir / "ncm.pkl"
    if ncm_path.exists():
        print("Timing MLP + NCM ...")
        with open(ncm_path, "rb") as f:
            ncm: NearestClassMean = pickle.load(f)
        p_closed_ncm = ncm.predict_proba(Z_val)[:, 0]
        infer_ms  = time_embed_then_classify(encoder, ncm, X_val, y_val)
        upd_ms, n_r1 = time_cl_update(encoder, X_train, y_train, NearestClassMean)
        model_kb = artifact_size_kb([models_dir / "encoder.pt",
                                      models_dir / "encoder_config.json",
                                      ncm_path])

        entries.append({
            "name":           "MLP + NCM",
            "auc_roc":        auc_from_scores(p_closed_ncm, y_val),
            "f1":             f1_from_scores(p_closed_ncm, y_val),
            "infer_ms":       infer_ms,
            "us_per_sample":  infer_ms * 1000 / n_val,
            "scale_min":      (infer_ms / 1000 / n_val * SCALE_TARGET) / 60,
            "model_kb":       model_kb,
            "update_method":  "incremental",
            "update_ms":      upd_ms,
            "needs_old_data": False,
            "incremental":    True,
        })

    # -----------------------------------------------------------------------
    # MLP + SLDA
    # -----------------------------------------------------------------------
    slda_path = models_dir / "slda.pkl"
    if slda_path.exists():
        print("Timing MLP + SLDA ...")
        with open(slda_path, "rb") as f:
            slda: StreamingLDA = pickle.load(f)
        p_closed_slda = slda.predict_proba(Z_val)[:, 0]
        infer_ms  = time_embed_then_classify(encoder, slda, X_val, y_val)
        upd_ms, n_r1 = time_cl_update(encoder, X_train, y_train, StreamingLDA)
        model_kb = artifact_size_kb([models_dir / "encoder.pt",
                                      models_dir / "encoder_config.json",
                                      slda_path])

        entries.append({
            "name":           "MLP + SLDA",
            "auc_roc":        auc_from_scores(p_closed_slda, y_val),
            "f1":             f1_from_scores(p_closed_slda, y_val),
            "infer_ms":       infer_ms,
            "us_per_sample":  infer_ms * 1000 / n_val,
            "scale_min":      (infer_ms / 1000 / n_val * SCALE_TARGET) / 60,
            "model_kb":       model_kb,
            "update_method":  "incremental",
            "update_ms":      upd_ms,
            "needs_old_data": False,
            "incremental":    True,
        })

    # --- Print table ---
    if entries:
        print_table(entries)

        # Speed ratios vs NCM
        ncm_entry = next((e for e in entries if e["name"] == "MLP + NCM"), None)
        if ncm_entry:
            print("  Update speed ratios (vs NCM update):")
            for e in entries:
                if not np.isnan(e["update_ms"]) and e["update_ms"] > 0:
                    ratio = e["update_ms"] / ncm_entry["update_ms"]
                    print(f"    {e['name']:<18}  {e['update_ms']:>8.1f} ms  "
                          f"({ratio:.0f}x vs NCM)")
            print()

    # --- Save ---
    out_path = models_dir / "cost_table.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2, default=lambda v: None if np.isnan(v) else v)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
