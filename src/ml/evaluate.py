"""
evaluate.py
-----------
Evaluate all models on the validation set and print a comparison table.

Models compared (closed class = positive):
    GBM          — GradientBoostingClassifier on raw features (run gbm.py first)
    MLP head     — encoder's own classification head
    MLP + NCM    — encoder embeddings → Nearest Class Mean
    MLP + SLDA   — encoder embeddings → Streaming LDA
    MLP + QDA    — encoder embeddings → Streaming QDA

Metrics (all for closed, label=0, as positive class):
    AUC-ROC, AUC-PR, F1, Precision, Recall @ optimal F1 threshold

Run:
    python src/split.py data/project_c_samples.json   # once
    python src/train.py                                # trains MLP + fits NCM/SLDA
    python src/gbm.py                                  # trains GBM baseline
    python src/evaluate.py                             # compare all models
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from encoder import PlaceDataset, PlaceEncoder, load_splits
from ncm import NearestClassMean
from qda import StreamingQDA
from slda import StreamingLDA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_f1_threshold(y_true_binary: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return (best_f1, best_threshold).

    Args:
        y_true_binary: 1 where sample is the positive class, 0 otherwise
        scores:        predicted probability of being the positive class
    """
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, 199):
        preds = (scores >= thr).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true_binary, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def evaluate_scores(
    name: str,
    y_val: np.ndarray,
    p_closed: np.ndarray,
) -> dict:
    """Compute all metrics for a model given P(closed) scores.

    Args:
        name:     model name (for display)
        y_val:    ground-truth labels (0=closed, 1=open)
        p_closed: predicted probability of being closed
    """
    # Binary indicator: 1 where sample is closed
    is_closed = (y_val == 0).astype(int)

    auc_roc = roc_auc_score(is_closed, p_closed)
    auc_pr  = average_precision_score(is_closed, p_closed)
    f1, thr = best_f1_threshold(is_closed, p_closed)

    preds = (p_closed >= thr).astype(int)
    prec  = precision_score(is_closed, preds, zero_division=0)
    rec   = recall_score(is_closed, preds, zero_division=0)

    return {
        "name": name,
        "auc_roc": auc_roc,
        "auc_pr":  auc_pr,
        "f1":      f1,
        "prec":    prec,
        "rec":     rec,
        "thr":     thr,
    }


def print_results(results: list[dict]) -> None:
    header = f"{'Model':<18}  {'AUC-ROC':>7}  {'AUC-PR':>6}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}  {'Thr':>5}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['name']:<18}  {r['auc_roc']:>7.4f}  {r['auc_pr']:>6.4f}  "
            f"{r['f1']:>6.4f}  {r['prec']:>6.4f}  {r['rec']:>6.4f}  {r['thr']:>5.2f}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    splits_dir: str | Path = "splits",
    models_dir: str | Path = "models",
    evals_dir: str | Path = "models/evals",
) -> None:
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    evals_dir = Path(evals_dir)
    evals_dir.mkdir(exist_ok=True)

    for p in [splits_dir, models_dir / "encoder.pt", models_dir / "ncm.pkl", models_dir / "slda.pkl", models_dir / "qda.pkl"]:
        if not Path(p).exists():
            sys.exit(f"Missing: {p}  — run split.py then train.py first")

    # --- Load data ---
    _X_train, X_val, _y_train, y_val = load_splits(splits_dir)
    print(f"Val set: {len(y_val)} samples  |  closed={(y_val==0).sum()}  open={(y_val==1).sum()}")

    results = []
    val_info = {
        "val_total":  int(len(y_val)),
        "val_closed": int((y_val == 0).sum()),
        "val_open":   int((y_val == 1).sum()),
    }

    # --- Load OHE transformer (needed for GBM / XGBoost inference) ---
    ohe_path = models_dir / "ohe.pkl"
    if ohe_path.exists():
        with open(ohe_path, "rb") as f:
            ohe = pickle.load(f)
        from gbm import apply_ohe
        X_val_ohe = apply_ohe(X_val, ohe)
    else:
        ohe = None
        X_val_ohe = None

    # --- 1a. GBM (load pre-trained from models/gbm.pkl) ---
    gbm_path = models_dir / "gbm.pkl"
    if gbm_path.exists() and X_val_ohe is not None:
        with open(gbm_path, "rb") as f:
            gbm = pickle.load(f)
        p_closed_gbm = 1.0 - gbm.predict_proba(X_val_ohe)[:, 1]
        results.append(evaluate_scores("GBM", y_val, p_closed_gbm))
    else:
        print("GBM not found — run: python src/gbm.py   (skipping)")

    # --- 1b. XGBoost (load pre-trained from models/xgb.pkl) ---
    xgb_path = models_dir / "xgb.pkl"
    if xgb_path.exists() and X_val_ohe is not None:
        with open(xgb_path, "rb") as f:
            xgb = pickle.load(f)
        p_closed_xgb = 1.0 - xgb.predict_proba(X_val_ohe)[:, 1]
        results.append(evaluate_scores("XGBoost", y_val, p_closed_xgb))
    else:
        print("XGBoost not found — run: python src/gbm.py   (skipping)")

    # --- 2. Load encoder ---
    with open(models_dir / "encoder_config.json") as f:
        ecfg = json.load(f)
    device = torch.device("cpu")
    encoder = PlaceEncoder(
        cat_vocab_size=ecfg["cat_vocab_size"],
        n_numeric=ecfg["n_numeric"],
        cfg=ecfg["model"],
    ).to(device)
    encoder.load_state_dict(torch.load(models_dir / "encoder.pt", map_location=device))
    encoder.eval()

    # --- 3. MLP head ---
    ds_val  = PlaceDataset(X_val, y_val)
    from torch.utils.data import DataLoader
    loader  = DataLoader(ds_val, batch_size=256, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            logits = encoder(x_num, x_cat)
            all_logits.append(torch.softmax(logits, dim=1).numpy())
    probs_mlp = np.concatenate(all_logits, axis=0)   # (N, 2)
    p_closed_mlp = probs_mlp[:, 0]                   # P(class=0 / closed)
    results.append(evaluate_scores("MLP head", y_val, p_closed_mlp))

    # --- 4. NCM on embeddings ---
    Z_val = np.load(models_dir / "embeddings_val.npy")
    with open(models_dir / "ncm.pkl", "rb") as f:
        ncm: NearestClassMean = pickle.load(f)
    probs_ncm    = ncm.predict_proba(Z_val)          # (N, 2), columns = [class0, class1]
    p_closed_ncm = probs_ncm[:, 0]
    results.append(evaluate_scores("MLP + NCM", y_val, p_closed_ncm))

    # --- 5. SLDA on embeddings ---
    with open(models_dir / "slda.pkl", "rb") as f:
        slda: StreamingLDA = pickle.load(f)
    probs_slda    = slda.predict_proba(Z_val)
    p_closed_slda = probs_slda[:, 0]
    results.append(evaluate_scores("MLP + SLDA", y_val, p_closed_slda))

    # --- 6. QDA on embeddings ---
    with open(models_dir / "qda.pkl", "rb") as f:
        qda: StreamingQDA = pickle.load(f)
    probs_qda    = qda.predict_proba(Z_val)
    p_closed_qda = probs_qda[:, 0]
    results.append(evaluate_scores("MLP + QDA", y_val, p_closed_qda))

    # --- Print table ---
    print_results(results)

    # --- Targets summary ---
    targets = {"AUC-ROC": 0.80, "AUC-PR": 0.50, "F1": 0.40}
    print("Target check (MLP + SLDA):")
    slda_r = next(r for r in results if r["name"] == "MLP + SLDA")
    for label, (key, target) in zip(
        ["AUC-ROC", "AUC-PR", "F1"],
        [("auc_roc", 0.80), ("auc_pr", 0.50), ("f1", 0.40)]
    ):
        val = slda_r[key]
        status = "PASS" if val >= target else "MISS"
        print(f"  {label}: {val:.4f}  (target {target})  [{status}]")

    # --- Save results ---
    out = evals_dir / f'eval_results_{int(time())}.json'
    with open(out, "w") as f:
        json.dump({"val_set": val_info, "models": results}, f, indent=2)
    print(f"\nResults saved to {out}\n")

if __name__ == "__main__":
    main()
