"""
category_eval.py
----------------
Per-category evaluation on the validation set.

For every category that has at least one closed example in the val set,
reports Recall and (where possible) F1.  Categories with fewer than 5 closed
examples are flagged as low-n — interpret with caution.

Uses the MLP head (best AUC) and MLP + SLDA (deployment candidate) so results
for both models are visible side-by-side.

Run:
    python src/train.py    (must have been run first)
    python src/category_eval.py

Outputs (models/):
    category_eval.json  — per-category metrics for both models
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from encoder import PlaceDataset, PlaceEncoder, load_splits, N_NUMERIC
from evaluate import best_f1_threshold
from ncm import NearestClassMean
from slda import StreamingLDA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_categories(X_val: np.ndarray, enc) -> list[str]:
    """Decode integer category column back to label strings.

    Args:
        X_val: feature matrix; column N_NUMERIC is the encoded category int.
        enc:   fitted LabelEncoder from split.py.

    Returns:
        List of category name strings, length == len(X_val).
    """
    cat_ints = X_val[:, N_NUMERIC].astype(int)
    classes = list(enc.classes_)
    return [classes[i] if i < len(classes) else "OOV" for i in cat_ints]


def get_mlp_scores(
    encoder: PlaceEncoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    """Return P(closed) from the MLP head for each validation sample.

    Args:
        encoder: trained PlaceEncoder in eval mode.
        X_val:   float32 feature matrix (N, 20).
        y_val:   int64 label array (N,).

    Returns:
        p_closed: float64 array (N,).
    """
    ds = PlaceDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            probs = torch.softmax(encoder(x_num, x_cat), dim=1).numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)[:, 0]


def per_category_metrics(
    categories: list[str],
    y_val: np.ndarray,
    p_closed: np.ndarray,
    threshold: float,
    model_name: str,
) -> list[dict[str, Any]]:
    """Compute per-category classification metrics.

    Args:
        categories: decoded category name per val sample.
        y_val:      ground-truth labels (0=closed, 1=open).
        p_closed:   predicted P(closed) per sample.
        threshold:  decision threshold for binary classification.
        model_name: model identifier (for output labeling).

    Returns:
        List of per-category dicts sorted by closed count descending.
    """
    is_closed = (y_val == 0).astype(int)
    y_pred = (p_closed >= threshold).astype(int)

    groups: dict[str, dict[str, list]] = defaultdict(
        lambda: {"true": [], "pred": [], "score": []}
    )
    for cat, true, pred, score in zip(categories, is_closed, y_pred, p_closed):
        groups[cat]["true"].append(true)
        groups[cat]["pred"].append(pred)
        groups[cat]["score"].append(float(score))

    rows = []
    for cat, g in groups.items():
        y_true = np.array(g["true"])
        y_pred_cat = np.array(g["pred"])
        n_closed = int(y_true.sum())
        n_open = int((y_true == 0).sum())
        if n_closed == 0:
            continue  # no closed examples — skip

        tp = int(((y_true == 1) & (y_pred_cat == 1)).sum())
        fn = int(((y_true == 1) & (y_pred_cat == 0)).sum())
        fp = int(((y_true == 0) & (y_pred_cat == 1)).sum())
        recall = recall_score(y_true, y_pred_cat, zero_division=0)
        prec   = precision_score(y_true, y_pred_cat, zero_division=0)
        f1     = f1_score(y_true, y_pred_cat, zero_division=0)
        rows.append({
            "category": cat,
            "model": model_name,
            "n_closed": n_closed,
            "n_open": n_open,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "recall": round(float(recall), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "reliable": n_closed >= 5,
        })

    rows.sort(key=lambda r: r["n_closed"], reverse=True)
    return rows


def print_category_table(
    rows_mlp: list[dict],
    rows_slda: list[dict],
) -> None:
    """Print side-by-side MLP head vs MLP+SLDA per-category recall.

    Args:
        rows_mlp:  per-category metrics for the MLP head model.
        rows_slda: per-category metrics for MLP + SLDA.
    """
    # Build lookup by category for SLDA
    slda_by_cat = {r["category"]: r for r in rows_slda}

    print(f"\n  {'Category':<32}  {'cl':>4}  {'op':>4}  "
          f"{'MLP Rec':>8}  {'MLP F1':>7}  "
          f"{'SLDA Rec':>9}  {'SLDA F1':>8}  Note")
    print("  " + "-" * 92)

    for r in rows_mlp:
        cat = r["category"]
        s = slda_by_cat.get(cat, {})
        slda_rec = s.get("recall", float("nan"))
        slda_f1  = s.get("f1",     float("nan"))
        note = "" if r["reliable"] else "  * low-n"

        slda_rec_str = f"{slda_rec:>9.3f}" if not np.isnan(slda_rec) else f"{'—':>9}"
        slda_f1_str  = f"{slda_f1:>8.3f}"  if not np.isnan(slda_f1)  else f"{'—':>8}"

        print(f"  {cat:<32}  {r['n_closed']:>4}  {r['n_open']:>4}  "
              f"{r['recall']:>8.3f}  {r['f1']:>7.3f}  "
              f"{slda_rec_str}  {slda_f1_str}{note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    splits_dir: str | Path = "splits",
    models_dir: str | Path = "models",
) -> None:
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)

    for p in [splits_dir, models_dir / "encoder.pt", models_dir / "slda.pkl"]:
        if not Path(p).exists():
            sys.exit(f"Missing: {p}  — run split.py then train.py first")

    # --- Load splits ---
    _X_train, X_val, _y_train, y_val = load_splits(splits_dir)
    with open(splits_dir / "category_encoder.pkl", "rb") as f:
        cat_enc = pickle.load(f)

    is_closed = (y_val == 0).astype(int)
    n_closed = int(is_closed.sum())
    n_open   = int((is_closed == 0).sum())
    print(f"Val set: {len(y_val)} samples  |  closed={n_closed}  open={n_open}")

    categories = decode_categories(X_val, cat_enc)
    n_cats_with_closed = sum(
        1 for cat in set(categories)
        if sum(1 for c, cl in zip(categories, is_closed) if c == cat and cl) > 0
    )
    print(f"Categories with ≥ 1 closed val example: {n_cats_with_closed}")
    print(f"(OKR threshold of 20 closed in val not reachable at this dataset size)")

    # --- Load MLP encoder ---
    with open(models_dir / "encoder_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cpu")
    encoder = PlaceEncoder(cat_vocab_size=cfg["cat_vocab_size"]).to(device)
    encoder.load_state_dict(torch.load(models_dir / "encoder.pt", map_location=device,
                                       weights_only=True))
    encoder.eval()

    # --- MLP head scores ---
    p_mlp = get_mlp_scores(encoder, X_val, y_val)
    _f1_mlp, thr_mlp = best_f1_threshold(is_closed, p_mlp)
    print(f"\nMLP head threshold: {thr_mlp:.3f}  F1={_f1_mlp:.4f}")

    # --- MLP + SLDA scores ---
    Z_val = np.load(models_dir / "embeddings_val.npy")
    with open(models_dir / "slda.pkl", "rb") as f:
        slda: StreamingLDA = pickle.load(f)
    p_slda = slda.predict_proba(Z_val)[:, 0]
    _f1_slda, thr_slda = best_f1_threshold(is_closed, p_slda)
    print(f"MLP + SLDA threshold: {thr_slda:.3f}  F1={_f1_slda:.4f}")

    # --- Per-category metrics ---
    rows_mlp  = per_category_metrics(categories, y_val, p_mlp,  thr_mlp,  "MLP head")
    rows_slda = per_category_metrics(categories, y_val, p_slda, thr_slda, "MLP+SLDA")

    # --- Print table ---
    print(f"\n{'='*62}")
    print(f"  Per-Category Evaluation  (sorted by # closed in val)")
    print(f"  * low-n = fewer than 5 closed examples, metrics unreliable")
    print(f"{'='*62}")
    print_category_table(rows_mlp, rows_slda)

    # --- Summary: categories where both models miss everything ---
    zero_recall = [r for r in rows_mlp if r["recall"] == 0.0 and r["n_closed"] >= 3]
    if zero_recall:
        print(f"\n  Categories with zero recall (≥3 closed): "
              f"{', '.join(r['category'] for r in zero_recall)}")

    perfect_recall = [r for r in rows_mlp if r["recall"] == 1.0 and r["reliable"]]
    if perfect_recall:
        print(f"  Categories with perfect recall (≥5 closed): "
              f"{', '.join(r['category'] for r in perfect_recall)}")

    # --- Save results ---
    out_path = models_dir / "category_eval.json"
    with open(out_path, "w") as f:
        json.dump({"mlp_head": rows_mlp, "mlp_slda": rows_slda}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
