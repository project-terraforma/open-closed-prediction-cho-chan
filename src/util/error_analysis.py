"""
error_analysis.py
-----------------
Structured error analysis: MLP head (best model) on the validation set.

For each val sample classifies as TP / FP / FN / TN at the optimal F1 threshold
and prints feature-level profiles to explain failure patterns:
  - False Negatives vs True Positives  (why some closures are missed)
  - False Positives vs True Negatives  (why some open places are flagged)
  - Category breakdown                 (which categories produce the most errors)

Run:
    python src/train.py    (must have been run first)
    python src/error_analysis.py

Outputs (models/):
    error_analysis.json  — per-sample error type, score, category
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from encoder import PlaceDataset, PlaceEncoder, load_splits, N_NUMERIC
from evaluate import best_f1_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_categories(X_val: np.ndarray, enc) -> list[str]:
    """Decode integer category column back to human-readable labels.

    Args:
        X_val: feature matrix; last column (index N_NUMERIC) is category int.
        enc:   fitted LabelEncoder from split.py.

    Returns:
        List of category strings, length == len(X_val).
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
        p_closed: float64 array of shape (N,).
    """
    ds = PlaceDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            probs = torch.softmax(encoder(x_num, x_cat), dim=1).numpy()
            all_probs.append(probs)
    probs_all = np.concatenate(all_probs, axis=0)   # (N, 2)
    return probs_all[:, 0]                           # P(closed = class 0)


def assign_error_types(
    is_closed: np.ndarray,
    y_pred: np.ndarray,
) -> list[str]:
    """Assign TP / FP / FN / TN to each sample.

    Args:
        is_closed: binary array, 1 = ground-truth closed.
        y_pred:    binary array, 1 = predicted closed.

    Returns:
        List of strings ("TP", "FP", "FN", "TN").
    """
    labels = []
    for true, pred in zip(is_closed, y_pred):
        if true and pred:
            labels.append("TP")
        elif not true and pred:
            labels.append("FP")
        elif true and not pred:
            labels.append("FN")
        else:
            labels.append("TN")
    return labels


def print_feature_profile(
    X_val: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    feat_names: list[str],
    label_a: str,
    label_b: str,
) -> None:
    """Print mean feature values for two groups, sorted by abs difference.

    Args:
        X_val:      feature matrix (numeric columns only, first N_NUMERIC).
        mask_a/b:   boolean masks selecting each group.
        feat_names: list of all feature names (only first N_NUMERIC are used).
        label_a/b:  display names for each group.
    """
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        print("  (one group is empty — skipping)")
        return

    numeric_names = feat_names[:N_NUMERIC]
    means_a = X_val[mask_a, :N_NUMERIC].mean(axis=0)
    means_b = X_val[mask_b, :N_NUMERIC].mean(axis=0)
    diffs = means_a - means_b

    ranked = sorted(
        zip(numeric_names, means_a, means_b, diffs),
        key=lambda x: abs(x[3]),
        reverse=True,
    )

    col_w = max(len(label_a), len(label_b), 12)
    header = f"  {'Feature':<28}  {label_a:>{col_w}}  {label_b:>{col_w}}  {'diff':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, ma, mb, d in ranked:
        marker = "  <--" if abs(d) > 0.10 else ""
        print(f"  {name:<28}  {ma:>{col_w}.3f}  {mb:>{col_w}.3f}  {d:>+8.3f}{marker}")


def print_category_breakdown(
    categories: list[str],
    error_types: list[str],
    is_closed: np.ndarray,
) -> None:
    """Print per-category recall and error counts.

    Only shows categories with at least one closed example in the val set.

    Args:
        categories:  decoded category string per val sample.
        error_types: TP/FP/FN/TN per val sample.
        is_closed:   binary 1 = ground-truth closed.
    """
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "n_closed": 0, "n_open": 0}
    )
    for cat, et, cl in zip(categories, error_types, is_closed):
        cat_stats[cat][et] += 1
        if cl:
            cat_stats[cat]["n_closed"] += 1
        else:
            cat_stats[cat]["n_open"] += 1

    with_closed = [
        (cat, s) for cat, s in cat_stats.items() if s["n_closed"] > 0
    ]
    with_closed.sort(key=lambda x: x[1]["n_closed"], reverse=True)

    print(f"\n  {'Category':<32}  {'cl':>4}  {'op':>4}  {'TP':>4}  {'FN':>4}  "
          f"{'FP':>4}  {'Recall':>7}  Note")
    print("  " + "-" * 80)
    for cat, s in with_closed:
        recall = s["TP"] / s["n_closed"]
        note = "  * low-n" if s["n_closed"] < 5 else ""
        print(f"  {cat:<32}  {s['n_closed']:>4}  {s['n_open']:>4}  "
              f"{s['TP']:>4}  {s['FN']:>4}  {s['FP']:>4}  {recall:>7.2f}{note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    splits_dir: str | Path = "splits",
    models_dir: str | Path = "models",
) -> None:
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)

    for p in [splits_dir, models_dir / "encoder.pt", models_dir / "encoder_config.json"]:
        if not Path(p).exists():
            sys.exit(f"Missing: {p}  — run split.py then train.py first")

    # --- Load splits and metadata ---
    _X_train, X_val, _y_train, y_val = load_splits(splits_dir)
    with open(splits_dir / "feature_names.json") as f:
        feat_names: list[str] = json.load(f)
    with open(splits_dir / "category_encoder.pkl", "rb") as f:
        cat_enc = pickle.load(f)

    is_closed = (y_val == 0).astype(int)
    n_closed = int(is_closed.sum())
    n_open = int((is_closed == 0).sum())
    print(f"Val set: {len(y_val)} samples  |  closed={n_closed}  open={n_open}")

    # --- Load encoder ---
    with open(models_dir / "encoder_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cpu")
    encoder = PlaceEncoder(cat_vocab_size=cfg["cat_vocab_size"]).to(device)
    encoder.load_state_dict(torch.load(models_dir / "encoder.pt", map_location=device,
                                       weights_only=True))
    encoder.eval()

    # --- Predictions ---
    p_closed = get_mlp_scores(encoder, X_val, y_val)
    f1, thr = best_f1_threshold(is_closed, p_closed)
    y_pred = (p_closed >= thr).astype(int)

    print(f"Threshold: {thr:.3f}  |  F1={f1:.4f}")

    # --- Error types ---
    error_types = assign_error_types(is_closed, y_pred)
    tp = np.array([e == "TP" for e in error_types])
    fp = np.array([e == "FP" for e in error_types])
    fn = np.array([e == "FN" for e in error_types])
    tn = np.array([e == "TN" for e in error_types])

    # ---------------------------------------------------------------------------
    # 1. Confusion matrix
    # ---------------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Confusion Matrix  (closed = positive class)")
    print(f"{'='*62}")
    print(f"  TP  correctly caught closures   : {tp.sum():>4}  / {n_closed} closed")
    print(f"  FN  missed closures             : {fn.sum():>4}  / {n_closed} closed")
    print(f"  FP  open places falsely flagged : {fp.sum():>4}  / {n_open} open")
    print(f"  TN  correctly cleared open      : {tn.sum():>4}  / {n_open} open")
    recall_val = tp.sum() / max(n_closed, 1)
    prec_val = tp.sum() / max(tp.sum() + fp.sum(), 1)
    print(f"\n  Recall    : {recall_val:.3f}")
    print(f"  Precision : {prec_val:.3f}")
    print(f"  F1        : {f1:.4f}")

    # ---------------------------------------------------------------------------
    # 2. Feature profile: FN vs TP
    # ---------------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  False Negatives vs True Positives")
    print(f"  Why are some closed places MISSED?")
    print(f"  (n_FN={fn.sum()}  n_TP={tp.sum()}  — higher FN value = harder to detect)")
    print(f"{'='*62}")
    if fn.sum() > 0:
        print_feature_profile(X_val, fn, tp, feat_names, "FN(missed)", "TP(caught)")
    else:
        print("  No false negatives at this threshold.")

    # ---------------------------------------------------------------------------
    # 3. Feature profile: FP vs TN
    # ---------------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  False Positives vs True Negatives")
    print(f"  Why are some OPEN places flagged as closed?")
    print(f"  (n_FP={fp.sum()}  n_TN={tn.sum()}  — higher FP value = more closed-like)")
    print(f"{'='*62}")
    if fp.sum() > 0:
        print_feature_profile(X_val, fp, tn, feat_names, "FP(flagged)", "TN(cleared)")
    else:
        print("  No false positives at this threshold.")

    # ---------------------------------------------------------------------------
    # 4. Category breakdown
    # ---------------------------------------------------------------------------
    categories = decode_categories(X_val, cat_enc)
    print(f"\n{'='*62}")
    print(f"  Category Error Breakdown")
    print(f"  (categories with ≥ 1 closed val sample, sorted by closed count)")
    print(f"  * low-n = fewer than 5 closed examples, metrics unreliable")
    print(f"{'='*62}")
    print_category_breakdown(categories, error_types, is_closed)

    # ---------------------------------------------------------------------------
    # 5. Save per-sample results
    # ---------------------------------------------------------------------------
    records = [
        {
            "idx": i,
            "true_label": "closed" if y_val[i] == 0 else "open",
            "p_closed": float(p_closed[i]),
            "predicted": "closed" if y_pred[i] == 1 else "open",
            "error_type": error_types[i],
            "category": categories[i],
        }
        for i in range(len(y_val))
    ]
    out_path = models_dir / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump({"threshold": thr, "f1": f1, "samples": records}, f, indent=2)
    print(f"\nPer-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
