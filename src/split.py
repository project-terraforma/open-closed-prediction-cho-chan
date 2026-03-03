"""
split.py
--------
Stratified 80/20 train/val split with category encoding.

Outputs to splits/:
    X_train.npy, y_train.npy  — training set (float32 / int64)
    X_val.npy,   y_val.npy    — validation set
    category_encoder.pkl      — LabelEncoder fitted on train only
    feature_names.json        — ordered feature names (matches column order in X)

primary_category is label-encoded to int (for MLP embedding layer).
Numerical features are left unscaled — BatchNorm inside the encoder handles this.

Run:
    python src/split.py data/project_c_samples.json
    python src/split.py data/project_c_samples.json --augment data/parquet_augment.json

With --augment:
    The original file is split 80/20 as usual; the val set is kept as-is
    (permanent hard-case benchmark).  All records from the augment file go
    directly into the train set — none contaminate val.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import load_dataset

RANDOM_SEED = 42
VAL_SIZE = 0.20

# Canonical feature order — numerical first, category last (easier for embedding indexing)
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
CATEGORICAL_FEATURES = ["primary_category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # category always last


def make_splits(
    data_path: str | Path,
    augment_path: str | Path | None = None,
    out_dir: str | Path = "splits",
) -> dict:
    """Load dataset, encode, split, and save.

    If augment_path is given, ALL records from that file are added to the
    train set only.  The val set comes exclusively from data_path (80/20
    stratified split), keeping it as a stable hard-case benchmark.

    Returns a dict with keys: X_train, X_val, y_train, y_val, encoder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # --- Load original data and split 80/20 ---
    print(f"Loading {data_path} ...")
    X_df, y = load_dataset(data_path)

    idx = np.arange(len(y))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    X_train_df = X_df.iloc[idx_train].reset_index(drop=True)
    X_val_df   = X_df.iloc[idx_val].reset_index(drop=True)
    y_train    = y[idx_train]
    y_val      = y[idx_val]

    # --- Append augment records to train only ---
    if augment_path is not None:
        print(f"Loading augment records from {Path(augment_path).name} ...")
        X_aug_df, y_aug = load_dataset(augment_path)
        X_train_df = pd.concat([X_train_df, X_aug_df], ignore_index=True)
        y_train    = np.concatenate([y_train, y_aug])
        print(f"  +{len(y_aug):,} records  "
              f"({(y_aug==0).sum():,} closed, {(y_aug==1).sum():,} open)")

    # --- Encode primary_category (fit on combined train only) ---
    enc = LabelEncoder()
    X_train_df["primary_category"] = enc.fit_transform(X_train_df["primary_category"])
    # Val: unseen categories map to a fallback index (len(classes_))
    val_cats = X_val_df["primary_category"].map(
        {c: i for i, c in enumerate(enc.classes_)}
    ).fillna(len(enc.classes_)).astype(int)
    X_val_df = X_val_df.copy()
    X_val_df["primary_category"] = val_cats

    # --- Convert to float32 arrays in canonical order ---
    X_train = X_train_df[ALL_FEATURES].to_numpy(dtype=np.float32)
    X_val   = X_val_df[ALL_FEATURES].to_numpy(dtype=np.float32)

    # --- Save ---
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)

    with open(out_dir / "category_encoder.pkl", "wb") as f:
        pickle.dump(enc, f)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    # --- Report ---
    aug_note = f"  (val = original {data_path} benchmark only)" if augment_path else ""
    print(f"\nSplit complete  (seed={RANDOM_SEED}){aug_note}")
    print(f"  Train: {len(y_train):>6,}  |  closed={(y_train==0).sum():,}  open={(y_train==1).sum():,}")
    print(f"  Val:   {len(y_val):>6,}  |  closed={(y_val==0).sum():,}  open={(y_val==1).sum():,}")
    print(f"  X shape: {X_train.shape[1]} features  ({len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")
    print(f"  Category vocab size: {len(enc.classes_)} (+1 OOV)")
    print(f"\nSaved to {out_dir}/")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "encoder": enc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data",    type=Path, nargs="?",
                        default=Path("data/project_c_samples.json"))
    parser.add_argument("--augment", type=Path, default=None,
                        help="JSONL of extra records added to train only")
    args = parser.parse_args()
    make_splits(args.data, augment_path=args.augment)
