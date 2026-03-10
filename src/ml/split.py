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
    python src/split.py data/project_c_samples.json --augment data/yelp_features.jsonl
    python src/split.py data/project_c_samples.json --augment data/parquet_augment.json data/yelp_features.jsonl
    python src/split.py data/project_c_samples.json --include-conf

With --augment:
    The original file is split 80/20 as usual; the val set is kept as-is
    (permanent hard-case benchmark).  All records from the augment file(s) go
    directly into the train set — none contaminate val.
    Multiple files can be passed: --augment file1.jsonl file2.jsonl ...

With --include-conf:
    Include the 5 source/record confidence features (max_source_confidence,
    min_source_confidence, mean_source_confidence, confidence_spread, confidence).
    Default is to exclude them — see engineer note on static-confidence providers.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import load_dataset

RANDOM_SEED = 42
VAL_SIZE = 0.20

# Canonical feature order — numerical first, category last (easier for embedding indexing)
# Confidence features are excluded by default (see --include-conf flag).
CONF_FEATURES = [
    "max_source_confidence",
    "min_source_confidence",
    "mean_source_confidence",
    "confidence_spread",
    "confidence",
]
NUMERIC_FEATURES_BASE = [
    "source_count",
    "has_meta",
    "has_microsoft",
    "has_only_meta",
    "has_single_source",
    "msft_update_age_days",
    "msft_fresh",
    "n_sources_with_update_time",  # potential removal: identical to has_microsoft in this labeled dataset
    "min_update_age_days",         # potential removal: identical to msft_update_age_days in this labeled dataset
    "max_update_age_days",         # potential removal: identical to msft_update_age_days in this labeled dataset
    "conf_x_source",
    "has_website",
    "has_phone",
    "has_socials",
    "has_brand",
    "website_count",
    "phone_count",
    "completeness_score",
    "has_alternate_categories",
    "alternate_category_count",
    "address_completeness",
]
CATEGORICAL_FEATURES = ["primary_category"]


def make_splits(
    data_path: str | Path,
    augment_paths: list[str | Path] | None = None,
    out_dir: str | Path = "splits",
    include_conf: bool = False,
    exclude: list[str] | None = None,
    fe=None,
) -> dict:
    """Load dataset, encode, split, and save.

    If augment_paths is given, ALL records from those files are added to the
    train set only.  The val set comes exclusively from data_path (80/20
    stratified split), keeping it as a stable hard-case benchmark.

    Args:
        data_path:     Primary data file (JSONL or GeoJSON depending on schema).
        augment_paths: Optional list of supplementary files added to train only.
        out_dir:       Directory to write split arrays and encoder.
        include_conf:  If True, include conf features (no-op for non-overture schemas).
        exclude:       Feature names to drop after all features are assembled.
        fe:            Alternate feature engineering module (e.g. sf_feature_engineering).
                       Must export load_dataset, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                       and CONF_FEATURES.  None → default Overture feature engineering.

    Returns:
        Dict with keys: X_train, X_val, y_train, y_val, encoder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Resolve feature engineering source
    _load      = fe.load_dataset          if fe else load_dataset
    _num_base  = getattr(fe, "NUMERIC_FEATURES",    NUMERIC_FEATURES_BASE) if fe else NUMERIC_FEATURES_BASE
    _cat_feats = getattr(fe, "CATEGORICAL_FEATURES", CATEGORICAL_FEATURES)  if fe else CATEGORICAL_FEATURES
    _conf      = getattr(fe, "CONF_FEATURES",        CONF_FEATURES)         if fe else CONF_FEATURES
    cat_col    = _cat_feats[0] if _cat_feats else "primary_category"

    numeric_features = _num_base + (_conf if include_conf else [])
    all_features = numeric_features + _cat_feats

    # --- Load original data and split 80/20 ---
    print(f"Loading {data_path} ...")
    X_df, y = _load(data_path)

    idx = np.arange(len(y))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    X_train_df = X_df.iloc[idx_train].reset_index(drop=True)
    X_val_df   = X_df.iloc[idx_val].reset_index(drop=True)
    y_train    = y[idx_train]
    y_val      = y[idx_val]

    # --- Append each augment file to train only ---
    if augment_paths:
        for aug_path in augment_paths:
            print(f"Loading augment records from {Path(aug_path).name} ...")
            X_aug_df, y_aug = _load(aug_path)
            X_train_df = pd.concat([X_train_df, X_aug_df], ignore_index=True)
            y_train    = np.concatenate([y_train, y_aug])
            print(f"  +{len(y_aug):,} records  "
                  f"({(y_aug==0).sum():,} closed, {(y_aug==1).sum():,} open)")

    # --- Encode categorical column (fit on combined train only) ---
    enc = LabelEncoder()
    X_train_df[cat_col] = enc.fit_transform(X_train_df[cat_col])
    # Val: unseen categories map to a fallback index (len(classes_))
    val_cats = X_val_df[cat_col].map(
        {c: i for i, c in enumerate(enc.classes_)}
    ).fillna(len(enc.classes_)).astype(int)
    X_val_df = X_val_df.copy()
    X_val_df[cat_col] = val_cats

    # --- category_closure_rate: fraction closed per category, fit on train only ---
    # Use raw string category to avoid confusion with the encoded int above.
    # Unseen categories on val get the global train closure rate as fallback.
    raw_cat_train = X_df.iloc[idx_train][cat_col].reset_index(drop=True)
    if augment_paths:
        # raw_cat_train must match the full X_train_df; augment rows have no raw original
        # so just recompute from the combined df before encoding was done.
        # We already encoded in place, so fall back: re-extract from X_train_df cat_col
        # which now holds ints — use them to look up enc.classes_ for the string.
        raw_cat_train = pd.Series(
            enc.classes_[X_train_df[cat_col].clip(upper=len(enc.classes_)-1).astype(int)],
            name=cat_col,
        )
    closure_map = (
        pd.DataFrame({"cat": raw_cat_train, "closed": (y_train == 0).astype(float)})
        .groupby("cat")["closed"].mean()
        .to_dict()
    )
    global_rate = float((y_train == 0).mean())
    X_train_df["category_closure_rate"] = raw_cat_train.map(closure_map).fillna(global_rate).values
    # Val: map using the same dict; fallback to global train rate for unseen categories
    X_val_df["category_closure_rate"] = (
        X_df.iloc[idx_val][cat_col].reset_index(drop=True)
        .map(closure_map).fillna(global_rate).values
    )
    # Save closure map for inference
    with open(out_dir / "category_closure_rate.pkl", "wb") as f:
        pickle.dump({"map": closure_map, "global_rate": global_rate}, f)

    numeric_features = numeric_features + ["category_closure_rate"]

    # --- Apply feature exclusions ---
    if exclude:
        exclude_set = set(exclude)
        dropped = [f for f in numeric_features if f in exclude_set]
        numeric_features = [f for f in numeric_features if f not in exclude_set]
        if dropped:
            print(f"  Excluded features: {dropped}")

    all_features = numeric_features + _cat_feats

    # --- Convert to float32 arrays in canonical order ---
    X_train = X_train_df[all_features].to_numpy(dtype=np.float32)
    X_val   = X_val_df[all_features].to_numpy(dtype=np.float32)

    # --- Save ---
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)

    with open(out_dir / "category_encoder.pkl", "wb") as f:
        pickle.dump(enc, f)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(all_features, f, indent=2)

    # --- Report ---
    conf_note = "  (+conf)" if include_conf else "  (no-conf)"
    aug_note = "  (val = original benchmark only)" if augment_paths else ""
    print(f"\nSplit complete  (seed={RANDOM_SEED}){conf_note}{aug_note}")
    print(f"  Train: {len(y_train):>6,}  |  closed={(y_train==0).sum():,}  open={(y_train==1).sum():,}")
    print(f"  Val:   {len(y_val):>6,}  |  closed={(y_val==0).sum():,}  open={(y_val==1).sum():,}")
    print(f"  X shape: {X_train.shape[1]} features  ({len(numeric_features)} numeric + {len(CATEGORICAL_FEATURES)} categorical)")
    print(f"  Category vocab size: {len(enc.classes_)} (+1 OOV)")
    print(f"\nSaved to {out_dir}/")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "encoder": enc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, nargs="?",
                        default=Path("data/project_c_samples.json"))
    parser.add_argument(
        "--schema", choices=["overture", "sf"], default="overture",
        help="Feature engineering schema: 'overture' (default) or 'sf' for the "
             "SF registered business GeoJSON dataset",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/train.yaml"),
        help="YAML config file — reads features.include_conf and features.exclude "
             "(default: config/train.yaml)",
    )
    parser.add_argument(
        "--augment", type=Path, nargs="+", default=None,
        help="One or more files added to train only",
    )
    parser.add_argument(
        "--include-conf", action="store_true", default=None,
        help="Include the 5 confidence features (overrides config; no-op for --schema sf)",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Feature names to drop (additive with config features.exclude)",
    )
    args = parser.parse_args()

    # Load config defaults
    cfg_features: dict = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg_features = yaml.safe_load(f).get("features", {})

    include_conf = cfg_features.get("include_conf", False)
    if args.include_conf is not None:   # CLI flag overrides config
        include_conf = args.include_conf

    exclude: list[str] = list(cfg_features.get("exclude") or [])
    if args.exclude:                    # CLI --exclude is additive
        exclude += args.exclude

    # Load alternate feature engineering module if requested
    import importlib
    fe_module = None
    if args.schema == "sf":
        fe_module = importlib.import_module("sf_feature_engineering")
        print(f"Schema: sf  ({len(fe_module.NUMERIC_FEATURES)} numeric + "
              f"{len(fe_module.CATEGORICAL_FEATURES)} categorical features)")

    make_splits(
        args.data,
        augment_paths=args.augment,
        include_conf=include_conf,
        exclude=exclude or None,
        fe=fe_module,
    )
