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

Run (Overture, default):
    python src/ml/split.py data/project_c_samples.json
    python src/ml/split.py data/project_c_samples.json --augment data/yelp_features.jsonl
    python src/ml/split.py data/project_c_samples.json --augment data/parquet_augment.json data/yelp_features.jsonl
    python src/ml/split.py data/project_c_samples.json --include-conf

Run (SF registered businesses):
    python src/ml/split.py --source sf --db data/sf_registered_businesses.ddb
    python src/ml/split.py --source sf --db data/sf_registered_businesses.ddb --out-dir splits/sf
    python src/ml/split.py --source sf --db data/sf_registered_businesses.ddb --sample 50000

    Outputs to splits/sf/ (separate from Overture splits).
    Encodes 5 SF categorical columns: naic_code, lic, business_zip,
    supervisor_district, neighborhood. All encoders saved to splits/sf/.

With --sample N (SF only):
    Randomly samples N records per class for the TRAIN set, producing a 50/50
    balanced train set (N open + N closed = 2N total). Val is always kept at
    the natural SF distribution (~60% closed / ~40% open) so val AUC measures
    real discrimination performance.

    Why 50/50 for train: the SF dataset is ~60% closed / ~40% open, which already
    differs from the Overture deployment distribution (~90% open). Balancing train
    forces the MLP encoder to learn discriminative features for both classes rather
    than relying on the majority-class shortcut. See make_sf_splits() for full notes.

    The spatial feature cache (built from the full 318k dataset) is reused
    regardless of --sample — sampling only slices the cached feature matrix.

With --augment (Overture only):
    The original file is split 80/20 as usual; the val set is kept as-is
    (permanent hard-case benchmark).  All records from the augment file(s) go
    directly into the train set — none contaminate val.
    Multiple files can be passed: --augment file1.jsonl file2.jsonl ...

With --include-conf (Overture only):
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
    "msft_update_age_days",
    "n_sources_with_update_time",  # potential removal: identical to has_microsoft in this labeled dataset
    "min_update_age_days",         # potential removal: identical to msft_update_age_days in this labeled dataset
    "max_update_age_days",         # potential removal: identical to msft_update_age_days in this labeled dataset
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
) -> dict:
    """Load dataset, encode, split, and save.

    If augment_paths is given, ALL records from those files are added to the
    train set only.  The val set comes exclusively from data_path (80/20
    stratified split), keeping it as a stable hard-case benchmark.

    Args:
        data_path:     Primary JSONL file (e.g. project_c_samples.json).
        augment_paths: Optional list of supplementary JSONL files whose records
                       are appended to train only (e.g. yelp_features.jsonl).
        out_dir:       Directory to write split arrays and encoder.
        include_conf:  If True, include the 5 confidence features. Default False.

    Returns:
        Dict with keys: X_train, X_val, y_train, y_val, encoder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    numeric_features = NUMERIC_FEATURES_BASE + (CONF_FEATURES if include_conf else [])
    all_features = numeric_features + CATEGORICAL_FEATURES

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

    # --- Append each augment file to train only ---
    if augment_paths:
        for aug_path in augment_paths:
            print(f"Loading augment records from {Path(aug_path).name} ...")
            X_aug_df, y_aug = load_dataset(aug_path)
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

    # --- category_closure_rate: fraction closed per category, fit on train only ---
    # Use raw string category to avoid confusion with the encoded int above.
    # Unseen categories on val get the global train closure rate as fallback.
    raw_cat_train = X_df.iloc[idx_train]["primary_category"].reset_index(drop=True)
    if augment_paths:
        # raw_cat_train must match the full X_train_df; augment rows have no raw original
        # so just recompute from the combined df before encoding was done.
        # We already encoded in place, so fall back: re-extract from X_train_df "primary_category"
        # which now holds ints — use them to look up enc.classes_ for the string.
        raw_cat_train = pd.Series(
            enc.classes_[X_train_df["primary_category"].clip(upper=len(enc.classes_)-1).astype(int)],
            name="primary_category",
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
        X_df.iloc[idx_val]["primary_category"].reset_index(drop=True)
        .map(closure_map).fillna(global_rate).values
    )
    # Save closure map for inference
    with open(out_dir / "category_closure_rate.pkl", "wb") as f:
        pickle.dump({"map": closure_map, "global_rate": global_rate}, f)

    numeric_features = numeric_features + ["category_closure_rate"]
    all_features     = numeric_features + CATEGORICAL_FEATURES

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


# ---------------------------------------------------------------------------
# SF registered businesses split
# ---------------------------------------------------------------------------

# Categorical columns in the SF feature set — each gets its own LabelEncoder,
# fit on train only. Unseen values on val map to len(classes_) (OOV index).
SF_CATEGORICAL_FEATURES = [
    "naic_code",
    "lic",
    "business_zip",
    "supervisor_district",
    "neighborhood",
]


def make_sf_splits(
    db_path: str | Path = "data/sf_registered_businesses.ddb",
    out_dir: str | Path = "splits/sf",
    n_per_class: int | None = None,
) -> dict:
    """Load SF registered businesses, encode categoricals, split 80/20, and save.

    --- Class balance note ---
    The SF dataset is ~60% closed / ~40% open. This already differs substantially
    from the Overture deployment distribution (~90% open / ~10% closed). Training
    on the raw SF distribution would bias the encoder toward predicting closed.

    A 50/50 balanced sample (--sample N) is recommended for the TRAIN set because:
      1. The MLP encoder learns better discriminative embeddings when both classes
         are equally represented — it can't rely on the majority-class shortcut.
      2. The class_weights in train.py already handle imbalance via weighted
         cross-entropy; with 50/50 the weights become 1.0/1.0, simplifying training.
      3. NCM/SLDA class means are more stable when computed from equal-size groups.

    Keep the VAL set at the natural SF distribution (do NOT balance it) so that
    val AUC reflects real-world discrimination performance.

    When the model is eventually applied to Overture (~90% open), confidence
    scores will need post-hoc calibration (e.g. Platt scaling / isotonic
    regression) against Overture-labeled examples to correct the prior shift.

    Output is compatible with train.py / encoder.py without any changes:
      - naic_code is placed last (X[:, -1]) as the single primary category,
        matching encoder.py's PlaceDataset convention.
      - naic_code encoder is saved as category_encoder.pkl (train.py hardcodes
        this filename).
      - NaN in numeric/spatial columns is imputed with per-column train medians
        (saved to numeric_medians.json) so BatchNorm1d does not receive NaN.
      - The other 4 SF categoricals (lic, business_zip, supervisor_district,
        neighborhood) remain as label-encoded ints in the numeric block.

    Args:
        db_path:    Path to the DuckDB file produced by fetch-sf-registered-businesses.py.
        out_dir:    Directory to write split arrays and encoders (default: splits/sf/).
        n_per_class: If set, randomly sample this many records per class for TRAIN
            after the 80/20 split. Val is always kept at the natural SF distribution.
            See class balance note above and module docstring for --sample details.

    Returns:
        Dict with keys: X_train, X_val, y_train, y_val, encoders.
    """
    from sf_feature_engineering import load_dataset as sf_load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SF dataset from {db_path} ...")
    X_df, y = sf_load_dataset(db_path)
    print(f"Loaded {len(y):,} records  |  open={y.sum():,}  closed={(y==0).sum():,}")

    idx = np.arange(len(y))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    X_train_df = X_df.iloc[idx_train].reset_index(drop=True)
    X_val_df   = X_df.iloc[idx_val].reset_index(drop=True)
    y_train    = y[idx_train]
    y_val      = y[idx_val]

    # --- Optional: balance TRAIN by sampling N records per class ---
    # Applied after 80/20 split so the val set is unaffected and reflects the
    # natural SF distribution (~60% closed / ~40% open). This gives an honest
    # val AUC while the encoder trains on a balanced signal.
    #
    # The spatial feature cache (built over all 318k businesses) is reused
    # regardless of --sample — sampling just slices the cached feature matrix.
    if n_per_class is not None:
        rng = np.random.default_rng(RANDOM_SEED)
        open_idx_train   = np.where(y_train == 1)[0]
        closed_idx_train = np.where(y_train == 0)[0]
        n_open   = min(n_per_class, len(open_idx_train))
        n_closed = min(n_per_class, len(closed_idx_train))
        keep_idx = np.sort(np.concatenate([
            rng.choice(open_idx_train,   size=n_open,   replace=False),
            rng.choice(closed_idx_train, size=n_closed, replace=False),
        ]))
        X_train_df = X_train_df.iloc[keep_idx].reset_index(drop=True)
        y_train    = y_train[keep_idx]
        print(f"  Train balanced: {n_open:,} open + {n_closed:,} closed = {len(y_train):,} total")
        print(f"  Val unchanged:  {(y_val==1).sum():,} open + {(y_val==0).sum():,} closed (natural SF distribution)")

    # --- Encode each SF categorical: fit on train, apply to val ---
    encoders = {}
    for col in SF_CATEGORICAL_FEATURES:
        enc = LabelEncoder()
        # Fill NaN with a sentinel string so LabelEncoder sees a complete array
        train_vals = X_train_df[col].fillna("__MISSING__").astype(str)
        X_train_df[col] = enc.fit_transform(train_vals)

        val_map = {c: i for i, c in enumerate(enc.classes_)}
        X_val_df[col] = (
            X_val_df[col].fillna("__MISSING__").astype(str)
            .map(val_map)
            .fillna(len(enc.classes_))  # OOV → one past the last known index
            .astype(int)
        )
        encoders[col] = enc

    # Save one encoder file per categorical column
    for col, enc in encoders.items():
        with open(out_dir / f"{col}_encoder.pkl", "wb") as f:
            pickle.dump(enc, f)

    # Save naic_code encoder as category_encoder.pkl — train.py loads this
    # filename unconditionally (train.py:189). naic_code is also placed last
    # in the feature matrix (X[:, -1]) to match PlaceDataset's single-category
    # convention in encoder.py.
    with open(out_dir / "category_encoder.pkl", "wb") as f:
        pickle.dump(encoders["naic_code"], f)

    # --- Reorder columns: naic_code last so X[:, -1] = category index ---
    non_cat_cols = [c for c in X_df.columns if c != "naic_code"]
    all_features = non_cat_cols + ["naic_code"]

    # --- NaN imputation on numeric columns (train medians, fit on train only) ---
    # Spatial KNN features return NaN when 0 neighbors exist at a radius.
    # PyTorch BatchNorm1d propagates NaN → kills gradients. Impute before saving.
    # train_medians saved to numeric_medians.json for inference-time use.
    numeric_cols = non_cat_cols  # naic_code (last) is categorical, skip it
    train_medians = X_train_df[numeric_cols].median()
    X_train_df[numeric_cols] = X_train_df[numeric_cols].fillna(train_medians)
    X_val_df[numeric_cols]   = X_val_df[numeric_cols].fillna(train_medians)
    train_medians.to_json(out_dir / "numeric_medians.json")

    # --- Convert to float32 arrays ---
    X_train = X_train_df[all_features].to_numpy(dtype=np.float32)
    X_val   = X_val_df[all_features].to_numpy(dtype=np.float32)

    # --- Save ---
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"\nSplit complete  (seed={RANDOM_SEED})")
    print(f"  Train: {len(y_train):>6,}  |  closed={(y_train==0).sum():,}  open={(y_train==1).sum():,}")
    print(f"  Val:   {len(y_val):>6,}  |  closed={(y_val==0).sum():,}  open={(y_val==1).sum():,}")
    print(f"  X shape: {X_train.shape[1]} features  (last col = naic_code category)")
    print(f"  naic_code vocab: {len(encoders['naic_code'].classes_)} (+1 OOV)")
    print(f"\nSaved to {out_dir}/")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "encoders": encoders,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["overture", "sf"], default="overture",
        help="Which dataset pipeline to use (default: overture)",
    )

    # Overture-only arguments
    parser.add_argument("data", type=Path, nargs="?",
                        default=Path("data/project_c_samples.json"))
    parser.add_argument(
        "--augment", type=Path, nargs="+", default=None,
        help="(Overture only) One or more JSONL files added to train only "
             "(e.g. --augment data/yelp_features.jsonl data/parquet_augment.json)",
    )
    parser.add_argument(
        "--include-conf", action="store_true", default=False,
        help="(Overture only) Include the 5 confidence features (excluded by default)",
    )

    # SF-only arguments
    parser.add_argument(
        "--db", type=Path, default=Path("data/sf_registered_businesses.ddb"),
        help="(SF only) Path to the SF registered businesses DuckDB file",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for splits (default: splits/ for overture, splits/sf/ for sf)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N", dest="n_per_class",
        help="(SF only) Sample N records per class for TRAIN (e.g. --sample 50000 → "
             "50k open + 50k closed = 100k balanced train set). "
             "Val is always kept at the natural SF distribution. "
             "Default: use all train data at natural distribution.",
    )

    args = parser.parse_args()

    if args.source == "sf":
        out_dir = args.out_dir or Path("splits/sf")
        make_sf_splits(db_path=args.db, out_dir=out_dir, n_per_class=args.n_per_class)
    else:
        out_dir = args.out_dir or Path("splits")
        make_splits(args.data, augment_paths=args.augment,
                    include_conf=args.include_conf, out_dir=out_dir)
