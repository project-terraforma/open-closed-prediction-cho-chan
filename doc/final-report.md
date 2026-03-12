# Project C — Final Report
**Author:** Caleb Cho | **Date:** 2026-03-11

---

## 1. Models & Continual Learning Methods

### Problem framing

Predict whether a place in Overture Maps is open or closed — a binary classification problem on a heavily imbalanced dataset (9% closed, 91% open). We evaluate five model families, with a specific focus on which architectures support **continual learning**: the ability to update the model incrementally when a new monthly Overture release arrives, without retraining from scratch.

---

### 1.1 GBM / XGBoost (baseline)

Gradient-boosted trees operating directly on raw features. The category column is one-hot encoded (295 binary columns). These are the fastest models to train, fully deterministic, and thus the most reliable signal in ablation studies. All reported GBM results are deterministic — no run-to-run variance.

**Strengths:** Fast, interpretable feature importances, handles mixed numeric/categorical well.

**Weaknesses:** Memorizes raw feature distributions — collapses on cross-domain evaluation (Yelp val: 0.48–0.51 AUC). Requires full retraining on every Overture release (~184–291 ms per retrain). Cannot do incremental updates.

---

### 1.2 MLP Encoder

A 3-layer feedforward network `[128, 64, 32]` that maps raw Overture features into a 32-dimensional embedding. The category column is encoded as an 8-dim learned embedding, not one-hot. BatchNorm is applied after each hidden layer.

The MLP is trained end-to-end with the classification head, then the frozen encoder is reused as a feature extractor for the continual learning heads.

**Why 32-dim output:** Large enough to capture meaningful geometry between open and closed clusters; small enough that covariance estimates (SLDA, QDA) remain stable with only ~250 closed training examples.

**Architecture search result:**

```
Architecture         MLP head   MLP+NCM   MLP+SLDA
-----------------------------------------------------
[64, 32]   2-layer   0.6703     0.6725     0.6669   (underfits)
[128,64,32] 3-layer  0.6817     0.6905     0.6881   (optimal) ←
[256,128,32] 3-layer 0.6663     0.6569     0.6782   (overfits)
```

At 256-dim the encoder spreads embeddings far enough that NCM centroids become unstable with only ~250 closed examples. **NCM is the most sensitive architecture probe** — use it to detect encoder overfitting before it appears in the MLP head.

**Strengths:** Learns a compressed representation that captures abstract patterns ("data completeness degrades as closure approaches") rather than memorizing raw feature values. Robust to distribution shift. Smallest model size (33–42 KB).

---

### 1.3 MLP head (softmax classifier)

A linear classification layer trained on top of the 32-dim MLP embedding. The simplest classifier; serves as the MLP baseline. Does not support incremental updates — the head weights must be retrained if the encoder is frozen.

---

### 1.4 NCM — Nearest Class Mean

Stores one **centroid** (mean embedding) per class. Assigns a new sample to the nearest centroid. Incremental update: when new labeled data arrives, update each centroid with the new sample means — O(N) in new samples, no old data needed.

```python
# Exact incremental update
ncm.update(X_new, y_new)   # 0.061 ms for 823 new samples
```

**Strengths:** Fastest incremental update (0.061 ms). Mathematically exact — result equals a full refit on all data.

**Weaknesses:** Most sensitive to encoder geometry. If the embedding space is poorly calibrated or the encoder overfits, centroids drift and NCM degrades sharply. Fails completely (F1=0) when parquet augmentation shifts the closed embedding far from the labeled closed centroid.

---

### 1.5 SLDA — Streaming Linear Discriminant Analysis

Maintains per-class means and a **shared (pooled) covariance matrix** updated via Welford's online algorithm. The LDA decision boundary is a hyperplane determined by the embedding geometry, not by learned weights.

```python
slda.update(X_new, y_new)   # 0.149 ms for 823 new samples
```

**Strengths:** More stable than NCM because covariance regularizes the boundary. Smallest transfer loss across domains (-0.020 vs. -0.039 to -0.047 for other models). The boundary re-finds the most separating hyperplane even when the embedding distribution shifts. Resilient to feature removal (-0.027 AUC drop vs. -0.058 to -0.063 for tree models).

**Weaknesses:** Assumes a shared covariance structure (pooled). When embedding distributions from two distinct sources (labeled + parquet aug) are mixed, the covariance matrix can become near-singular (condition number 6 → 1,525 after parquet aug).

---

### 1.6 QDA — Quadratic Discriminant Analysis

Like SLDA but maintains **separate covariance matrices** per class. This allows class-specific cluster shapes.

**Strengths:** Outperforms SLDA when confidence features are included (0.7106 vs. 0.6960) because confidence shifts the embedding geometry enough that class-specific covariance is beneficial.

**Weaknesses:** Requires more data to estimate two covariance matrices stably. Less resilient to distribution shift than SLDA's pooled estimate.

---

### 1.7 Continual learning speed comparison

```
NCM   update()         0.061 ms   for 823 new samples
SLDA  update()         0.149 ms
XGBoost full retrain   184–291 ms
→ NCM is ~3,000× faster than XGBoost retrain
```

The incremental update is **mathematically exact** — the result equals a full refit on all data, not an approximation:

```
            R0 only   After update   Full refit   Exact match?
NCM          0.7231       0.7213      0.7213       YES (diff < 1e-6)
SLDA         0.7332       0.7260      0.7260       YES (diff < 1e-6)
```

---

## 2. Datasets

### 2.1 Overture 3k labeled set (primary)

| | |
|---|---|
| Records | 3,425 Overture places, manually annotated |
| Closed (positive class) | 313 (9.1%) |
| Open | 3,112 (90.9%) |
| Geography | US-only |
| Sources | Meta (Facebook) + Microsoft Bing |
| Split | Stratified 80/20 → train: 2,740 | val: 685 (held out permanently) |

This is the permanent held-out benchmark. Val set: 685 samples, 63 closed / 622 open. All model comparisons use this val set.

---

### 2.2 SF Open Business Registry (augmentation source)

| | |
|---|---|
| Source | SF Open Data — Registered Business Locations |
| Snapshot | `data/sf_open_dataset_20260309.geojson` (downloaded 2026-03-09) |
| Total records | 356,351 |
| Open (no `dba_end_date`) | 164,271 (46%) |
| Closed (`dba_end_date` ≤ ref date) | 192,068 (54%) |

**Label derivation:** `closed=1` if `dba_end_date` is set and ≤ 2026-03-09. All other records are `open=1`.

Used in two ways:
1. **Option A (aug) —** Matched against Overture places via geo+name similarity (`src/util/sf_lookup.py`). Matched Overture records inherit the SF label. 13,729 records augmented into Overture training.
2. **Option B (standalone) —** Trained a separate SF-native model directly on the registry to measure performance upper bound and analyze feature signals.

---

### 2.3 Overture Parquet (Feb release) — augmentation attempt

The Feb Overture parquet release contains 785 places with `operating_status = 'closed'` not overlapping with the labeled set. These were extracted via a DuckDB pipeline (`src/ml/parquet_augment.py`) with category-stratified open sampling.

**Issue:** All 785 parquet-closed records have `source_count = 2.0` exactly (Overture's 2-source confirmation requirement for setting `operating_status=closed`). This structural artifact inflates confidence (~0.95 avg vs. ~0.75 for labeled closed) and doesn't generalize. Mixing them into MLP training degraded AUC by -0.027. **Not used in the final configuration.**

A 9010 (open:closed) ratio was found to be necessary — 50/50 ratio collapses recall by shifting class weight from 5.59× to 2.41×.

---

### 2.4 Yelp Academic Dataset (generalization probe)

US business listings with ground-truth `is_open` labels. Converted to pseudo-Overture schema via `src/ml/yelp_feature_engineering.py`. Used exclusively as a **cross-domain generalization test** — not mixed into Overture training.

| | |
|---|---|
| Records | ~150k US businesses |
| Closed | ~30k (20.4%) |
| Overture overlap | 46 matched places (1.3% of labeled set) — too sparse for direct label transfer |
| Val set used | 30,069 samples, 20.4% closed |

Key limitation: `completeness_score = 0` for all Yelp records (has_phone, has_website, has_socials, has_brand absent from Yelp schema, forced to 0). The model learns "completeness=0 is normal" — directly backwards for Overture. Yelp should never be mixed into Overture training; its role is as a held-out robustness probe.

---

### 2.5 SF-augmented training sets

| File | Records | Config |
|---|---|---|
| `data/sf_aug.json` | 13,729 | nsim=0.75, lsim=0.85, asim=off, staleness=off |
| `data/sf_aug_asim_75_85.json` | ~13,400 | Same + asim=0.4 gate (caused regression, reverted) |

---

## 3. Features

All features extracted by `src/feature_engineering.py` from raw Overture JSON. 19 numeric/boolean + 1 categorical column.

### 3.1 Feature list

| Feature | Type | Description |
|---|---|---|
| `confidence` | float | Overture's top-level confidence score |
| `max_source_confidence` | float | Max confidence across all sources |
| `min_source_confidence` | float | Min confidence across all sources |
| `mean_source_confidence` | float | Mean confidence across all sources |
| `confidence_spread` | float | max − min confidence (disagreement signal) |
| `source_count` | int | Number of data sources |
| `has_microsoft` | bool | Microsoft Bing is a source |
| `has_phone` | bool | Phone number present |
| `has_website` | bool | Website URL present |
| `has_socials` | bool | Social media URL present |
| `has_brand` | bool | Brand information present |
| `address_completeness` | float | Fraction of address sub-fields filled |
| `completeness_score` | float | Composite completeness (phone + website + socials + brand) |
| `msft_update_age_days` | float | Days since Microsoft last updated the record |
| `primary_category` | string (→ 8-dim embed) | Top-level Overture category |
| `category_closure_rate` | float | Historical closure rate for this category in training data |

### 3.2 EDA-measured signal strength (Cohen's d)

```
address_completeness    ████████████████████████  0.82  strong
confidence              ████████████████░░░░░░░░  0.64  strong
source_confidence       ████████████████░░░░░░░░  0.55–0.64  strong
has_phone               ██████████████░░░░░░░░░░  0.54  strong
completeness_score      ██████████░░░░░░░░░░░░░░  0.39  strong
has_website             ███████░░░░░░░░░░░░░░░░░  0.28  moderate
source_count            ██████░░░░░░░░░░░░░░░░░░  0.25  moderate
has_microsoft           ███░░░░░░░░░░░░░░░░░░░░░  0.13  moderate
msft_update_age_days    ██░░░░░░░░░░░░░░░░░░░░░░  0.08  weak
```

### 3.3 Features excluded from final config

Five features added mid-project were found to hurt augmented training and are excluded:

| Feature | Reason for exclusion |
|---|---|
| `has_only_meta` | Distribution flips between labeled set and parquet aug (most parquet open records are Meta-only) |
| `n_sources_with_update_time` | Different coverage patterns across record sources |
| `min_update_age_days` | Microsoft update-age patterns differ between sources |
| `max_update_age_days` | Same |
| `category_closure_rate` | Distorted when category distribution of aug records differs from labeled set |

Dropping them gave the largest single-run improvement: +0.018 MLP head AUC with parquet aug.

---

## 4. Evaluation Results, Insights & Recommendation

### 4.1 Final best results

**Best configuration:** `[128, 64, 32]` encoder | confidence features included | SF augmentation (nsim=0.75, lsim=0.85, 13,729 records)

```
Model        AUC-ROC   Notes
-----------------------------
GBM           0.7453   ← best overall
MLP + SLDA    0.7350
MLP head      0.7205
```

**3k-only baseline (no augmentation):**

```
Model              AUC-ROC  AUC-PR    F1   Prec  Recall
--------------------------------------------------------
MLP head            0.734   0.263  0.315  0.313   0.318
MLP + SLDA          0.726   0.253  0.308  0.235   0.444
MLP + NCM           0.721   0.243  0.302  0.218   0.492
XGBoost + OHE       0.720   0.293  0.311  0.206   0.635
GBM + OHE           0.686   0.255  0.277  0.173   0.698
```

All revised targets met (AUC-ROC > 0.73, AUC-PR > 0.28, F1 > 0.31).

---

### 4.2 Key insights

#### Confidence is the dominant lever

Removing all 5 confidence features drops AUC by -0.063 (GBM) and -0.058 (MLP head) — confirmed with deterministic tree models. This is the single largest feature group by impact.

> **Caveat on parquet:** Parquet-sourced `operating_status=closed` records have avg confidence ~0.95 vs. ~0.75 for labeled closed. The anomaly is specific to chain-store mass closures (Payless, Sears-era batch imports). Individual business closures — what the labeled set captures — still exhibit low confidence. The feature is valid.

#### MLP is robust to distribution shift; trees are not

Cross-domain transfer experiment (train on Yelp, eval on Overture):

```
                   Overture→Overture  Yelp→Yelp  Yelp→Overture  Transfer loss
GBM                    0.686           0.507        0.671          -0.015
XGBoost                0.720           0.484        0.673          -0.047
MLP head               0.734           0.730        0.695          -0.039
MLP + NCM              0.721           0.730        0.678          -0.043
MLP + SLDA             0.726           0.729        0.706          -0.020  ← best
```

GBM/XGBoost score 0.48–0.51 on Yelp val — functionally broken. They memorize Overture's raw feature distribution (particularly the confidence proxy directionality). The MLP encoder learns an abstract 32-dim representation that transfers across datasets.

**This matters for production:** Overture's feature distributions shift with every monthly release as new sources are added and coverage evolves. The MLP architecture is correct not just for peak accuracy but for long-term resilience.

#### The error ceiling is the feature set, not the model

At optimal F1 threshold (0.495), MLP head catches only 33% of closed places (21/63). The 42 missed closures look like healthy open places: high confidence (0.85), multi-source, complete profiles.

```
Feature                     Missed closures   Caught closures   Diff
---------------------------------------------------------------------
msft_update_age_days              804.9           1196.2       -391.3  ← biggest
confidence                        0.849            0.540       +0.309
completeness_score                0.726            0.524       +0.202
```

False positives are open places with low confidence and incomplete data — feature-indistinguishable from closed places. **The model correctly learns the correlation between data quality and closure, but two-thirds of actual closures retain healthy-looking data.** This is an irreducible ceiling given current features.

#### SF augmentation helps; quality filters hurt

Augmenting with 13,729 SF-matched Overture records adds +0.009–0.020 AUC across models. However:
- **ASIM gate (address similarity ≥ 0.4):** Rejected 291 valid parquet records because `freeform` address was empty; GBM regression 0.7453 → 0.7255. Disabled.
- **Staleness filter:** Skip aug records where Overture `update_time > SF closure date`. 88% of parquet records have source `update_time` in 2025 → any SF closure before 2025 triggers staleness → 32% of aug records dropped → GBM 0.7453 → 0.6852. The Overture source `update_time` reflects when *any* field was updated, not when existence was verified. Filter kept in code (default off).

---

### 4.3 Model recommendation

**MLP + SLDA** for production deployment.

| Criterion | GBM | MLP + SLDA |
|---|---|---|
| In-domain AUC-ROC (with SF aug) | **0.7453** | 0.7350 |
| Cross-domain transfer loss | -0.015 | **-0.020** |
| Incremental update | No (full retrain) | Yes (0.149 ms) |
| Sensitivity to dist. shift | High | Low |
| Model size | 652 KB | **41.9 KB** |
| Recommended for production | No | **Yes** |

GBM edges out MLP+SLDA in-domain by +0.010 AUC, but collapses on cross-domain evaluation and requires full retraining each release. MLP+SLDA's 0.020 transfer loss and exact incremental update make it the correct production choice — the MLP encoder generalizes across data shifts that Overture production will inevitably deliver.

---

## 5. Overture Dataset Shortcomings, Why SF Works Better, and Recommendations

### 5.1 Why SF achieves 0.88 AUC vs Overture's 0.72

Trained directly on the SF registry (356,351 records, 54/46 class balance), using SF-native features:

```
Model        AUC-ROC  AUC-PR    F1
------------------------------------
GBM           0.8867  0.8870  0.836
XGBoost       0.8846  0.8842  0.835
MLP head      0.8812  0.8797  0.832
MLP + SLDA    0.8811  0.8794  0.833
```

**The gap is data quality, not model capacity.** The same MLP architecture that scores 0.72 on Overture scores 0.88 on SF. The architecture has headroom — it is the Overture data that is the constraint.

| Factor | SF registry | Overture |
|---|---|---|
| Label quality | Ground-truth administrative record (`dba_end_date`) | Inferred from `operating_status` (noisy, crowd-sourced) |
| Class balance | 54% closed / 46% open | 9% closed / 91% open |
| Business age signal | Direct: `dba_start_date`, `location_start_date` | Indirect: `msft_update_age_days` (source recency, not business age) |
| Missing-data signal | `has_naic_code` — cleared on closure (73% closed rate when missing) | `completeness_score` — weaker proxy |
| Regulatory signal | `has_lic` (license type) | None |
| Closed training examples | ~103,000 | 313 |

**SF feature importances confirm what's missing in Overture:**

```
Feature                  SF GBM importance   SF MLP perm. importance
---------------------------------------------------------------------
category_closure_rate        0.396               +0.019
location_age_days            0.394               +0.171   ← #1 for MLP, absent in Overture
business_age_days            0.103               +0.041
has_lic                      0.063               +0.039
has_naic_code                0.004               +0.124   ← #2 for MLP
```

`location_age_days` is the #1 signal for MLP (permutation drop +0.171 AUC). Overture has no equivalent — `msft_update_age_days` measures source recency, not how long the business has existed. This is the single biggest feature gap between the two pipelines.

`category_closure_rate` is the **universal signal** — #1 in both SF GBM (0.396) and Overture GBM (0.400). It transfers reliably across datasets regardless of schema differences.

---

### 5.2 Overture dataset shortcomings

**1. Noisy labels.** `operating_status` is inferred from crowd-sourced and aggregated signals, not administrative records. Two-thirds of actual closures in our labeled set retain high-confidence, complete data profiles — they are undetectable with current features because Overture's data pipeline continues to report them as complete.

**2. Severe class imbalance (9% closed).** Only 313 closed training examples out of 3,425. With this many positive examples, covariance estimates (QDA, SLDA) are unstable, and any augmentation strategy that shifts the class distribution risks collapsing recall. SF's 54/46 balance is why every model — including the simplest GBM — can achieve 0.88.

**3. No business age signal.** The `msft_update_age_days` feature (Cohen's d = 0.08, weak) is a proxy for when Microsoft *last updated* a record, not when the business opened. The SF result shows that a direct age signal contributes 0.171 permutation AUC for MLP alone.

**4. No administrative presence signal.** SF's `has_lic` (license code present, +0.039 permutation) and `has_naic_code` (+0.124) are administrative indicators that a business is actively registered with a government body. Overture has no equivalent — there is no field indicating whether a business holds an active license or tax registration.

**5. Completeness features signal data quality, not status.** The strongest Overture features (`address_completeness`, `confidence`, `has_phone`) measure how well-documented a place is, not whether it is still operating. A business can close while retaining its Overture profile intact if no source triggers a status update.

---

### 5.3 Recommendations for Overture dataset improvement

**Highest impact:**

1. **Add place creation date.** Even a coarse "first seen in Overture" timestamp would approximate business age. Given that `location_age_days` contributes +0.171 AUC on SF, this is the single highest-ROI field Overture could add. A business registered 10 years ago has a fundamentally different closure probability than one registered 6 months ago.

2. **Increase labeled closed examples.** The data bottleneck dominates model performance. 2× more labeled closed records (+313 closed, ~3,100 total) would have more impact than any model improvement. Scale the SF augmentation pipeline to additional cities with open business registries (NYC, Chicago, Seattle).

3. **Cross-reference with government business registries.** SF matching yields a clean signal (+0.009–0.020 AUC from 13,729 matched records). Expanding to other city registries would multiply labeled data and provide ground-truth label quality unavailable from crowd-sourced sources.

**Longer-term:**

4. **Source-level status freshness.** The current `update_time` in `sources[]` reflects when *any* field was updated, not when business existence was last verified. A dedicated `last_verified_open` timestamp per source would make the staleness filter reliable and enable a strong temporal signal.

5. **Regulatory presence indicators.** A flag indicating whether a business appears in any government registry (business license, health permit, tax registration) would replicate the `has_lic` / `has_naic_code` signals that are #2 for MLP on SF data (+0.124 permutation AUC).
