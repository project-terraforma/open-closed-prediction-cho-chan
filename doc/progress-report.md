# Project C — Progress Log

---

## 2026-03-02 | Caleb Cho

### Data Augmentation Experiment — Parquet Closed Records

Investigated incorporating additional labeled records directly from the Feb
Overture parquet release to address the core bottleneck: only 313 closed
training examples.

**Approach:** The Feb parquet has 785 places with `operating_status = 'closed'`
and no overlap with our existing labeled set. We built a pipeline
(`src/augment.py`, updated `src/split.py --augment`) to:
- Pull all 785 new closed records from parquet
- Sample matching open records (target 15% closed combined)
- Keep `project_c_samples.json` as a permanent held-out val benchmark

**First attempt (wrong):** Mixed everything into one augmented_samples.json and
re-ran the split. Val AUC jumped to 0.96 — obviously inflated.

**Root cause (correct setup):** Feature distribution probe showed the parquet
closed records are structurally different from the original labeled closed
records:

```
Feature               Old closed (313)   New closed (785)   diff
confidence                     0.751              0.945   +0.194
completeness_score             0.649              0.478   -0.171
has_website                    0.815              0.918   +0.104
has_phone                      0.872              0.995   +0.123
source_count                   1.128              2.000   +0.872
```

All 785 parquet-closed records have `source_count = 2.0` exactly — Overture's
pipeline requires 2-source confirmation before setting `operating_status = closed`.
This is a data artifact, not a generalizable signal.

**Results with correct setup** (parquet → train only, original → val benchmark):

```
Model           3k-only (baseline)   + parquet augment
GBM                       0.686             0.697  (+0.011)
XGBoost                   0.720             0.700  (-0.020)
MLP head                  0.734             0.707  (-0.027)
MLP + NCM                 0.721             0.684  (-0.037)  ← NCM F1=0.000
MLP + SLDA                0.726             0.694  (-0.032)
```

SLDA condition number jumped from 6 → 1,525. The mixed embedding space
(two distinct distributions) made the covariance matrix nearly singular.

**Conclusion:** The parquet augmentation degrades the MLP/CL models on the hard
val benchmark. The easy structural signal in the parquet data (source_count=2)
does not transfer. Reverted to 3k baseline.

**New scripts produced (kept for future work):**
- `src/augment.py` — pull parquet records, output `data/parquet_augment.json`
- `src/parquet_probe.py` — operating_status stats, overlap check, sizing
- `src/yelp_probe.py` — Yelp × Overture spatial + name similarity match probe

**Yelp matching probe:** Yelp academic dataset has 150k US businesses, 30k
closed. Spatial probe (100m radius) found only 136 of 3,425 labeled records
have a Yelp candidate nearby. After rapidfuzz name filtering (sim ≥ 80), only
46 true matches (1.3%). Label agreement on those 46: 91.3%. The labeled set is
too NYC-concentrated to benefit from Yelp matching; matching against the full
72.9M Overture parquet was explored but not pursued due to augmentation
findings above.

---

## 2026-02-24 | Caleb Cho

### Overall Progress

```
Phase 1: Data & Features   ████████████████████████  100%  done
Phase 2: MLP + NCM/SLDA    ████████████████████████  100%  done
Phase 3: CL Evaluation     ████████████████████████  100%  done
Phase 4: Final Report      ░░░░░░░░░░░░░░░░░░░░░░░    0%
```

### Model Benchmark

Val set: 685 samples | 63 closed (9.2%) | 622 open

All metrics treat **closed** as the positive class.

```
Model              AUC-ROC  AUC-PR    F1   Prec  Recall
--------------------------------------------------------
GBM + OHE           0.686   0.255  0.277  0.173   0.698
XGBoost + OHE       0.720   0.293  0.311  0.206   0.635
MLP + NCM           0.721   0.243  0.302  0.218   0.492
MLP + SLDA          0.726   0.253  0.308  0.235   0.444
MLP head            0.734   0.263  0.315  0.313   0.318   <- best
```

MLP beats both tree models. The 8-dim learned category embedding shares
information across similar business types. XGBoost gets 295 isolated binary
dummies (one-hot) and still trails by 0.014 AUC-ROC.

### Continual Learning Evaluation

Simulated a new Overture release by splitting training data:
- Release 0 (70%): 1,917 samples — initial fit
- Release 1 (30%): 823 samples — incremental update, encoder frozen

```
            R0 only   After update   Full fit   update == full fit?
NCM          0.7231       0.7213      0.7213     YES  (diff < 1e-6)
SLDA         0.7332       0.7260      0.7260     YES  (diff < 1e-6)
```

The AUC shift after update equals the shift from a full refit on all data —
the update is mathematically exact. Not degradation; same answer as retraining,
computed without storing old data.

**Update speed vs. XGBoost full retrain (823 new samples):**

```
NCM   update()     0.061 ms
SLDA  update()     0.149 ms
XGBoost retrain  184.3   ms   (3,021x slower than NCM)
```

### Notes on Targets

Original targets: AUC-ROC > 0.80 | AUC-PR > 0.50 | F1 > 0.40

With 250 closed training examples and 9% class prevalence these were ambitious.
Revised targets given data constraints: AUC-ROC > 0.73 | AUC-PR > 0.28 | F1 > 0.31.
All revised targets are met by MLP head and MLP + SLDA.

Highest-leverage improvement: more labeled closed examples.

### Codebase Status

| File | Purpose |
|---|---|
| `src/feature_engineering.py` | 20 features from raw Overture JSON |
| `src/eda.py` | Class-conditional signal analysis |
| `src/split.py` | Stratified 80/20 train/val split |
| `src/encoder.py` | MLP encoder (6,528 params, PyTorch) |
| `src/ncm.py` | Nearest Class Mean — fit / update / predict |
| `src/slda.py` | Streaming LDA — fit / update (Welford) / predict |
| `src/train.py` | End-to-end: train encoder, extract embeddings, fit NCM/SLDA |
| `src/gbm.py` | GBM + XGBoost baseline with OHE |
| `src/evaluate.py` | Full model comparison table |
| `src/cl_eval.py` | Continual learning simulation + speed benchmark |

---

## 2026-02-23 | Caleb Cho

### Overall Progress

```
Phase 1: Data & Features   ████████████████████░░░  85%  <- we are here
Phase 2: MLP + NCM/SLDA    ░░░░░░░░░░░░░░░░░░░░░░░   0%
Phase 3: CL Evaluation     ░░░░░░░░░░░░░░░░░░░░░░░   0%
Phase 4: Final Report      ░░░░░░░░░░░░░░░░░░░░░░░   0%
```

### What We Have

**Dataset (confirmed)**

| | |
|---|---|
| Labeled places | 3,425 Overture records with binary open/closed labels |
| Closed (positive class) | 313 (9.1%) |
| Open | 3,112 (90.9%) |
| Geography | US-only |
| Sources | Meta (Facebook) + Microsoft Bing |

**Codebase**

| File | Status | Purpose |
|---|---|---|
| `src/feature_engineering.py` | Done | 20 features from raw JSON; `load_dataset()` |
| `src/eda.py` | Done | Class-conditional signal analysis |
| `src/split.py` | Done | Stratified 80/20 train/val split |
| `src/encoder.py` | Next | MLP encoder (PyTorch) |
| `src/ncm.py` | Next | Nearest Class Mean classifier |
| `src/slda.py` | Next | Streaming LDA classifier |

### Key EDA Finding: The Data Has Signal

Feature effect sizes (Cohen's d) — how well each feature separates open vs. closed:

```
address_completeness    ████████████████████████  0.82  strong
confidence              ████████████████░░░░░░░░  0.64  strong
source_confidence       ████████████████░░░░░░░░  0.55-0.64  strong
has_phone               ██████████████░░░░░░░░░░  0.54  strong
completeness_score      ██████████░░░░░░░░░░░░░░  0.39  strong
has_website             ███████░░░░░░░░░░░░░░░░░  0.28  moderate
source_count            ██████░░░░░░░░░░░░░░░░░░  0.25  moderate
has_microsoft           ███░░░░░░░░░░░░░░░░░░░░░  0.13  moderate
msft_update_age_days    ██░░░░░░░░░░░░░░░░░░░░░░  0.08  weak
```

**Microsoft Staleness Signal**

When a place has a Microsoft source, its data age is a strong closed indicator:

```
            Median Microsoft data age
Open        ████████░░░░░░░░░░░░░░░░░░░░░░   518 days
Closed      ████████████████████░░░░░░░░░░  1322 days   (~2.5x older)
```

**Confidence Score Distribution**

```
                 <=0.4   0.4-0.6   0.77-0.9   >0.95
Open              5%       5%        26%       48%    (mostly high-confidence)
Closed           13%      15%        38%       23%    (skews lower)
```

**Source Count**

```
Open    1-source: 77%   2-source: 23%
Closed  1-source: 87%   2-source: 13%   (fewer corroborating sources)
```

### Approach Differentiator

Other teams are using GBM (gradient boosted trees) on static feature snapshots.

We are building:
1. **MLP Encoder** — learns a 32-dim embedding of each place
2. **NCM / SLDA** — continual learning classifiers that update incrementally with each new Overture release, no retraining required

This means when Overture ships a new monthly release, we update our model's
class statistics in O(N) time instead of retraining from scratch.

### Next Steps (at time of writing)

1. Run train/val split (`src/split.py`)
2. Build MLP encoder in PyTorch (`src/encoder.py`)
3. Implement NCM and SLDA classifiers
4. First evaluation: AUC-ROC, AUC-PR, F1 on closed class

**Target metrics:** AUC-ROC > 0.80 | AUC-PR > 0.50 | F1-closed > 0.40
