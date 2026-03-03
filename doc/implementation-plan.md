# Project C — Implementation Plan

> Track progress by replacing `[ ]` with `[x]` as tasks are completed.

---

## Dataset Status (confirmed)

| | Value |
|---|---|
| **Labeled dataset** | `data/project_c_samples.json` — 3,425 records |
| **Closed (open=0)** | 313 (9.1%) |
| **Open (open=1)** | 3,112 (90.9%) |
| **Geography** | US-only |
| **Sources present** | `meta`, `Microsoft` (no Foursquare in this split) |
| **Meta update_time** | Always `2025-02-24` (batch date, not freshness) |
| **Microsoft update_time** | Real timestamps (2013–2024) — usable as staleness signal |
| **Full Overture release** | 72.9M places, 785 closed + 20 temp closed in `operating_status` |
| **Cross-release** | Jan→Feb: 445k removed (dedup/source drop, NOT closures), 0 status changes |

---

## Phase 1: Data Pipeline & Feature Engineering ✓

### 1.1 Data Access & Extraction
- [x] Identify labeled dataset: `project_c_samples.json` (3,425 records, binary `open` label)
- [x] Confirm schema: sources, confidence, categories, websites, socials, phones, addresses, geometry
- [x] Analyze cross-release diff (Jan vs. Feb 2026) — confirmed operating_status is static
- [x] Confirm removed places are not closures (dedup/source drops, all labeled "open")

### 1.2 Feature Engineering
- [x] Create `src/feature_engineering.py` with `extract_features(record) -> dict`
- [x] 20 features across 4 groups: source (8), completeness (6), category (3), confidence+address (3)
- [x] Dropped `has_address` (constant, all 1.0) and `has_email` (all-zero, d=0.000)
- [x] Smoke test passed — 3,425 records, 20 features, zero nulls

### 1.3 Exploratory Data Analysis
- [x] Class distribution: 313 closed / 3,112 open (9.1% / 90.9%)
- [x] Top signals: address_completeness (d=0.82), confidence (d=0.64), has_phone (d=0.54)
- [x] Microsoft staleness: closed places have 2.5x older Microsoft timestamps (1,322 vs 518 days)
- [x] Findings documented in `memory/MEMORY.md` and `doc/progress-report.md`

### 1.4 Train/Val Split
- [x] `src/split.py` — stratified 80/20, seed=42, LabelEncoder for primary_category, saves to `splits/`
- [x] Train: 2,740 (closed=250, open=2,490) | Val: 685 (closed=63, open=622)

---

## Phase 2: MLP Encoder + NCM/SLDA ✓

### 2.1 MLP Encoder
- [x] `src/encoder.py` — `PlaceEncoder`, `PlaceDataset`, `class_weights`, `load_splits`
- [x] Architecture: 19 numeric → BN | category → Embedding(294+1, 8) | concat(27) → Linear(64) → BN → ReLU → Dropout(0.3) → Linear(32) → BN → ReLU → head(2)
- [x] 6,528 params | class weights: closed=5.48x, open=0.55x
- [x] Smoke test passed

### 2.2 MLP Encoder Training
- [x] `src/train.py` — Adam lr=1e-3 + weight_decay=1e-4, cosine LR, early stopping on val AUC (patience=20)
- [x] Best run: epoch 11, val_loss=0.608, val_auc=0.734
- [x] Extracts 32-dim embeddings → `models/embeddings_train.npy`, `models/embeddings_val.npy`
- [x] Saves `models/encoder.pt`, `models/encoder_config.json`, `models/train_log.json`

### 2.3 NCM Implementation
- [x] `src/ncm.py` — `fit()`, `update()` (online mean formula), `predict_proba()` (softmax over neg distances)
- [x] Smoke test: val acc=1.000, incremental diff=0.000000
- [x] Fitted on training embeddings → `models/ncm.pkl`

### 2.4 SLDA Implementation
- [x] `src/slda.py` — `fit()`, `update()` (parallel Welford), `predict_proba()` (Mahalanobis + log prior)
- [x] Smoke test: val acc=1.000, scatter diff≈0, condition number=6.2
- [x] Fitted on training embeddings → `models/slda.pkl`

### 2.5 GBM / XGBoost Baseline
- [x] `src/gbm.py` — OHE for primary_category (295 dummies), balanced sample weights
- [x] Trains sklearn GBM and XGBoost, saves `models/gbm.pkl`, `models/xgb.pkl`, `models/ohe.pkl`
- [x] XGBoost instaled: `pip install xgboost`

### 2.6 Evaluation
- [x] `src/evaluate.py` — loads all models, prints comparison table
- [x] Results (val set, closed as positive):

```
Model              AUC-ROC  AUC-PR    F1   Prec  Recall
--------------------------------------------------------
GBM + OHE           0.686   0.255  0.277  0.173   0.698
XGBoost + OHE       0.720   0.293  0.311  0.206   0.635
MLP + NCM           0.721   0.243  0.302  0.218   0.492
MLP + SLDA          0.726   0.253  0.308  0.235   0.444
MLP head            0.734   0.263  0.315  0.313   0.318   <- best
```

- [x] Key finding: MLP beats both tree models — learned category embedding outperforms OHE

---

## Phase 3: Continual Learning Evaluation ✓

### 3.1 Simulated Release Update
- [x] `src/cl_eval.py` — splits train into Release 0 (70%) + Release 1 (30%)
- [x] Fits NCM/SLDA on R0, calls `update()` with R1 (encoder frozen throughout)
- [x] Verified: incremental update == full fit (mean diff < 1e-6)

### 3.2 Results

```
            R0 only   After update   Full fit   update == full fit?
NCM          0.7231       0.7213      0.7213     YES  (diff < 1e-6)
SLDA         0.7332       0.7260      0.7260     YES  (diff < 1e-6)
```

- [x] AUC shift is within statistical noise (63 val closed examples)
- [x] Speed: NCM update 0.061ms | SLDA 0.149ms | XGBoost retrain 184.3ms (3,021x slower)

### 3.3 Model Refinement (done during Phase 2)
- [x] Fixed early stopping to monitor val AUC instead of val loss
- [x] Added weight_decay=1e-4 to Adam to reduce overfitting
- [x] Raised patience from 10 → 20

---

## Phase 4: Final Evaluation & Write-Up

### 4.1 Final Report
- [ ] Update `doc/progress-report.md` for final submission (done incrementally as log)
- [ ] Feature importance writeup (EDA findings → model signal)
- [ ] Recommendations for production deployment

### 4.2 Production Scoring (optional)
- [ ] Score full Overture release with selected model (MLP head or MLP+SLDA)
- [ ] Analyze score distribution
- [ ] Benchmark per-place inference latency at scale

### 4.3 Extensions (optional)
- [ ] t-SNE / UMAP of encoder embeddings colored by open/closed
- [ ] LLM label augmentation for unlabeled Overture places
- [ ] Yelp matching pipeline for US places

---

## Deliverables Checklist

| Deliverable | Status |
|---|---|
| `src/feature_engineering.py` — 20 features from Overture JSON | [x] |
| `src/encoder.py` — MLP encoder (PyTorch, 6,528 params) | [x] |
| `src/ncm.py` — Nearest Class Mean, incremental update | [x] |
| `src/slda.py` — Streaming LDA, Welford covariance update | [x] |
| `src/train.py` — end-to-end training pipeline | [x] |
| `src/gbm.py` — GBM + XGBoost baseline with OHE | [x] |
| `src/evaluate.py` — full model comparison table | [x] |
| `src/cl_eval.py` — CL simulation + speed benchmark | [x] |
| `doc/progress-report.md` — dated team log | [x] |
| CL simulation results (update == full fit, 3,021x speedup) | [x] |
| Scored full release (parquet with confidence scores) | [ ] |
| Final report | [ ] |
