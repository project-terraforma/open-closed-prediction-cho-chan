# Project C — Implementation Plan

> Track progress by replacing `[ ]` with `[x]` as tasks are completed.

---

## Phase 1: Data Pipeline & Feature Engineering (Week 1)

### 1.1 Data Access & Extraction
- [ ] Resolve DuckDB httpfs bug (update DuckDB or use `overturemaps` CLI as fallback)
- [ ] Export labeled ~5k sample with all schema fields + `sources` field to local parquet/csv
- [ ] Export a larger unlabeled sample (e.g., 50k places across categories/geographies) for EDA and later label augmentation
- [ ] Verify exported data has all expected columns: id, name, confidence, categories, basic_category, taxonomy, websites, socials, emails, phones, brand, addresses, sources, operating_status, geometry

### 1.2 Feature Engineering
- [ ] Create `feature_engineering.py` with `extract_features(record) -> feature_vector` function
- [ ] Implement source-level features:
  - [ ] `source_count` — number of distinct source providers
  - [ ] `has_meta`, `has_microsoft`, `has_foursquare` — binary flags per provider
  - [ ] `max_source_confidence`, `min_source_confidence`, `mean_source_confidence`
  - [ ] `confidence_spread` — max minus min source confidence
- [ ] Implement completeness features:
  - [ ] `has_website`, `has_phone`, `has_email`, `has_socials`, `has_brand`
  - [ ] `completeness_score` — fraction of optional fields that are non-null
  - [ ] `phone_count`, `website_count`
- [ ] Implement category & identity features:
  - [ ] `primary_category` — integer-encoded
  - [ ] `category_depth` — depth in taxonomy hierarchy
  - [ ] `has_alternate_categories`
  - [ ] `confidence` — Overture's existing confidence field
- [ ] Implement address features:
  - [ ] `has_address`
  - [ ] `address_completeness` — fraction of address subfields populated
  - [ ] `country` — integer-encoded country code
- [ ] Build a feature matrix from the full labeled sample: `X` (N x ~25-30), `y` (N,)

### 1.3 Exploratory Data Analysis
- [ ] Class distribution: confirm ~90/10 open/closed split
- [ ] Feature distributions: histograms for each numerical feature, value counts for categoricals
- [ ] Correlation matrix across numerical features
- [ ] Class-conditional feature distributions (open vs. closed) — identify which features have signal
- [ ] Missing value analysis — which fields are commonly null?
- [ ] Source analysis: distribution of source_count, per-provider frequency
- [ ] Save EDA results as notebook or report

### 1.4 Train/Val Split
- [ ] Create stratified 80/20 train/val split (preserving open/closed ratio)
- [ ] Create 5-fold stratified CV splits for later use
- [ ] Save splits with fixed random seed for reproducibility

---

## Phase 2: MLP Encoder + NCM/SLDA (Week 2)

### 2.1 MLP Encoder Implementation
- [ ] Install PyTorch
- [ ] Create `encoder.py` with `PlaceEncoder` class:
  - [ ] Embedding layers for categorical features (category: 8-dim, country: 4-dim)
  - [ ] BatchNorm for numerical features
  - [ ] Linear(~30 → 64) → BatchNorm → ReLU → Dropout(0.3)
  - [ ] Linear(64 → 32) → BatchNorm → ReLU
  - [ ] Classification head: Linear(32 → 2) for training only
- [ ] Create `dataset.py` with PyTorch Dataset class for place features
- [ ] Implement weighted cross-entropy loss (closed class weight ~9x)

### 2.2 MLP Encoder Training
- [ ] Train encoder end-to-end with classification head
- [ ] Optimizer: Adam, lr=1e-3, cosine annealing scheduler
- [ ] Early stopping on validation loss (patience=10)
- [ ] Run 5-fold stratified CV
- [ ] Log training curves (loss, AUC per epoch)
- [ ] Save best encoder weights

### 2.3 NCM Implementation
- [ ] Create `ncm.py` with `NearestClassMean` class:
  - [ ] `fit(embeddings, labels)` — compute class means
  - [ ] `predict_proba(embeddings)` — softmax over negative distances
  - [ ] `update(new_embeddings, new_labels)` — incremental mean update
- [ ] Extract embeddings from frozen encoder for all labeled places
- [ ] Fit NCM on training embeddings
- [ ] Evaluate NCM on validation set

### 2.4 SLDA Implementation
- [ ] Create `slda.py` with `StreamingLDA` class:
  - [ ] `fit(embeddings, labels)` — compute class means + shared covariance
  - [ ] `predict_proba(embeddings)` — LDA decision rule → sigmoid
  - [ ] `update(new_embeddings, new_labels)` — incremental mean + covariance update (Welford's)
- [ ] Fit SLDA on training embeddings
- [ ] Evaluate SLDA on validation set

### 2.5 Encoder Evaluation
- [ ] Evaluate MLP+NCM and MLP+SLDA on validation set:
  - [ ] AUC-ROC (target: > 0.80)
  - [ ] AUC-PR (target: > 0.50)
  - [ ] F1 for closed class (target: > 0.40)
  - [ ] ECE (target: < 0.05)
- [ ] Generate confusion matrices
- [ ] Per-category AUC breakdown
- [ ] Per-source-count AUC breakdown
- [ ] NCM vs. SLDA comparison — does covariance structure help?

### 2.6 Embedding Analysis
- [ ] t-SNE or UMAP visualization of encoder embeddings colored by open/closed
- [ ] Silhouette score for cluster separation
- [ ] Nearest-neighbor purity (5-NN, same-label fraction)
- [ ] Save visualizations

---

## Phase 3: Continual Learning Evaluation & Refinement (Week 3)

### 3.1 Simulated Continual Learning
- [ ] Split labeled data: 70% initial train, 30% simulated "new release"
- [ ] Train encoder + NCM/SLDA on initial 70%
- [ ] Incrementally update NCM/SLDA with 30% new data
- [ ] Compare incremental update vs. full retrain from scratch
- [ ] Verify: incremental NCM/SLDA matches full retrain (expected for these methods)
- [ ] Document CL results

### 3.2 Model Refinement
- [ ] If embeddings don't separate well: experiment with encoder depth/width
- [ ] If NCM/SLDA underperform: try Mahalanobis distance for NCM, regularized covariance for SLDA
- [ ] If calibration is poor: apply temperature scaling as post-hoc fix
- [ ] Re-run evaluation after refinements

### 3.3 LLM Label Augmentation (Optional)
- [ ] Sample unlabeled places from full Overture release (focus on ambiguous confidence 0.4-0.7)
- [ ] Design prompt template for LLM annotation
- [ ] Validate LLM accuracy: run LLM on the known 5k sample, compare against ground truth
- [ ] If LLM accuracy is acceptable (> 80% agreement):
  - [ ] Generate LLM labels for unlabeled sample
  - [ ] Add to training set with 0.5x weight
  - [ ] Retrain encoder + NCM/SLDA with augmented labels
  - [ ] Compare augmented vs. original performance
- [ ] If LLM accuracy is poor: skip augmentation, document findings

---

## Phase 4: Final Evaluation & Write-Up (Week 4)

### 4.1 Final Model Selection
- [ ] Select best classifier (NCM vs. SLDA) based on evaluation metrics
- [ ] Run final evaluation on held-out test set (not used during any tuning)
- [ ] Generate final metrics table: AUC-ROC, AUC-PR, F1-closed, ECE

### 4.2 Production Scoring
- [ ] Score full Overture release with selected model
- [ ] Analyze score distribution (histogram)
- [ ] Spot-check: manually inspect 50 high-confidence-open and 50 high-confidence-closed places
- [ ] Verify inference speed: benchmark per-place latency

### 4.3 Write-Up & Deliverables
- [ ] Final report with model results and CL advantage
- [ ] Document continual learning advantage with simulated results
- [ ] Feature importance analysis (what drives open/closed?)
- [ ] Recommendations for production deployment
- [ ] Clean up code, add docstrings to main modules
- [ ] Push final code and report to repo

---

## Optional: GBM Baseline Comparison

> If time permits, train a GBM baseline to benchmark our approach against the industry standard for tabular data.

### GBM Training
- [ ] Install xgboost (or lightgbm)
- [ ] Train XGBoost classifier on engineered features with `scale_pos_weight=9`
- [ ] Tune hyperparameters: `n_estimators`, `max_depth`, `learning_rate`
- [ ] Run 5-fold stratified cross-validation

### GBM Evaluation
- [ ] Compute AUC-ROC, AUC-PR, F1-closed, ECE
- [ ] Generate confusion matrix and reliability diagram

### GBM Interpretability
- [ ] Feature importance (built-in gain/weight)
- [ ] SHAP values analysis
- [ ] Per-category and per-source-count AUC breakdown

### Head-to-Head: MLP+CL vs. GBM
- [ ] Compare NCM, SLDA, GBM on same val set across all metrics
- [ ] Document where CL approach wins (adaptability) and where GBM wins (raw accuracy)

### Optional: Yelp Matching Pipeline (US-only)
- [ ] Load Yelp Academic Dataset
- [ ] Implement fuzzy name matching (Jaro-Winkler > 0.85) + geographic proximity (< 100m)
- [ ] Match Overture places to Yelp businesses
- [ ] Extract Yelp features: `yelp_is_open`, `yelp_stars`, `yelp_review_count`
- [ ] Add `yelp_matched` flag and Yelp features to feature matrix
- [ ] Retrain models with Yelp features and compare

---

## Deliverables Checklist

| Deliverable | Status |
|---|---|
| `feature_engineering.py` — feature extraction from Overture records | [ ] |
| `encoder.py` — MLP encoder (PyTorch) | [ ] |
| `ncm.py` — Nearest Class Mean classifier | [ ] |
| `slda.py` — Streaming LDA classifier | [ ] |
| `train.py` — end-to-end training pipeline | [ ] |
| `evaluate.py` — evaluation metrics and plots | [ ] |
| EDA notebook/report | [ ] |
| Embedding visualizations (t-SNE/UMAP) | [ ] |
| CL simulation results | [ ] |
| Final report | [ ] |
| Scored full release (parquet with confidence scores) | [ ] |
