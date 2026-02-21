# Project C — Open & Closed Predictions: MLP Encoder with Continual Learning

> **Approach C:** MLP Encoder + NCM/SLDA Continual Learning with GBM Baseline
>
> Overture Maps Foundation · February 2026
>
> Caleb Cho, Kevin Chan

---

## Table of Contents

1. [Problem Context & Data Constraints](#1-problem-context--data-constraints)
2. [Why Not Approaches A & B](#2-why-not-approaches-a--b)
3. [Feature Engineering](#3-feature-engineering)
4. [Architecture Overview](#4-architecture-overview)
5. [Component 1 — MLP Encoder](#5-component-1--mlp-encoder)
6. [Component 2 — Continual Learning Classifiers (NCM / SLDA)](#6-component-2--continual-learning-classifiers-ncm--slda)
7. [Component 3 — GBM Baseline](#7-component-3--gbm-baseline)
8. [Label Augmentation Strategy](#8-label-augmentation-strategy)
9. [Training & Inference Pipeline](#9-training--inference-pipeline)
10. [Evaluation Plan](#10-evaluation-plan)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [Implementation Timeline](#12-implementation-timeline)

---

## 1. Problem Context & Data Constraints

The Overture Maps places dataset requires a confidence score for place existence (open vs. closed) published as a first-class schema field. The score must satisfy:

| Constraint | Requirement | Implication |
|---|---|---|
| Generalization | Works across all categories and geographies | Cannot hard-code rules per category or country |
| Scale | Scores 100M+ places per release | Sub-millisecond inference per record; no external API calls at runtime |
| Stability | Scores do not fluctuate wildly across releases | Model must be robust to minor data distribution shifts |
| Interpretability | Justifiable as a public field | Consumers need to understand what the score means |
| Robustness | Tolerates upstream source churn | No single-source dependency; graceful degradation when sources drop |
| Adaptability | Improves as new releases arrive | Must incorporate new data without full retraining or catastrophic forgetting |

### What We Have

- **~5k labeled places** with `operating_status` (open / permanently_closed / temporarily_closed), labeled by an external provider using agents + web search (noisy ground truth)
- **~90/10 open/closed class split** reflecting production distribution
- **Single Overture release snapshot** (2026-01-21.0) — no multi-release version history
- **Three upstream data sources** visible in the `sources` field: Meta (Facebook), Microsoft (Bing), Foursquare — each with dataset name, record ID, and confidence score
- **Yelp Academic Dataset** as external enrichment (US-only, has `is_open`, `stars`, `review_count`, `attributes`, `hours`)
- **No access** to the external provider's upstream signals (last-verified date, verification flags, etc.)

### What We Do Not Have

- Individual per-source records (we see aggregated place records with source metadata, not raw source-level records)
- Multi-release version history (cannot track places across releases)
- The external provider's labeling model or features

---

## 2. Why Not Approaches A & B

The original proposal document described two approaches. After investigating the actual data available, both were found to be infeasible:

| Approach | Core Requirement | Why It Fails |
|---|---|---|
| **A: Contrastive Embeddings** | Needs individual source records per place to construct positive/negative pairs for contrastive pre-training | We only see aggregated place records with source metadata (provider name, confidence). We cannot access the raw records from Meta, Microsoft, or Foursquare separately. |
| **B: Survival Model** | Needs release-over-release version history to construct temporal person-period dataset and estimate closure timing | We have a single release snapshot. There is no version history or multi-release data to model time-to-event. |

This proposal presents **Approach C**, designed around the data we actually have. It differentiates from the GBM approaches other teams are exploring by introducing a neural encoder with continual learning capabilities.

---

## 3. Feature Engineering

All features are derived from a single Overture release snapshot. Features are organized by what they measure.

### 3.1 Source-Level Features (from `sources` field)

Each place's `sources` field contains a list of contributing data providers with metadata. We extract:

| Feature | Type | Description |
|---|---|---|
| `source_count` | Numerical | Number of distinct source providers (max observed: 3 — Meta, Microsoft, Foursquare) |
| `has_meta` | Binary | Whether Meta (Facebook) is a contributing source |
| `has_microsoft` | Binary | Whether Microsoft (Bing) is a contributing source |
| `has_foursquare` | Binary | Whether Foursquare is a contributing source |
| `max_source_confidence` | Numerical | Highest confidence score among all sources |
| `min_source_confidence` | Numerical | Lowest confidence score among all sources |
| `mean_source_confidence` | Numerical | Average confidence score across sources |
| `confidence_spread` | Numerical | max - min source confidence (source disagreement signal) |

### 3.2 Completeness Features (from record attributes)

These measure how "filled in" the place record is. The intuition: well-maintained open businesses tend to have more complete records.

| Feature | Type | Description |
|---|---|---|
| `has_website` | Binary | Whether the place has a website URL |
| `has_phone` | Binary | Whether the place has a phone number |
| `has_email` | Binary | Whether the place has an email address |
| `has_socials` | Binary | Whether the place has social media URLs |
| `has_brand` | Binary | Whether the place has brand information |
| `has_hours` | Binary | Whether the place has operating hours (if available) |
| `completeness_score` | Numerical | Fraction of optional fields that are non-null (0.0 to 1.0) |
| `phone_count` | Numerical | Number of phone numbers listed |
| `website_count` | Numerical | Number of websites listed |

### 3.3 Category & Identity Features

| Feature | Type | Description |
|---|---|---|
| `primary_category` | Categorical | Primary category (e.g., "restaurant", "retail") — encoded as integer index |
| `category_depth` | Numerical | Depth in taxonomy hierarchy (more specific = deeper) |
| `has_alternate_categories` | Binary | Whether alternate categories exist |
| `confidence` | Numerical | Overture's existing confidence field (current heuristic score) |

### 3.4 Address Features

| Feature | Type | Description |
|---|---|---|
| `has_address` | Binary | Whether any address is present |
| `address_completeness` | Numerical | Fraction of address subfields populated (freeform, locality, region, country, postcode) |
| `country` | Categorical | Country code — encoded as integer index |

### 3.5 Optional: Yelp Enrichment Features (US-only)

If a place can be matched to the Yelp Academic Dataset (by name + location proximity):

| Feature | Type | Description |
|---|---|---|
| `yelp_matched` | Binary | Whether a Yelp match was found |
| `yelp_is_open` | Binary | Yelp's own open/closed label |
| `yelp_stars` | Numerical | Star rating (1-5) |
| `yelp_review_count` | Numerical | Number of reviews |

> **Note:** Yelp features are US-only. The model must work without them. They are included as optional enrichment that the model can learn to use when available and ignore when absent (set to 0/NaN with a `yelp_matched=0` flag).

### Total Feature Count: ~25-30 dimensions

---

## 4. Architecture Overview

The approach has three components working together:

```
                    ┌─────────────────────────────────────┐
                    │         Raw Place Record            │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      Feature Engineering            │
                    │  (extract ~25-30 dim feature vec)   │
                    └──────────────┬──────────────────────┘
                                   │
                         feature vector x
                                   │
                    ┌──────────────┴──────────────────────┐
                    │                                      │
                    ▼                                      ▼
        ┌───────────────────┐               ┌──────────────────────┐
        │   MLP Encoder     │               │     GBM Baseline     │
        │  (THE model)      │               │   (XGBoost/LightGBM) │
        │                   │               │                      │
        │  x → 64 → 32     │               │  x → open/closed     │
        │  (learned repr)   │               │  (direct prediction) │
        └────────┬──────────┘               └──────────┬───────────┘
                 │                                      │
            embedding z                            P(open|x)
                 │
        ┌────────┴──────────┐
        │                    │
        ▼                    ▼
  ┌───────────┐      ┌───────────┐
  │    NCM    │      │   SLDA    │
  │ Nearest   │      │ Streaming │
  │ Class     │      │ LDA      │
  │ Mean      │      │          │
  └─────┬─────┘      └─────┬────┘
        │                    │
   P(open|z)            P(open|z)
        │                    │
        └────────┬───────────┘
                 │
        Compare all three
        classifiers on eval
```

**Why this design:**
- The **MLP encoder** learns nonlinear feature interactions that raw tabular features miss (e.g., a restaurant with 1 source and no phone is different from a hospital with 1 source and no phone)
- **NCM/SLDA** on top of the encoder enables continual learning — when new releases arrive with new/updated labels, the classifier adapts without retraining the encoder and without catastrophic forgetting
- The **GBM baseline** provides a benchmark and is the industry standard for tabular data — it should be beaten or at least matched to justify the neural approach
- **Continual learning** is the key differentiator from other teams' GBM approaches

---

## 5. Component 1 — MLP Encoder

### 5.1 What It Is

A small feedforward neural network that transforms the raw feature vector into a learned embedding. The encoder IS the model — it captures the patterns that distinguish open from closed places.

### 5.2 Architecture

```
Input (25-30 dim)
    │
    ├── Categorical features → Embedding layers
    │     category: vocab_size → 8-dim embedding
    │     country:  vocab_size → 4-dim embedding
    │
    ├── Numerical features → BatchNorm / quantile normalization
    │
    └── Concatenate all → ~30 dim
                │
                ▼
        ┌───────────────┐
        │  Linear(30→64) │
        │  BatchNorm     │
        │  ReLU          │
        │  Dropout(0.3)  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  Linear(64→32) │
        │  BatchNorm     │
        │  ReLU          │
        └───────┬───────┘
                │
                ▼
          32-dim embedding z
```

### 5.3 Training the Encoder

The encoder is trained end-to-end with a classification head attached:

```
32-dim embedding → Linear(32→2) → softmax → cross-entropy loss
```

Training details:
- **Data:** ~5k labeled places, stratified 80/20 train/val split (maintaining 90/10 open/closed ratio)
- **Class imbalance handling:** Weighted cross-entropy loss — closed class gets ~9x weight
- **Optimizer:** Adam, lr=1e-3 with cosine annealing
- **Regularization:** Dropout(0.3) on first hidden layer, early stopping on val loss
- **Epochs:** ~50-100 with early stopping (patience=10)
- **Validation:** 5-fold stratified cross-validation to estimate generalization

After training, the classification head is discarded. The encoder's job is done — it has learned to map place features into a 32-dim space where open and closed places are separable.

### 5.4 Parameter Count

| Layer | Parameters |
|---|---|
| Category embedding | ~200 × 8 = 1,600 |
| Country embedding | ~50 × 4 = 200 |
| Linear(30→64) + bias | 30 × 64 + 64 = 1,984 |
| BatchNorm(64) | 128 |
| Linear(64→32) + bias | 64 × 32 + 32 = 2,080 |
| BatchNorm(32) | 64 |
| **Total** | **~6,056** |

This is deliberately tiny. A model this small is nearly impossible to overfit on 5k examples (especially with dropout and early stopping), and inference is a single matrix multiply — trivially sub-millisecond for 100M+ places.

### 5.5 Why Not Just Use the Classification Head?

The classification head (Linear(32→2)) would give a direct prediction. But it has two problems:
1. **No continual learning** — when new data arrives, you retrain from scratch and risk catastrophic forgetting
2. **No adaptability** — the decision boundary is fixed

By extracting the 32-dim embedding and using NCM/SLDA on top, we get a classifier that can be updated incrementally with zero risk of forgetting.

---

## 6. Component 2 — Continual Learning Classifiers (NCM / SLDA)

### 6.1 The Core Idea

Continual learning (CL) is a **training strategy**, not a model. The model is the MLP encoder. CL addresses this problem: Overture releases new data every few months. The source ecosystem changes. New places appear, old ones close. How do we update the model without:
- Retraining from scratch every time (expensive, wasteful)
- Forgetting what the model already learned (catastrophic forgetting)

NCM and SLDA are CL methods that operate on the encoder's embeddings. They maintain running statistics that can be updated incrementally.

### 6.2 NCM — Nearest Class Mean

**How it works:**

1. Pass all labeled places through the frozen encoder to get 32-dim embeddings
2. Compute the mean embedding for each class:
   - `mu_open = mean(z_i for all open places)`
   - `mu_closed = mean(z_i for all closed places)`
3. At inference, classify a new place by which class mean its embedding is closest to (Euclidean or Mahalanobis distance)

**Continual update:** When new labeled data arrives (new release, new labels from LLM augmentation, etc.):
- Update `mu_open` and `mu_closed` as running means: `mu_new = (n * mu_old + sum(z_new)) / (n + n_new)`
- No retraining. No gradient computation. Just arithmetic.

**Scoring:** Convert distance to probability via softmax over negative distances:
```
P(open|z) = exp(-d(z, mu_open)) / (exp(-d(z, mu_open)) + exp(-d(z, mu_closed)))
```

### 6.3 SLDA — Streaming Linear Discriminant Analysis

**How it works:**

1. Pass all labeled places through the frozen encoder to get 32-dim embeddings
2. Compute per-class means AND a shared covariance matrix across all classes
3. Classify using the LDA decision rule (linear boundary in embedding space)

**Advantage over NCM:** SLDA uses the covariance structure of the embeddings, not just the means. If some embedding dimensions are more discriminative than others (likely), SLDA captures this. It's the Mahalanobis distance version of NCM.

**Continual update:** Both the class means and the covariance matrix can be updated incrementally using Welford's online algorithm:
```
# Update mean
mu_new = mu_old + (z_new - mu_old) / n

# Update covariance (rank-1 update)
S_new = S_old + (z_new - mu_old)(z_new - mu_new)^T
```

**Scoring:** LDA gives log-odds directly, which convert to probability via sigmoid.

### 6.4 NCM vs. SLDA

| Dimension | NCM | SLDA |
|---|---|---|
| Complexity | Trivial — store 2 mean vectors (64 floats) | Moderate — store 2 mean vectors + 32x32 covariance (1,088 floats) |
| Discriminative power | Lower — treats all embedding dims equally | Higher — weights dimensions by discriminative power |
| Update cost | O(d) per new sample | O(d^2) per new sample |
| When to prefer | Quick baseline, very few labeled samples | More labeled data available, embedding dims have varying importance |

**Recommendation:** Implement both. NCM is the simpler starting point. SLDA should outperform NCM once the encoder produces good embeddings. Compare empirically.

### 6.5 When to Update vs. When to Retrain

The continual learning pipeline operates at two timescales:

| Trigger | Action | What Changes |
|---|---|---|
| New labeled data within same release | Update NCM/SLDA statistics | Class means and covariance |
| New Overture release (quarterly) | Re-extract features, re-run encoder, update NCM/SLDA | Feature vectors, embeddings, class statistics |
| Major source ecosystem change (e.g., Foursquare drops out) | Retrain encoder from scratch on accumulated labels | Everything |

The encoder itself is retrained only when the feature distribution shifts fundamentally. Day-to-day adaptation happens purely through NCM/SLDA updates.

---

## 7. Component 3 — GBM Baseline

### 7.1 Why GBM

Gradient Boosted Machines (XGBoost or LightGBM) are the industry standard for tabular classification. They:
- Handle mixed feature types (categorical + numerical) natively
- Require minimal preprocessing (no normalization, no embeddings)
- Are highly interpretable (feature importance, SHAP values)
- Train in seconds on 5k samples
- Achieve state-of-the-art on most tabular benchmarks

GBM serves as the **bar to beat**. If the MLP+NCM/SLDA approach doesn't match or exceed GBM, the added complexity isn't justified.

### 7.2 Configuration

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=9,         # handle 90/10 imbalance
    eval_metric='auc',
    early_stopping_rounds=10,
    use_label_encoder=False
)
```

### 7.3 GBM Limitations (Why We Still Want MLP+CL)

| Limitation | Impact |
|---|---|
| No continual learning | Must retrain from scratch each release |
| No embedding reuse | Embeddings from MLP encoder can be reused for other tasks (conflation, dedup) |
| Feature interactions are axis-aligned | Tree splits are one feature at a time; MLP captures arbitrary nonlinear interactions |
| Other teams are already doing it | No differentiation in our proposal |

---

## 8. Label Augmentation Strategy

### 8.1 The Problem

~5k labeled places is small. The 90/10 split means only ~500 closed examples. The model needs more signal, especially for the minority class.

### 8.2 LLM as Label Annotator

Use a large language model to expand the labeled dataset. This is a **data strategy**, not a model strategy — it benefits all three classifiers (NCM, SLDA, GBM) equally.

**Process:**
1. Sample unlabeled places from the full Overture release
2. For each place, construct a prompt with its attributes (name, category, address, sources, completeness)
3. Ask the LLM: "Based on the following place attributes, is this place likely open, permanently closed, or temporarily closed? Explain your reasoning."
4. Use LLM predictions as noisy labels with lower weight during training

**What the LLM sees (example prompt):**
```
Place: "Joe's Diner"
Category: restaurant
Address: 123 Main St, Anytown, NY
Sources: 1 (Meta only, confidence 0.72)
Has website: No
Has phone: No
Has brand: No

Based on these attributes, is this place likely open or closed?
```

**Key constraints:**
- LLM is used for **label generation only** — the labels are baked into the training set
- At inference time, the model uses only the engineered features — no LLM calls
- LLM-generated labels are weighted lower than ground-truth labels (e.g., 0.5x weight) to account for noise
- Focus LLM annotation on ambiguous cases (confidence 0.4-0.7) and underrepresented categories/geographies

### 8.3 Yelp Cross-Reference (US-only)

For US places matchable to Yelp:
- Use Yelp's `is_open` field as an additional label source
- Match by name similarity (Jaro-Winkler > 0.85) + geographic proximity (< 100m)
- Yelp labels treated as ground truth quality (weight 1.0) since they're from an independent data source

---

## 9. Training & Inference Pipeline

### 9.1 Training Pipeline

```
Step 1: Feature Extraction
    Overture parquet → DuckDB query → extract features → feature matrix X (N×30)

Step 2: Label Preparation
    Ground truth labels (~5k) + LLM-augmented labels + Yelp labels → label vector y

Step 3: Train MLP Encoder
    (X, y) → MLP encoder training → frozen encoder weights
    Output: encoder that maps X → 32-dim embedding Z

Step 4: Extract Embeddings
    X → frozen encoder → Z (N×32)

Step 5: Fit NCM/SLDA
    (Z, y) → compute class means (+ covariance for SLDA)

Step 6: Train GBM Baseline
    (X, y) → XGBoost training → fitted GBM model

Step 7: Evaluate All
    Compare NCM, SLDA, GBM on held-out validation set
```

### 9.2 Inference Pipeline (Production)

```
New place record
    │
    ▼
Feature extraction (sub-microsecond)
    │
    ▼
MLP forward pass: x → z (sub-millisecond, single matrix multiply)
    │
    ▼
NCM/SLDA: z → P(open|z) (sub-microsecond, distance computation)
    │
    ▼
Publish confidence score
```

**Total inference cost per place: sub-millisecond.** For 100M places, this is ~hours on a single CPU, or minutes with batched GPU inference.

### 9.3 Continual Update Pipeline (Per New Release)

```
New Overture release arrives
    │
    ▼
Extract features for all places in new release
    │
    ▼
Run frozen encoder → new embeddings
    │
    ▼
If new labels available:
    Update NCM/SLDA statistics incrementally
    │
    ▼
Score all places with updated classifier
    │
    ▼
Monitor for distribution drift (embedding space shift)
    │
    ▼
If drift exceeds threshold:
    Retrain encoder on accumulated labels
    Refit NCM/SLDA from scratch
```

---

## 10. Evaluation Plan

### 10.1 Primary Metrics

| Metric | Target | What It Measures |
|---|---|---|
| AUC-ROC | > 0.80 | Ranking quality — can the model separate open from closed? |
| AUC-PR | > 0.50 | Precision-recall tradeoff for the minority class (closed) |
| F1 (closed class) | > 0.40 | How well we detect closed places specifically |
| ECE (Expected Calibration Error) | < 0.05 | Are the probabilities trustworthy? |

### 10.2 Stratified Analysis

- **Per category:** AUC for top-20 categories. Flag any with AUC < 0.65.
- **Per geography:** AUC per top-10 countries (if data covers multiple countries).
- **By source count:** AUC for 1-source vs 2-source vs 3-source places.

### 10.3 Head-to-Head Comparison

| Comparison | Question Answered |
|---|---|
| NCM vs. SLDA | Does covariance structure help? |
| NCM vs. GBM | Does CL approach match tabular SOTA? |
| SLDA vs. GBM | Can CL approach beat tabular SOTA? |
| All three on minority class | Who finds closed places best? |

### 10.4 Embedding Quality

- **t-SNE / UMAP visualization** of encoder embeddings colored by open/closed
- **Silhouette score** measuring cluster separation in embedding space
- **Nearest-neighbor purity:** For each place, what fraction of its 5 nearest neighbors in embedding space share the same label?

### 10.5 Continual Learning Evaluation

Simulate the CL scenario:
1. Train on 70% of labeled data
2. Hold out 30% as "new release" data
3. Update NCM/SLDA with the new data incrementally
4. Compare performance vs. retraining from scratch
5. Measure: Does incremental update match full retrain? (It should, for NCM/SLDA)

---

## 11. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| 5k labels insufficient for MLP training | High | Tiny model (~6k params) with heavy regularization. LLM label augmentation. Yelp cross-reference. If still insufficient, fall back to GBM which needs fewer examples. |
| 90/10 imbalance causes model to predict all-open | High | Weighted loss (9x for closed). Evaluate on AUC-PR and F1-closed, not accuracy. Stratified splits in all CV folds. |
| MLP encoder underperforms GBM | Medium | Expected outcome — GBM is hard to beat on tabular data. The value proposition is CL adaptability, not single-snapshot accuracy. If gap is > 5% AUC, reconsider. |
| Encoder embeddings don't separate classes well | Medium | Increase encoder depth/width. Add contrastive loss as auxiliary objective. Worst case: use raw features with SLDA (no encoder). |
| Source ecosystem changes (e.g., Foursquare drops out) | Medium | `has_foursquare` feature gracefully degrades to 0 for all places. Monitor per-source feature importance. Trigger encoder retrain if a major source disappears. |
| LLM label augmentation introduces systematic bias | Medium | Weight LLM labels at 0.5x. Validate LLM labels against ground truth on the known 5k sample before using them. Track LLM agreement rate with ground truth. |
| Yelp enrichment creates US-vs-international performance gap | Low | `yelp_matched` flag lets model learn separate decision boundaries. Evaluate US vs. non-US performance separately. |
| NCM/SLDA too simple — can't capture complex boundaries | Medium | SLDA captures linear boundaries in embedding space, which is often sufficient when the encoder does its job. If not, add a small classification head and fine-tune with experience replay (another CL technique). |

---

## 12. Implementation Timeline

Assumes access to the ~5k labeled sample and the full Overture release.

| Week | Tasks |
|---|---|
| **Week 1** | **Data pipeline & feature engineering.** Write DuckDB queries to extract all features from parquet. Parse `sources` field. Build `extract_features()` function. Create train/val splits. Exploratory data analysis — feature distributions, correlations, class separability. |
| **Week 2** | **GBM baseline.** Train XGBoost on engineered features. Run 5-fold CV. Compute AUC-ROC, AUC-PR, F1. Feature importance analysis (SHAP). This is the bar to beat. Begin Yelp matching pipeline (US places). |
| **Week 3** | **MLP encoder + NCM/SLDA.** Implement encoder in PyTorch. Train end-to-end. Extract embeddings. Fit NCM and SLDA. Compare against GBM. Embedding visualization (t-SNE). Simulated CL evaluation. |
| **Week 4** | **Label augmentation & polish.** LLM label augmentation experiment. Retrain all models with augmented labels. Final head-to-head comparison. Write-up results. Score full release with best model. |

---

## Appendix A: Key Differentiators from Other Teams

Other teams are exploring GBM-only approaches. This proposal differentiates by:

1. **Continual learning capability** — NCM/SLDA on encoder embeddings enables incremental adaptation to new releases without retraining. No other team is addressing the "how do we update the model" question.

2. **Reusable embeddings** — The 32-dim place embeddings from the encoder are useful beyond open/closed prediction. They can be applied to place conflation, deduplication, and categorization tasks.

3. **GBM as baseline, not the answer** — We include GBM for rigorous comparison, but the goal is to demonstrate that a CL-capable architecture can match or exceed GBM while providing adaptability.

## Appendix B: Technology Stack

| Component | Tool |
|---|---|
| Data querying | DuckDB with `spatial`, `httpfs` extensions |
| Data format | Overture parquet on S3 (us-west-2) |
| Feature engineering | Python (pandas, numpy) |
| MLP encoder | PyTorch |
| NCM/SLDA | Custom implementation (< 100 lines each) |
| GBM baseline | XGBoost or LightGBM |
| Evaluation | scikit-learn (metrics), matplotlib (plots) |
| Label augmentation | Claude / GPT-4 API |
| Yelp matching | Fuzzy string matching (rapidfuzz) + geospatial (haversine) |
