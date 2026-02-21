# Project C — Open & Closed Predictions: Model Approach Proposals

> **Approach A:** Contrastive Place Embeddings with Self-Supervised Pre-training
> **Approach B:** Discrete-Time Survival Model with Hazard-Based Scoring
>
> Overture Maps Foundation · February 2026

---

## Table of Contents

1. [Problem Context & Design Constraints](#1-problem-context--design-constraints)
2. [Shared Feature Taxonomy](#2-shared-feature-taxonomy)
3. [Approach A — Contrastive Place Embeddings](#3-approach-a--contrastive-place-embeddings)
   - 3.1 Architecture Overview
   - 3.2 Pre-training: Source-Agreement Contrastive Task
   - 3.3 Fine-tuning & Calibration
   - 3.4 Input Feature Vector
   - 3.5 Evaluation Plan
   - 3.6 Risks & Mitigations
4. [Approach B — Discrete-Time Survival Model](#4-approach-b--discrete-time-survival-model)
   - 4.1 Architecture Overview
   - 4.2 Temporal Data Construction
   - 4.3 Model Specification
   - 4.4 Scoring at Inference Time
   - 4.5 Evaluation Plan
   - 4.6 Risks & Mitigations
5. [Comparative Analysis](#5-comparative-analysis)
6. [Implementation Timeline](#6-implementation-timeline)

---

## 1. Problem Context & Design Constraints

The Overture Maps places dataset requires a confidence score for place existence (open vs. closed) that can be published as a first-class schema field. The score must satisfy five simultaneous constraints that are in tension with each other:

| Constraint | Requirement | Implication |
|---|---|---|
| Generalization | Works across all categories and geographies | Cannot hard-code rules per category or country |
| Scale | Scores 100M+ places per release | Sub-millisecond inference per record; no external API calls at runtime |
| Stability | Scores do not fluctuate wildly across releases | Model must be robust to minor data distribution shifts |
| Interpretability | Justifiable as a public field | Consumers need to understand what the score means and why |
| Robustness | Tolerates upstream source churn | No single-source dependency; graceful degradation when sources drop |

Given that other teams are already exploring gradient-boosted tree approaches, this proposal focuses on two complementary alternatives that offer distinct modeling philosophies and differentiated insights for the final recommendation.

---

## 2. Shared Feature Taxonomy

Both approaches draw from the same universe of signals, organized by computational cost at scale. This shared taxonomy ensures fair comparison and allows feature engineering effort to be reused.

| Tier | Signal Family | Examples | Cost per Place |
|---|---|---|---|
| Tier 1 — Intrinsic | Attributes of the record itself | Source count, recency of last update, completeness (has phone / website / hours), category, brand presence | O(1) |
| Tier 2 — Cross-source | Agreement across sources | Source concordance count, freshness delta between sources, name/address variance across sources | O(k), k = sources |
| Tier 3 — External | Active verification probes | Website HTTP status, domain WHOIS expiry, phone HLR validation, search engine presence | O(1) but $ |

> **Design Principle:** Tier 3 signals are used for label generation and calibration validation, never as runtime features. This prevents coupling the production score to external API availability and cost.

---

## 3. Approach A — Contrastive Place Embeddings

### 3.1 Architecture Overview

This approach uses representation learning to capture rich interactions between place attributes. The architecture has three stages, each building on the previous:

| Stage | What It Does | Labels Required | Compute |
|---|---|---|---|
| 1. Place Encoder | Small feed-forward network (3 layers, 64-dim hidden, 32-dim output) ingests a fixed-width feature vector per place | None | Minimal |
| 2. Contrastive Pre-training | Self-supervised task: predict whether two source records for the same place agree on key attributes. Uses NT-Xent contrastive loss. | None — runs on all 100M+ places | GPU, one-time |
| 3. Fine-tuning + Calibration | Freeze encoder, attach 2-layer MLP head, train on labeled sample. Apply temperature scaling for calibration. | ~5k labeled places | CPU, minutes |

**Key insight:** The contrastive pre-training step teaches the encoder a general sense of "data health" and "source consistency" without any open/closed labels. This representation transfers powerfully to the downstream classification task, making the model far more label-efficient than a supervised-only approach.

### 3.2 Pre-training: Source-Agreement Contrastive Task

The self-supervised pre-training task is the core innovation. For each place with multiple source records, we construct:

- **Positive pairs:** Two source records from the same place that agree on name and address (high concordance).
- **Negative pairs:** A source record from the target place paired with a random source record from a different place.

The encoder is trained with NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss to pull positive pairs together and push negative pairs apart in embedding space. After pre-training, places with consistent, well-maintained data cluster together, while places with sparse or conflicting data are pushed to different regions.

> **Why This Beats Pure Supervised Learning:** With only ~5k labeled examples, a supervised model sees a tiny slice of category and geography combinations. Pre-training on 100M+ places exposes the encoder to the full diversity of the dataset. The encoder learns that a well-maintained cafe in Tokyo and a well-maintained barbershop in Lagos share structural similarities in their source records, even though a supervised model would never see both.

### 3.3 Fine-tuning & Calibration

After pre-training, the encoder weights are frozen (or lightly fine-tuned with a low learning rate). A small classification head is attached:

- **Architecture:** 32-dim embedding → ReLU → 16-dim hidden → 2-class logits. Total trainable parameters: ~600.
- **Training:** Cross-entropy loss on the ~5k labeled sample, stratified by category × open/closed. Use 5-fold cross-validation to estimate generalization.
- **Calibration:** Apply temperature scaling — learn a single scalar T such that `softmax(logits / T)` minimizes negative log-likelihood on a held-out calibration set. Target: Expected Calibration Error (ECE) < 0.03.

Temperature scaling is re-fit each release on a fresh calibration sample (labels refreshed via Tier 3 probes on a rotating subset). This absorbs distributional drift without retraining the full model.

### 3.4 Input Feature Vector

The encoder ingests a fixed-width vector per place, combining learned embeddings for categorical features with normalized numerical features:

| Feature | Type | Dimensionality | Notes |
|---|---|---|---|
| Category | Categorical embedding | 8-dim | Learned; similar categories get similar embeddings |
| Country | Categorical embedding | 4-dim | Captures geography-specific base rates |
| Primary source | Categorical embedding | 4-dim | Encodes source identity and reliability |
| Source count | Numerical (quantile-normalized) | 1-dim | Number of distinct sources |
| Max source freshness | Numerical (quantile-normalized) | 1-dim | Days since most recent source update |
| Completeness score | Numerical (quantile-normalized) | 1-dim | Fraction of non-null optional fields |
| Source name agreement | Numerical (quantile-normalized) | 1-dim | Jaro-Winkler similarity of name across sources |
| Source address agreement | Numerical (quantile-normalized) | 1-dim | Token-level Jaccard of address across sources |
| Version | Numerical (quantile-normalized) | 1-dim | Record version number |
| Place age (releases) | Numerical (quantile-normalized) | 1-dim | Consecutive releases this place has appeared in |
| Has website | Binary | 1-dim | Boolean flag |
| Has phone | Binary | 1-dim | Boolean flag |
| Has hours | Binary | 1-dim | Boolean flag |
| Has brand | Binary | 1-dim | Boolean flag |

**Total input dimensionality: ~30.** The encoder is deliberately small (~50K parameters) to ensure inference scales to 100M+ places on commodity hardware.

### 3.5 Evaluation Plan

1. **Primary metric:** AUC-ROC on stratified 5-fold cross-validation over the labeled sample.
2. **Calibration:** Reliability diagram with 10 bins; max calibration error < 0.05 per bin; ECE < 0.03 overall.
3. **Category breakdown:** AUC per top-20 categories; flag any category with AUC < 0.70 for investigation.
4. **Geography breakdown:** AUC per top-10 countries; quantify variance across regions.
5. **Embedding quality:** Visualize t-SNE of place embeddings colored by open/closed; measure silhouette score.
6. **Stability test:** If release-over-release data is available, re-score with the same model and measure score drift (cosine similarity of embedding vectors, KL divergence of score distributions).

### 3.6 Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Small labeled dataset (5k) causes overfitting of classifier head | Medium | Pre-training provides strong initialization; fine-tuning touches only ~600 parameters. Regularize with dropout (0.3) and early stopping. |
| Interpretability gap vs. tree models | Medium | Publish top-3 feature attributions (via integrated gradients) alongside each score. Cluster embeddings into human-labeled archetypes. |
| Pre-training requires GPU compute | Low | One-time cost (~4–8 hours on a single GPU for 100M places). Incremental updates warm-start from previous encoder. |
| Category embedding cold-start for rare categories | Low | Use hierarchical category structure: rare subcategories inherit from parent category embedding. |
| Source ecosystem changes shift embedding space | Medium | Monitor embedding drift via cosine similarity across releases. Trigger re-pre-training if drift exceeds threshold. |

---

## 4. Approach B — Discrete-Time Survival Model

### 4.1 Architecture Overview

This approach reframes the problem entirely. Instead of asking *"is this place open right now?"*, it asks *"what is the probability this place is still open at time t?"* — and derives the confidence score from the survival function S(t).

Closures are inherently temporal events. A restaurant last updated 3 months ago is fundamentally different from one last updated 3 years ago, even if their static attributes are identical. Survival models are built to reason about this — time is the structural backbone, not just another feature.

> **Core Advantage:** Calibration is built into the model by construction. A survival function S(t) directly outputs the probability that a place is still open at time t. There is no need for isotonic regression or temperature scaling — the output IS a probability.

### 4.2 Temporal Data Construction

The key data engineering step is converting the static labeled sample into a temporal (person-period) dataset. Each place contributes one row per time period (Overture release) it has been observed:

| Field | Description | Source |
|---|---|---|
| `place_id` | Unique place identifier | Overture place ID |
| `period` | Release index (t = 1, 2, 3, ...) | Derived from release history |
| `event` | 1 if the place closed in this period, 0 otherwise | Approximated from version history / source dropout timestamps |
| `censored` | 1 if the place is still open at end of observation (right-censored) | From current label: open = 1 at labeling time |
| `features_t` | Time-varying covariates as of this period | Source count, freshness, completeness at each release |

**Estimating closure timing:** For places labeled as closed, approximate the closure period as the release in which the record stopped being updated by any source (version number freezes, source freshness stops advancing). For places labeled as open, they are right-censored at the current period.

**Handling right-censoring:** This is a natural strength of survival models. Many places in the dataset are currently open — we don't know when (or if) they'll close. Unlike a binary classifier that throws away this temporal information, the survival model uses it properly: these places contribute to the likelihood up to the current time but don't force a closure prediction.

### 4.3 Model Specification

Two model variants are proposed, ranging from interpretable to flexible:

#### Variant B1: Cox Proportional Hazards (Interpretable)

The Cox model estimates the hazard function h(t|X) = h₀(t) · exp(βᵀX), where h₀(t) is the baseline hazard (the risk of closing for a "reference" place) and β captures how each feature increases or decreases that risk.

- **Output:** Hazard ratios. Example: "A place losing its phone number has a 2.3× higher closure hazard." This is directly interpretable and documentable.
- **Assumption:** Proportional hazards — the effect of features is constant over time. This may not hold (e.g., source freshness matters more as places age). Test with Schoenfeld residuals.
- **Scale:** Fits in seconds on 5k places. Inference is a dot product per place — trivially scales to 100M+.

#### Variant B2: Discrete-Time Hazard with Neural Network (Flexible)

Replace the linear predictor βᵀX with a small neural network. Each time period is modeled as a binary classification ("did the place close in this period?"), with period indicators and features as inputs.

- **Architecture:** Input features + period embedding → 2-layer MLP (64 → 32 → 1) → sigmoid. Loss: binary cross-entropy weighted by inverse censoring probability.
- **Advantage:** Captures non-proportional hazards and complex feature interactions. A restaurant with 1 source behaves differently from a hospital with 1 source at different time horizons.
- **Scale:** Comparable to Approach A. Small network, single forward pass per place.

### 4.4 Scoring at Inference Time

At scoring time, compute the survival probability for each place at the current release:

1. Compute the feature vector for the place as of the current release.
2. Feed it through the fitted model to get the hazard h(t) for the current period.
3. Compute the cumulative survival: S(t) = ∏ (1 − h(s)) for s = 1 to t, where t is the place's age in releases.
4. Publish S(t) as the confidence score. S(t) = 0.92 means "we estimate a 92% probability this place is still open."

> **Interpretability Advantage:** The survival framework provides a natural narrative: "This place has been in the dataset for 12 releases. Based on its category (restaurant), source count (1), and data freshness (last updated 8 months ago), its estimated survival probability is 0.64." This is far more intuitive than a black-box classifier score.

### 4.5 Evaluation Plan

1. **Primary metric:** Concordance index (C-index) — the survival-analysis equivalent of AUC. Measures whether the model correctly ranks places by closure risk.
2. **Calibration:** Compare predicted survival curves against observed Kaplan-Meier curves within category × geography strata.
3. **Time-dependent AUC:** Evaluate discrimination at specific time horizons (t+1 release, t+3 releases, t+6 releases).
4. **Proportional hazards test:** For Variant B1, run Schoenfeld residual tests. If violated, this motivates Variant B2.
5. **Brier score:** Assess calibration and discrimination jointly across time points.
6. **Feature importance:** Hazard ratios (B1) or permutation importance (B2) to identify key drivers.

### 4.6 Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Temporal data construction is complex | High | Start with a simplified version: estimate closure time from version freeze date. Validate against manual inspection of ~100 places. |
| Limited release history available | High | If only 1–2 releases are available, use source-level timestamps as proxy periods. Fall back to Approach A if insufficient temporal signal. |
| Proportional hazards assumption violated (B1) | Medium | Use Schoenfeld residual test. If violated, use Variant B2 or add time-interaction terms. |
| Small labeled sample limits survival curve estimation | Medium | Pool categories hierarchically for baseline hazard estimation. Use shrinkage priors (penalized Cox model). |
| Score instability if baseline hazard is re-estimated each release | Low | Fix the baseline hazard from the initial training; only update coefficients via incremental re-estimation. |

---

## 5. Comparative Analysis

The two approaches offer complementary strengths. The choice depends on data availability and the team's priority between innovation and interpretability.

| Dimension | Approach A: Contrastive Embeddings | Approach B: Survival Model |
|---|---|---|
| Core philosophy | Learn a general representation of "place data health" via self-supervised learning | Model closure as a time-to-event process; derive score from survival function |
| Label efficiency | Excellent — pre-training uses 0 labels on 100M+ places | Moderate — requires temporal labels (closure timing), not just binary open/closed |
| Calibration | Post-hoc (temperature scaling) | Built-in (survival function is a probability by construction) |
| Interpretability | Requires post-hoc attribution (integrated gradients) | Native — hazard ratios are directly interpretable |
| Data requirements | Standard labeled sample + full Overture release for pre-training | Release-over-release version history to construct temporal dataset |
| Compute (training) | GPU for pre-training (one-time, ~4–8 hours) | CPU-only, fits in minutes |
| Compute (inference) | Sub-millisecond per place (small forward pass) | Sub-millisecond per place (dot product or small forward pass) |
| Scalability to 100M+ | Trivial | Trivial |
| Extensibility | High — encoder can ingest text, images, or other modalities | Moderate — adding features is straightforward but architecture is time-centric |
| Reusable artifacts | Place embeddings reusable for conflation, categorization, deduplication | Hazard estimates reusable for data freshness monitoring |
| Risk profile | Medium — pre-training is novel but fine-tuning is standard | Medium–High — depends on temporal data availability |

> **Recommendation:** If release-over-release version history is readily available, Approach B offers a more elegant and naturally calibrated solution. If not, Approach A is the safer bet — it works with the data you have today and produces reusable embeddings as a bonus. Both approaches are differentiated from the GBM work being done by other teams.

---

## 6. Implementation Timeline

Both approaches can be executed within a focused 4-week sprint. The timeline assumes access to the ~5k labeled sample and the full Overture release.

| Week | Approach A: Contrastive Embeddings | Approach B: Survival Model |
|---|---|---|
| Week 1 | Feature engineering + data pipeline. Extract Tier 1/Tier 2 features from parquet. Build training/validation splits. | Temporal data construction. Parse version history, estimate closure timing. Build person-period dataset. Validate on ~100 places. |
| Week 2 | Contrastive pre-training on full release. Experiment with NT-Xent hyperparameters (temperature, batch size, negative sampling). | Fit Cox model (Variant B1). Run proportional hazards tests. Compute hazard ratios. Begin Variant B2 if PH assumption is violated. |
| Week 3 | Fine-tune classifier head on labeled sample. Run 5-fold CV. Apply temperature scaling. Evaluate calibration and per-category AUC. | Fine-tune neural hazard model (Variant B2). Run C-index, Brier score, and time-dependent AUC evaluations. Compare B1 vs. B2. |
| Week 4 | Embedding analysis (t-SNE, clustering). Write-up and comparison against GBM baseline from other teams. Final report. | Score full release with survival probabilities. Write-up and comparison against GBM baseline. Final report. |