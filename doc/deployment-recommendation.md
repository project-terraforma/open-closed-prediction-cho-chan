# Deployment Recommendation — Project C
v1 · 2026-03-02 · Caleb Cho

---

## Summary

**Recommended model: MLP + SLDA**

The MLP + SLDA pipeline delivers AUC-ROC 0.726 (second-best overall, 0.008 behind
the MLP head), supports mathematically exact incremental updates at each Overture
release, and requires no access to historical data after initial training. It is
the only option that can scale month-over-month without growing operational costs.

---

## Model Comparison

```
Model              AUC-ROC  AUC-PR    F1   Prec  Recall   Incremental?
-----------------------------------------------------------------------
GBM + OHE           0.686   0.255  0.277  0.173   0.698   No
XGBoost + OHE       0.720   0.293  0.311  0.206   0.635   No
MLP + NCM           0.721   0.243  0.302  0.218   0.492   Yes
MLP + SLDA          0.726   0.253  0.308  0.235   0.444   Yes   <- RECOMMENDED
MLP head            0.734   0.263  0.315  0.313   0.318   No
```

Val set: 685 samples | 63 closed (9.2%) | 622 open.
Metrics treat closed as the positive class at the optimal F1 threshold.

### Why not MLP head?

The MLP head has the best raw accuracy (AUC 0.734) but requires retraining the
encoder whenever new labeled data arrives. At 100M+ places per Overture release,
that means maintaining a labeled training pipeline, storing old data, and running
a full train/val cycle each month. For a 0.008 AUC gain over SLDA, this is not
worth the operational overhead.

### Why not XGBoost?

XGBoost (AUC 0.720) is close to SLDA (AUC 0.726) in accuracy but 3,000x slower
to update. A full retrain on 2,740 training samples takes ~184 ms; an SLDA update
on 823 new samples takes 0.15 ms. At scale, XGBoost update cost grows linearly
with the size of historical data. SLDA update cost is O(N_new) and requires no
historical data — only the new batch's embeddings.

### Why SLDA over NCM?

SLDA (AUC 0.726) modestly outperforms NCM (AUC 0.721). Both support exact
incremental updates. SLDA models within-class covariance, which provides a
better discriminant boundary when class distributions differ in shape. NCM
(0.061 ms update) is marginally faster than SLDA (0.149 ms update) but the
accuracy advantage of SLDA is consistent.

---

## Recommended Pipeline

```
[Monthly Overture release]
        |
        v
feature_engineering.py          Extract 19 numeric + 1 category feature per new place
        |
        v
PlaceEncoder.encode()           Frozen MLP encoder  →  32-dim embedding
(encoder.pt — never retrained)
        |
        v
StreamingLDA.update()           O(N_new) update, no old data needed
(slda.pkl — updated in-place)  Updates class means and scatter matrices
        |
        v
StreamingLDA.predict_proba()    Score all 100M+ places
        |
        v
p_closed > threshold            Binary open/closed prediction
```

### Threshold choice

The optimal F1 threshold for SLDA on the current val set is around 0.25–0.35
(varies slightly by run). For production:

- **Recall-oriented** (minimize missed closures): lower threshold ~0.15. Catches
  more closed places at the cost of more false flags.
- **Precision-oriented** (minimize false flags): higher threshold ~0.45. Fewer
  false alerts but misses more genuinely closed places.
- **Balanced (recommended starting point)**: use the optimal F1 threshold from
  `evaluate.py` on the current val benchmark, and re-evaluate after each update.

### What "incremental update" means operationally

1. New Overture release arrives (~monthly).
2. Extract embeddings for new/changed places using the frozen encoder.
3. If any of those places have ground-truth labels (closed/open), call:
   ```python
   slda.update(Z_new_labeled, y_new_labeled)
   ```
   This updates class statistics in 0.15 ms for a batch of ~800 samples.
4. Re-score all 100M places. Constant-time per place, feasible in minutes on CPU.
5. No old data required; no encoder retraining required.

---

## Feature Priorities

Features are ranked by Cohen's d (class-conditional separation) on the labeled set:

| Rank | Feature                | Cohen's d | Notes |
|------|------------------------|-----------|-------|
| 1    | `address_completeness` | 0.82      | Closed: 0.90 vs Open: 0.99 — strongest single signal |
| 2    | `confidence`           | 0.64      | Overture's own quality score |
| 3    | `max_source_confidence`| 0.55–0.64 | Source-level quality signals |
| 4    | `has_phone`            | 0.54      | 87% closed have phone vs 97% open |
| 5    | `completeness_score`   | 0.39      | Fraction of optional fields present |
| 6    | `has_website`          | 0.28      | Secondary completeness signal |
| 7    | `source_count`         | 0.25      | Single-source places more likely closed |
| 8    | `msft_update_age_days` | 0.08*     | *Diluted by 60% -1.0 values (no Microsoft source) |

**Microsoft staleness (conditioned on having Microsoft source):**
Closed places with a Microsoft source have data that is ~2.5x older (1,322 vs
518 median days). This signal is strong when the source is present but diluted
by the majority of places that lack it.

**Features to add in future iterations:**

1. `has_msft_and_old` — interaction: is_microsoft AND update_age > 730 days.
   This isolates the staleness signal from the no-Microsoft noise.
2. `days_since_any_source_update` — minimum across all sources.
3. `cross_release_status_change` — did `operating_status` change between
   the previous Overture release and the current one? Zero cost if releases
   are tracked.
4. `review_recency` — if a place has been reviewed on any platform in the last
   N months, it was almost certainly open at that time.

---

## What Data Would Most Improve Performance

The model's ceiling is set by data, not architecture. Current constraints:

| Bottleneck | Impact | Fix |
|---|---|---|
| Only 313 closed training examples | Class imbalance drives recall/F1 instability | More labeled closed places |
| US-only, NYC-heavy geography | Poor generalization to other regions | Global labeled sample |
| No freshness or activity signals | Missing temporal dimension | Review recency, status change history |
| Indirect closed indicators | Features overlap between open/closed | Direct observation (field audits, trusted signals) |

**Highest-leverage action:** Obtain 1,000+ additional human-labeled closed
examples, distributed across geography and business categories. A 4x increase
in closed training examples would be expected to meaningfully narrow the gap
between current performance and the original AUC target of 0.80.

**Second-highest-leverage action:** Add temporal features. The Microsoft
staleness signal (2.5x age gap) already demonstrates that time-based signals
are informative. Tracking when a place was last observed "active" (reviewed,
claimed, updated by owner) would provide the most direct closure signal available
without field verification.

---

## Risk Factors and Limitations

**Class imbalance:** At 9% closed prevalence, even a small number of false
positives at scale can represent a large absolute count. At 100M places with 9%
closed (~9M closed), an FP rate of 1% would flag 910k open places incorrectly.
Threshold tuning and regular val-set monitoring are essential.

**Distribution shift:** The labeled set is US-only and has two sources (Meta,
Microsoft). The model has not been validated on international places or places
with other source combinations (Foursquare, etc.). Recall may be lower in
underrepresented geographies.

**Parquet augmentation failure:** An experiment adding 785 automatically-labeled
closed places from the Feb parquet (where `operating_status = 'closed'` was set
by Overture's own pipeline) degraded MLP and SLDA performance. All 785 records
had `source_count = 2.0` — a pipeline artifact — making them structurally
different from the hand-labeled closed set. Any future augmentation from
Overture's own `operating_status` field should be treated with caution until
a distribution check is performed.

**Encoder staleness:** The MLP encoder is frozen after initial training. As
Overture's data coverage evolves (new sources, new feature distributions),
encoder quality may degrade. Re-evaluate encoder quality every 6–12 months.

---

## Operational Integration

### Minimal deployment (scoring only)

```
artifacts needed:   encoder.pt  encoder_config.json  slda.pkl  category_encoder.pkl
inference:          ~0.07 ms per sample end-to-end (embed + classify)
100M place scoring: feasible in ~7 minutes single-threaded on CPU
storage:            < 1 MB for all model artifacts
```

### With monthly update

```
inputs per release:  new labeled samples (if any)  — even 10–20 new labels help
update time:         0.15 ms for a batch of 800 samples
data retention:      none required (Welford online algorithm)
encoder retraining:  not required
```

### Decision output

```python
p_closed = slda.predict_proba(Z)[:, 0]    # P(closed) in [0, 1]
is_closed = p_closed > threshold           # bool flag per place
```

The raw `p_closed` score can also be used as a confidence-weighted signal
downstream (e.g., deprioritize places with p_closed > 0.5 in search ranking)
without committing to a hard threshold.
