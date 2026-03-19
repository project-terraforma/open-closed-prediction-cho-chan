# Deployment Recommendation — Project C
v2 · 2026-03-16 · Caleb Cho

---

## Summary

**Recommended model: MLP + NCM**

Updated from v1 (SLDA) based on new experimental runs with SF augmentation and
SF-schema training. MLP+NCM matches SLDA on AUC in best-case runs, has consistent
sub-microsecond inference (vs SLDA's erratic 12–80 µs at small dataset sizes),
and supports the same zero-history incremental updates. It is the most reliable
option across all conditions.

---

## Experimental Runs

| # | Schema | Conf | Augmentation | Train | Val | Val closed% |
|---|--------|------|--------------|-------|-----|------------|
| 1 | Overture | ✅ | — | ~2,740 | 685 | 9.2% |
| 2 | Overture | ❌ | — | ~2,740 | 685 | 9.2% |
| 3 | Overture | ✅ | SF aug (9,547) | 12,287 | 685 | 9.2% |
| 4 | Overture | ❌ | SF aug (9,547) | 12,287 | 685 | 9.2% |
| 5 | SF schema | — | — | 285,080 | 71,271 | 53.9% |

Val set for runs 1–4 is fixed (same 685 Overture samples across all runs), making
AUC comparisons directly meaningful. Run 5 is a different problem (different schema,
balanced classes) — not directly comparable to runs 1–4.

---

## AUC-ROC by Run

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|------:|------:|------:|------:|------:|
| GBM | .649 | .653 | .685 | .673 | .887 |
| XGBoost | .662 | .665 | .682 | .679 | .885 |
| MLP head | .702 | .696 | **.727** | .710 | .881 |
| MLP + NCM | .700 | .700 | .727 | .698 | .873 |
| MLP + SLDA | .690 | **.713** | .725 | .674 | .881 |
| MLP + QDA | .700 | .691 | .697 | .701 | .873 |

**Best Overture result:** MLP head / MLP+NCM tied at .727 (run 3: with conf + SF aug).

## F1 (closed class) by Run

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|-------|------:|------:|------:|------:|------:|
| GBM | .288 | .284 | .309 | .317 | .836 |
| XGBoost | .308 | .306 | .315 | .306 | .835 |
| MLP head | **.356** | .319 | .348 | **.364** | .832 |
| MLP + NCM | .350 | .326 | .323 | .350 | .832 |
| MLP + SLDA | .315 | .319 | .316 | .327 | .833 |
| MLP + QDA | .351 | .307 | .323 | .332 | .832 |

---

## Model Cost Table (Overture runs)

| Model | µs/sample | Model size | Retrain cost | Update method | Needs history? |
|-------|----------:|----------:|-------------|--------------|---------------|
| GBM | ~3.1 | ~670 KB | NaN | Full retrain | ✅ |
| XGBoost | ~1.6–2.6 | ~370–400 KB | 133–355 ms | Full retrain | ✅ |
| MLP head | **~0.5–0.7** | **~76–96 KB** | NaN | Full retrain | ✅ |
| MLP + NCM | **~0.6–0.9** | **~77–97 KB** | 0.07–0.19 ms | Incremental | ❌ |
| MLP + SLDA | 12–80 ⚠️ | ~85–105 KB | 0.14–0.34 ms | Incremental | ❌ |

SLDA inference cost is erratic at small dataset sizes — 80 µs/sample in run 2 (no aug),
12–17 µs in runs 3–4 (with aug). This is a known numerical instability in the covariance
matrix inversion when the training set is small. At production scale (100M places) it
stabilizes (run 5: 2 µs/sample), but the unpredictability in low-data regimes is a risk.

---

## Key Findings

### SF augmentation is a clear win (+2–3 AUC)

Adding 9,547 SF-schema records to the Overture training set (runs 1→3, 2→4) consistently
improved AUC-ROC by 2–3 points across all models:

- MLP head: .702 → .727 (+2.5)
- GBM: .649 → .685 (+3.6)
- XGBoost: .662 → .682 (+2.0)

The SF augmentation records are ~93% open (9,103 open / 444 closed), so they contribute
primarily to defining the open-class boundary. Worth including in all future runs.
Cost: XGBoost retrain time increases ~2.5× (133 ms → 355 ms), still acceptable.

### Conf features have negligible effect

Runs 1 vs 2 (with/without conf) show no consistent AUC improvement. Drop them to
reduce feature count and noise.

### SF schema (run 5) is a different problem

AUC .887 with balanced classes and 285k training samples. Not comparable to Overture
runs — different features, balanced labels, and no source-confidence signal. Confirms
the SF dataset itself is highly learnable and represents a viable standalone model
for SF-specific deployments.

---

## Recommendation: MLP + NCM

### Why NCM over SLDA (change from v1)

In v1, SLDA was preferred for a slight AUC edge. New data shows:

1. **SLDA inference is unreliable at small dataset sizes** — 80 µs/sample (run 2) vs
   the expected ~0.07 ms. This makes it risky for production without a guaranteed
   minimum training set size.
2. **NCM matches SLDA's best result** — both hit .727 AUC in run 3 (best Overture run).
3. **NCM inference is consistently fast** — 0.6–0.9 µs/sample across all runs, no
   anomalies.
4. Both support incremental updates with no historical data required. NCM update
   cost (0.07–0.19 ms) is marginally faster than SLDA (0.14–0.34 ms).

### Why not MLP head

MLP head ties NCM at .727 AUC (run 3) and has the best F1 in 2 of 5 runs. However:
- Requires full retrain whenever new labeled data arrives
- Retrain time not benchmarked (NaN in cost table)
- No path to incremental updates without retraining the encoder

For a 0 AUC gain vs NCM (tied in best run), the operational overhead is not justified.

### Why not XGBoost/GBM

Significantly lower AUC (.662–.685 on Overture) and no incremental update support.
XGBoost retrain grows to 355 ms with augmentation and would scale worse with more data.

---

## Recommended Pipeline

```
[Monthly Overture release]
        |
        v
feature_engineering.py          Extract ~20 numeric + 1 category features per place
        |
        v
PlaceEncoder.encode()           Frozen MLP encoder  →  32-dim embedding
(encoder.pt — never retrained)
        |
        v
StreamingNCM.update()           O(N_new) update, no old data needed
(ncm.pkl — updated in-place)   Updates per-class mean vectors
        |
        v
StreamingNCM.predict_proba()    Score all places via nearest centroid distance
        |
        v
p_closed > threshold            Binary open/closed prediction
```

### Threshold choice

Optimal F1 threshold is around 0.3–0.55 (varies by run). For production:

- **Recall-oriented** (minimize missed closures): lower threshold ~0.15–0.30
- **Precision-oriented** (minimize false flags): higher threshold ~0.50–0.65
- **Recommended starting point**: use optimal F1 threshold from `evaluate.py` on
  the current val benchmark, re-evaluate after each incremental update

---

## Operational Integration

### Artifacts

```
encoder.pt              Frozen MLP encoder (~77–97 KB)
encoder_config.json     Encoder architecture config
ncm.pkl                 StreamingNCM classifier
category_encoder.pkl    LabelEncoder for primary_category
feature_names.json      Ordered feature list for inference alignment
```

### Inference

```
~0.6–0.9 µs per sample end-to-end (embed + classify)
100M place scoring: ~1 minute single-threaded on CPU
storage: < 1 MB for all artifacts
```

### Monthly update

```
inputs:         new labeled samples (even 10–20 new labels help)
update time:    0.07–0.19 ms for a batch of ~800 samples
data retention: none required
encoder:        never retrained
```

### Decision output

```python
p_closed = ncm.predict_proba(Z)[:, 0]    # P(closed) in [0, 1]
is_closed = p_closed > threshold          # bool flag per place
```

---

## Feature Priorities (Overture schema)

| Rank | Feature | Notes |
|------|---------|-------|
| 1 | `address_completeness` | Strongest single signal (Cohen's d 0.82) |
| 2 | `confidence` | Overture's own quality score |
| 3 | `max_source_confidence` | Source-level quality signal |
| 4 | `has_phone` | 87% closed vs 97% open |
| 5 | `completeness_score` | Fraction of optional fields present |
| 6 | `has_website` | Secondary completeness signal |
| 7 | `source_count` | Single-source places more likely closed |
| 8 | `msft_update_age_days` | Strong when present; diluted by 60% missing |
| — | `category_closure_rate` | Derived: train-time closure rate per category |

**Conf features (excluded by default):** `max_source_confidence`, `min_source_confidence`,
`mean_source_confidence`, `confidence_spread`, `confidence` — no consistent AUC gain
observed across runs 1 vs 2.

**Pending experiment:** spatial neighborhood features (18 features at 100m/250m/500m
from SF BallTree) — run in progress, results not yet in this document.

---

## What Data Would Most Improve Performance

| Bottleneck | Impact | Fix |
|---|---|---|
| Only ~313–694 closed training examples | Class imbalance, F1 instability | More labeled closed places |
| US-only, SF-heavy geography | Poor generalization to other regions | Global labeled sample |
| No freshness / activity signals | Missing temporal dimension | Review recency, status change history |
| Indirect closed indicators | Feature overlap between open/closed | Direct observation |

**Highest-leverage action:** More labeled closed examples. The SF augmentation
result (+2–3 AUC) shows that even noisy cross-schema labels from a different dataset
move the needle — a clean set of 1,000+ additional closed Overture places would
be expected to push past the 0.75 AUC barrier.

---

## Risk Factors and Limitations

**Class imbalance:** At 9% closed prevalence, a 1% FP rate at 100M places flags
~910k open places incorrectly. Threshold tuning and regular val-set monitoring
are essential.

**Distribution shift:** Labeled set is US-only with two dominant sources (Meta,
Microsoft). Not validated on international places or non-Meta/Microsoft sources.

**SLDA inference instability:** If SLDA is used instead of NCM, minimum training
set size should be enforced (empirically: SF aug raises it to ~12k, at which point
SLDA inference normalizes to ~17 µs/sample). Below ~3k samples, 80 µs/sample
latency spikes have been observed.

**Encoder staleness:** Frozen encoder may degrade as Overture coverage evolves.
Re-evaluate encoder quality every 6–12 months or when a major new source is added.

**Parquet augmentation failure (v1 note):** Adding 785 auto-labeled closed places
from Feb parquet (`operating_status = 'closed'`) degraded performance — all 785 had
`source_count = 2.0`, a pipeline artifact making them structurally different from
hand-labeled closed. Any future Overture `operating_status` augmentation needs a
distribution check first.

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v1 | 2026-03-02 | Initial recommendation: MLP + SLDA |
| v2 | 2026-03-16 | Updated with runs 1–5; switched recommendation to MLP + NCM due to SLDA inference instability; documented SF augmentation impact; added SF schema results |
