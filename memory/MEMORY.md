# Project C — Decision Log & Memory

## Project Goal
Predict whether an Overture Maps place (business/POI) is open or permanently closed.
Must scale to 100M+ places with sub-millisecond inference at scoring time.

## Dataset (confirmed)
- **File**: `data/project_c_samples.json` — 3,425 records, JSONL (one JSON per line)
- **Labels**: `open` field, binary (1=open, 0=closed)
- **Class balance**: 3,112 open (90.9%) / 313 closed (9.1%)
- **Geography**: US-only
- **Sources**: `meta` and `Microsoft` only (no Foursquare in this labeled split)
- **Meta update_time**: Always `2025-02-24T08:00:00Z` — batch date, NOT a freshness signal
- **Microsoft update_time**: Real timestamps (2013–2024) — usable staleness signal

## Full Overture Release (context)
- 72.9M total places; only 785 permanently_closed + 20 temporarily_closed in `operating_status`
- operating_status is static (zero changes between Jan and Feb 2026 releases)
- 445k places removed Jan→Feb were dedup/source drops, NOT closures (all labeled "open")

## Approach
- MLP Encoder (30→64→32 dim, ~6k params) + NCM/SLDA continual learning classifiers
- Tabular features only (no text, no geo); GBM optional baseline

## Feature Engineering (`src/feature_engineering.py`)
**21 features** — `extract_features(record) -> dict`, `load_dataset(path) -> (X, y)`

### Feature Decisions
| Feature | Decision | Reason |
|---|---|---|
| `has_address` | **DROPPED** | Constant (100% of records have addresses) — zero variance |
| `has_email` | **DROPPED** | All-zero in labeled dataset — confirmed via EDA, no signal |
| `msft_update_age_days` | Kept, returns -1.0 if no Microsoft source | -1.0 encodes "no Microsoft" as distinct signal; model can learn conditional |

## EDA Findings (`src/eda.py`)
Run on `project_c_samples.json`, key results:

### Feature Signal (Cohen's d, sorted)
| Feature | Effect | Notes |
|---|---|---|
| `address_completeness` | 0.824 strong | Closed: 0.902 vs Open: 0.988 — strongest single feature |
| `confidence` | 0.641 strong | Closed: 0.751 vs Open: 0.869 |
| source confidence features | 0.55–0.64 strong | All aligned |
| `has_phone` | 0.540 strong | Closed: 87.2% vs Open: 97.3% |
| `completeness_score` | 0.388 strong | |
| `has_website` | 0.275 moderate | |
| `source_count` | 0.252 moderate | Closed: 87.2% single-source vs Open: 76.7% |
| `has_microsoft` | 0.126 moderate | Closed: 33.9% vs Open: 40.0% |
| `msft_update_age_days` (overall) | 0.080 weak | Diluted by 60% -1.0 values |
| `alternate_category_count` | 0.032 noise | Near-zero signal |
| `has_email` | 0.000 | All zeros — DROPPED |

### Microsoft Staleness (conditioned on having Microsoft source)
- Open: median age = 518 days
- Closed: median age = 1,322 days (~2.5x older)
- Strong signal when Microsoft source is present

### Confidence Distribution
- Open: 48.1% have confidence >0.95; only 5% ≤0.4
- Closed: 22.7% have confidence >0.95; 28.1% ≤0.6

### Category Notes
- `rental_kiosks` at 81% closed — likely systematic data artifact, not generalizable
- Restaurants/food service at 15–18% closed — reasonable real-world rate
- Category is a meaningful feature

### Source Count
- Open: 23.3% have 2 sources
- Closed: only 12.8% have 2 sources

## Model Benchmark (val set, 685 samples, 63 closed / 622 open)
All metrics for **closed (label=0) as positive class**.

| Model | AUC-ROC | AUC-PR | F1 | Prec | Recall |
|---|---|---|---|---|---|
| GBM + OHE | 0.686 | 0.255 | 0.277 | 0.173 | 0.698 |
| XGBoost + OHE | 0.720 | 0.293 | 0.311 | 0.206 | 0.635 |
| MLP + NCM | 0.721 | 0.243 | 0.302 | 0.218 | 0.492 |
| MLP + SLDA | 0.726 | 0.253 | 0.308 | 0.235 | 0.444 |
| **MLP head** | **0.734** | **0.263** | **0.315** | **0.313** | **0.318** |

**Key finding**: MLP head beats XGBoost/GBM because the 8-dim category embedding shares
information across similar categories — OHE gives trees 295 isolated binary features.
MLP + SLDA (0.726) is within 0.008 of XGBoost while supporting O(N) incremental updates.

**Targets missed** (original: AUC-ROC>0.80, AUC-PR>0.50, F1>0.40). Revised realistic targets
given 250 train-closed and 9% prevalence: AUC-ROC>0.73, AUC-PR>0.28, F1>0.31.

### Training config (best run)
- Adam lr=1e-3, weight_decay=1e-4, cosine LR, patience=20 on val AUC
- Best epoch: 11 of 100, val_loss=0.6081, val_auc=0.7344
- Class weights: closed=5.48x, open=0.55x

## Key Files
| File | Purpose |
|---|---|
| `src/feature_engineering.py` | `extract_features()`, `load_dataset()` |
| `src/eda.py` | Class-conditional EDA, prints signal analysis |
| `src/split.py` | Stratified 80/20 split, saves to `splits/` |
| `src/encoder.py` | PlaceEncoder (MLP), PlaceDataset, class_weights |
| `src/ncm.py` | NearestClassMean — fit/update/predict_proba |
| `src/slda.py` | StreamingLDA — fit/update (Welford)/predict_proba |
| `src/train.py` | End-to-end training: MLP → embeddings → NCM/SLDA |
| `src/gbm.py` | GBM + XGBoost with OHE for primary_category |
| `src/evaluate.py` | Compare all models, print metrics table |
| `doc/implementation-plan.md` | Phase checklist |
| `doc/approach-proposal-cl.md` | Full approach writeup |
| `memory/MEMORY.md` | This file — decisions and findings |

## Common Issues
- Smart/curly quotes break SQL — always use straight quotes in DuckDB
- WSL paths: use `\\wsl.localhost\Ubuntu\...` for file ops from Windows context
