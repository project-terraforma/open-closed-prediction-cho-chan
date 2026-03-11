# Spatial KNN Neighborhood Features
SF Registered Business Dataset — Feature Documentation

## Introduction

A place does not exist in isolation. Whether a business is open or closed is
heavily influenced by the health of the commercial environment around it.
A restaurant in a block where 40% of businesses have closed in the last year
is fundamentally different from an identical restaurant on a thriving street —
even if their individual attributes (name, category, source count) look the same.

This feature set captures that neighborhood context. Using the SF Registered
Business Locations dataset as a spatial reference pool (356,000+ businesses),
we build a BallTree spatial index and, for every query place, compute
neighborhood health statistics at three radii:

- `100m` — same block / immediate surroundings
- `250m` — local neighborhood cluster
- `500m` — district-level commercial environment

The result is 33 spatial features per place (11 features × 3 radii) that
describe the closure rate, vitality, economic diversity, and commercial
character of the area surrounding each business.

These features are SF-only (the reference dataset is SF-specific) and are
designed to supplement — not replace — the core Overture record features.
For Overture places outside SF, all spatial features default to NaN, and
the model falls back to the Overture-native feature set.

---

## How It Feeds Into the Pipeline

1. `fetch-sf-registered-businesses.py` ingests the raw GeoJSON into DuckDB and derives a confident open/closed label for each SF business.

2. `spatial_knn_features.py` builds a BallTree over all 356k SF businesses and exposes a `transform(query_df)` method.

3. During feature engineering, for any Overture place with a lat/lon in SF, `transform()` is called to append the 33 spatial features to the existing 24 Overture features — bringing the total feature vector to 57 dimensions.

4. The combined feature vector is passed to the MLP encoder, GBM baseline, and NCM/SLDA classifiers as normal. The spatial features are treated as regular input dimensions — no special handling required.

5. For the continual learning pipeline, the BallTree is rebuilt each time the SF dataset is refreshed (it is a live dataset), ensuring neighborhood statistics stay current without retraining the encoder.

---

## Feature Reference

Feature names follow the pattern: `<feature_name>_<radius>m`
e.g. `closure_rate_100m`, `closure_rate_250m`, `closure_rate_500m`

### Closure Signals

**`closure_rate_{r}m`**
Proportion of neighboring businesses with a known status that are marked closed — the core neighborhood health signal indicating area-level decline.

**`admin_closed_rate_{r}m`**
Proportion of neighbors flagged as administratively closed by the city, meaning they stopped filing or communicating with the tax office for 3+ years — a leading indicator of silent business failure in the area.

### Vitality Signals

**`n_businesses_{r}m`**
Total count of SF-registered businesses within the radius, measuring commercial density — sparse areas may have less reliable neighborhood signals and are flagged via NaN when count is zero.

**`new_business_rate_{r}m`**
Proportion of neighboring businesses that opened within the last year, where a high rate suggests an active and growing commercial environment.

**`median_business_age_{r}m`**
Median age in days of neighboring businesses, where a high median indicates a stable, established area with long-running tenants.

**`business_age_std_{r}m`**
Standard deviation of business ages in the neighborhood, where a high value signals a transitional area with significant turnover and churn.

### Diversity Signal

**`naics_diversity_{r}m`**
Count of unique NAICS industry codes among neighbors, measuring economic diversity — highly diverse areas tend to be more resilient to sector-specific downturns than single-category commercial corridors.

### Tax Signals

**`parking_tax_rate_{r}m`**
Proportion of neighboring businesses that pay the SF parking tax, serving as a proxy for car-oriented commercial areas such as auto dealerships, garages, and parking-dependent retail.

**`tot_tax_rate_{r}m`**
Proportion of neighboring businesses that pay the transient occupancy tax, serving as a proxy for hotel and short-term rental density in the area.

### Category-Specific Signal

**`same_category_closure_rate_{r}m`**
Closure rate restricted to neighbors sharing the same NAICS industry code as the query place, capturing how businesses of the same type are faring in the immediate area — more targeted than the overall closure rate. Returns NaN if the query place has no NAICS code or no same-category neighbors exist within the radius.

---

## Implementation Notes

- All features return NaN (not 0) when there are zero neighbors at a radius. This distinction matters for XGBoost/GBM: zero businesses and unknown are different signals and should be treated differently by the model.

- `closure_rate` and `same_category_closure_rate` are computed only over neighbors with a confidently known status (open or closed). Uncertain records — those that quietly stopped operating without officially closing — are excluded so they do not dilute the closure rate toward open.

- The BallTree is built once and reused across all query points and all radii, querying at the maximum radius (500m) and using binary search (`searchsorted`) to find radius boundaries for 100m and 250m — avoiding redundant tree traversals.

- Source file: `src/ml/spatial_knn_features.py`
- Reference data: `data/sf_registered_businesses.ddb` (generated by `src/util/fetch-sf-registered-businesses.py`)
- Raw dataset: [Registered Business Locations — San Francisco](https://data.sfgov.org/Economy-and-Community/Registered-Business-Locations-San-Francisco/g8m3-pdis/about_data) (SF Open Data Portal)
