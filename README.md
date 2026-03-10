# open-closed-prediction-cho-chan

Project C for Winter 2026 — Overture Maps Foundation

Caleb Cho, Kevin Chan

## OKRs

**Objective 1:** Deliver a scalable, continual-learning model that improves over baseline for open/closed prediction.
- KR 1.1: Achieve AUC-ROC ≥ 0.78 and F1 ≥ 0.40, with ≥ 5% AUC-ROC improvement over GBM.
- KR 1.2: Evaluate per-category performance only if at least 20 closed samples exist.
- KR 1.3: Maintain scalability to 100M+ places with efficient inference and incremental updates per release.

**Objective 2:** Make a data-driven recommendation on the optimal model and feature set.
- KR 2.1: Identify and rank the top 5 predictive features.
- KR 2.2: Benchmark model architectures and produce a cost vs accuracy comparison.
- KR 2.3: Deliver a final deployment recommendation supported by empirical evidence.

## Goal

Predict whether a place (business/service/amenity) is open or permanently closed using its attributes and cross-source signals. The solution must scale to score 100M+ places per release with sub-millisecond inference.

## Approach

We use a small MLP encoder (~6k parameters) trained on place features to produce 32-dim embeddings, then classify with continual learning methods (Nearest Class Mean and Streaming LDA) that can adapt incrementally as new Overture releases arrive without retraining the encoder or risking catastrophic forgetting.

See [approach-proposal-cl.md](approach-proposal-cl.md) for the full proposal and [implementation-plan.md](implementation-plan.md) for the task checklist.

## Datasets

All data files live in the `data/` folder (gitignored).

### 1. Overture Maps Places

Places pulled from the Overture Maps S3 parquet via DuckDB (see `nyc_small.sql`). GeoJSON FeatureCollection format.

| Field | Type | Description |
|---|---|---|
| `id` | string | Overture place ID |
| `name` | string | Primary name of the place |
| `confidence` | float | Overture's existing confidence score (0-1) |
| `categories` | object | `{primary: string, alternate: [string]}` |
| `basic_category` | string | Simplified category name |
| `taxonomy` | object | `{primary: string, hierarchy: [string], alternates: [string]}` |
| `websites` | array | List of website URLs |
| `socials` | array | List of social media URLs (mostly Facebook) |
| `emails` | array | List of email addresses |
| `phones` | array | List of phone numbers |
| `brand` | object | `{names: {primary: string}}` or null |
| `addresses` | array | `[{freeform, locality, postcode, region, country}]` |
| `sources` | array | Contributing data sources (see below) |
| `operating_status` | string | `"open"`, `"permanently_closed"`, or `"temporarily_closed"` |
| `geometry` | GeoJSON Point | Coordinates `[lon, lat]` |

**Sources array** — each entry:
| Field | Type | Description |
|---|---|---|
| `dataset` | string | Provider name: `meta`, `Microsoft`, `Foursquare`, or `Overture` |
| `record_id` | string | Provider's internal record ID |
| `update_time` | string | ISO timestamp of last update from this source |
| `confidence` | float | Source-level confidence score |
| `property` | string | What the source contributes (empty = full record, `/properties/existence` = existence vote, `/properties/confidence` = Overture's own confidence entry) |

### 2. Labeled Sample (parquet)

~3k place pairs for entity matching validation. Each row is a pair of places with a match label. **Note:** This dataset is for matching validation (label 1=match, 0=no match), not directly for open/closed prediction. However, the `id`-side fields contain the same Overture attributes we use for feature engineering.

| Field | Type | Description |
|---|---|---|
| `label` | int | 1 = match, 0 = no match (entity matching label) |
| `id` | string | Overture place ID (candidate) |
| `sources` | string (JSON) | Source array for the candidate place — entries have `dataset` (`meta`/`msft`), `record_id`, `update_time`, optionally `confidence` and `property` |
| `names` | string (JSON) | `{primary: string}` |
| `categories` | string (JSON) | `{primary: string, alternate: [string]}` |
| `confidence` | float | Overture confidence (0-1) |
| `websites` | string (JSON) | Array of URLs or null |
| `socials` | string (JSON) | Array of social URLs or null |
| `emails` | string (JSON) | Array of emails or null |
| `phones` | string (JSON) | Array of phone numbers or null |
| `brand` | string (JSON) | `{names: {primary: string}}` or null |
| `addresses` | string (JSON) | `[{freeform, locality, postcode, region, country}]` |
| `base_*` | same as above | Corresponding fields for the base place in the pair |

### 3. SF Registered Business Locations (GeoJSON)

Source: [SF Open Data — Registered Business Locations](https://data.sfgov.org/Economy-and-Community/Registered-Business-Locations-San-Francisco/g8m3-pdis)
Snapshot: `data/sf_open_dataset_20260309.geojson` (downloaded 2026-03-09)

Used to augment training labels via geo+name matching against Overture places (`src/util/sf_lookup.py`).

**Label derivation:** A business is labelled `closed=1` if `dba_end_date` is set and ≤ reference date (2026-03-09). Otherwise `open=1`.

**Schema** (33 columns):

| API Field | Display Name | Type | Description |
|---|---|---|---|
| `certificate_number` | Business Account Number | Text | 7-digit account number |
| `ttxid` | Location Id | Text | Location identifier |
| `ownership_name` | Ownership Name | Text | Business owner(s) name |
| `dba_name` | DBA Name | Text | Doing Business As / location name |
| `full_business_address` | Street Address | Text | Business street address |
| `city` / `state` / `business_zip` | City / State / Zip | Text | Location fields |
| `dba_start_date` | Business Start Date | Timestamp | Start of business |
| `dba_end_date` | Business End Date | Timestamp | End date — **key label field** (null = open) |
| `location_start_date` | Location Start Date | Timestamp | Start at this location |
| `location_end_date` | Location End Date | Timestamp | End at this location |
| `administratively_closed` | Administratively Closed | Text | Marked closed by TTX after 3 years of non-filing |
| `naic_code` / `naic_code_description` | NAICS Code | Text | Industry classification |
| `lic` / `lic_code_description` | LIC Code | Text | License type |
| `parking_tax` | Parking Tax | Bool | Pays parking tax |
| `transient_occupancy_tax` | Transient Occupancy Tax | Bool | Pays hotel/TOT tax |
| `location` | Business Location | Point | Lat/lon for mapping |
| `business_corridor` | Business Corridor | Text | Named commercial corridor (if any) |
| `neighborhoods_analysis_boundaries` | Neighborhood | Text | SF analysis neighborhood |
| `supervisor_district` | Supervisor District | Text | Board of Supervisors district |
| `community_benefit_district` | Community Benefit District | Text | CBD boundary |
| `data_as_of` / `data_loaded_at` | Timestamps | Timestamp | Source/portal update times |

**Record counts (snapshot 2026-03-09):**

| Status | Count |
|---|---|
| Total records | 356,351 |
| Open (no `dba_end_date`) | 164,271 |
| Closed (`dba_end_date` ≤ 2026-03-09) | 192,068 |
| Future end date (> ref) | 12 |

**`dba_end_date` distribution for closed records (selected years):**

| Year | Count | Notes |
|---|---|---|
| pre-2010 | ~800 | Sparse; legacy records |
| 2011 | 1,350 | |
| 2012 | 2,346 | |
| 2013 | 4,677 | |
| 2014 | 9,368 | |
| 2015 | 12,526 | |
| 2016 | 11,714 | |
| 2017 | 12,551 | |
| **2018** | **34,723** | **Anomalous spike — likely batch data correction** |
| 2019 | 18,958 | |
| 2020 | 14,591 | |
| 2021 | 16,275 | |
| 2022 | 16,034 | |
| 2023 | 12,499 | |
| 2024 | 12,154 | |
| 2025 | 9,268 | |
| 2026 | 1,043 | |

> **Data quality note:** The 2018 spike (34,723 closures) is ~2x the surrounding years and is likely a mass administrative correction or batch import artifact, not a real closure wave. Closure labels from that year may have lower reliability. The staleness filter in `sf_lookup.py` (`--use-overture-label` / default: skip) helps handle cases where Overture's `update_time` post-dates the SF closure date.

### 4. Yelp Academic Dataset (JSONL)

JSONL format (one JSON object per line). US-only business listings. Used as external enrichment and an independent label source for open/closed status.

| Field | Type | Description |
|---|---|---|
| `business_id` | string | Yelp's internal ID |
| `name` | string | Business name |
| `address` | string | Street address |
| `city` | string | City |
| `state` | string | US state code |
| `postal_code` | string | ZIP code |
| `latitude` | float | Latitude |
| `longitude` | float | Longitude |
| `stars` | float | Star rating (1.0-5.0) |
| `review_count` | int | Number of reviews |
| `is_open` | int | 1 = open, 0 = closed |
| `attributes` | object | Business attributes (varies: `ByAppointmentOnly`, `BusinessAcceptsCreditCards`, `WiFi`, `BusinessParking`, etc.) |
| `categories` | string | Comma-separated category list |
| `hours` | object | `{Monday: "8:0-22:0", ...}` or null |
