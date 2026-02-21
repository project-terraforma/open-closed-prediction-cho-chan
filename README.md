# open-closed-prediction-cho-chan

Project C for Winter 2026 — Overture Maps Foundation

Caleb Cho, Kevin Chan

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

### 3. Yelp Academic Dataset (JSONL)

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
