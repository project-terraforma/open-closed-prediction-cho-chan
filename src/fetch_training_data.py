"""
fetch_training_data.py

Grabs a balanced sample of US places from the latest Overture release
and saves it to data/overture_us_balanced_{release}.parquet for training.

A few things that are different from overture.parquet:
  - 'name'            : just the primary name string instead of the full names struct
  - 'basic_category'  : simpler category field that overture added recently
  - 'taxonomy'        : full category hierarchy (also new)
  - 'operating_status': the raw overture field ("open" / "permanently_closed" / "temporarily_closed")
  - 'open'            : 1 = open, 0 = not open (what we're trying to predict)

Usage:
    python fetch_training_data.py
"""

import duckdb
from pathlib import Path
from obstore.store import S3Store

# how many records to grab per class — keeps the dataset 50/50 so the model
# doesn't just learn to always predict "open". lower this for a quick test run.
SAMPLE_PER_CLASS = 50_000

# pull the latest release from the same S3 bucket my teammate uses
store = S3Store("overturemaps-us-west-2", region="us-west-2", skip_signature=True)
releases = store.list_with_delimiter("release/")
latest_release = sorted(releases.get("common_prefixes"), reverse=True)[0].split("/")[1]

S3_PATH = f"s3://overturemaps-us-west-2/release/{latest_release}/theme=places/type=place/*"

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)
output_path = str(data_dir / f"overture_us_balanced_{latest_release}.parquet")

print("Setting up DuckDB...")
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region = 'us-west-2';")

print(f"Target : {SAMPLE_PER_CLASS:,} open  +  {SAMPLE_PER_CLASS:,} not-open  =  {SAMPLE_PER_CLASS * 2:,} total rows")
print(f"Source : {S3_PATH}")
print(f"Output : {output_path}")
print()
print("Fetching (this may take a few minutes over S3)...")

# LIMIT just takes the first N rows from whatever parquet files get scanned first.
# since the files are partitioned by geography, this probably skews toward certain
# regions in the US — not perfectly random, but good enough to start training with.
# if we need a proper random sample later we can switch to USING SAMPLE RESERVOIR.
query = f"""
COPY (
    WITH not_open AS (
        SELECT
            id,
            names.primary                   AS name,
            confidence,
            CAST(categories  AS JSON)       AS categories,
            basic_category,
            CAST(taxonomy    AS JSON)       AS taxonomy,
            CAST(websites    AS JSON)       AS websites,
            CAST(socials     AS JSON)       AS socials,
            CAST(emails      AS JSON)       AS emails,
            CAST(phones      AS JSON)       AS phones,
            CAST(brand       AS JSON)       AS brand,
            CAST(addresses   AS JSON)       AS addresses,
            CAST(sources     AS JSON)       AS sources,
            operating_status,
            0                               AS open,
            geometry
        FROM read_parquet('{S3_PATH}', hive_partitioning=1)
        WHERE addresses IS NOT NULL
          AND len(addresses) > 0
          AND addresses[1].country = 'US'
          AND operating_status IN ('permanently_closed', 'temporarily_closed')
        LIMIT {SAMPLE_PER_CLASS}
    ),
    open_places AS (
        SELECT
            id,
            names.primary                   AS name,
            confidence,
            CAST(categories  AS JSON)       AS categories,
            basic_category,
            CAST(taxonomy    AS JSON)       AS taxonomy,
            CAST(websites    AS JSON)       AS websites,
            CAST(socials     AS JSON)       AS socials,
            CAST(emails      AS JSON)       AS emails,
            CAST(phones      AS JSON)       AS phones,
            CAST(brand       AS JSON)       AS brand,
            CAST(addresses   AS JSON)       AS addresses,
            CAST(sources     AS JSON)       AS sources,
            operating_status,
            1                               AS open,
            geometry
        FROM read_parquet('{S3_PATH}', hive_partitioning=1)
        WHERE addresses IS NOT NULL
          AND len(addresses) > 0
          AND addresses[1].country = 'US'
          AND operating_status = 'open'
        LIMIT {SAMPLE_PER_CLASS}
    )
    SELECT * FROM not_open
    UNION ALL
    SELECT * FROM open_places
) TO '{output_path}' (FORMAT PARQUET);
"""

con.execute(query)

print(f"\nDone. Saved to: {output_path}")
print("Run `python viewData.py data/overture_us_balanced.parquet` to inspect the result.")
