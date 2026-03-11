"""
Ingest the SF Registered Business Locations dataset into DuckDB.

Source: https://data.sfgov.org/Economy-and-Community/Registered-Business-Locations-San-Francisco/g8m3-pdis/about_data
Download the GeoJSON export and place it at GEOJSON_PATH before running.
"""
import duckdb, os

GEOJSON_PATH = "data/Registered_Business_Locations_SF_.geojson"
DB_PATH = "data/sf_registered_businesses.ddb"

os.makedirs("data", exist_ok=True)

conn = duckdb.connect(DB_PATH)

conn.sql("INSTALL spatial; LOAD spatial;")
conn.sql("INSTALL json; LOAD json;")

print(f"Loading {GEOJSON_PATH}...")

conn.sql(f"""
    CREATE OR REPLACE TABLE sf_registered_businesses AS (
        SELECT
            -- Identifiers (use for dedup/joins, not model features)
            uniqueid,           -- formula: ttxid-certificate_number
            certificate_number, -- 7-digit business account number
            ttxid,              -- location identifier

            -- Business identity
            ownership_name,     -- business owner(s) name
            dba_name,           -- doing business as / location name (fuzzy name match for overture)

            -- Physical address
            full_business_address,
            city,
            state,
            business_zip,

            -- Business lifecycle dates (key signals for open/closed)
            TRY_CAST(dba_start_date      AS TIMESTAMP) AS dba_start_date,      -- when the business account opened
            TRY_CAST(dba_end_date        AS TIMESTAMP) AS dba_end_date,        -- when the business account closed (null = still active)
            TRY_CAST(location_start_date AS TIMESTAMP) AS location_start_date, -- when business started at this location
            TRY_CAST(location_end_date   AS TIMESTAMP) AS location_end_date,   -- when business left this location (not necessarily closed)

            -- Closure flag (no filing/contact with TTX for 3+ years, or notified closed by another city dept)
            administratively_closed,

            -- Mailing address (may differ from business address)
            mailing_address_1,
            mail_city,
            mail_state,
            mail_zipcode,

            -- Industry classification (NAICS) — maps to Overture category
            naic_code,                      -- NAICS industry code
            naic_code_description,          -- "Multiple" if more than one
            naics_code_descriptions_list,   -- semicolon-separated list of all NAICS descriptions

            -- License type
            lic,                            -- license code(s), space-separated if multiple
            lic_code_description,           -- "Multiple" if more than one
            lic_code_descriptions_list,     -- semicolon-separated list of all license descriptions

            -- Tax flags (business type indicators)
            TRY_CAST(parking_tax             AS BOOLEAN) AS parking_tax,            -- pays SF parking tax
            TRY_CAST(transient_occupancy_tax AS BOOLEAN) AS transient_occupancy_tax, -- pays hotel/short-term rental tax

            -- SF geographic boundaries
            business_corridor,                  -- SF business corridor (nullable)
            neighborhoods_analysis_boundaries,  -- SF analysis neighborhood
            supervisor_district,                -- SF supervisor district
            community_benefit_district,         -- SF community benefit district (nullable)

            -- Data freshness metadata
            TRY_CAST(data_as_of    AS TIMESTAMP) AS data_as_of,       -- when source system last updated this record
            TRY_CAST(data_loaded_at AS TIMESTAMP) AS data_loaded_at,  -- when record was loaded to open data portal

            -- Geometry (lat/lng point)
            geom
            -- FIX: ST_Read() on GeoJSON exposes all feature properties as flat top-level
            -- columns directly — do NOT use properties->>'column_name' extraction syntax
            -- (causes "Referenced column 'properties' not found" Binder Error).
            -- Similarly, geom is already a geometry object; do NOT wrap it in
            -- ST_GeomFromGeoJSON() — that also fails with ST_Read() output.
        FROM ST_Read('{GEOJSON_PATH}')
    );
""")

count = conn.sql("SELECT COUNT(*) FROM sf_registered_businesses").fetchone()[0]
print(f"Loaded {count:,} records.")

# Derive open/closed label
conn.sql("""
    CREATE OR REPLACE VIEW sf_businesses_labeled AS (
        SELECT
            *,
            CASE
                WHEN dba_end_date IS NOT NULL AND dba_end_date <= CURRENT_TIMESTAMP THEN 'closed'
                WHEN administratively_closed = 'true' THEN 'closed'
                WHEN dba_end_date IS NULL
                     AND location_end_date IS NULL
                     AND administratively_closed IS NULL
                     AND data_as_of >= CURRENT_TIMESTAMP - INTERVAL '6 months' THEN 'open'
                ELSE NULL  -- uncertain: exclude from training
            END AS status
        FROM sf_registered_businesses
    );
""")

summary = conn.sql("""
    SELECT
        COALESCE(status, 'uncertain (excluded)') AS status,
        COUNT(*) AS count
    FROM sf_businesses_labeled
    GROUP BY status
    ORDER BY status
""").fetchall()

print("\nLabel distribution:")
for row in summary:
    print(f"  {row[0]}: {row[1]:,}")

print(f"\nDatabase saved to: {DB_PATH}")
print("Tables:  sf_registered_businesses")
print("Views:   sf_businesses_labeled")
