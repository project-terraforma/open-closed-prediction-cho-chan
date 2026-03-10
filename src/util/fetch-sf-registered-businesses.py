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
            properties->>'uniqueid'                     AS uniqueid,           -- formula: ttxid-certificate_number
            properties->>'certificate_number'           AS certificate_number, -- 7-digit business account number
            properties->>'ttxid'                        AS ttxid,              -- location identifier

            -- Business identity
            properties->>'ownership_name'               AS ownership_name,     -- business owner(s) name
            properties->>'dba_name'                     AS dba_name,           -- doing business as / location name (fuzzy name match for overture)

            -- Physical address
            properties->>'full_business_address'        AS full_business_address,
            properties->>'city'                         AS city,
            properties->>'state'                        AS state,
            properties->>'business_zip'                 AS business_zip,

            -- Business lifecycle dates (key signals for open/closed)
            TRY_CAST(properties->>'dba_start_date'      AS TIMESTAMP) AS dba_start_date,      -- when the business account opened
            TRY_CAST(properties->>'dba_end_date'        AS TIMESTAMP) AS dba_end_date,        -- when the business account closed (null = still active)
            TRY_CAST(properties->>'location_start_date' AS TIMESTAMP) AS location_start_date, -- when business started at this location
            TRY_CAST(properties->>'location_end_date'   AS TIMESTAMP) AS location_end_date,   -- when business left this location (not necessarily closed)

            -- Closure flag (no filing/contact with TTX for 3+ years, or notified closed by another city dept)
            properties->>'administratively_closed'      AS administratively_closed,

            -- Mailing address (may differ from business address)
            properties->>'mailing_address_1'            AS mailing_address_1,
            properties->>'mail_city'                    AS mail_city,
            properties->>'mail_state'                   AS mail_state,
            properties->>'mail_zipcode'                 AS mail_zipcode,

            -- Industry classification (NAICS) — maps to Overture category
            properties->>'naic_code'                    AS naic_code,                      -- NAICS industry code
            properties->>'naic_code_description'        AS naic_code_description,          -- "Multiple" if more than one
            properties->>'naics_code_descriptions_list' AS naics_code_descriptions_list,   -- semicolon-separated list of all NAICS descriptions

            -- License type
            properties->>'lic'                          AS lic,                            -- license code(s), space-separated if multiple
            properties->>'lic_code_description'         AS lic_code_description,           -- "Multiple" if more than one
            properties->>'lic_code_descriptions_list'   AS lic_code_descriptions_list,     -- semicolon-separated list of all license descriptions

            -- Tax flags (business type indicators)
            TRY_CAST(properties->>'parking_tax'         AS BOOLEAN) AS parking_tax,            -- pays SF parking tax
            TRY_CAST(properties->>'transient_occupancy_tax' AS BOOLEAN) AS transient_occupancy_tax, -- pays hotel/short-term rental tax

            -- SF geographic boundaries
            properties->>'business_corridor'            AS business_corridor,                  -- SF business corridor (nullable)
            properties->>'neighborhoods_analysis_boundaries' AS neighborhoods_analysis_boundaries, -- SF analysis neighborhood
            properties->>'supervisor_district'          AS supervisor_district,                -- SF supervisor district
            properties->>'community_benefit_district'   AS community_benefit_district,         -- SF community benefit district (nullable)

            -- Data freshness metadata
            TRY_CAST(properties->>'data_as_of'          AS TIMESTAMP) AS data_as_of,       -- when source system last updated this record
            TRY_CAST(properties->>'data_loaded_at'      AS TIMESTAMP) AS data_loaded_at,   -- when record was loaded to open data portal

            -- Geometry (lat/lng point)
            ST_GeomFromGeoJSON(geometry::VARCHAR)       AS geom
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
