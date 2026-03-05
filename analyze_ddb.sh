#!/bin/bash
export PATH="$HOME/.duckdb/cli/latest:$PATH"
latest="data/release_2026-02-18.0.ddb"
second="data/release_2026-01-21.0.ddb"

duckdb -c "
LOAD spatial;
ATTACH '$latest' AS feb;
ATTACH '$second' AS jan;

/*
-- Total place count per release
SELECT 'Feb count' as label, COUNT(*) as cnt FROM feb.place
UNION ALL
SELECT 'Jan count', COUNT(*) FROM jan.place;

-- Operating status breakdown per release
SELECT 'Feb' as release, operating_status, COUNT(*) as cnt FROM feb.place GROUP BY operating_status
UNION ALL
SELECT 'Jan', operating_status, COUNT(*) FROM jan.place GROUP BY operating_status
ORDER BY release, operating_status;

-- Places in Jan but not Feb (removed)
SELECT COUNT(*) AS removed_count FROM jan.place j
LEFT JOIN feb.place f ON j.id = f.id
WHERE f.id IS NULL;

-- Places in Feb but not Jan (newly added)
SELECT COUNT(*) AS added_count FROM feb.place f
LEFT JOIN jan.place j ON f.id = j.id
WHERE j.id IS NULL;

-- Places where confidence score changed between releases
SELECT COUNT(*) AS confidence_changed FROM jan.place j
JOIN feb.place f ON j.id = f.id
WHERE j.confidence IS DISTINCT FROM f.confidence;

-- Sample 100 removed places with confidence = 0 (NYC bbox only)
SELECT j.id, j.names.primary as name, j.categories.primary as category, j.operating_status, j.confidence
FROM jan.place j
LEFT JOIN feb.place f ON j.id = f.id
WHERE f.id IS NULL
  AND j.confidence = 0
  AND j.bbox.xmin BETWEEN -74.05 AND -73.75
  AND j.bbox.ymin BETWEEN 40.60 AND 40.90
LIMIT 100;


-- Sample 20 places with "Vinted" in the name that are present in Feb release (NYC bbox only)
SELECT id, names.primary as name, confidence
FROM feb.place
WHERE names.primary ILIKE '%Vinted%'
  AND bbox.xmin BETWEEN -74.05 AND -73.75
  AND bbox.ymin BETWEEN 40.60 AND 40.90
LIMIT 20;

-- Places where operating_status changed between releases
SELECT j.operating_status AS jan_status, f.operating_status AS feb_status, COUNT(*) AS cnt
FROM jan.place j
JOIN feb.place f ON j.id = f.id
WHERE j.operating_status IS DISTINCT FROM f.operating_status
GROUP BY j.operating_status, f.operating_status
ORDER BY cnt DESC;

-- What values does operating_status have?
SELECT 'jan' as release, operating_status, COUNT(*) as cnt FROM jan.place GROUP BY operating_status;
SELECT 'feb' as release, operating_status, COUNT(*) as cnt FROM feb.place GROUP BY operating_status;

-- How many IDs match between releases?
SELECT COUNT(*) AS matched_ids FROM jan.place j JOIN feb.place f ON j.id = f.id;

-- Count of Feb entries with operating_status = closed
SELECT COUNT(*) AS feb_closed_count FROM feb.place WHERE operating_status = 'closed';

-- Describe schema to check for a region column
DESCRIBE feb.place;

-- Sample coordinates of closed entries to see where they are
SELECT bbox.xmin AS lon, bbox.ymin AS lat, names.primary AS name, addresses[1].region AS state
FROM feb.place
WHERE operating_status = 'closed'
LIMIT 50;


-- Operating status stats per state (Feb)
SELECT
  addresses[1].region AS state,
  COUNT(*) FILTER (WHERE operating_status = 'open') AS open,
  COUNT(*) FILTER (WHERE operating_status = 'closed') AS closed,
  COUNT(*) FILTER (WHERE operating_status = 'temporarily_closed') AS temp,
  COUNT(operating_status) AS total
FROM feb.place
GROUP BY state
ORDER BY total DESC;
*/

SELECT
  addresses[1].region AS state,
  COUNT(*) FILTER (WHERE operating_status = 'temporarily closed') AS temp_closed
FROM feb.place
GROUP BY state
ORDER BY temp_closed DESC;

/*
-- Closed entries by region with confidence stats
SELECT
  addresses[1].region AS region,
  COUNT(*) AS closed_count,
  ROUND(AVG(confidence), 4) AS avg_confidence,
  ROUND(MIN(confidence), 4) AS min_confidence,
  ROUND(MAX(confidence), 4) AS max_confidence
FROM feb.place
WHERE operating_status = 'closed'
GROUP BY region
ORDER BY closed_count DESC;

-- 3 sample closed entries from each of the top-10 regions by closed count
WITH top_regions AS (
  SELECT addresses[1].region AS region
  FROM feb.place
  WHERE operating_status = 'closed'
  GROUP BY region
  ORDER BY COUNT(*) DESC
  LIMIT 10
),
ranked AS (
  SELECT p.*, addresses[1].region AS region,
         ROW_NUMBER() OVER (PARTITION BY addresses[1].region ORDER BY p.id) AS rn
  FROM feb.place p
  WHERE operating_status = 'closed'
    AND addresses[1].region IN (SELECT region FROM top_regions)
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn <= 3
ORDER BY region, rn;

-- Export all closed entries to JSON in native format (matches project_c_samples.json)
COPY (
  SELECT
    id, geometry, bbox, 'place' AS "type", 0 AS version, sources, names, categories,
    confidence, websites, socials, emails, phones, brand, addresses,
    CASE WHEN operating_status = 'closed' THEN 0 ELSE 1 END AS open
  FROM feb.place
  WHERE operating_status = 'closed'
) TO 'data/overture-feb-release-closed.json' (FORMAT JSON, ARRAY false);
*/
"
