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
*/ 
-- Count of Feb entries with operating_status = closed
SELECT COUNT(*) AS feb_closed_count FROM feb.place WHERE operating_status = 'closed';
"
