#!/bin/bash
latest="data/release_2026-02-18.0.ddb"
second="data/release_2026-01-21.0.ddb"

duckdb -c "
LOAD spatial;
ATTACH '$latest' AS feb;
ATTACH '$second' AS jan;

SELECT 'Feb count' as label, COUNT(*) as cnt FROM feb.place
UNION ALL
SELECT 'Jan count', COUNT(*) FROM jan.place;

SELECT 'Feb' as release, operating_status, COUNT(*) as cnt FROM feb.place GROUP BY operating_status
UNION ALL
SELECT 'Jan', operating_status, COUNT(*) FROM jan.place GROUP BY operating_status
ORDER BY release, operating_status;

SELECT COUNT(*) AS removed_count FROM jan.place j
LEFT JOIN feb.place f ON j.id = f.id
WHERE f.id IS NULL;

SELECT COUNT(*) AS added_count FROM feb.place f
LEFT JOIN jan.place j ON f.id = j.id
WHERE j.id IS NULL;

SELECT j.operating_status AS jan_status, f.operating_status AS feb_status, COUNT(*) AS cnt
FROM jan.place j
JOIN feb.place f ON j.id = f.id
WHERE j.operating_status IS DISTINCT FROM f.operating_status
GROUP BY j.operating_status, f.operating_status
ORDER BY cnt DESC;

SELECT COUNT(*) AS confidence_changed FROM jan.place j
JOIN feb.place f ON j.id = f.id
WHERE j.confidence IS DISTINCT FROM f.confidence;
"
