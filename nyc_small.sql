LOAD spatial; -- noqa
LOAD httpfs;

SET s3_region='us-west-2';

COPY(                                       -- COPY <query> TO <output> saves the results to disk.
  SELECT
    id,
    names.primary as name,
    confidence,
    CAST(categories AS JSON) as categories,
    basic_category,
    CAST(taxonomy AS JSON) as taxonomy,
    CAST(websites AS JSON) as websites,
    CAST(socials AS JSON) as socials,
    CAST(emails AS JSON) as emails,
    CAST(phones AS JSON) as phones,
    CAST(brand AS JSON) as brand,
    CAST(addresses AS JSON) as addresses,
    CAST(sources AS JSON) as sources,
    operating_status,
    geometry
  FROM
    read_parquet('s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*', filename=true, hive_partitioning=1)
  WHERE
    categories.primary = 'restaurant'
    AND bbox.xmin BETWEEN -75 AND -74       -- Only use the bbox min values
    AND bbox.ymin BETWEEN 40 AND 41         -- because they are point geometries.
) TO 'nyc_small.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON');
-- TO 'nyc_restaurants_small.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON');
