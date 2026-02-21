import duckdb, json, os
from obstore.store import S3Store

os.makedirs("data", exist_ok=True)

store = S3Store("overturemaps-us-west-2", region="us-west-2", skip_signature=True)

releases = store.list_with_delimiter("release/")

output = {}

for idx, release in enumerate(sorted(releases.get("common_prefixes"), reverse=True)):
    path = release.split("/")[1]
    if idx == 0:
        output["latest"] = path
        output["releases"] = []
    output["releases"].append(path)

    print(f"  {idx}: {path}")

with open("data/releases.json", "w") as output_file:
    output_file.write(json.dumps(output, indent=4))

choice = int(input("\nSelect a release number: "))
selected = output["releases"][choice]
print(f"\nUsing release: {selected}")

conn = duckdb.connect(f"data/release_{selected}.ddb")

conn.sql(
    f"""
INSTALL spatial;
LOAD spatial;
CREATE OR REPLACE VIEW place AS (
  SELECT * FROM read_parquet('s3://overturemaps-us-west-2/release/{selected}/theme=places/type=place/*.parquet')
);
"""
)