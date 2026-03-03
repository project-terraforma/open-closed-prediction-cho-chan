import argparse, duckdb, json, os
from obstore.store import S3Store

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true", help="Download parquet files locally via obstore instead of querying S3 directly")
args = parser.parse_args()

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

if args.download:
    parquet_dir = f"data/parquet/{selected}"
    os.makedirs(parquet_dir, exist_ok=True)

    prefix = f"release/{selected}/theme=places/type=place/"
    listing = store.list_with_delimiter(prefix)
    objects = listing.get("objects", [])

    print(f"\nFound {len(objects)} files, downloading...")
    for obj in objects:
        path = obj["path"]
        if not path.endswith(".parquet"):
            continue
        filename = os.path.basename(path)
        local_path = os.path.join(parquet_dir, filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"  Skipping {filename} (already exists)")
            continue
        print(f"  Downloading {filename}...")
        result = store.get(path)
        with open(local_path, "wb") as f:
            f.write(result.bytes())
    print("Download complete!")

    parquet_glob = f"{parquet_dir}/*.parquet"
else:
    parquet_glob = f"s3://overturemaps-us-west-2/release/{selected}/theme=places/type=place/*.parquet"

conn = duckdb.connect(f"data/release_{selected}.ddb")

conn.sql(
    f"""
INSTALL spatial;
LOAD spatial;
CREATE OR REPLACE VIEW place AS (
  SELECT * FROM read_parquet('{parquet_glob}')
);
"""
)
