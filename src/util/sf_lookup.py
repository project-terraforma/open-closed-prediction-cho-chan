"""
sf_lookup.py
------------
Match Overture place records to the SF registered business registry.

For each Overture place, finds the nearest SF business within a radius using a
KD-tree, then scores candidates by name similarity (same logic as fsq_lookup.py).
A matched SF record contributes its dba_end_date label (closed/open) to the
Overture record — enabling Option A augmentation without schema conversion.

Two input modes:
  --overture  : match against a labeled NDJSON file (e.g. project_c_samples.json)
                → validates SF labels vs Overture labels
  --from-parquet : query Overture parquet files for SF-area places via DuckDB,
                then match → produces large augmentation set with SF-derived labels

Output NDJSON usage:
  1. Validation  — compare overture_open vs sf_open (agreement rate)
  2. Augmentation — matched records written with open=sf_open, ready for
                    split.py --augment

Run:
    # Validate against labeled NDJSON:
    python src/util/sf_lookup.py --overture data/project_c_samples.json --verbose

    # Build augmentation from parquet (main use case):
    python src/util/sf_lookup.py --from-parquet data/parquet/2026-02-18.0/*.zstd.parquet \\
        --augment-output data/sf_aug.json

    # Adjust match thresholds:
    python src/util/sf_lookup.py --from-parquet data/parquet/2026-02-18.0/*.zstd.parquet \\
        --min-nsim 0.65 --min-lsim 0.85 --augment-output data/sf_aug.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Reuse name similarity logic from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from fsq_lookup import name_similarity  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SF_DEFAULT     = Path("data/sf_open_dataset_20260309.geojson")
OVERTURE_DEFAULT = Path("data/project_c_samples.json")
OUTPUT_DEFAULT = Path("data/sf_matches.json")

DEFAULT_RADIUS   = 100    # metres — SF is dense, keep tight
MIN_NSIM_DEFAULT = 0.75   # name similarity floor (AND with lsim)
MIN_LSIM_DEFAULT = 0.85   # location similarity floor (AND with nsim); 0.85 -> dist <= 15m

_REFERENCE_DATE = datetime(2026, 3, 9, tzinfo=timezone.utc)

# Approximate metres per degree at SF latitude (~37.77°)
_LAT_M = 111_111.0
_LON_M = 111_111.0 * math.cos(math.radians(37.77))


# SF city bounding box (tight — city limits only)
SF_BBOX = {"lat_min": 37.70, "lat_max": 37.82, "lon_min": -122.52, "lon_max": -122.35}

_CLOSED_STATUSES = {"closed", "permanently_closed"}


# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def _df_to_records(df) -> list[dict]:
    """Convert DuckDB/pandas DataFrame to JSON-serializable dicts.

    Handles binary (bytearray) columns that appear in Overture parquet IDs
    by converting them to hex strings.
    """
    import numpy as np

    df = df.copy()
    for col in df.columns:
        sample = next(
            (v for v in df[col]
             if v is not None and not (isinstance(v, float) and np.isnan(v))),
            None,
        )
        if isinstance(sample, (bytes, bytearray)):
            df[col] = df[col].apply(
                lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else None
            )
    return json.loads(df.to_json(orient="records"))


def _resolve_parquet_glob(path_or_glob: str) -> str:
    """If path_or_glob is a directory, return a glob for all parquet files inside it."""
    p = Path(path_or_glob)
    if p.is_dir():
        return str(p / "*.zstd.parquet")
    return path_or_glob


def load_overture_sf_parquet(glob_pattern: str, bbox: dict = SF_BBOX) -> list[dict]:
    """Query Overture parquet files for places within the SF city bounding box.

    Args:
        glob_pattern: DuckDB-compatible glob, e.g. 'data/parquet/2026-02-18.0/*.zstd.parquet'
        bbox:         lat/lon bounding box dict with lat_min/lat_max/lon_min/lon_max

    Returns:
        List of Overture place dicts (same structure as project_c_samples.json records).
        Each dict has 'open' set from operating_status (closed/permanently_closed → 0, else → 1).
    """
    import duckdb

    print(f"Querying Overture parquet for SF-area places ...")
    con = duckdb.connect()
    sql = f"""
        SELECT *
        FROM read_parquet('{glob_pattern}')
        WHERE bbox.ymin >= {bbox['lat_min']}
          AND bbox.ymax <= {bbox['lat_max']}
          AND bbox.xmin >= {bbox['lon_min']}
          AND bbox.xmax <= {bbox['lon_max']}
    """
    df = con.execute(sql).df()
    con.close()

    records = _df_to_records(df)

    # Add open label from operating_status (used as fallback; overridden by sf_open on match)
    for rec in records:
        status = (rec.get("operating_status") or "").lower()
        rec["open"] = 0 if status in _CLOSED_STATUSES else 1

    print(f"  {len(records):,} Overture records in SF bbox")
    return records


# ---------------------------------------------------------------------------
# SF index
# ---------------------------------------------------------------------------

def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    # pandas to_json() serializes datetime columns as Unix milliseconds (int)
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        try:
            return datetime.fromtimestamp(s / 1000, tz=timezone.utc)
        except (ValueError, OSError):
            return None
    try:
        s_str = str(s).replace("Z", "+00:00")
        # Strip sub-second precision (.000) — Python < 3.11 fromisoformat can't parse it
        dot = s_str.find(".")
        if dot != -1:
            plus = s_str.find("+", dot)
            s_str = s_str[:dot] + (s_str[plus:] if plus != -1 else "")
        dt = datetime.fromisoformat(s_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _sf_label(props: dict) -> int:
    """0 = closed (dba_end_date in past), 1 = open."""
    dt = _parse_dt(props.get("dba_end_date"))
    return 0 if (dt is not None and dt <= _REFERENCE_DATE) else 1


def build_sf_index(sf_path: Path) -> tuple[list[dict], np.ndarray, cKDTree]:
    """Load SF GeoJSON and build a KD-tree over records that have lat/lon.

    Returns:
        records:  list of SF property dicts (parallel to tree points)
        coords_m: (N, 2) array of [lat_m, lon_m] approximate-metres coords
        tree:     cKDTree built on coords_m
    """
    print(f"Loading SF dataset from {sf_path} ...")
    with open(sf_path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    coords: list[tuple[float, float]] = []

    for feature in data["features"]:
        geom = feature.get("geometry")
        if not geom or geom.get("type") != "Point":
            continue
        xy = geom.get("coordinates") or []
        if len(xy) < 2:
            continue
        lon, lat = float(xy[0]), float(xy[1])
        props = dict(feature.get("properties") or {})
        props["_lat"] = lat
        props["_lon"] = lon
        records.append(props)
        coords.append((lat * _LAT_M, lon * _LON_M))

    coords_m = np.array(coords, dtype=np.float64)
    tree = cKDTree(coords_m)
    print(f"  {len(records):,} SF records with geometry indexed")
    return records, coords_m, tree


# ---------------------------------------------------------------------------
# Per-record matching
# ---------------------------------------------------------------------------

def find_sf_match(
    lat: float,
    lon: float,
    name: str,
    tree: cKDTree,
    sf_records: list[dict],
    radius_m: float,
    min_nsim: float,
    min_lsim: float,
) -> tuple[dict, float, float, float] | None:
    """Find the best-matching SF record for an Overture place.

    Queries the KD-tree for all SF records within radius_m, then picks
    the candidate with the highest NSIM (name similarity) and accepts only if:
        nsim >= min_nsim  AND  lsim >= min_lsim

    where lsim = max(0, 1 - dist_m / radius_m).

    Returns:
        (sf_props, nsim, dist_m, lsim) or None if no qualifying match.
    """
    query_pt = np.array([lat * _LAT_M, lon * _LON_M])
    idxs = tree.query_ball_point(query_pt, r=radius_m)
    if not idxs:
        return None

    best_sim  = -1.0
    best_dist = None
    best_rec  = None

    for idx in idxs:
        sf_props = sf_records[idx]
        sf_name  = sf_props.get("dba_name") or ""
        sim      = name_similarity(name, sf_name)
        if sim > best_sim:
            best_sim  = sim
            best_rec  = sf_props
            sf_lat    = sf_props["_lat"]
            sf_lon    = sf_props["_lon"]
            dlat      = (lat - sf_lat) * _LAT_M
            dlon      = (lon - sf_lon) * _LON_M
            best_dist = math.hypot(dlat, dlon)

    if best_rec is None or best_dist is None:
        return None

    lsim = max(0.0, 1.0 - best_dist / radius_m)
    if best_sim >= min_nsim and lsim >= min_lsim:
        return best_rec, best_sim, best_dist, lsim
    return None


# ---------------------------------------------------------------------------
# Overture record helpers
# ---------------------------------------------------------------------------

def _overture_coords(rec: dict) -> tuple[float, float] | None:
    """Extract (lat, lon) from Overture record bbox.

    Handles both NDJSON format (bbox.ymin etc.) and parquet-serialized format
    (same dict structure after JSON round-trip via _df_to_records).
    """
    bbox = rec.get("bbox")
    if not bbox:
        return None
    lat = (bbox.get("ymin", 0) + bbox.get("ymax", 0)) / 2
    lon = (bbox.get("xmin", 0) + bbox.get("xmax", 0)) / 2
    return lat, lon


def _overture_name(rec: dict) -> str:
    """Extract primary name. Works for both NDJSON and parquet-serialized records."""
    names = rec.get("names") or {}
    # NDJSON format: {"primary": "Foo"}
    # Parquet format after JSON: {"primary": "Foo", "common": [...], ...}
    return names.get("primary", "") if isinstance(names, dict) else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    sf_path: Path = SF_DEFAULT,
    overture_path: Path | None = OVERTURE_DEFAULT,
    parquet_glob: str | None = None,
    output_path: Path = OUTPUT_DEFAULT,
    augment_output: Path | None = None,
    radius: float = DEFAULT_RADIUS,
    min_nsim: float = MIN_NSIM_DEFAULT,
    min_lsim: float = MIN_LSIM_DEFAULT,
    min_asim: float = 0.0,
    use_overture_label: bool = False,
    limit: int = 0,
    verbose: bool = False,
) -> None:
    # Build spatial index
    sf_records, coords_m, tree = build_sf_index(sf_path)

    # Load Overture records — from parquet or NDJSON
    print()
    if parquet_glob:
        overture = load_overture_sf_parquet(_resolve_parquet_glob(parquet_glob))
    else:
        print(f"Loading Overture records from {overture_path} ...")
        overture = []
        with open(overture_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    overture.append(json.loads(line))
        print(f"  {len(overture):,} Overture records")

    if limit:
        overture = overture[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    no_coords = 0
    no_match  = 0
    asim_rejected = 0
    agree = 0    # sf_open == overture_open
    disagree = 0

    aug_lines: list[str] = []
    aug_skipped_stale = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(overture):
            overture_id   = rec.get("id", "unknown")
            overture_name = _overture_name(rec)
            overture_open = rec.get("open")
            coords        = _overture_coords(rec)

            if not verbose:
                print(f"\r  Matching {i+1}/{len(overture)} ...", end="", flush=True)

            if not coords or not overture_name:
                no_coords += 1
                row = {
                    "overture_id": overture_id,
                    "overture_name": overture_name,
                    "overture_open": overture_open,
                    "sf_match": None,
                }
                out.write(json.dumps(row) + "\n")
                continue

            lat, lon = coords
            result = find_sf_match(lat, lon, overture_name, tree, sf_records,
                                   radius_m=radius, min_nsim=min_nsim, min_lsim=min_lsim)

            if result is None:
                no_match += 1
                row = {
                    "overture_id": overture_id,
                    "overture_name": overture_name,
                    "overture_open": overture_open,
                    "lat": lat,
                    "lon": lon,
                    "sf_match": None,
                }
                out.write(json.dumps(row) + "\n")
                if verbose:
                    print(f"  [{i+1}] NO MATCH  {overture_name[:50]}")
                continue

            sf_props, nsim, dist_m, lsim = result

            # ASIM gate: reject if address similarity is too low (different unit/building)
            # Extract addresses here so we can compute asim before incrementing matched
            addrs = rec.get("addresses") or []
            ov_addr = ""
            if addrs and isinstance(addrs, list) and isinstance(addrs[0], dict):
                ov_addr = addrs[0].get("freeform") or ""
            sf_addr = (sf_props.get("full_business_address") or "").strip()
            asim = round(name_similarity(ov_addr, sf_addr), 4) if ov_addr and sf_addr else None

            if min_asim > 0.0 and asim is not None and asim < min_asim:
                asim_rejected += 1
                no_match += 1
                row = {
                    "overture_id": overture_id,
                    "overture_name": overture_name,
                    "overture_open": overture_open,
                    "lat": lat,
                    "lon": lon,
                    "sf_match": None,
                    "asim_rejected": True,
                    "asim": asim,
                }
                out.write(json.dumps(row) + "\n")
                if verbose:
                    print(f"  [{i+1}] ASIM-REJECT asim={asim:.2f}  '{overture_name[:35]}'")
                continue

            sf_open = _sf_label(sf_props)
            matched += 1

            if overture_open == sf_open:
                agree += 1
            else:
                disagree += 1

            # Staleness: days between SF closure date and Overture update_time
            # Positive = Overture updated after SF closure (label uncertainty)
            sf_staleness_days = None
            if sf_open == 0:
                sf_end = _parse_dt(sf_props.get("dba_end_date"))
                # update_time may be top-level (NDJSON) or nested in sources (parquet)
                ov_upd_raw = rec.get("update_time")
                if not ov_upd_raw:
                    # extract from primary source (property == "") — parquet records
                    for src in (rec.get("sources") or []):
                        if isinstance(src, dict) and src.get("property") == "":
                            ov_upd_raw = src.get("update_time")
                            break
                ov_upd = _parse_dt(ov_upd_raw)
                if sf_end and ov_upd:
                    sf_staleness_days = (ov_upd - sf_end).days

            row = {
                # Overture identity + original label
                "overture_id":      overture_id,
                "overture_name":    overture_name,
                "overture_open":    overture_open,
                "overture_address": ov_addr,
                "lat": lat,
                "lon": lon,
                # Match quality
                "nsim":   round(nsim, 4),
                "lsim":   round(lsim, 4),
                "asim":   asim,
                "dist_m": round(dist_m, 1),
                # SF-derived label
                "sf_open":             sf_open,
                "sf_name":             sf_props.get("dba_name"),
                "sf_lat":              sf_props.get("_lat"),
                "sf_lon":              sf_props.get("_lon"),
                "sf_dba_end_date":     sf_props.get("dba_end_date"),
                "sf_staleness_days":   sf_staleness_days,
                "sf_cert":             sf_props.get("certificate_number"),
                "sf_naic":             sf_props.get("naic_code_description"),
                # Full SF record for inspection
                "sf_record": {k: v for k, v in sf_props.items()
                              if not k.startswith("_")},
            }
            out.write(json.dumps(row) + "\n")

            # Augmentation record: full Overture record with open=sf_open
            if augment_output is not None:
                is_stale = (sf_open == 0 and sf_staleness_days is not None and sf_staleness_days > 0)
                if is_stale and not use_overture_label:
                    aug_skipped_stale += 1   # skip — uncertain label, keep in match report only
                else:
                    label = 1 if (is_stale and use_overture_label) else sf_open
                    aug_rec = dict(rec)
                    aug_rec["open"]               = label
                    aug_rec["_sf_nsim"]           = round(nsim, 4)
                    aug_rec["_sf_lsim"]           = round(lsim, 4)
                    aug_rec["_sf_dist"]           = round(dist_m, 1)
                    aug_rec["_sf_staleness_days"] = sf_staleness_days
                    aug_lines.append(json.dumps(aug_rec))

            if verbose:
                agree_tag = "Y" if overture_open == sf_open else "N"
                print(f"  [{i+1}] {agree_tag} nsim={nsim:.2f} lsim={lsim:.2f} dist={dist_m:.0f}m  "
                      f"'{overture_name[:35]}' -> '{sf_props.get('dba_name', '')[:35]}'  "
                      f"overture={overture_open} sf={sf_open}")

    if not verbose:
        print("\r" + " " * 50 + "\r", end="")

    # Write augmentation file
    if augment_output and aug_lines:
        augment_output.parent.mkdir(parents=True, exist_ok=True)
        with open(augment_output, "w", encoding="utf-8") as f:
            f.write("\n".join(aug_lines) + "\n")
        print(f"Augmentation JSONL: {augment_output}  ({len(aug_lines):,} records)")

    total = len(overture)
    print(f"\n{'='*55}")
    print(f"  Total Overture records:  {total:,}")
    print(f"  No coords/name:          {no_coords:,}  ({100*no_coords/total:.1f}%)")
    print(f"  No SF match:             {no_match:,}  ({100*no_match/total:.1f}%)")
    if asim_rejected:
        print(f"    ASIM rejected:         {asim_rejected:,}")
    print(f"  Matched:                 {matched:,}  ({100*matched/total:.1f}%)")
    if matched:
        print(f"    Agreement (labels):    {agree:,}  ({100*agree/matched:.1f}%)")
        print(f"    Disagreement:          {disagree:,}  ({100*disagree/matched:.1f}%)")
    if augment_output:
        print(f"  Aug written: {len(aug_lines):,}  |  Skipped (stale label): {aug_skipped_stale:,}"
              + ("  [use-overture-label=on]" if use_overture_label else ""))
    print(f"  Radius: {radius}m  |  min-nsim: {min_nsim}  |  min-lsim: {min_lsim}  |  min-asim: {min_asim}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Match Overture places to SF business registry via lat/lon + name",
    )
    ap.add_argument("--sf", type=Path, default=SF_DEFAULT,
                    help=f"SF GeoJSON file (default: {SF_DEFAULT})")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--overture", type=Path, default=None,
                     help="Overture NDJSON file to match against (e.g. project_c_samples.json). "
                          "Use for label validation.")
    src.add_argument("--from-parquet", dest="parquet_glob", type=str, default=None,
                     help="Directory or DuckDB glob for Overture parquet files. "
                          "If a directory is given, all *.zstd.parquet files inside are used. "
                          "Example: data/parquet/2026-02-18.0")

    ap.add_argument("--output", type=Path, default=OUTPUT_DEFAULT,
                    help=f"Match report NDJSON (default: {OUTPUT_DEFAULT})")
    ap.add_argument("--augment-output", type=Path, default=None,
                    help="Write matched Overture records with open=sf_open as JSONL "
                         "for use with split.py --augment")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                    help=f"Geo search radius in metres (default: {DEFAULT_RADIUS})")
    ap.add_argument("--min-nsim", type=float, default=MIN_NSIM_DEFAULT,
                    help=f"Min name similarity (AND with lsim, default: {MIN_NSIM_DEFAULT})")
    ap.add_argument("--min-lsim", type=float, default=MIN_LSIM_DEFAULT,
                    help=f"Min location similarity (AND with nsim, default: {MIN_LSIM_DEFAULT})")
    ap.add_argument("--min-asim", type=float, default=0.0,
                    help="Min address similarity to accept a match (0=disabled, default: 0.0)")
    ap.add_argument("--use-overture-label", action="store_true",
                    help="When Overture is fresher than SF closure date, write open=1 instead of skipping")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N Overture records (0 = all)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print each match/no-match result")
    args = ap.parse_args()

    if args.parquet_glob is None and args.overture is None:
        args.overture = OVERTURE_DEFAULT   # default fallback

    main(
        sf_path=args.sf,
        overture_path=args.overture,
        parquet_glob=args.parquet_glob,
        output_path=args.output,
        augment_output=args.augment_output,
        radius=args.radius,
        min_nsim=args.min_nsim,
        min_lsim=args.min_lsim,
        min_asim=args.min_asim,
        use_overture_label=args.use_overture_label,
        limit=args.limit,
        verbose=args.verbose,
    )
