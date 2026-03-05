"""
fsq_lookup.py
-------------
Look up places from project_c_samples.json via the Foursquare Places API.

Matches Overture records by name + lat/lon to retrieve FSQ signals:
    date_closed, categories, chains, location, social_media

Outputs NDJSON to data/fsq_matches.json with both Overture ID and FSQ data.

Requires:
    export FSQ_LEGACY_API_KEY="your-api-key"

Run:
    python src/util/fsq_lookup.py                          # default: first 50 from project_c_samples.json
    python src/util/fsq_lookup.py --limit 200              # first 200
    python src/util/fsq_lookup.py --input data/overture-feb-release-closed.json --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv

root = Path(__file__).parent.parent.parent
load_dotenv(root / ".env")

API_KEY = os.environ.get("FSQ_DATA_API_KEY", "")
BASE_URL = "https://places-api.foursquare.com/places/search"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Places-Api-Version": "2025-06-17",
    "accept": "application/json",
}

# Fields to request (Pro tier)
FIELDS = ",".join([
    "fsq_place_id",
    "name",
    "location",
    "categories",
    "chains",
    "date_closed",
    "website",
    "tel",
    "email",
    "social_media",
    "link",
    "photos",
])

OUTPUT_DEFAULT = Path("data/fsq_matches.json")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

MIN_SIM_GEO = 0.55
MIN_SIM_NAME = 0.75
DEFAULT_RADIUS = 500
DEFAULT_FIRST_WORD_WEIGHT = 0.6

_STRIP_RE = re.compile(r"[^a-z0-9 ]")
_ARTICLES = {"the", "a", "an"}


def _normalize(name: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    return _STRIP_RE.sub("", name.lower()).strip()


def _first_content_word(normalized: str) -> str:
    """Return the first non-article word from a normalized name."""
    words = normalized.split()
    for w in words:
        if w not in _ARTICLES:
            return w
    return words[0] if words else ""


def name_similarity(a: str, b: str, first_word_weight: float = DEFAULT_FIRST_WORD_WEIGHT) -> float:
    """Return 0-1 similarity score between two place names.

    Blends full-name similarity with first-content-word matching to
    penalise cases where only a generic suffix (e.g. "Restaurant and
    Lounge") is shared but the distinctive leading word differs.

    Leading articles (the, a, an) are skipped for the first-word
    comparison so that "The Regency House" matches "Regency House
    Bingo" on the distinctive word "regency".

    Score = (1 - first_word_weight) * full_sim + first_word_weight * first_word_sim

    Also returns 1.0 if one normalized name is a substring of the
    other (handles 'Panda Express' matching 'Panda Express #1234').
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0

    full_sim = SequenceMatcher(None, na, nb).ratio()

    # First-content-word comparison (skip articles)
    wa = _first_content_word(na)
    wb = _first_content_word(nb)
    if wa and wb:
        first_sim = SequenceMatcher(None, wa, wb).ratio()
    else:
        first_sim = 0.0

    return (1 - first_word_weight) * full_sim + first_word_weight * first_sim


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _do_search(params: dict) -> list[dict]:
    """Execute a FSQ search and return list of result dicts."""
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return []


def search_place(
    name: str,
    lat: float,
    lon: float,
    radius: int = 500,
) -> dict | None:
    """Search FSQ for a place by name near a coordinate.

    Args:
        name:   place name to search for
        lat:    latitude (WGS84)
        lon:    longitude (WGS84)
        radius: search radius in meters (default 500m)

    Returns:
        Best-matching FSQ result dict, or None if no results.
    """
    params = {
        "query": name,
        "ll": f"{lat},{lon}",
        "radius": radius,
        "limit": 1,
        "fields": FIELDS,
    }
    results = _do_search(params)
    return results[0] if results else None


def search_place_by_name(
    name: str,
    near: str | None = None,
) -> dict | None:
    """Fallback: search FSQ by name only (no lat/lon constraint).

    Uses the 'near' param with the place's region/state if available,
    otherwise searches globally.

    Args:
        name: place name to search for
        near: optional locality hint, e.g. "New York, NY"

    Returns:
        Best-matching FSQ result dict, or None if no results.
    """
    params = {
        "query": name,
        "limit": 1,
        "fields": FIELDS,
    }
    if near:
        params["near"] = near
    results = _do_search(params)
    return results[0] if results else None


def extract_coords(record: dict) -> tuple[float, float] | None:
    """Get (lat, lon) from an Overture record's bbox.

    Args:
        record: parsed Overture place JSON

    Returns:
        (lat, lon) tuple or None if missing.
    """
    bbox = record.get("bbox")
    if not bbox:
        return None
    lat = (bbox.get("ymin", 0) + bbox.get("ymax", 0)) / 2
    lon = (bbox.get("xmin", 0) + bbox.get("xmax", 0)) / 2
    return lat, lon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    input_path: str | Path = "data/project_c_samples.json",
    output_path: str | Path = OUTPUT_DEFAULT,
    limit: int = 50,
    delay: float = 0.2,
    min_sim_geo: float = MIN_SIM_GEO,
    min_sim_name: float = MIN_SIM_NAME,
    radius: int = DEFAULT_RADIUS,
    first_word_weight: float = DEFAULT_FIRST_WORD_WEIGHT,
    verbose: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not API_KEY:
        sys.exit("ERROR: Set FSQ_DATA_API_KEY environment variable first.\n"
                 "  Add FSQ_DATA_API_KEY=your-key to .env")

    print(f"API key loaded: {API_KEY[:8]}...{API_KEY[-4:]}  (len={len(API_KEY)})")
    print(f"First-word weight: {first_word_weight}")

    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")

    # Load records with balanced open/closed split
    all_records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    if limit:
        open_recs = [r for r in all_records if r.get("open") == 1]
        closed_recs = [r for r in all_records if r.get("open") == 0]
        half = limit // 2
        records = closed_recs[:half] + open_recs[: limit - half]
        print(f"Balanced sample: {min(half, len(closed_recs))} closed + "
              f"{min(limit - half, len(open_recs))} open "
              f"(from {len(closed_recs)} closed / {len(open_recs)} open total)")
    else:
        records = all_records

    print(f"Loaded {len(records)} records from {input_path}")
    print(f"Output: {output_path}\n")

    matched = 0
    matched_fallback = 0
    not_found = 0
    low_sim = 0
    fsq_closed_count = 0

    with open(output_path, "w") as out:
        for i, rec in enumerate(records):
            overture_id = rec.get("id", "unknown")
            name = (rec.get("names") or {}).get("primary", "")
            coords = extract_coords(rec)
            label = rec.get("open")

            if not name or not coords:
                if verbose:
                    print(f"  [{i+1}/{len(records)}] SKIP (missing name/coords): {overture_id}")
                continue

            lat, lon = coords
            # Build a 'near' hint from Overture address for fallback
            addr = rec.get("addresses") or []
            near_hint = None
            if addr and isinstance(addr, list) and len(addr) > 0:
                a = addr[0]
                parts = [a.get("locality", ""), a.get("region", "")]
                near_hint = ", ".join(p for p in parts if p) or None

            if verbose:
                print(f"  [{i+1}/{len(records)}] {name[:40]:<40s}  ({lat:.4f}, {lon:.4f})", end="")
            else:
                print(f"\r  Processing {i+1}/{len(records)} ...", end="", flush=True)

            fsq_result = search_place(name, lat, lon, radius=radius)
            match_type = "geo"
            sim = 0.0
            need_fallback = False

            if fsq_result:
                fsq_name = fsq_result.get("name", "")
                sim = name_similarity(name, fsq_name, first_word_weight)
                if sim >= min_sim_geo:
                    matched += 1
                    closed_tag = ""
                    if fsq_result.get("date_closed"):
                        fsq_closed_count += 1
                        closed_tag = f"  [CLOSED {fsq_result['date_closed']}]"
                    if verbose:
                        print(f"  -> {fsq_name}  (sim={sim:.2f}){closed_tag}")
                else:
                    # geo match returned wrong place — try name fallback
                    if verbose:
                        print(f"  -> REJECTED {fsq_name}  (sim={sim:.2f})", end="")
                    need_fallback = True
            else:
                if verbose:
                    print(f"  -> no geo match", end="")
                need_fallback = True

            # Fallback: name-only search
            if need_fallback:
                time.sleep(delay)  # rate-limit before fallback call
                fsq_result = search_place_by_name(name, near=near_hint)
                match_type = "name"
                sim = 0.0

                if fsq_result:
                    fsq_name = fsq_result.get("name", "")
                    sim = name_similarity(name, fsq_name, first_word_weight)
                    if sim >= min_sim_name:
                        matched_fallback += 1
                        closed_tag = ""
                        if fsq_result.get("date_closed"):
                            fsq_closed_count += 1
                            closed_tag = f"  [CLOSED {fsq_result['date_closed']}]"
                        if verbose:
                            print(f"  -> FALLBACK {fsq_name}  (sim={sim:.2f}){closed_tag}")
                    else:
                        low_sim += 1
                        if verbose:
                            print(f"  -> FALLBACK REJECTED {fsq_name}  (sim={sim:.2f})")
                        fsq_result = None
                else:
                    not_found += 1
                    if verbose:
                        print(f"  -> NO MATCH")

            # Extract FSQ closed status
            fsq_date_closed = None
            fsq_closed = None
            latest_photo_date = None
            if fsq_result:
                fsq_date_closed = fsq_result.get("date_closed")
                fsq_closed = fsq_date_closed is not None
                # Latest photo created_at
                photos = fsq_result.get("photos") or []
                if photos:
                    dates = [p["created_at"] for p in photos if p.get("created_at")]
                    if dates:
                        latest_photo_date = max(dates)

            # Write combined record
            row = {
                "overture_id": overture_id,
                "overture_name": name,
                "overture_open": label,
                "lat": lat,
                "lon": lon,
                "name_similarity": round(sim, 4),
                "match_type": match_type if fsq_result else None,
                "fsq_closed": fsq_closed,
                "fsq_date_closed": fsq_date_closed,
                "fsq_latest_photo": latest_photo_date,
                "fsq_match": fsq_result,
            }
            out.write(json.dumps(row) + "\n")

            # Rate-limit
            time.sleep(delay)

    if not verbose:
        print("\r" + " " * 40 + "\r", end="")  # clear progress line

    print(f"Done. Matched (geo): {matched}  |  Matched (fallback): {matched_fallback}"
          f"  |  Rejected (low sim): {low_sim}  |  Not found: {not_found}"
          f"  |  Total: {len(records)}")
    print(f"FSQ closed: {fsq_closed_count}")
    print(f"Similarity threshold: geo={min_sim_geo}  name={min_sim_name}  first_word_weight={first_word_weight}")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Look up Overture places in Foursquare API")
    ap.add_argument("--input", default="data/project_c_samples.json",
                    help="Input NDJSON file (default: data/project_c_samples.json)")
    ap.add_argument("--output", default=str(OUTPUT_DEFAULT),
                    help="Output NDJSON file (default: data/fsq_matches.json)")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max records to look up (default: 50)")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="Seconds between API calls (default: 0.2)")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                    help=f"Geo search radius in meters (default: {DEFAULT_RADIUS})")
    ap.add_argument("--min-sim-geo", type=float, default=MIN_SIM_GEO,
                    help=f"Min similarity for geo matches (default: {MIN_SIM_GEO})")
    ap.add_argument("--min-sim-name", type=float, default=MIN_SIM_NAME,
                    help=f"Min similarity for name-only fallback (default: {MIN_SIM_NAME})")
    ap.add_argument("--first-word-weight", type=float, default=DEFAULT_FIRST_WORD_WEIGHT,
                    help=f"Weight for first-content-word similarity (default: {DEFAULT_FIRST_WORD_WEIGHT})")
    ap.add_argument("--verbose", action="store_true",
                    help="Show individual lookup results")
    args = ap.parse_args()
    main(input_path=args.input, output_path=args.output,
         limit=args.limit, delay=args.delay, radius=args.radius,
         min_sim_geo=args.min_sim_geo, min_sim_name=args.min_sim_name,
         first_word_weight=args.first_word_weight, verbose=args.verbose)
