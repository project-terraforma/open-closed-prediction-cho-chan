"""
fsq_probe.py
------------
Query Foursquare Places API for all restaurants in San Francisco
and tally open vs closed counts using the date_closed field.

The FSQ search endpoint caps results at 50 per request, so we paginate
by shifting a geographic cursor across an SF lat/lon grid.

Run:
    python src/fsq_probe.py
    python src/fsq_probe.py --category 13065 --grid-steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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

# FSQ category 13065 = "Restaurant" (top-level dining parent)
DEFAULT_CATEGORY = "13065"

# Pro-tier fields – we only need name + date_closed for tallying
FIELDS = ",".join([
    "fsq_place_id",
    "name",
    "date_closed",
    "categories",
])

# San Francisco bounding box (approx)
SF_SW = (37.7079, -122.5140)   # southwest corner
SF_NE = (37.8120, -122.3570)   # northeast corner

DEFAULT_GRID = 6          # 6x6 = 36 grid cells
DEFAULT_RADIUS = 800      # meters per cell search
MAX_PER_REQUEST = 50      # FSQ hard limit
DEFAULT_DELAY = 0.25      # seconds between requests


def _search(params: dict) -> list[dict]:
    """Execute a single FSQ search request."""
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except requests.RequestException as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        return []


def grid_search(
    category: str,
    grid_steps: int,
    radius: int,
    delay: float,
) -> dict[str, dict]:
    """Sweep an lat/lon grid over SF, collecting unique places.

    Returns dict keyed by fsq_place_id -> place dict.
    """
    lat_step = (SF_NE[0] - SF_SW[0]) / grid_steps
    lon_step = (SF_NE[1] - SF_SW[1]) / grid_steps

    seen: dict[str, dict] = {}
    total_requests = 0

    for i in range(grid_steps):
        for j in range(grid_steps):
            lat = SF_SW[0] + lat_step * (i + 0.5)
            lon = SF_SW[1] + lon_step * (j + 0.5)

            params = {
                "categories": category,
                "ll": f"{lat},{lon}",
                "radius": radius,
                "limit": MAX_PER_REQUEST,
                "fields": FIELDS,
            }

            results = _search(params)
            total_requests += 1
            new = 0
            for place in results:
                pid = place.get("fsq_place_id")
                if pid and pid not in seen:
                    seen[pid] = place
                    new += 1

            cell = f"[{i},{j}]"
            print(
                f"  grid {cell:>6s}  ll=({lat:.4f},{lon:.4f})  "
                f"returned={len(results):>2d}  new={new:>2d}  "
                f"cumulative={len(seen)}"
            )

            if delay > 0:
                time.sleep(delay)

    print(f"\nTotal API requests: {total_requests}")
    return seen


def tally(places: dict[str, dict]) -> None:
    """Print open/closed breakdown."""
    closed = []
    open_places = []

    for pid, p in places.items():
        date_closed = p.get("date_closed")
        if date_closed:
            closed.append(p)
        else:
            open_places.append(p)

    total = len(places)
    n_closed = len(closed)
    n_open = len(open_places)

    print(f"\n{'='*50}")
    print(f"FSQ Restaurants in San Francisco")
    print(f"{'='*50}")
    print(f"  Total unique places : {total}")
    print(f"  Open (no date_closed): {n_open}  ({100*n_open/total:.1f}%)" if total else "")
    print(f"  Closed (date_closed) : {n_closed}  ({100*n_closed/total:.1f}%)" if total else "")
    print(f"{'='*50}")

    if closed:
        print(f"\nClosed restaurants ({n_closed}):")
        for p in sorted(closed, key=lambda x: x.get("date_closed", "")):
            cats = ", ".join(c["name"] for c in p.get("categories", []))
            print(
                f"  {p.get('date_closed','?'):>10s}  "
                f"{p.get('name','?'):<40s}  [{cats}]"
            )


FIELDS_WITH_LOCATION = ",".join([
    "fsq_place_id",
    "name",
    "date_closed",
    "categories",
    "location",
])

# Major US cities for broad closed-place probe
US_CITIES = [
    ("New York",        40.7128, -74.0060),
    ("Los Angeles",     34.0522, -118.2437),
    ("Chicago",         41.8781, -87.6298),
    ("Houston",         29.7604, -95.3698),
    ("Phoenix",         33.4484, -112.0740),
    ("Philadelphia",    39.9526, -75.1652),
    ("San Antonio",     29.4241, -98.4936),
    ("San Diego",       32.7157, -117.1611),
    ("Dallas",          32.7767, -96.7970),
    ("San Francisco",   37.7749, -122.4194),
    ("Seattle",         47.6062, -122.3321),
    ("Denver",          39.7392, -104.9903),
    ("Boston",          42.3601, -71.0589),
    ("Atlanta",         33.7490, -84.3880),
    ("Miami",           25.7617, -80.1918),
    ("Detroit",         42.3314, -83.0458),
    ("Minneapolis",     44.9778, -93.2650),
    ("Portland",        45.5152, -122.6784),
    ("Las Vegas",       36.1699, -115.1398),
    ("New Orleans",     29.9511, -90.0715),
]


def probe_closed(
    category: str,
    limit: int,
    delay: float,
) -> None:
    """Search across major US cities for restaurants with date_closed.

    FSQ search has no server-side filter for date_closed, so we pull
    `limit` results per city (max 50) and check client-side.
    """
    per_city = min(limit, MAX_PER_REQUEST)
    seen: dict[str, dict] = {}
    closed: list[dict] = []

    print(f"Probing {len(US_CITIES)} US cities, {per_city} results each...\n")

    for city_name, lat, lon in US_CITIES:
        params = {
            "categories": category,
            "ll": f"{lat},{lon}",
            "radius": 50_000,          # 50 km – cover entire metro
            "limit": per_city,
            "fields": FIELDS_WITH_LOCATION,
        }
        results = _search(params)
        new = 0
        new_closed = 0
        for p in results:
            pid = p.get("fsq_place_id")
            if pid and pid not in seen:
                seen[pid] = p
                new += 1
                if p.get("date_closed"):
                    closed.append(p)
                    new_closed += 1

        print(
            f"  {city_name:<16s}  returned={len(results):>2d}  "
            f"new={new:>2d}  closed={new_closed}  "
            f"cumulative={len(seen)}  total_closed={len(closed)}"
        )
        if delay > 0:
            time.sleep(delay)

    # Report
    print(f"\n{'='*60}")
    print(f"Probe: restaurants with date_closed across {len(US_CITIES)} cities")
    print(f"{'='*60}")
    print(f"  Total unique places scanned: {len(seen)}")
    print(f"  With date_closed set       : {len(closed)}")
    print(f"{'='*60}")

    if closed:
        print(f"\nClosed restaurants ({len(closed)}):")
        print(f"  {'date_closed':>12s}  {'name':<40s}  {'state':<6s}  categories")
        print(f"  {'-'*12}  {'-'*40}  {'-'*6}  {'-'*20}")
        for p in sorted(closed, key=lambda x: x.get("date_closed", "")):
            loc = p.get("location", {})
            state = loc.get("region", "?")
            cats = ", ".join(c["name"] for c in p.get("categories", []))
            print(
                f"  {p.get('date_closed','?'):>12s}  "
                f"{p.get('name','?'):<40s}  {state:<6s}  {cats}"
            )
    else:
        print("\nNo restaurants found with date_closed set.")
        print("FSQ may not populate date_closed for searchable places.")


def analyze_matches(input_path: str) -> None:
    """Analyze fsq_matches.json output from fsq_lookup.py.

    Reports match rates by Overture open/closed label, FSQ closed
    entries, and match-type breakdown per class.
    """
    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    open_recs = [r for r in records if r["overture_open"] == 1]
    closed_recs = [r for r in records if r["overture_open"] == 0]

    open_matched = sum(1 for r in open_recs if r["match_type"] is not None)
    closed_matched = sum(1 for r in closed_recs if r["match_type"] is not None)

    print(f"{'='*55}")
    print(f"FSQ Match Analysis  ({input_path})")
    print(f"{'='*55}")
    print(f"  Closed (Overture): {closed_matched}/{len(closed_recs)} matched"
          f"  ({100*closed_matched/len(closed_recs):.1f}%)" if closed_recs else "")
    print(f"  Open   (Overture): {open_matched}/{len(open_recs)} matched"
          f"  ({100*open_matched/len(open_recs):.1f}%)" if open_recs else "")
    total_matched = open_matched + closed_matched
    print(f"  Total            : {total_matched}/{len(records)} matched"
          f"  ({100*total_matched/len(records):.1f}%)")

    # Match type breakdown per class
    print(f"\n  {'class':<8s}  {'geo':>5s}  {'name':>5s}  {'none':>5s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*5}")
    for label, recs in [("closed", closed_recs), ("open", open_recs)]:
        types = Counter(r["match_type"] for r in recs)
        print(f"  {label:<8s}  {types.get('geo',0):>5d}  {types.get('name',0):>5d}"
              f"  {types.get(None,0):>5d}")

    # FSQ closed entries
    fsq_closed = [r for r in records if r.get("fsq_closed")]
    print(f"\n  FSQ date_closed set: {len(fsq_closed)}")
    if fsq_closed:
        for r in fsq_closed:
            m = r.get("fsq_match", {})
            print(f"    Overture: {r['overture_name']}  (open={r['overture_open']})")
            print(f"    FSQ:      {m.get('name','?')}  "
                  f"date_closed={r['fsq_date_closed']}  sim={r['name_similarity']}")
            print()
    print(f"{'='*55}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe FSQ for SF restaurants and tally open/closed."
    )
    parser.add_argument(
        "--category", default=DEFAULT_CATEGORY,
        help=f"FSQ category ID (default: {DEFAULT_CATEGORY} = Restaurant)",
    )
    parser.add_argument(
        "--grid-steps", type=int, default=DEFAULT_GRID,
        help=f"Grid divisions per axis (default: {DEFAULT_GRID}; "
             f"total cells = N²)",
    )
    parser.add_argument(
        "--radius", type=int, default=DEFAULT_RADIUS,
        help=f"Search radius in meters per grid cell (default: {DEFAULT_RADIUS})",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between API calls (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional: save raw results as NDJSON file",
    )
    parser.add_argument(
        "--probe-closed", action="store_true",
        help="Search 20 major US cities for any restaurant with date_closed",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max results per city in --probe-closed mode (default: 50, max: 50)",
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        metavar="FILE",
        help="Analyze fsq_matches.json output from fsq_lookup.py",
    )
    args = parser.parse_args()

    if args.analyze:
        analyze_matches(args.analyze)
        return

    if not API_KEY:
        sys.exit("ERROR: FSQ_DATA_API_KEY not set. Add it to .env")

    print(f"API key loaded: {API_KEY[:10]}...{API_KEY[-4:]}  (len={len(API_KEY)})")

    if args.probe_closed:
        print(f"Mode: probe-closed | Category: {args.category}\n")
        probe_closed(
            category=args.category,
            limit=args.limit,
            delay=args.delay,
        )
        return

    print(f"Category: {args.category}")
    print(f"Grid: {args.grid_steps}x{args.grid_steps} = {args.grid_steps**2} cells")
    print(f"Radius: {args.radius}m | Delay: {args.delay}s\n")

    places = grid_search(
        category=args.category,
        grid_steps=args.grid_steps,
        radius=args.radius,
        delay=args.delay,
    )

    tally(places)

    if args.output:
        out = Path(args.output)
        with open(out, "w") as f:
            for p in places.values():
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(places)} records to {out}")


if __name__ == "__main__":
    main()
