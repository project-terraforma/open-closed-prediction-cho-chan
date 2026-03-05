"""
json_probe.py
-------------
Tally open/closed counts, region breakdown, and category breakdown for any NDJSON place file.

Run:
    python src/util/json_probe.py data/project_c_samples.json
    python src/util/json_probe.py data/overture-feb-release-closed.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(path: str | Path) -> None:
    path = Path(path)
    open_count = 0
    closed_count = 0
    region_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Tally open/closed
            label = r.get("open")
            if label == 1:
                open_count += 1
            elif label == 0:
                closed_count += 1

            # Tally region
            addresses = r.get("addresses") or []
            region = addresses[0].get("region", "unknown") if addresses else "unknown"
            region_counter[region] += 1

            # Tally primary category
            categories = r.get("categories") or {}
            category = categories.get("primary", "unknown") or "unknown"
            category_counter[category] += 1

    total = open_count + closed_count
    print(f"File: {path}")
    print(f"Total: {total:,}  |  open: {open_count:,}  |  closed: {closed_count:,}\n")

    print(f"{'region':<10} {'count':>8} {'pct':>7}")
    print("-" * 27)
    for region, count in region_counter.most_common():
        pct = count / total * 100 if total else 0
        print(f"{region:<10} {count:>8,} {pct:>6.1f}%")

    print(f"\n{'category':<40} {'closed_count':>12}")
    print("-" * 54)
    for category, count in category_counter.most_common():
        print(f"{category:<40} {count:>12,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tally open/closed and region from NDJSON place file")
    ap.add_argument("file", help="Path to NDJSON place file")
    args = ap.parse_args()
    main(args.file)
