"""
sf_match_analysis.py
--------------------
Inspect sf_lookup.py match report to assess false positive rate and tune thresholds.

Shows:
  1. Similarity score distribution (histogram)
  2. Distance distribution
  3. Threshold sensitivity — match count remaining at each min-sim cut
  4. Lowest-similarity matches (most likely false positives)
  5. Label agreement rate by similarity bucket (where overture_open is ground truth)

Run:
    python src/util/sf_match_analysis.py data/sf_matches.json
    python src/util/sf_match_analysis.py data/sf_matches.json --top 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def histogram(values: list[float], bins: list[float], label: str = "sim") -> None:
    if not values:
        return
    counts = [0] * len(bins)
    for v in values:
        for i, b in enumerate(bins):
            if v <= b:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    total = len(values)
    for i, (b, c) in enumerate(zip(bins, counts)):
        lo = bins[i-1] if i > 0 else 0.0
        bar = "#" * min(int(c / max(counts) * 40), 40)
        print(f"    {lo:.2f}-{b:.2f}  {c:>5,}  ({100*c/total:4.1f}%)  {bar}")


def _print_match_table(
    rows: list[dict],
    indent: int = 2,
    show_coords: bool = False,
    show_addr: bool = False,
) -> None:
    pad = " " * indent
    sub = " " * (indent + 2)
    print(f"{pad}{'sim':>5}  {'lsim':>5}  {'dist':>5}  {'ok':>2}  {'sf':>2}  {'ov':>2}  "
          f"{'Overture name':<35}  {'SF name':<35}")
    print(pad + "-" * 101)
    for r in rows:
        nsim    = r.get("nsim", r.get("name_similarity", 0))
        lsim    = r.get("lsim", max(0.0, 1.0 - r.get("dist_m", 0) / 100.0))
        dist    = r.get("dist_m", 0)
        sf_open = r.get("sf_open", "?")
        ov_open = r.get("overture_open", "?")
        agree   = "Y" if sf_open == ov_open else "N" if isinstance(sf_open, int) and isinstance(ov_open, int) else " "
        ov_name = (r.get("overture_name") or "")[:35]
        sf_name = (r.get("sf_name") or "")[:35]
        print(f"{pad}{nsim:>5.3f}  {lsim:>5.3f}  {dist:>5.0f}  {agree}  {sf_open!s:>2}  {ov_open!s:>2}  "
              f"{ov_name:<35}  {sf_name:<35}")
        if show_coords:
            ov_lat = r.get("lat")
            ov_lon = r.get("lon")
            sf_lat = r.get("sf_lat")
            sf_lon = r.get("sf_lon")
            ov_pos = f"{ov_lat:.5f},{ov_lon:.5f}" if ov_lat is not None else "?"
            sf_pos = f"{sf_lat:.5f},{sf_lon:.5f}" if sf_lat is not None else "?"
            print(f"{sub} \t\t\t\t ov: {ov_pos:<29}  sf: {sf_pos}")
        if show_addr:
            sf_rec    = r.get("sf_record") or {}
            ov_addr   = (r.get("overture_address") or "").strip()
            sf_addr   = (sf_rec.get("full_business_address") or "").strip()
            ov_addr_s = ov_addr[:29] if ov_addr else "(no address)"
            sf_addr_s = sf_addr[:50] if sf_addr else "(no address)"
            print(f"{sub} \t\t\t\t ov: {ov_addr_s:<29}  sf: {sf_addr_s}")


def main(match_path: Path, top_n: int = 20, show_coords: bool = False, show_addr: bool = False) -> None:
    print(f"Reading {match_path} ...")
    matched: list[dict] = []
    total = 0
    with open(match_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            if r.get("sf_match") is not None or r.get("sf_open") is not None:
                matched.append(r)

    print(f"  Total records in file: {total:,}")
    print(f"  Matched records:       {len(matched):,}  ({100*len(matched)/total:.1f}%)")

    if not matched:
        print("No matched records found.")
        return

    sims  = [r.get("nsim", r.get("name_similarity", 0)) for r in matched]
    dists = [r["dist_m"] for r in matched]

    # -----------------------------------------------------------------------
    section("NSIM (NAME SIMILARITY) DISTRIBUTION")
    bins_sim = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    histogram(sims, bins_sim, "sim")
    print(f"\n  min={min(sims):.3f}  median={sorted(sims)[len(sims)//2]:.3f}  "
          f"max={max(sims):.3f}  mean={sum(sims)/len(sims):.3f}")

    # -----------------------------------------------------------------------
    section("DISTANCE DISTRIBUTION (metres)")
    bins_dist = [10, 25, 50, 75, 100, 150, 200, 500]
    histogram(dists, bins_dist, "dist")
    print(f"\n  min={min(dists):.1f}m  median={sorted(dists)[len(dists)//2]:.1f}m  "
          f"max={max(dists):.1f}m  mean={sum(dists)/len(dists):.1f}m")

    # -----------------------------------------------------------------------
    section("THRESHOLD SENSITIVITY")
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    print(f"\n  {'min-sim':>8}  {'kept':>7}  {'dropped':>7}  {'kept%':>6}")
    print("  " + "-" * 36)
    for thr in thresholds:
        kept = sum(1 for s in sims if s >= thr)
        dropped = len(sims) - kept
        marker = " <- min-sim" if abs(thr - 0.85) < 0.01 else " <- sim-floor" if abs(thr - 0.60) < 0.01 else ""
        print(f"  {thr:>8.2f}  {kept:>7,}  {dropped:>7,}  {100*kept/len(sims):>5.1f}%{marker}")

    # -----------------------------------------------------------------------
    section(f"LOWEST NSIM MATCHES (potential false positives, bottom {top_n})")
    by_sim = sorted(matched, key=lambda r: r.get("nsim", r.get("name_similarity", 0)))[:top_n]
    _print_match_table(by_sim, show_coords=show_coords, show_addr=show_addr)

    # -----------------------------------------------------------------------
    section("SAMPLE MATCHES BY NSIM BUCKET (3 examples each)")
    import random; random.seed(42)
    buckets_sample = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75),
                      (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    for lo, hi in buckets_sample:
        sub = [r for r in matched if lo <= r.get("nsim", r.get("name_similarity", 0)) < hi]
        if not sub:
            continue
        sample = random.sample(sub, min(3, len(sub)))
        print(f"\n  nsim {lo:.2f}-{hi:.2f}  (n={len(sub):,})")
        _print_match_table(sample, indent=4, show_coords=show_coords, show_addr=show_addr)

    # -----------------------------------------------------------------------
    # Agreement analysis — only meaningful where overture_open is ground-truth labeled
    gt_records = [r for r in matched if r.get("overture_open") is not None
                  and r.get("sf_open") is not None]
    if gt_records:
        section("LABEL AGREEMENT BY NSIM BUCKET (ground-truth Overture only)")
        buckets = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
        print(f"\n  {'nsim range':>12}  {'n':>5}  {'agree':>6}  {'agree%':>7}")
        print("  " + "-" * 38)
        for lo, hi in buckets:
            sub = [r for r in gt_records if lo <= r.get("nsim", r.get("name_similarity", 0)) < hi]
            if not sub:
                continue
            agree_n = sum(1 for r in sub if r["sf_open"] == r["overture_open"])
            print(f"  {lo:.2f}-{hi:.2f}      {len(sub):>5,}  {agree_n:>6,}  "
                  f"{100*agree_n/len(sub):>6.1f}%")

        total_agree = sum(1 for r in gt_records if r["sf_open"] == r["overture_open"])
        print(f"\n  Overall agreement: {total_agree}/{len(gt_records)} "
              f"({100*total_agree/len(gt_records):.1f}%)")

    # -----------------------------------------------------------------------
    section("SF LABEL DISTRIBUTION IN MATCHED SET")
    sf_labels = [r["sf_open"] for r in matched if r.get("sf_open") is not None]
    closed_n = sf_labels.count(0)
    open_n   = sf_labels.count(1)
    print(f"\n  sf_open=0 (closed): {closed_n:,}  ({100*closed_n/len(sf_labels):.1f}%)")
    print(f"  sf_open=1 (open):   {open_n:,}  ({100*open_n/len(sf_labels):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse sf_lookup.py match report")
    ap.add_argument("path", nargs="?", default="data/sf_matches.json",
                    help="Match NDJSON file (default: data/sf_matches.json)")
    ap.add_argument("--top", type=int, default=20,
                    help="Bottom-N false-positive candidates to show (default: 20)")
    ap.add_argument("--verbose-coords", action="store_true", default=True,
                    help="Show lat/lon coordinates under each match row")
    ap.add_argument("--verbose-addr", action="store_true", default=True,
                    help="Show street addresses under each match row")
    args = ap.parse_args()
    p = Path(args.path)
    if not p.exists():
        sys.exit(f"File not found: {p}")
    main(p, top_n=args.top, show_coords=args.verbose_coords, show_addr=args.verbose_addr)
