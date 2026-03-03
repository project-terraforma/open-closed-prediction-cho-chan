import json, numpy as np
from pathlib import Path
from feature_engineering import extract_features

old, new = [], []
existing_ids = set()

with open("data/project_c_samples.json") as f:
    for line in f:
        r = json.loads(line)
        existing_ids.add(r["id"])
        if r["open"] == 0:
            old.append(extract_features(r))

with open("data/augmented_samples.json") as f:
    for line in f:
        r = json.loads(line)
        if r["open"] == 0 and r["id"] not in existing_ids:
            new.append(extract_features(r))

print(f"Old closed: {len(old)}, New closed: {len(new)}")
for feat in ["confidence", "completeness_score", "has_website", "has_phone", "source_count"]:
    o = np.mean([r[feat] for r in old])
    n = np.mean([r[feat] for r in new])
    print(f"  {feat:30s}  old={o:.3f}  new={n:.3f}  diff={n-o:+.3f}")
