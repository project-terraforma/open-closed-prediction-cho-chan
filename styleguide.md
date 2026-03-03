# Style Guide

Conventions used across this project. Follow these when adding or modifying code.

---

## File & Directory Naming

- **snake_case** for all files and folders: `feature_engineering.py`, `train_log.json`.
- Keep names short and descriptive — prefer `encoder.py` over `place_encoder_module.py`.
- Group related scripts under `src/`, data under `data/`, saved artifacts under `models/`.

## Module Docstrings

Every Python file starts with a triple-quoted module docstring containing:

1. Filename
2. One-line summary
3. Brief description of what the module does or its public API
4. `Run:` section showing exact CLI invocation
5. `Outputs:` section listing artifacts produced (if any)

```python
"""
train.py
--------
End-to-end training pipeline.

Run:
    python src/train.py

Outputs saved to models/:
    encoder.pt  — best encoder weights
"""
```

## Imports

Order (separated by blank lines):

1. `from __future__ import annotations`
2. Standard library (`argparse`, `json`, `sys`, `pathlib`, …)
3. Third-party (`numpy`, `torch`, `sklearn`, …)
4. Local project imports (`from encoder import PlaceEncoder`)

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from encoder import PlaceEncoder
```

## Naming Conventions

| Element            | Style           | Example                        |
| ------------------ | --------------- | ------------------------------ |
| Files / modules    | `snake_case`    | `feature_engineering.py`       |
| Functions          | `snake_case`    | `extract_features()`           |
| Variables          | `snake_case`    | `max_src_conf`                 |
| Constants          | `UPPER_SNAKE`   | `N_NUMERIC = 19`               |
| Private constants  | `_UPPER_SNAKE`  | `_REFERENCE_DATE`              |
| Classes            | `PascalCase`    | `PlaceEncoder`, `PlaceDataset` |
| Type aliases       | `PascalCase`    | n/a                            |

## Type Hints

- Use modern syntax enabled by `from __future__ import annotations` (e.g., `list[str]`, `dict[str, Any]`, `str | Path`).
- Annotate all public function signatures (args + return).
- Use `tuple[float, float]` over `Tuple[float, float]`.

```python
def train(
    splits_dir: str | Path = "splits",
    out_dir: str | Path = "models",
    max_epochs: int = 100,
) -> tuple:
```

## Docstrings (Functions & Classes)

Use Google-style docstrings with `Args:` and `Returns:` sections.

```python
def extract_features(record: dict[str, Any]) -> dict[str, Any]:
    """Return a flat feature dict from one Overture place JSON record.

    Args:
        record: Parsed JSON dict for a single place.

    Returns:
        Dict mapping feature name -> scalar (int, float, or str).
    """
```

## Code Layout

- **Section dividers**: use comment banners to group related code.

```python
# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
```

- **Blank lines**: two blank lines between top-level definitions, one within classes.
- **Line length**: aim for ≤ 100 characters; hard limit at 120.

## CLI Scripts

- Use `argparse` for any configurable parameters.
- Provide sensible defaults so scripts work with zero flags.
- Guard entry points with `if __name__ == "__main__":`.

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(lr=args.lr)
```

## Paths

- Use `pathlib.Path` instead of `os.path`.
- Accept `str | Path` in function signatures; convert early with `Path(...)`.

## Data & Artifacts

- Raw / input data lives in `data/`.
- Train/val splits live in `splits/`.
- Trained model weights, configs, and logs live in `models/`.
- Use `.npy` for numpy arrays, `.json` for configs/logs, `.pkl` for sklearn objects, `.pt` for PyTorch state dicts.

## Misc

- Prefer `numpy` for array math; avoid raw Python loops over large arrays.
- Use `torch.no_grad()` decorator for all eval/inference functions.
- Print progress to stdout (epoch, loss, metrics) so runs are observable.
