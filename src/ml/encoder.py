"""
encoder.py
----------
MLP encoder for Overture place features.

Architecture:
    n_numeric features  → BatchNorm1d
    1 category int      → Embedding(cat_vocab_size+1, embed_dim)
    concat              → [Linear(h) → BN → ReLU → Dropout] × (L-1)
                        →  Linear(h) → BN → ReLU              ← embed (z)
    z                   → Linear(2)                            ← head (training only)

hidden_dims controls layer sizes; last element is the embedding dimension.
Default: [128, 64, 32]  (3 layers, 32-dim output — compatible with NCM/SLDA/QDA)
Small/original:   [64, 32]

The encoder produces a hidden_dims[-1]-dim embedding used downstream by NCM/SLDA.
At inference time, call encoder.encode(x_num, x_cat) to get embeddings.

Input convention (matches split.py output):
    X[:, :-1]  — float32 numeric features
    X[:, -1]   — int64 category index (last column)

n_numeric is inferred automatically from X.shape[1] - 1 (PlaceDataset) and
must be passed explicitly to PlaceEncoder (saved in encoder_config.json).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Default model config — mirrors config/train.yaml model section.
# Used as fallback when no cfg is passed (e.g. smoke test).
DEFAULT_MODEL_CFG: dict = {
    "hidden_dims": [128, 64, 32],  # last = embedding dim; original 2-layer: [64, 32]
    "embed_dim": 8,
    "dropout": 0.3,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlaceDataset(Dataset):
    """PyTorch Dataset wrapping the numpy arrays produced by split.py.

    Args:
        X: float32 array of shape (N, F). Last column is category index.
        y: int64 array of shape (N,). Labels: 1=open, 0=closed.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        n_numeric = X.shape[1] - 1  # last column is always category
        self.x_num = torch.from_numpy(X[:, :n_numeric].astype(np.float32))
        self.x_cat = torch.from_numpy(X[:, n_numeric].astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PlaceEncoder(nn.Module):
    """MLP encoder producing hidden_dims[-1]-dim place embeddings.

    Args:
        cat_vocab_size: Number of unique training categories (from LabelEncoder).
                        An extra OOV slot is added automatically (+1).
        n_numeric:      Number of numeric input features.
        cfg:            Model config dict — keys: hidden_dims, embed_dim, dropout.
                        Defaults to DEFAULT_MODEL_CFG if None.
                        Typically the ``model`` section from config/train.yaml.
    """

    def __init__(self, cat_vocab_size: int, n_numeric: int, cfg: dict | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DEFAULT_MODEL_CFG
        hidden_dims = list(cfg["hidden_dims"])
        embed_dim   = cfg.get("embed_dim", 8)
        dropout     = cfg.get("dropout", 0.3)

        self.hidden_dims = hidden_dims
        self.n_numeric = n_numeric
        self.cat_embedding = nn.Embedding(cat_vocab_size + 1, embed_dim)  # +1 = OOV

        self.bn_input = nn.BatchNorm1d(n_numeric)

        # Build layers dynamically from hidden_dims
        dims = [n_numeric + embed_dim] + hidden_dims
        self.fcs   = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_dims])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden_dims) - 1)])

        self.head = nn.Linear(hidden_dims[-1], 2)

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return hidden_dims[-1]-dim embeddings (no classification head)."""
        x = torch.cat([self.bn_input(x_num), self.cat_embedding(x_cat)], dim=1)

        for i, (fc, bn) in enumerate(zip(self.fcs, self.bns)):
            x = F.relu(bn(fc(x)))
            if i < len(self.drops):          # dropout after every layer except last
                x = self.drops[i](x)
        return x

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return class logits for training."""
        return self.head(self.encode(x_num, x_cat))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_weights(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle 9:1 imbalance.

    Returns tensor([w_closed, w_open]) on the given device.
    """
    n = len(y_train)
    n_open = (y_train == 1).sum()
    n_closed = (y_train == 0).sum()
    w_open = n / (2 * n_open)
    w_closed = n / (2 * n_closed)
    return torch.tensor([w_closed, w_open], dtype=torch.float32, device=device)


def load_splits(splits_dir: str | Path = "splits") -> tuple:
    """Load numpy arrays saved by split.py.

    Returns:
        X_train, X_val, y_train, y_val  (numpy arrays)
    """
    d = Path(splits_dir)
    return (
        np.load(d / "X_train.npy"),
        np.load(d / "X_val.npy"),
        np.load(d / "y_train.npy"),
        np.load(d / "y_val.npy"),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    splits_dir = Path("splits")
    if not splits_dir.exists():
        print("Run split.py first to generate splits/")
    else:
        X_train, X_val, y_train, y_val = load_splits(splits_dir)

        with open(splits_dir / "feature_names.json") as f:
            feat_names = json.load(f)

        # Infer vocab size from saved category encoder
        import pickle
        with open(splits_dir / "category_encoder.pkl", "rb") as f:
            enc = pickle.load(f)
        cat_vocab_size = len(enc.classes_)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PlaceEncoder(cat_vocab_size=cat_vocab_size, n_numeric=X_train.shape[1]-1).to(device)

        print(f"PlaceEncoder")
        print(f"  cat_vocab_size : {cat_vocab_size} (+1 OOV)")
        print(f"  parameters     : {model.param_count():,}")
        print(f"  device         : {device}")

        # Forward pass smoke test
        ds = PlaceDataset(X_train[:8], y_train[:8])
        x_num, x_cat, y_b = ds[0:8] if hasattr(ds, '__getitem__') else next(iter(ds))
        # Manual batch
        x_num_b = torch.stack([ds[i][0] for i in range(8)]).to(device)
        x_cat_b = torch.stack([ds[i][1] for i in range(8)]).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(x_num_b, x_cat_b)
            z = model.encode(x_num_b, x_cat_b)

        print(f"\nForward pass (batch=8):")
        print(f"  logits shape   : {logits.shape}")
        print(f"  embedding shape: {z.shape}")
        print(f"  logits sample  : {logits[0].cpu().numpy()}")

        weights = class_weights(y_train, device)
        print(f"\nClass weights: closed={weights[0]:.3f}  open={weights[1]:.3f}")
