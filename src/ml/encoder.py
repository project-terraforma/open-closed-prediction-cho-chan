"""
encoder.py
----------
MLP encoder for Overture place features.

Architecture:
    n_numeric features  → BatchNorm1d
    1 category int      → Embedding(cat_vocab_size+1, 8)
    concat              → Linear(64) → BN → ReLU → Dropout(0.3)
                        → Linear(32) → BN → ReLU   ← 32-dim embedding (z)
    z                   → Linear(2)                 ← classification head (training only)

The encoder produces a 32-dim embedding used downstream by NCM/SLDA.
At inference time, call encoder.encode(x_num, x_cat) to get embeddings.

Input convention (matches split.py output):
    X[:, :-1]  — float32 numeric features
    X[:, -1]   — int64 category index (last column)

n_numeric is inferred automatically from X.shape[1] - 1 (PlaceDataset) and
must be passed explicitly to PlaceEncoder (saved in encoder_config.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

EMBED_DIM = 8       # category embedding dimension
HIDDEN1 = 64
HIDDEN2 = 32        # output embedding dimension
DROPOUT = 0.3


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
    """MLP encoder producing 32-dim place embeddings.

    Args:
        cat_vocab_size: Number of unique training categories (from LabelEncoder).
                        An extra OOV slot is added automatically (+1).
        embed_dim:      Dimension of the category embedding (default 8).
    """

    def __init__(self, cat_vocab_size: int, n_numeric: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()

        self.n_numeric = n_numeric
        self.cat_embedding = nn.Embedding(cat_vocab_size + 1, embed_dim)  # +1 = OOV

        input_dim = n_numeric + embed_dim

        self.bn_input = nn.BatchNorm1d(n_numeric)

        self.fc1 = nn.Linear(input_dim, HIDDEN1)
        self.bn1 = nn.BatchNorm1d(HIDDEN1)
        self.drop1 = nn.Dropout(DROPOUT)

        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.bn2 = nn.BatchNorm1d(HIDDEN2)

        self.head = nn.Linear(HIDDEN2, 2)

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return 32-dim embeddings (no classification head).

        Args:
            x_num: float32 tensor (B, 19)
            x_cat: int64 tensor  (B,)

        Returns:
            z: float32 tensor (B, 32)
        """
        x_num = self.bn_input(x_num)
        cat_emb = self.cat_embedding(x_cat)          # (B, 8)
        x = torch.cat([x_num, cat_emb], dim=1)       # (B, 27)

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))  # (B, 64)
        z = F.relu(self.bn2(self.fc2(x)))               # (B, 32)
        return z

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Return class logits for training.

        Args:
            x_num: float32 tensor (B, 19)
            x_cat: int64 tensor  (B,)

        Returns:
            logits: float32 tensor (B, 2)
        """
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
