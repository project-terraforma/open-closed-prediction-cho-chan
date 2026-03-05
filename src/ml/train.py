"""
train.py
--------
End-to-end training pipeline.

Steps:
    1. Load splits from splits/  (run split.py first if missing)
    2. Train PlaceEncoder end-to-end with weighted CrossEntropy + Adam + cosine LR
    3. Early stopping on validation loss (patience=10)
    4. Extract 32-dim embeddings from best frozen encoder
    5. Fit NearestClassMean, StreamingLDA, and StreamingQDA on training embeddings
    6. Save all artifacts to models/

Run:
    python src/split.py data/project_c_samples.json   # once
    python src/train.py

Optional flags:
    --splits  splits   directory with X_train.npy etc  (default: splits)
    --out     models   output directory                 (default: models)
    --epochs  100      max training epochs              (default: 100)
    --lr      1e-3     learning rate                    (default: 1e-3)
    --batch   64       batch size                       (default: 64)
    --patience 10      early-stop patience (val loss)   (default: 10)

Outputs saved to models/:
    encoder.pt           — best encoder weights (state_dict)
    encoder_config.json  — cat_vocab_size (for reloading the model)
    ncm.pkl              — fitted NearestClassMean
    slda.pkl             — fitted StreamingLDA
    qda.pkl              — fitted StreamingQDA
    embeddings_train.npy — (N_train, 32) training embeddings
    embeddings_val.npy   — (N_val,   32) validation embeddings
    train_log.json       — per-epoch metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from encoder import PlaceDataset, PlaceEncoder, class_weights, load_splits
from ncm import NearestClassMean
from qda import StreamingQDA
from slda import StreamingLDA

import random

def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss with optional class weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The focal term (1 - p_t)^gamma downweights easy examples so training
    focuses on hard, ambiguous cases — useful for the 9:1 closed/open imbalance.
    gamma=0 reduces to standard weighted cross-entropy.

    Args:
        weight: per-class weights (same as nn.CrossEntropyLoss weight)
        gamma:  focusing exponent (0 = off, 2 = typical)
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 0.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # p_t: model confidence in the correct class (separate from class weighting)
        pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        # per-sample weighted CE (applies alpha_t class weight)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: PlaceEncoder,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_num, x_cat), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: PlaceEncoder,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (val_loss, val_auc_roc)."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        logits = model(x_num, x_cat)
        total_loss += criterion(logits, y).item() * len(y)
        # P(class=1 / open) — AUC is symmetric, same as 1-P(closed)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader.dataset), float(auc)


@torch.no_grad()
def extract_embeddings(
    model: PlaceEncoder,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    zs = []
    for x_num, x_cat, _y in loader:
        zs.append(model.encode(x_num.to(device), x_cat.to(device)).cpu().numpy())
    return np.concatenate(zs, axis=0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def train(
    splits_dir: str | Path = "splits",
    out_dir: str | Path = "models",
    max_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 20,
    gamma: float = 0.0,
) -> tuple:
    _seed_everything(42)

    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)

    if not splits_dir.exists():
        sys.exit(f"splits/ not found — run: python src/split.py data/project_c_samples.json")

    out_dir.mkdir(exist_ok=True)

    # --- Load splits ---
    X_train, X_val, y_train, y_val = load_splits(splits_dir)
    with open(splits_dir / "category_encoder.pkl", "rb") as f:
        cat_enc = pickle.load(f)
    cat_vocab_size = len(cat_enc.classes_)

    print(f"Train: {len(y_train):,}  (closed={( y_train==0).sum()}  open={(y_train==1).sum()})")
    print(f"Val:   {len(y_val):,}  (closed={(y_val==0).sum()}  open={(y_val==1).sum()})")
    print(f"Cat vocab size: {cat_vocab_size} (+1 OOV)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Datasets / loaders ---
    ds_train = PlaceDataset(X_train, y_train)
    ds_val   = PlaceDataset(X_val,   y_val)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    # --- Model, loss, optimizer, scheduler ---
    model = PlaceEncoder(cat_vocab_size=cat_vocab_size, n_numeric=X_train.shape[1]-1).to(device)
    weights = class_weights(y_train, device)
    criterion = FocalLoss(weight=weights, gamma=gamma)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    print(f"Params: {model.param_count():,}  |  "
          f"Class weights: closed={weights[0]:.3f}  open={weights[1]:.3f}")
    print(f"Loss: FocalLoss(gamma={gamma})  {'(= weighted CE)' if gamma == 0 else ''}")
    print(f"Max epochs: {max_epochs}  |  Patience: {patience}  |  LR: {lr}  |  WD: 1e-4  |  Batch: {batch_size}\n")

    # --- Training loop ---
    # Monitor val_auc (not val_loss) — with class-weighted loss on imbalanced data,
    # loss and AUC often diverge; AUC directly measures what we care about.
    best_val_auc = 0.0
    best_val_loss_at_best = float("inf")
    best_epoch = 0
    patience_counter = 0
    log: list[dict] = []

    header = f"{'Ep':>4}  {'TrainLoss':>9}  {'ValLoss':>8}  {'ValAUC':>7}  {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, loader_train, criterion, optimizer, device)
        val_loss, val_auc = eval_epoch(model, loader_val, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_auc": val_auc})

        marker = " *" if val_auc > best_val_auc else ""
        print(f"{epoch:>4}  {train_loss:>9.4f}  {val_loss:>8.4f}  {val_auc:>7.4f}  {lr_now:>8.6f}{marker}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss_at_best = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "encoder.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping (patience={patience}, best epoch={best_epoch})")
                break

    # --- Reload best weights ---
    model.load_state_dict(torch.load(out_dir / "encoder.pt", map_location=device))
    print(f"\nBest  epoch={best_epoch}  val_loss={best_val_loss_at_best:.4f}  val_auc={best_val_auc:.4f}")

    # Save encoder config and log
    with open(out_dir / "encoder_config.json", "w") as f:
        json.dump({"cat_vocab_size": cat_vocab_size, "n_numeric": X_train.shape[1]-1}, f)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # --- Extract embeddings (shuffle=False to preserve order) ---
    print("\nExtracting embeddings ...")
    loader_train_ord = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    Z_train = extract_embeddings(model, loader_train_ord, device)
    Z_val   = extract_embeddings(model, loader_val, device)
    print(f"  Z_train: {Z_train.shape}   Z_val: {Z_val.shape}")

    np.save(out_dir / "embeddings_train.npy", Z_train)
    np.save(out_dir / "embeddings_val.npy",   Z_val)

    # --- Fit NCM ---
    print("\nFitting NCM ...")
    ncm = NearestClassMean()
    ncm.fit(Z_train, y_train)
    ncm.summary()
    with open(out_dir / "ncm.pkl", "wb") as f:
        pickle.dump(ncm, f)

    # --- Fit SLDA ---
    print("\nFitting SLDA ...")
    slda = StreamingLDA()
    slda.fit(Z_train, y_train)
    slda.summary()
    with open(out_dir / "slda.pkl", "wb") as f:
        pickle.dump(slda, f)

    # --- Fit QDA ---
    print("\nFitting QDA ...")
    qda = StreamingQDA()
    qda.fit(Z_train, y_train)
    qda.summary()
    with open(out_dir / "qda.pkl", "wb") as f:
        pickle.dump(qda, f)

    print(f"\nAll artifacts saved to {out_dir}/")
    return model, ncm, slda, qda, Z_train, Z_val, y_train, y_val


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlaceEncoder + fit NCM/SLDA/QDA")
    parser.add_argument("--splits",  default="splits", help="splits/ directory")
    parser.add_argument("--out",     default="models",  help="output directory")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--patience",type=int,   default=20)
    parser.add_argument("--gamma",   type=float, default=0.0,
                        help="Focal loss gamma (0 = standard weighted CE, 2 = focal;)")
    args = parser.parse_args()

    train(
        splits_dir=args.splits,
        out_dir=args.out,
        max_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        patience=args.patience,
        gamma=args.gamma,
    )
