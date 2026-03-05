"""
ncm.py
------
Nearest Class Mean (NCM) classifier for continual learning.

Classifies by finding the class whose mean embedding is closest (Euclidean).
Supports incremental updates — class means can be updated with new data
without retraining the encoder.

Usage:
    ncm = NearestClassMean()
    ncm.fit(z_train, y_train)           # set initial class means
    probs = ncm.predict_proba(z_val)    # shape (N, 2)
    preds = ncm.predict(z_val)          # shape (N,)

    # Continual learning update with new release:
    ncm.update(z_new, y_new)            # incremental mean update, O(N)
"""

from __future__ import annotations

import numpy as np


class NearestClassMean:
    """Nearest Class Mean classifier.

    Attributes:
        means_:   dict mapping class label -> mean embedding (ndarray, shape (D,))
        counts_:  dict mapping class label -> number of samples seen
        classes_: sorted list of known class labels
    """

    def __init__(self) -> None:
        self.means_: dict[int, np.ndarray] = {}
        self.counts_: dict[int, int] = {}
        self.classes_: list[int] = []

    # ------------------------------------------------------------------
    # Fit / update
    # ------------------------------------------------------------------

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "NearestClassMean":
        """Compute class means from scratch.

        Args:
            Z: float32 embeddings, shape (N, D)
            y: int labels, shape (N,)

        Returns:
            self
        """
        self.means_ = {}
        self.counts_ = {}

        for cls in np.unique(y):
            mask = y == cls
            self.means_[int(cls)] = Z[mask].mean(axis=0)
            self.counts_[int(cls)] = int(mask.sum())

        self.classes_ = sorted(self.means_.keys())
        return self

    def update(self, Z: np.ndarray, y: np.ndarray) -> "NearestClassMean":
        """Incrementally update class means with new embeddings.

        Uses the online mean formula:
            μ_new = (n_old * μ_old + Σ z_i) / (n_old + n_new)

        This is the key continual learning operation — O(N) time,
        no need to store or revisit old data.

        Args:
            Z: new embeddings, shape (N, D)
            y: labels for new embeddings, shape (N,)

        Returns:
            self
        """
        for cls in np.unique(y):
            mask = y == cls
            z_new = Z[mask]
            n_new = int(mask.sum())

            if int(cls) in self.means_:
                n_old = self.counts_[int(cls)]
                mu_old = self.means_[int(cls)]
                self.means_[int(cls)] = (n_old * mu_old + z_new.sum(axis=0)) / (n_old + n_new)
                self.counts_[int(cls)] = n_old + n_new
            else:
                # New class not seen during fit
                self.means_[int(cls)] = z_new.mean(axis=0)
                self.counts_[int(cls)] = n_new
                self.classes_ = sorted(self.means_.keys())

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return class probabilities via softmax over negative squared distances.

        Args:
            Z:           embeddings, shape (N, D)
            temperature: softmax temperature (higher = softer, default 1.0)

        Returns:
            probs: shape (N, len(classes_)), columns ordered by self.classes_
        """
        if not self.means_:
            raise RuntimeError("Call fit() before predict_proba().")

        # Compute squared Euclidean distance to each class mean
        # dists[i, c] = ||z_i - μ_c||²
        means = np.stack([self.means_[c] for c in self.classes_])  # (C, D)
        diff = Z[:, None, :] - means[None, :, :]                   # (N, C, D)
        dists = (diff ** 2).sum(axis=2)                            # (N, C)

        # Softmax over negative distances (closer = higher probability)
        logits = -dists / temperature
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return predicted class labels (nearest mean).

        Args:
            Z: embeddings, shape (N, D)

        Returns:
            labels: shape (N,)
        """
        probs = self.predict_proba(Z)
        idx = probs.argmax(axis=1)
        classes = np.array(self.classes_)
        return classes[idx]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        print("NearestClassMean")
        for cls in self.classes_:
            mu = self.means_[cls]
            n = self.counts_[cls]
            print(f"  class={cls}  n={n:>6}  mean_norm={np.linalg.norm(mu):.4f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Toy embeddings: class 0 centered at -1, class 1 at +1
    z0 = rng.normal(-1.0, 0.5, size=(50, 32)).astype(np.float32)
    z1 = rng.normal(+1.0, 0.5, size=(150, 32)).astype(np.float32)
    Z = np.vstack([z0, z1])
    y = np.array([0] * 50 + [1] * 150, dtype=np.int64)

    ncm = NearestClassMean()
    ncm.fit(Z[:160], y[:160])
    ncm.summary()

    probs = ncm.predict_proba(Z[160:])
    preds = ncm.predict(Z[160:])
    acc = (preds == y[160:]).mean()
    print(f"\nVal accuracy on toy data: {acc:.3f}")

    # Test incremental update
    ncm2 = NearestClassMean()
    ncm2.fit(Z[:80], y[:80])
    ncm2.update(Z[80:160], y[80:160])  # incremental

    # Means should match full fit
    diff_0 = np.abs(ncm.means_[0] - ncm2.means_[0]).max()
    diff_1 = np.abs(ncm.means_[1] - ncm2.means_[1]).max()
    print(f"\nIncremental vs full-fit mean diff: class0={diff_0:.6f}  class1={diff_1:.6f}")
    print("(should be near zero)")
