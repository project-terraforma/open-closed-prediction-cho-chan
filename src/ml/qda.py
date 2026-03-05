"""
qda.py
------
Streaming Quadratic Discriminant Analysis (QDA) for continual learning.

Like SLDA but each class has its own covariance matrix instead of a shared
pooled one. This allows a curved (quadratic) decision boundary that can
better capture classes with different embedding distributions.

Trade-off vs SLDA:
    - More expressive boundary (class-specific covariance)
    - Requires more data per class to estimate covariance reliably
    - Uses higher regularization by default (reg=0.1) to compensate for
      the closed class having only ~250 samples in 32-dim embedding space

QDA score for class c:
    s_c(z) = -0.5 * (z-μ_c)^T Σ_c^{-1} (z-μ_c)  -- Mahalanobis term
             -0.5 * log|Σ_c|                        -- volume penalty
             + log π_c                              -- class prior

Supports the same incremental update interface as StreamingLDA.

Usage:
    qda = StreamingQDA()
    qda.fit(z_train, y_train)
    probs = qda.predict_proba(z_val)    # shape (N, 2)
    preds = qda.predict(z_val)          # shape (N,)

    # Continual learning update:
    qda.update(z_new, y_new)
"""

from __future__ import annotations

import numpy as np


class StreamingQDA:
    """Streaming Quadratic Discriminant Analysis classifier.

    Each class has its own regularized covariance matrix. The decision
    boundary is quadratic (curved), unlike SLDA's linear boundary.

    Attributes:
        means_:   dict mapping class label -> mean embedding (D,)
        counts_:  dict mapping class label -> number of samples seen
        scatter_: dict mapping class label -> within-class scatter S_c (D, D)
                  (unnormalized; class covariance = S_c / (n_c - 1) + reg * I)
        classes_: sorted list of known class labels
        reg:      ridge regularization added to each class covariance diagonal
                  (higher than SLDA default due to small per-class sample counts)
    """

    def __init__(self, reg: float = 0.1) -> None:
        self.reg = reg
        self.means_: dict[int, np.ndarray] = {}
        self.counts_: dict[int, int] = {}
        self.scatter_: dict[int, np.ndarray] = {}
        self.classes_: list[int] = []

    # ------------------------------------------------------------------
    # Fit / update  (identical to StreamingLDA — scatter storage is shared)
    # ------------------------------------------------------------------

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "StreamingQDA":
        """Compute per-class means and scatter matrices from scratch.

        Args:
            Z: float32 embeddings, shape (N, D)
            y: int labels, shape (N,)

        Returns:
            self
        """
        self.means_ = {}
        self.counts_ = {}
        self.scatter_ = {}

        for cls in np.unique(y):
            mask = y == cls
            z_c = Z[mask]
            n_c = int(mask.sum())
            mu_c = z_c.mean(axis=0)
            diff = z_c - mu_c
            S_c = diff.T @ diff          # (D, D) within-class scatter

            self.means_[int(cls)] = mu_c
            self.counts_[int(cls)] = n_c
            self.scatter_[int(cls)] = S_c

        self.classes_ = sorted(self.means_.keys())
        return self

    def update(self, Z: np.ndarray, y: np.ndarray) -> "StreamingQDA":
        """Incrementally update means and scatter matrices (parallel Welford).

        Parallel Welford formula:
            n      = n_old + n_new
            μ      = (n_old·μ_old + n_new·μ_new) / n
            δ      = μ_new − μ_old
            S      = S_old + S_new + (n_old·n_new / n) · outer(δ, δ)

        Args:
            Z: new embeddings, shape (N, D)
            y: labels, shape (N,)

        Returns:
            self
        """
        for cls in np.unique(y):
            mask = y == cls
            z_new = Z[mask]
            n_new = int(mask.sum())
            mu_new = z_new.mean(axis=0)
            diff_new = z_new - mu_new
            S_new = diff_new.T @ diff_new

            if int(cls) in self.means_:
                n_old = self.counts_[int(cls)]
                mu_old = self.means_[int(cls)]
                n_combined = n_old + n_new

                delta = mu_new - mu_old
                mu_combined = (n_old * mu_old + n_new * mu_new) / n_combined
                S_combined = (
                    self.scatter_[int(cls)]
                    + S_new
                    + (n_old * n_new / n_combined) * np.outer(delta, delta)
                )

                self.means_[int(cls)] = mu_combined
                self.counts_[int(cls)] = n_combined
                self.scatter_[int(cls)] = S_combined
            else:
                self.means_[int(cls)] = mu_new
                self.counts_[int(cls)] = n_new
                self.scatter_[int(cls)] = S_new
                self.classes_ = sorted(self.means_.keys())

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _class_covariance(self, cls: int) -> np.ndarray:
        """Per-class regularized covariance matrix (D, D)."""
        n_c = self.counts_[cls]
        D = self.means_[cls].shape[0]
        dof = max(n_c - 1, 1)
        Sigma_c = self.scatter_[cls] / dof + self.reg * np.eye(D, dtype=np.float64)
        return Sigma_c

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Return class probabilities via softmax over QDA discriminant scores.

        QDA score for class c:
            s_c(z) = -0.5 * (z-μ_c)^T Σ_c^{-1} (z-μ_c)
                     -0.5 * log|Σ_c|
                     + log π_c

        Args:
            Z: embeddings, shape (N, D)

        Returns:
            probs: float32 array, shape (N, len(classes_))
        """
        if not self.means_:
            raise RuntimeError("Call fit() before predict_proba().")

        N_total = sum(self.counts_.values())
        Z64 = Z.astype(np.float64)
        N = Z64.shape[0]
        C = len(self.classes_)
        scores = np.zeros((N, C), dtype=np.float64)

        for i, cls in enumerate(self.classes_):
            mu_c = self.means_[cls]
            Sigma_c = self._class_covariance(cls)

            # Centered data: (N, D)
            Z_centered = Z64 - mu_c[None, :]

            # Solve Σ_c V = Z_centered.T → V shape (D, N)
            V = np.linalg.solve(Sigma_c, Z_centered.T)

            # Mahalanobis: diag of Z_centered @ Σ_c^{-1} @ Z_centered.T → (N,)
            maha = (Z_centered.T * V).sum(axis=0)

            # Log-determinant of Σ_c
            sign, logdet = np.linalg.slogdet(Sigma_c)
            if sign <= 0:
                logdet = 1e6   # degenerate — penalize heavily

            # Class prior
            log_prior = np.log(self.counts_[cls] / N_total)

            scores[:, i] = -0.5 * maha - 0.5 * logdet + log_prior

        # Numerically stable softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return predicted class labels (highest QDA score).

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
        print("StreamingQDA")
        print(f"  reg={self.reg}")
        for cls in self.classes_:
            mu = self.means_[cls]
            n = self.counts_[cls]
            Sigma_c = self._class_covariance(cls)
            cond = np.linalg.cond(Sigma_c)
            print(f"  class={cls}  n={n:>6}  mean_norm={np.linalg.norm(mu):.4f}"
                  f"  Sigma cond={cond:.1f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Toy: class 0 has tighter spread (like closed places in small minority)
    z0 = rng.normal(-1.0, 0.3, size=(50, 32)).astype(np.float32)
    z1 = rng.normal(+1.0, 0.8, size=(150, 32)).astype(np.float32)
    Z = np.vstack([z0, z1])
    y = np.array([0] * 50 + [1] * 150, dtype=np.int64)

    qda = StreamingQDA(reg=0.1)
    qda.fit(Z[:160], y[:160])
    qda.summary()

    probs = qda.predict_proba(Z[160:])
    preds = qda.predict(Z[160:])
    acc = (preds == y[160:]).mean()
    print(f"\nVal accuracy on toy data: {acc:.3f}")
