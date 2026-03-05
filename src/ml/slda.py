"""
slda.py
-------
Streaming Linear Discriminant Analysis (SLDA) for continual learning.

Like NCM but uses a shared pooled covariance matrix (Mahalanobis distance)
instead of Euclidean distance. Typically more accurate than NCM when class
distributions are roughly Gaussian with similar covariance.

Supports incremental updates via the parallel Welford algorithm — class means
and the pooled scatter matrix are updated in O(N·D²) time with no need to
store old data.

Usage:
    slda = StreamingLDA()
    slda.fit(z_train, y_train)
    probs = slda.predict_proba(z_val)    # shape (N, 2)
    preds = slda.predict(z_val)          # shape (N,)

    # Continual learning update with new release:
    slda.update(z_new, y_new)            # O(N·D²), no old data needed
"""

from __future__ import annotations

import numpy as np


class StreamingLDA:
    """Streaming Linear Discriminant Analysis classifier.

    Attributes:
        means_:   dict mapping class label -> mean embedding (D,)
        counts_:  dict mapping class label -> number of samples seen
        scatter_: dict mapping class label -> within-class scatter S_c (D, D)
                  (unnormalized; pooled covariance = Σ_c S_c / (N - C))
        classes_: sorted list of known class labels
        reg:      ridge regularization added to pooled covariance diagonal
    """

    def __init__(self, reg: float = 1e-4) -> None:
        self.reg = reg
        self.means_: dict[int, np.ndarray] = {}
        self.counts_: dict[int, int] = {}
        self.scatter_: dict[int, np.ndarray] = {}
        self.classes_: list[int] = []

    # ------------------------------------------------------------------
    # Fit / update
    # ------------------------------------------------------------------

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "StreamingLDA":
        """Compute class means and scatter matrices from scratch.

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
            z_c = Z[mask]                  # (n_c, D)
            n_c = int(mask.sum())
            mu_c = z_c.mean(axis=0)        # (D,)
            diff = z_c - mu_c              # (n_c, D)
            S_c = diff.T @ diff            # (D, D) within-class scatter

            self.means_[int(cls)] = mu_c
            self.counts_[int(cls)] = n_c
            self.scatter_[int(cls)] = S_c

        self.classes_ = sorted(self.means_.keys())
        return self

    def update(self, Z: np.ndarray, y: np.ndarray) -> "StreamingLDA":
        """Incrementally update means and scatter matrices (parallel Welford).

        Parallel Welford formula for merging two batches:
            n      = n_old + n_new
            μ      = (n_old·μ_old + n_new·μ_new) / n
            δ      = μ_new − μ_old
            S      = S_old + S_new + (n_old·n_new / n) · outer(δ, δ)

        This is exact — the combined scatter equals what a single full fit
        would produce.

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
                # New class not seen during fit
                self.means_[int(cls)] = mu_new
                self.counts_[int(cls)] = n_new
                self.scatter_[int(cls)] = S_new
                self.classes_ = sorted(self.means_.keys())

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pooled_covariance(self) -> np.ndarray:
        """Regularized pooled within-class covariance matrix (D, D)."""
        S_total = sum(self.scatter_.values())          # sum scatter across classes
        N_total = sum(self.counts_.values())
        C = len(self.classes_)
        dof = max(N_total - C, 1)                      # degrees of freedom
        D = next(iter(self.means_.values())).shape[0]
        Sigma = S_total / dof + self.reg * np.eye(D, dtype=np.float64)
        return Sigma

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Return class probabilities via softmax over LDA discriminant scores.

        LDA score for class c:
            score_c(z) = z^T Σ^{-1} μ_c  −  0.5 · μ_c^T Σ^{-1} μ_c  +  log π_c

        Uses np.linalg.solve (numerically stable; avoids explicit inverse).

        Args:
            Z: embeddings, shape (N, D)

        Returns:
            probs: float32 array, shape (N, len(classes_))
        """
        if not self.means_:
            raise RuntimeError("Call fit() before predict_proba().")

        Sigma = self._pooled_covariance()              # (D, D)
        N_total = sum(self.counts_.values())

        # Stack class means: (D, C)
        mu_stack = np.stack([self.means_[c] for c in self.classes_], axis=1)

        # Solve Σ W = mu_stack  →  W = Σ^{-1} mu_stack,  shape (D, C)
        W = np.linalg.solve(Sigma, mu_stack)

        # Linear term: z^T W,  shape (N, C)
        linear = Z.astype(np.float64) @ W

        # Quadratic term (scalar per class): 0.5 · μ_c^T W_c
        quadratic = 0.5 * (mu_stack * W).sum(axis=0)  # (C,)

        # Class prior: log π_c = log(n_c / N)
        priors = np.array([
            np.log(self.counts_[c] / N_total) for c in self.classes_
        ])                                             # (C,)

        logits = linear - quadratic[None, :] + priors[None, :]  # (N, C)

        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return predicted class labels (highest LDA score).

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
        Sigma = self._pooled_covariance()
        print("StreamingLDA")
        print(f"  D={Sigma.shape[0]}  reg={self.reg}")
        for cls in self.classes_:
            mu = self.means_[cls]
            n = self.counts_[cls]
            print(f"  class={cls}  n={n:>6}  mean_norm={np.linalg.norm(mu):.4f}")
        cond = np.linalg.cond(Sigma)
        print(f"  Sigma condition number: {cond:.1f}")


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

    slda = StreamingLDA()
    slda.fit(Z[:160], y[:160])
    slda.summary()

    probs = slda.predict_proba(Z[160:])
    preds = slda.predict(Z[160:])
    acc = (preds == y[160:]).mean()
    print(f"\nVal accuracy on toy data: {acc:.3f}")

    # Test incremental update — should match full fit exactly
    slda2 = StreamingLDA()
    slda2.fit(Z[:80], y[:80])
    slda2.update(Z[80:160], y[80:160])

    mean_diff_0 = np.abs(slda.means_[0] - slda2.means_[0]).max()
    mean_diff_1 = np.abs(slda.means_[1] - slda2.means_[1]).max()
    S_diff_0 = np.abs(slda.scatter_[0] - slda2.scatter_[0]).max()
    S_diff_1 = np.abs(slda.scatter_[1] - slda2.scatter_[1]).max()
    print(f"\nIncremental vs full-fit:")
    print(f"  mean diff:    class0={mean_diff_0:.6f}  class1={mean_diff_1:.6f}")
    print(f"  scatter diff: class0={S_diff_0:.6f}  class1={S_diff_1:.6f}")
    print("(should be near zero)")
