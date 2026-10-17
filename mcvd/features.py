"""Leakage-safe feature selection as an sklearn transformer, so it is re-fit inside every training fold."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def welch_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = X[y == 1], X[y == 0]
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    return (a.mean(0) - b.mean(0)) / np.sqrt(va / len(a) + vb / len(b) + 1e-12)


class TopKTTest(BaseEstimator, TransformerMixin):
    """Keep the k probes with the largest |Welch t| between classes, computed on the data passed to fit() only."""

    def __init__(self, k: int = 2000):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        t = welch_t(X, np.asarray(y))
        t = np.nan_to_num(t)
        self.idx_ = np.argsort(-np.abs(t))[: self.k]
        self.t_ = t[self.idx_]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx_]


class ColumnsAfter(BaseEstimator, TransformerMixin):
    """Pass through the trailing `n` covariate columns (age, sex) untouched by the CpG selector."""

    def __init__(self, n: int = 2):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, -self.n:]
