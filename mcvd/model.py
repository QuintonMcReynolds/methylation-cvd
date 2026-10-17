"""The deployable classifier: [age/sex residualisation] -> top-k CpG selection -> model, all fit on training data only.

Input: X = CpG M-values (n x p, columns in the training probe order); cov = (age, sex_male) columns appended.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import welch_t


class MethylClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, k: int = 1000, model: str = "en", C: float = 0.1, adjust_cov: bool = True, include_cov: bool = True,
                 n_cov: int = 2, n_var: int = 100_000, random_state: int = 0):
        self.k, self.model, self.C, self.adjust_cov, self.n_var = k, model, C, adjust_cov, n_var
        self.include_cov, self.n_cov, self.random_state = include_cov, n_cov, random_state

    def _split(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return Z[:, : -self.n_cov], Z[:, -self.n_cov:]      # views, no copy

    def _design(self, cov):
        return np.c_[np.ones(len(cov), dtype=np.float32), cov].astype(np.float32)

    @staticmethod
    def _top_var(X, n, chunk=100_000):
        """Most variable CpGs on the TRAINING samples (label-free), computed in column chunks to bound memory."""
        v = np.concatenate([X[:, i:i + chunk].var(0) for i in range(0, X.shape[1], chunk)])
        return np.sort(np.argsort(-v)[: min(n, len(v))])

    def fit(self, Z, y):
        Xfull, cov = self._split(Z)
        y = np.asarray(y)
        self.var_idx_ = self._top_var(Xfull, self.n_var)
        X = np.ascontiguousarray(Xfull[:, self.var_idx_])
        if self.adjust_cov:
            D = self._design(cov)
            self.beta_ = np.linalg.solve(D.T @ D, D.T @ X).astype(np.float32)            # per-CpG OLS on age, sex
            X = X - D @ self.beta_
        t = np.nan_to_num(welch_t(X, y))
        self.idx_ = np.argsort(-np.abs(t))[: self.k]
        F = X[:, self.idx_]
        if self.include_cov:
            F = np.c_[F, cov]
        self.scaler_ = StandardScaler().fit(F)
        F = self.scaler_.transform(F)
        if self.model == "en":
            self.clf_ = LogisticRegression(solver="saga", l1_ratio=0.5, C=self.C, max_iter=3000, random_state=self.random_state)
        elif self.model == "rf":
            self.clf_ = RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=self.random_state)
        elif self.model == "lgb":
            import lightgbm as lgb
            self.clf_ = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.3, verbose=-1, random_state=self.random_state)
        else:
            raise ValueError(self.model)
        self.clf_.fit(F, y)
        self.classes_ = self.clf_.classes_
        return self

    def _features(self, Z):
        Xfull, cov = self._split(Z)
        X = Xfull[:, self.var_idx_]
        if self.adjust_cov:
            X = X - self._design(cov) @ self.beta_
        F = X[:, self.idx_]
        if self.include_cov:
            F = np.c_[F, cov]
        return self.scaler_.transform(F)

    def predict_proba(self, Z):
        return self.clf_.predict_proba(self._features(Z))

    def predict(self, Z):
        return (self.predict_proba(Z)[:, 1] >= 0.5).astype(int)


class CovOnly(ClassifierMixin, BaseEstimator):
    """Baseline: logistic regression on age and sex only."""

    def __init__(self, n_cov: int = 2):
        self.n_cov = n_cov

    def fit(self, Z, y):
        cov = np.asarray(Z)[:, -self.n_cov:]
        self.s_ = StandardScaler().fit(cov)
        self.clf_ = LogisticRegression().fit(self.s_.transform(cov), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, Z):
        return self.clf_.predict_proba(self.s_.transform(np.asarray(Z)[:, -self.n_cov:]))
