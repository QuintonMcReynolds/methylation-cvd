"""Data and model drift monitoring for incoming methylation batches.

Fit once on the training data (DriftMonitor.fit), then call check(batch) for each new lab batch:
  sample level   Mahalanobis distance in the training PCA space (flags single odd samples: degraded DNA, wrong tissue)
  batch level    kernel MMD between the batch and the training reference in PCA space (permutation p-value), and
                 per-CpG two-sample KS tests on the model's CpGs with Benjamini-Hochberg FDR (which features moved)
  prediction     PSI of the risk-score distribution vs training
  performance    AUC on whatever labelled feedback has arrived for this batch (labels arrive late in practice)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


def _mmd2(x, y, gamma):
    def k(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        return np.exp(-gamma * np.maximum(d, 0))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return kxx.sum() / (len(x) * (len(x) - 1)) + kyy.sum() / (len(y) * (len(y) - 1)) - 2 * kxy.mean()


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, n + 1) / n)
    k = int(np.max(np.flatnonzero(passed))) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(new, edges)[0] / len(new) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


class DriftMonitor:
    def __init__(self, n_pcs: int = 20, n_perm: int = 300, seed: int = 0):
        self.n_pcs, self.n_perm, self.rng = n_pcs, n_perm, np.random.default_rng(seed)

    def fit(self, X_ref: np.ndarray, scores_ref: np.ndarray, model_cols: np.ndarray):
        """X_ref: training samples x monitored CpGs (e.g. the 50k most variable); model_cols: indexes of the model's CpGs."""
        # Fit the PCA on one half of the reference and compare new batches against the OTHER half. In-sample PCA scores are
        # inflated (the components partly fit that sample's noise), so comparing them with out-of-sample scores of a
        # perfectly clean batch looks like drift. Both sides of the test must be out-of-sample projections.
        perm = self.rng.permutation(len(X_ref))
        fit_i, ref_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
        self.mu = X_ref[fit_i].mean(0)
        self.pca = PCA(min(self.n_pcs, len(fit_i) - 1), random_state=0).fit(X_ref[fit_i] - self.mu)
        self.n_pcs = self.pca.n_components_
        self.ref_pc = self.pca.transform(X_ref[ref_i] - self.mu)
        cov = np.cov(self.ref_pc.T)
        self.prec = np.linalg.inv(cov + 1e-6 * np.eye(self.n_pcs))
        d = np.einsum("ij,jk,ik->i", self.ref_pc, self.prec, self.ref_pc)
        self.sample_thr = float(np.quantile(d, 0.99))
        sq = ((self.ref_pc[:, None] - self.ref_pc[None]) ** 2).sum(-1)
        self.gamma = 1.0 / float(np.median(sq[np.triu_indices(len(sq), 1)]))
        self.X_model_ref = X_ref[:, model_cols]
        self.model_cols = model_cols
        self.scores_ref = scores_ref[ref_i]
        return self

    def check(self, X: np.ndarray, scores: np.ndarray, labels: np.ndarray | None = None) -> dict:
        pc = self.pca.transform(X - self.mu)
        d = np.einsum("ij,jk,ik->i", pc, self.prec, pc)
        stat = _mmd2(self.ref_pc, pc, self.gamma)
        pool = np.concatenate([self.ref_pc, pc])
        n = len(self.ref_pc)
        null = []
        for _ in range(self.n_perm):
            p = self.rng.permutation(len(pool))
            null.append(_mmd2(pool[p[:n]], pool[p[n:]], self.gamma))
        p_mmd = (1 + np.sum(np.asarray(null) >= stat)) / (self.n_perm + 1)
        ks_p = np.array([ks_2samp(self.X_model_ref[:, j], X[:, c]).pvalue for j, c in enumerate(self.model_cols)])
        moved = bh(ks_p)
        out = {"n": int(len(X)), "outlier_samples": int((d > self.sample_thr).sum()), "batch_mmd_p": float(p_mmd),
               "batch_drift": bool(p_mmd < 0.01), "model_cpgs_shifted_fdr05": int(moved.sum()),
               "model_cpgs_total": int(len(self.model_cols)), "score_psi": psi(self.scores_ref, scores),
               "mean_risk": float(np.mean(scores))}
        # Hold results only on two independent signals: whole-batch MMD drift AND either >= 5 % of the model's CpGs shifted
        # (FDR 5 %) or >= 25 % of samples outlying. MMD alone false-alarmed on ~3 % of clean held-out batches (1 of 30
        # in reports/drift_calibration.json); a real technical artefact moves the model's CpGs too.
        out["hold"] = bool(out["batch_drift"] and (moved.mean() >= 0.05 or out["outlier_samples"] >= 0.25 * len(X)))
        if labels is not None and len(np.unique(labels)) == 2:
            out["auc_feedback"] = float(roc_auc_score(labels, scores))
        return out
