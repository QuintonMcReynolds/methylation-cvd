import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from mcvd.drift import DriftMonitor, bh, psi
from mcvd.features import TopKTTest, welch_t
from mcvd.model import MethylClassifier
from mcvd.registry import MethylPyfunc


def _noise(n=120, p=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p)).astype(np.float32)
    y = rng.integers(0, 2, n)
    cov = np.c_[rng.uniform(40, 60, n), rng.integers(0, 2, n)].astype(np.float32)
    return X, y, cov


def test_leaky_selection_inflates_auc_on_pure_noise():
    X, y, cov = _noise()
    Z = np.c_[X, cov]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    honest = cross_val_score(MethylClassifier(k=50, n_var=3000, include_cov=False), Z, y, cv=cv, scoring="roc_auc").mean()
    idx = np.argsort(-np.abs(welch_t(X, y)))[:50]                    # selection that has seen every label
    leaky = cross_val_score(MethylClassifier(k=50, n_var=50, include_cov=False), np.c_[X[:, idx], cov], y, cv=cv,
                            scoring="roc_auc").mean()
    assert abs(honest - 0.5) < 0.15 and leaky > 0.75


def test_selector_uses_only_training_rows():
    X, y, _ = _noise(p=200)
    s = TopKTTest(k=10).fit(X[:60], y[:60])
    assert set(s.idx_) == set(np.argsort(-np.abs(welch_t(X[:60], y[:60])))[:10])


def test_age_residualisation_removes_an_age_driven_cpg():
    rng = np.random.default_rng(1)
    n = 200
    age = rng.uniform(40, 60, n).astype(np.float32)
    y = (age + rng.normal(0, 3, n) > 50).astype(int)                 # outcome tracks age
    X = rng.normal(0, 1, (n, 500)).astype(np.float32)
    X[:, 0] = 0.2 * age + rng.normal(0, 0.1, n)                        # a pure age clock CpG
    Z = np.c_[X, age, rng.integers(0, 2, n)].astype(np.float32)
    adj = MethylClassifier(k=5, n_var=500, adjust_cov=True, include_cov=False).fit(Z, y)
    raw = MethylClassifier(k=5, n_var=500, adjust_cov=False, include_cov=False).fit(Z, y)
    assert 0 in raw.var_idx_[raw.idx_] and 0 not in adj.var_idx_[adj.idx_]


def _fitted_pyfunc():
    rng = np.random.default_rng(2)
    n, p = 150, 400
    X = rng.normal(0, 1.5, (n, p)).astype(np.float32)
    y = rng.integers(0, 2, n)
    X[y == 1, :20] += 1.0
    Z = np.c_[X, rng.uniform(40, 60, n), rng.integers(0, 2, n)].astype(np.float32)
    probes = [f"cg{i:08d}" for i in range(p)]
    m = MethylClassifier(k=20, n_var=p).fit(Z, y)
    py = MethylPyfunc(m, probes)
    py.set_reference(Z[:, m.var_idx_[m.idx_]])
    return py, Z, probes


def test_pyfunc_contract():
    py, Z, probes = _fitted_pyfunc()
    row = pd.DataFrame([dict(zip(py.required, Z[0, [probes.index(c) for c in py.required]], strict=True))]).assign(age=50.0, sex_male=1)
    assert 0 <= py.predict(None, row)["risk"].iloc[0] <= 1
    with pytest.raises(ValueError, match="missing"):
        py.predict(None, row.drop(columns=py.required[:5]))
    beta = row.copy()
    beta[py.required] = 0.5
    with pytest.raises(ValueError, match="beta"):
        py.predict(None, beta)


def test_drift_monitor_flags_shift_not_noise():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, (150, 300))
    mon = DriftMonitor(n_pcs=10, n_perm=150).fit(ref, rng.uniform(0, 1, 150), np.arange(20))
    clean = mon.check(rng.normal(0, 1, (40, 300)), rng.uniform(0, 1, 40))
    shifted = mon.check(rng.normal(0, 1, (40, 300)) * 0.9 + 0.5, rng.uniform(0, 1, 40))
    assert not clean["batch_drift"] and shifted["batch_drift"] and shifted["model_cpgs_shifted_fdr05"] > 10


def test_bh_and_psi():
    p = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    assert bh(p, 0.05).tolist() == [True, True, False, False, False]
    r = np.random.default_rng(0).uniform(0, 1, 1000)
    assert psi(r, r) < 0.01 and psi(r, r ** 3) > 0.2
    assert roc_auc_score([0, 1], [0.1, 0.9]) == 1.0
