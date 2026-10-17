"""Does methylation add anything beyond age and sex?

1. methylation only (CpGs residualised on age/sex inside each fold, no age/sex as model inputs) - signal beyond age
2. paired comparison on identical folds: age+sex vs age+sex+methylation (LightGBM and elastic net), with a paired
   bootstrap over samples on the out-of-fold predictions (5 repeats averaged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcvd.data import task  # noqa: E402
from mcvd.model import CovOnly, MethylClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def oof(Z, y, make, n_rep=5, seed=0):
    P = np.zeros((n_rep, len(y)))
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(Z, y)):
        P[i // 5, te] = make().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return P


def boot_diff(y, a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.asarray(d)
    return {"diff": float(np.mean(d)), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "p_one_sided": float((d <= 0).mean())}


def main(task_name: str):
    Z, y, _, _ = task(task_name)
    res = {"n": int(len(y)), "prevalence": float(y.mean())}
    base = oof(Z, y, lambda: CovOnly())
    res["age_sex_auc"] = float(np.mean([roc_auc_score(y, p) for p in base]))
    for name, kw in (("lightgbm", {"model": "lgb"}), ("elastic_net", {"model": "en", "C": 0.05, "k": 1000})):
        meth_only = oof(Z, y, lambda kw=kw: MethylClassifier(adjust_cov=True, include_cov=False, **kw))
        full = oof(Z, y, lambda kw=kw: MethylClassifier(adjust_cov=True, include_cov=True, **kw))
        res[name] = {"methylation_only_adjusted_auc": float(np.mean([roc_auc_score(y, p) for p in meth_only])),
                     "age_sex_plus_methylation_auc": float(np.mean([roc_auc_score(y, p) for p in full])),
                     "increment_vs_age_sex": boot_diff(y, base.mean(0), full.mean(0))}
        print(task_name, name, json.dumps(res[name]), flush=True)
    (ROOT / "reports" / f"incremental_{task_name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
