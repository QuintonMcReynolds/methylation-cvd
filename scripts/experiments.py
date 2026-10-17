"""Repeated, nested cross-validation for every configuration, logged to MLflow.

Outer: 5-fold x 5 repeats (stratified). Inner (elastic net only): 3-fold grid over k and C. Every step that looks at
labels (residualisation is label-free but still fit on train only; CpG selection; tuning) happens inside the training
fold. 'leaky' deliberately selects CpGs on ALL samples first, to show the optimism that creates.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcvd.data import task  # noqa: E402
from mcvd.features import welch_t  # noqa: E402
from mcvd.model import CovOnly, MethylClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")


def outer_cv(Z, y, make, n_rep=5, seed=0):
    aucs, oof = [], np.zeros((n_rep, len(y)))
    for r, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(Z, y)):
        m = make().fit(Z[tr], y[tr])
        oof[r // 5, te] = m.predict_proba(Z[te])[:, 1]
    for r in range(n_rep):
        aucs.append(roc_auc_score(y, oof[r]))
    return np.array(aucs), oof.mean(0)


def nested_en(**kw):
    # k = 10,000 fits took 10-30 s each (saga elastic net), making the nested grid take hours on a laptop; capped at 3,000
    grid = {"k": [250, 1000, 3000], "C": [0.05, 0.5]}
    return GridSearchCV(MethylClassifier(model="en", **kw), grid, cv=StratifiedKFold(3, shuffle=True, random_state=1),
                        scoring="roc_auc", n_jobs=1)


def permutation_null(Z, y, make, n_perm=50, seed=0):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1 << 30))).split(Z, yp):
            oof[te] = make().fit(Z[tr], yp[tr]).predict_proba(Z[te])[:, 1]
        null.append(roc_auc_score(yp, oof))
    return np.array(null)


def main(task_name: str = "generalized_vs_none", n_perm: int = 50):
    Z, y, probes, sheet = task(task_name)
    X, cov = Z[:, :-2], Z[:, -2:]
    mlflow.set_experiment(f"pesa-{task_name}")
    configs = {
        "age_sex_only": lambda: CovOnly(),
        "elastic_net_nested_adjusted": lambda: nested_en(adjust_cov=True),
        "elastic_net_k1000_unadjusted": lambda: MethylClassifier(k=1000, model="en", C=0.1, adjust_cov=False),
        "random_forest_k1000_adjusted": lambda: MethylClassifier(k=1000, model="rf", adjust_cov=True),
        "lightgbm_k1000_adjusted": lambda: MethylClassifier(k=1000, model="lgb", adjust_cov=True),
    }
    results = {}
    for name, make in configs.items():
        t0 = time.time()
        with mlflow.start_run(run_name=name):
            aucs, oof = outer_cv(Z, y, make, n_rep=3 if name.startswith("elastic_net_nested") else 5)
            mlflow.log_params({"config": name, "task": task_name, "n_samples": len(y), "n_probes": X.shape[1],
                               "cv": f"5-fold x {len(aucs)} repeats, selection inside folds"})
            mlflow.log_metrics({"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "auc_min": float(aucs.min())})
            pd.DataFrame({"sample_id": sheet.sample_id, "y": y, "oof_prob": oof}).to_parquet(ROOT / "reports" / f"oof_{task_name}_{name}.parquet")
            mlflow.log_artifact(str(ROOT / "reports" / f"oof_{task_name}_{name}.parquet"))
            mlflow.set_tag("data_manifest", json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())["source_sha256"]
                           ["GSE220622_processed_M-value.tsv.gz"])
        results[name] = {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std()), "per_repeat": aucs.round(4).tolist(),
                         "seconds": round(time.time() - t0)}
        print(name, json.dumps(results[name]), flush=True)
    # leaky: CpGs chosen on all samples, then ordinary CV
    with mlflow.start_run(run_name="LEAKY_selection_outside_cv"):
        v = np.concatenate([X[:, i:i + 100_000].var(0) for i in range(0, X.shape[1], 100_000)])
        vidx = np.sort(np.argsort(-v)[:100_000])
        idx = vidx[np.argsort(-np.abs(np.nan_to_num(welch_t(X[:, vidx], y))))[:1000]]
        Zl = np.c_[X[:, idx], cov].astype(np.float32)
        aucs, _ = outer_cv(Zl, y, lambda: MethylClassifier(k=1000, model="en", C=0.1, adjust_cov=False, n_var=1000))
        mlflow.log_metrics({"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std())})
        mlflow.set_tag("warning", "feature selection saw the test folds; for illustration only")
    results["LEAKY_selection_outside_cv"] = {"auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std())}
    print("LEAKY", json.dumps(results["LEAKY_selection_outside_cv"]), flush=True)
    # permutation null for the main adjusted elastic net (fixed hyperparameters for speed)
    if n_perm:
        null = permutation_null(Z, y, lambda: MethylClassifier(k=2000, model="en", C=0.05, adjust_cov=True), n_perm)
        obs = results["elastic_net_nested_adjusted"]["auc_mean"]
        results["permutation"] = {"null_mean": float(null.mean()), "null_95pct": float(np.quantile(null, 0.95)),
                                  "p_value": float((1 + (null >= obs).sum()) / (len(null) + 1)), "n_perm": int(len(null))}
        print("perm", json.dumps(results["permutation"]), flush=True)
    (ROOT / "reports" / f"experiments_{task_name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "generalized_vs_none", int(a[1]) if len(a) > 1 else 50)
