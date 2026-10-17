"""Production lifecycle on held-out samples: register -> monitor batches -> catch a technical batch effect ->
re-normalise -> collect delayed labels -> retrain challenger -> promote only if it is not worse.

Every step is logged to MLflow; promotions move the registry alias, so serving (which loads @champion) follows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcvd.data import task  # noqa: E402
from mcvd.drift import DriftMonitor  # noqa: E402
from mcvd.model import MethylClassifier  # noqa: E402
from mcvd.registry import MODEL_NAME, TRACKING, register  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"model": "en", "k": 2000, "C": 0.05, "adjust_cov": True}


def inject_batch_effect(Z: np.ndarray, type2: np.ndarray, shift: float = 0.35, scale: float = 0.9) -> np.ndarray:
    """A reagent-lot style artefact: Infinium type-II probes shifted and their dynamic range compressed."""
    Z = Z.copy()
    cols = np.flatnonzero(type2)
    Z[:, cols] = Z[:, cols] * scale + shift
    return Z


def renormalise(Zb: np.ndarray, ref_stats: dict, type2: np.ndarray) -> np.ndarray:
    """Align a held batch to the training reference, separately for Infinium type-I and type-II probes, by matching the
    batch's POOLED median and IQR to the training set's pooled median and IQR (inverts an affine per-type artefact).
    A first version rescaled to the median of per-probe IQRs, which is far narrower than the pooled spread and made
    every model CpG look shifted."""
    Zb = Zb.copy()
    for key, mask in (("II", type2), ("I", ~type2)):
        cols = np.flatnonzero(mask)
        x = Zb[:, cols]
        med, iqr = np.median(x), np.subtract(*np.percentile(x, [75, 25]))
        Zb[:, cols] = (x - med) / iqr * ref_stats[key]["iqr"] + ref_stats[key]["median"]
    return Zb


def paired_boot_diff(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    d = np.array(d)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main(task_name: str = "any_sa_vs_none", seed: int = 0):
    Z, y, probes, sheet = task(task_name)
    X = Z[:, :-2]
    ann = pd.read_parquet(ROOT / "data" / "processed" / "probes.parquet").set_index("probe")
    type2 = (ann.loc[probes, "Infinium_Design_Type"].astype(str) == "II").to_numpy()
    tr, stream = next(StratifiedShuffleSplit(1, test_size=0.4, random_state=seed).split(Z, y))
    rng = np.random.default_rng(seed)
    stream = rng.permutation(stream)
    batches = np.array_split(stream, 4)
    mlflow.set_tracking_uri(TRACKING)
    log = []
    champ = MethylClassifier(**CONFIG).fit(Z[tr], y[tr])
    v1 = register(champ, probes, Z[tr], task_name, CONFIG, {"n_train": len(tr)}, alias="champion")
    log.append({"event": "register", "version": v1, "alias": "champion", "n_train": len(tr)})
    mon_cols = champ.var_idx_[:50000]                                   # monitor 50k high-variance CpGs
    model_pos = np.searchsorted(mon_cols, champ.var_idx_[champ.idx_])
    model_pos = model_pos[(model_pos < len(mon_cols)) & (mon_cols[np.minimum(model_pos, len(mon_cols) - 1)] == champ.var_idx_[champ.idx_])]
    # reference scores must be OUT-OF-FOLD: in-sample predictions are overconfident, which made PSI ~4.5 on a clean batch
    from sklearn.model_selection import cross_val_predict
    oof_scores = cross_val_predict(MethylClassifier(**CONFIG), Z[tr], y[tr], cv=5, method="predict_proba")[:, 1]
    monitor = DriftMonitor().fit(X[tr][:, mon_cols], oof_scores, model_pos)
    ref_stats = {}
    for key, mask in (("II", type2), ("I", ~type2)):
        sub = X[tr][:, np.flatnonzero(mask)[::20]]                      # every 20th probe is plenty for pooled quantiles
        ref_stats[key] = {"median": float(np.median(sub)), "iqr": float(np.subtract(*np.percentile(sub, [75, 25])))}
    feedback_idx: list[int] = []
    pending: list[int] = []
    for b, idx in enumerate(batches, 1):
        Zb = Z[idx]
        if b == 2:
            Zb = np.c_[inject_batch_effect(Zb[:, :-2], type2), Zb[:, -2:]]
        s = champ.predict_proba(Zb)[:, 1]
        rep = monitor.check(Zb[:, :-2][:, mon_cols], s)
        entry = {"event": "batch", "batch": b, "injected_artifact": b == 2, **rep}
        if rep["hold"]:
            entry["action"] = "HOLD: drift detected, results not released; batch re-normalised and re-checked"
            Zb = np.c_[renormalise(Zb[:, :-2], ref_stats, type2), Zb[:, -2:]]
            s = champ.predict_proba(Zb)[:, 1]
            rep2 = monitor.check(Zb[:, :-2][:, mon_cols], s)
            entry["after_renormalisation"] = {k: rep2[k] for k in ("batch_mmd_p", "batch_drift", "hold", "model_cpgs_shifted_fdr05", "score_psi")}
        elif rep["batch_drift"]:
            entry["action"] = "released with a warning: MMD drift alone (no model-CpG shift, few outliers); logged for review"
        entry["auc_on_batch_once_labelled"] = float(roc_auc_score(y[idx], s)) if len(np.unique(y[idx])) == 2 else None
        log.append(entry)
        print(json.dumps(entry), flush=True)
        feedback_idx += pending            # labels for the previous batch arrive now (one-batch delay)
        pending = list(idx)
    # feedback loop: retrain on launch data + labelled feedback (batches 1-3), compare on batch 4 once its labels arrive
    chal = MethylClassifier(**CONFIG).fit(Z[np.r_[tr, feedback_idx]], y[np.r_[tr, feedback_idx]])
    ev = batches[-1]
    a, c = champ.predict_proba(Z[ev])[:, 1], chal.predict_proba(Z[ev])[:, 1]
    diff, lo, hi = paired_boot_diff(y[ev], a, c)
    v2 = register(chal, probes, Z[np.r_[tr, feedback_idx]], task_name, CONFIG,
                  {"n_train": len(tr) + len(feedback_idx), "auc_eval_batch": float(roc_auc_score(y[ev], c))}, alias="challenger")
    promote = lo > -0.02                                        # not credibly worse (and has seen more data)
    if promote:
        mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", v2)
    log.append({"event": "challenger", "version": v2, "n_train": len(tr) + len(feedback_idx), "eval_batch": 4,
                "auc_champion": float(roc_auc_score(y[ev], a)), "auc_challenger": float(roc_auc_score(y[ev], c)),
                "auc_diff": diff, "ci95": [lo, hi], "promoted": bool(promote),
                "rule": "promote if the lower 95% bound of (challenger - champion) AUC > -0.02"})
    print(json.dumps(log[-1]), flush=True)
    (ROOT / "reports" / f"production_{task_name}.json").write_text(json.dumps(log, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
