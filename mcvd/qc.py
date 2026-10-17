"""Lab-style QC on top of the authors' processing.

sex_check   A recorded-vs-predicted sex mismatch is the classic signature of a sample swap or mislabelled tube. Both
            public matrices had their X/Y probes removed, so sex is predicted out-of-fold from autosomal CpGs (weaker
            than the usual X/Y check; with raw IDATs one would use chrX/chrY signal directly).
probe_filter Drop probes whose target or single-base-extension site overlaps a common SNP (manifest SNP_DISTANCE
            <= 1 with minor-allele frequency >= 1 %): their 'methylation' partly measures genotype. Also non-CpG ('ch')
            probes. Returns a boolean mask over the processed probes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def sex_check(n_probes: int = 200, flag_below: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold sex prediction from AUTOSOMAL methylation (the processed and signal matrices both had X/Y probes
    removed). Within each fold: pick the n most sex-differential probes on the training samples only, fit a regularised
    logistic model, predict the held-out samples. Flag samples whose recorded sex gets out-of-fold probability < flag_below."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from .ingest import load_M
    M, probes, samples = load_M()
    sheet = pd.read_parquet(OUT / "samples.parquet").set_index("sample_id").loc[samples]
    y = (sheet.sex == "male").to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    pool = np.sort(rng.choice(M.shape[0], 150_000, replace=False))     # a random 150k-probe subset keeps this fast
    X = np.asarray(M[pool]).T.astype(np.float64)
    X = np.nan_to_num(X - np.nanmean(X, 0))
    p_male = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        a, b = X[tr][y[tr] == 1], X[tr][y[tr] == 0]
        t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-9)
        top = np.argsort(-np.abs(t))[:n_probes]
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(X[tr][:, top], y[tr])
        p_male[te] = clf.predict_proba(X[te][:, top])[:, 1]
    p_recorded = np.where(y == 1, p_male, 1 - p_male)
    return pd.DataFrame({"sample_id": samples, "p_male_oof": p_male, "p_recorded_sex": p_recorded,
                         "predicted_sex": np.where(p_male >= 0.5, "male", "female"), "sex_flag": p_recorded < flag_below})


def probe_filter(probes: pd.DataFrame, maf: float = 0.01) -> np.ndarray:
    snp = pd.to_numeric(probes.SNP_DISTANCE.astype(str).str.split(";").str[0], errors="coerce")
    f = pd.to_numeric(probes.SNP_MinorAlleleFrequency.astype(str).str.split(";").str[0], errors="coerce")
    snp_hit = (snp <= 1) & (f >= maf)
    non_cpg = ~probes.probe.str.startswith("cg")
    return (~snp_hit & ~non_cpg).to_numpy()


def main():
    sheet = pd.read_parquet(OUT / "samples.parquet")
    sx = sex_check()
    m = sheet.merge(sx, on="sample_id", how="left")
    m["sex_mismatch"] = m.sex_flag
    m.to_parquet(OUT / "samples_qc.parquet", index=False)
    probes = pd.read_parquet(OUT / "probes.parquet")
    keep = probe_filter(probes)
    np.save(OUT / "probe_keep.npy", keep)
    from sklearn.metrics import roc_auc_score
    rep = {"n_samples": int(len(m)), "sex_check_auc_oof": float(roc_auc_score(m.sex == "male", m.p_male_oof)),
           "sex_flags": m.loc[m.sex_mismatch, ["sample_id", "sex", "p_recorded_sex"]].round(3).to_dict(orient="records"),
           "n_probes": int(len(probes)), "n_probes_kept": int(keep.sum()),
           "dropped_snp_or_noncpg": int((~keep).sum())}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "qc.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
