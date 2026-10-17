"""Analysis matrix: QC-passed probes x samples, labels for each task."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import OUT, load_M

TASKS = {
    "generalized_vs_none": ({"Generalized"}, {"No"}),          # the matched extremes of the design
    "any_sa_vs_none": ({"Focal", "Intermediate", "Generalized"}, {"No"}),
}


@lru_cache(maxsize=1)
def matrix() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    M, probes, samples = load_M()
    keep = np.load(OUT / "probe_keep.npy")
    X = np.ascontiguousarray(np.asarray(M[keep]).T, dtype=np.float32)   # samples x probes
    X = np.where(np.isnan(X), np.nanmean(X, 0), X).astype(np.float32)
    sheet = pd.read_parquet(OUT / "samples_qc.parquet").set_index("sample_id").loc[samples].reset_index()
    return X, [p for p, k in zip(probes, keep, strict=True) if k], sheet


def task(name: str):
    """Returns Z = [CpG M-values | age | sex_male] (float32, built once, no extra copies), y, probe names, sample sheet."""
    X, probes, sheet = matrix()
    pos, neg = TASKS[name]
    m = np.flatnonzero(sheet.sa_extent.isin(pos | neg).to_numpy())
    y = sheet.sa_extent.iloc[m].isin(pos).astype(int).to_numpy()
    Z = np.empty((len(m), X.shape[1] + 2), dtype=np.float32)
    Z[:, :-2] = X[m]
    Z[:, -2] = sheet.age.iloc[m].to_numpy(dtype=np.float32)
    Z[:, -1] = (sheet.sex.iloc[m] == "male").to_numpy(dtype=np.float32)
    return Z, y, probes, sheet.iloc[m].reset_index(drop=True)


ROOT = Path(__file__).resolve().parents[1]
