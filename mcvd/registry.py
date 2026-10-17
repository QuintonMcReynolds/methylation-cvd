"""MLflow model registry: a self-validating pyfunc around MethylClassifier, registered with lineage and aliases.

The pyfunc accepts a DataFrame whose columns are CpG IDs (M-values) plus 'age' and 'sex_male'. It only needs the CpGs
the fitted model uses; it rejects requests missing too many of them, and flags beta-valued input (0..1) sent where
M-values are expected - the most common unit mistake with methylation data.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "pesa-sa-classifier"
TRACKING = f"sqlite:///{ROOT / 'mlflow.db'}"


class MethylPyfunc(PythonModel):  # type: ignore[misc]
    """Wraps a fitted MethylClassifier trained on the full probe set; serves from only the CpGs it actually uses."""

    def __init__(self, model, probes: list[str], max_missing: float = 0.05):
        self.max_missing = max_missing
        used = model.var_idx_[model.idx_]                   # positions in the full probe list actually used
        self.required = [probes[i] for i in used]
        # rebuild a compact model whose input is exactly `required` (+ cov): same coefficients, no 800k-wide matrix
        self.beta = model.beta_[:, model.idx_] if model.adjust_cov else None
        self.scaler, self.clf, self.include_cov = model.scaler_, model.clf_, model.include_cov
        self.train_median = None

    def set_reference(self, Xreq: np.ndarray):
        self.train_median = np.nanmedian(Xreq, 0)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        df = model_input
        missing = [c for c in self.required if c not in df.columns]
        if len(missing) / len(self.required) > self.max_missing:
            raise ValueError(f"{len(missing)} of {len(self.required)} required CpGs missing (limit {self.max_missing:.0%})")
        X = df.reindex(columns=self.required).to_numpy(np.float32)
        if np.nanmax(np.abs(X)) <= 1.0 and np.nanmin(X) >= 0.0:
            raise ValueError("input looks like beta values (all within 0..1); this model expects M-values = log2(beta/(1-beta))")
        if self.train_median is not None:
            X = np.where(np.isnan(X), self.train_median, X)          # impute the few allowed missing CpGs
        cov = df[["age", "sex_male"]].to_numpy(np.float32)
        if self.beta is not None:
            X = X - np.c_[np.ones(len(cov), dtype=np.float32), cov] @ self.beta
        F = np.c_[X, cov] if self.include_cov else X
        p = self.clf.predict_proba(self.scaler.transform(F))[:, 1]
        return pd.DataFrame({"risk": p, "missing_cpgs": len(missing)})


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def register(model, probes: list[str], Z_train: np.ndarray, task: str, config: dict, metrics: dict, alias: str) -> str:
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(f"pesa-{task}-registry")
    py = MethylPyfunc(model, probes)
    used = model.var_idx_[model.idx_]
    py.set_reference(Z_train[:, used])
    example = pd.DataFrame(Z_train[:2, used], columns=py.required).assign(age=Z_train[:2, -2], sex_male=Z_train[:2, -1])
    manifest = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())
    with mlflow.start_run(run_name=f"register-{alias}") as run:
        mlflow.log_params({**config, "task": task, "n_train": len(Z_train), "n_required_cpgs": len(py.required)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"git_sha": git_sha() or "n/a", "data_sha256": manifest["source_sha256"]["GSE220622_processed_M-value.tsv.gz"],
                         "intended_use": "research demo on public data; not for clinical use"})
        # Explicit signature: CpGs are OPTIONAL doubles (the model's own contract allows <= 5 % missing and imputes them;
        # an inferred signature made every CpG required and float32, so MLflow rejected JSON float64 input and any
        # sample with a single missing probe before the model's own checks could run). age and sex are required.
        from mlflow.models import ModelSignature
        from mlflow.types.schema import ColSpec, Schema, TensorSpec
        cols_in: list[ColSpec | TensorSpec] = [ColSpec("double", c, required=False) for c in py.required]
        cols_in += [ColSpec("double", "age"), ColSpec("double", "sex_male")]
        cols_out: list[ColSpec | TensorSpec] = [ColSpec("double", "risk"), ColSpec("long", "missing_cpgs")]
        sig = ModelSignature(inputs=Schema(cols_in), outputs=Schema(cols_out))
        info = mlflow.pyfunc.log_model(name="model", python_model=py, signature=sig, input_example=example.astype("float64"),
                                       registered_model_name=MODEL_NAME, pip_requirements=["scikit-learn", "numpy", "pandas", "lightgbm"])
    client = mlflow.MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    client.set_model_version_tag(MODEL_NAME, version, "run_id", run.info.run_id)
    return str(version)


def load(alias: str = "champion"):
    mlflow.set_tracking_uri(TRACKING)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
