"""Scoring service for the registered model (MLflow alias @champion by default).

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db uvicorn serve.app:app --port 8090

POST /score  {"samples": [{"sample_ref": "...", "age": 52, "sex_male": 1, "cpgs": {"cg000...": 1.23, ...}}, ...]}
  -> per sample: risk, missing CpG count, QC errors; plus the model version that produced it.
The audit log stores a SHA-256 of each request body, the model version and timestamps: no methylation data,
no identifiers beyond the caller's own opaque sample_ref.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path(os.environ.get("AUDIT_LOG", ROOT / "reports" / "audit.jsonl"))
app = FastAPI(title="methylation-cvd scorer", version="0.1.0",
              description="Research demo on public data (GEO GSE220622). Not a medical device; not for clinical use.")


class Sample(BaseModel):
    sample_ref: str = Field(max_length=64)
    age: float = Field(ge=18, le=110)
    sex_male: int = Field(ge=0, le=1)
    cpgs: dict[str, float]


class Batch(BaseModel):
    samples: list[Sample] = Field(min_length=1, max_length=500)


@lru_cache(maxsize=1)
def model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    alias = os.environ.get("MODEL_ALIAS", "champion")
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias("pesa-sa-classifier", alias)
    return mlflow.pyfunc.load_model(f"models:/pesa-sa-classifier@{alias}"), f"{mv.version}@{alias}"


@app.get("/health")
def health():
    m, v = model()
    return {"status": "ok", "model_version": v}


@app.post("/score")
async def score(batch: Batch, request: Request):
    m, version = model()
    body = await request.body()
    results = []
    for s in batch.samples:
        row = pd.DataFrame([{**s.cpgs, "age": s.age, "sex_male": float(s.sex_male)}]).astype("float64")
        try:
            out = m.predict(row).iloc[0]
            results.append({"sample_ref": s.sample_ref, "risk": float(out["risk"]), "missing_cpgs": int(out["missing_cpgs"]), "qc_error": None})
        except ValueError as e:                      # model-side input contract (coverage, units)
            results.append({"sample_ref": s.sample_ref, "risk": None, "missing_cpgs": None, "qc_error": str(e)})
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT, "a") as f:
        f.write(json.dumps({"ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "model_version": version,
                            "request_sha256": hashlib.sha256(body).hexdigest(), "n_samples": len(batch.samples),
                            "n_qc_errors": sum(r["qc_error"] is not None for r in results)}) + "\n")
    if all(r["qc_error"] for r in results):
        raise HTTPException(422, {"model_version": version, "results": results})
    return {"model_version": version, "results": results,
            "disclaimer": "Research demo on public data; not for clinical use."}
