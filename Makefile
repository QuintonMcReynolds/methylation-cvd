PY ?= .venv/bin/python
data:          ## download GSE220622 processed matrices + EPIC manifest (~3.5 GB)
	./scripts/download.sh
ingest:        ## stream into a float32 memmap + parquet sample/probe tables + manifest
	$(PY) -m mcvd.ingest
qc:            ## autosomal sex check, SNP-overlapping / non-CpG probe filter
	$(PY) -m mcvd.qc
experiments:   ## repeated nested CV for every model, leakage demo, permutation null (MLflow-logged)
	$(PY) -W ignore scripts/experiments.py generalized_vs_none 50 && $(PY) -W ignore scripts/experiments.py any_sa_vs_none 50
production:    ## register -> monitor batches -> drift hold + renormalise -> feedback retrain -> gated promotion
	$(PY) -W ignore scripts/simulate_production.py any_sa_vs_none
ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
serve:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(PY) -m uvicorn serve.app:app --port 8090
check:
	$(PY) -m ruff check mcvd scripts serve tests && $(PY) -m mypy && $(PY) -m pytest -q
.PHONY: data ingest qc experiments production ui serve check
