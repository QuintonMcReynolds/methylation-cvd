# Scoring service. Expects the MLflow tracking server / registry to be reachable (MLFLOW_TRACKING_URI) - on AWS that is
# an MLflow server with an RDS backend and S3 artifact store (deploy/AWS.md). Locally, mount mlflow.db + mlruns/.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_ALIAS=champion
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mcvd/ mcvd/
COPY serve/ serve/
RUN useradd -r -u 10001 app && mkdir -p /app/reports && chown -R app /app
USER app
EXPOSE 8090
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8090"]
