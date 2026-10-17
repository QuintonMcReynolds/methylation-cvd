# Deploying on AWS (design; not applied from this repo)

| piece | AWS service | notes |
|---|---|---|
| raw + processed data | S3 (versioned bucket, SSE-KMS, Object Lock for released results) | the ingest step streams from S3 the same way it streams the local gzip; the manifest records object version IDs instead of local hashes |
| large preprocessing | EMR / Glue (Spark) or AWS Batch | ingest is a per-block map (parse, cast, detection-p counts) and parallelises by probe block; at 391 samples one laptop suffices, at 10k+ samples it doesn't |
| experiment tracking + registry | MLflow server on ECS Fargate, backend RDS Postgres, artifact store S3 | `MLFLOW_TRACKING_URI=https://mlflow.internal`; registry aliases (`champion`, `challenger`) are the deployment contract |
| scoring service | ECS Fargate behind an internal ALB (this Dockerfile), or SageMaker endpoint from the MLflow pyfunc | loads `models:/pesa-sa-classifier@champion` at start-up; redeploy = move the alias + restart |
| monitoring job | EventBridge-scheduled ECS task running the drift monitor on each released batch | writes metrics to CloudWatch; alarms on batch drift / PSI / AUC on feedback |
| audit log | CloudWatch Logs -> S3 (immutable) | request hash, model version, timestamps; no methylation values or identifiers |

HIPAA-minded defaults: data never leaves the VPC (VPC endpoints for S3/ECR/CloudWatch), KMS everywhere, IAM roles per
task, BAA-covered services only, no PHI in logs, model versions immutable once released. A CLIA lab would additionally
lock the validated model version, require a documented validation report per version, and treat promotion as a change
control event rather than an automatic alias move.
