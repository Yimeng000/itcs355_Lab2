"""GCP adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install google-cloud-storage google-cloud-aiplatform
Docs: storage.Client for GCS; Artifact Registry push goes through `docker push` after
      `gcloud auth configure-docker <region>-docker.pkg.dev`.

Hints for Lab 1:
  * BLOB_URI looks like gs://bucket/prefix — parse it here, never in src/.
  * Artifact Registry paths are region-scoped:
        <region>-docker.pkg.dev/<project>/<repo>/<image>
    A common first failure is pushing to gcr.io out of habit; it is a different service.
  * push_image must return the digest reference, not the tag.
  * GCP calls them labels, not tags, and they must be lowercase with no spaces.
    cfg.tags(1) already satisfies that constraint — do not "improve" the values.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from google.cloud import storage
from google.cloud import aiplatform

from cloudlayer.base import CloudAdapter
from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job


class GcpAdapter(CloudAdapter):
    def upload(self, local_path: str, key: str) -> str:
        blob_uri = self.cfg.blob_uri
        bucket_name, prefix = blob_uri.removeprefix("gs://").split("/", 1)

        client = storage.Client(project=self.cfg.project_id)
        bucket = client.bucket(bucket_name)

        object_name = f"{prefix.rstrip('/')}/{key.lstrip('/')}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path)

        return f"gs://{bucket_name}/{object_name}"

    def download(self, uri: str, local_path: str) -> None:
        bucket_name, object_name = uri.removeprefix("gs://").split("/", 1)

        client = storage.Client(project=self.cfg.project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry.rstrip("/")
        image_name = local_tag.split(":")[0]
        image_tag = local_tag.rsplit(":", 1)[1]

        remote_tag = f"{registry}/{image_name}:{image_tag}"

        subprocess.run(
            ["docker", "tag", local_tag, remote_tag],
            check=True,
        )

        subprocess.run(
            ["docker", "push", remote_tag],
            check=True,
        )

        result = subprocess.run(
            ["docker", "inspect", "--format={{index .RepoDigests 0}}", remote_tag],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    def submit_training(self, image_uri: str, args: dict) -> str:
        aiplatform.init(
            project=self.cfg.project_id,
            location=self.cfg.region,
        )

        job = aiplatform.CustomContainerTrainingJob(
            display_name="itcs355-lab2-training",
            container_uri=image_uri,
            staging_bucket=f"gs://{self.cfg.project_id}",
        )

        job.run(
            args=[f"--{k.replace('_', '-')}={v}" for k, v in args.items()],
            machine_type="e2-standard-4",
            replica_count=1,
            scheduling_strategy=gca_custom_job.Scheduling.Strategy.SPOT,
            environment_variables={
                "CLOUD_PROVIDER": self.cfg.provider,
                "PROJECT_ID": self.cfg.project_id,
                "REGION": self.cfg.region,
                "BLOB_URI": self.cfg.blob_uri,
                "CONTAINER_REGISTRY": self.cfg.container_registry,
                "MLFLOW_TRACKING_URI": "sqlite:////app/reports/mlflow.db",
                "MODEL_REGISTRY_NAME": self.cfg.model_registry_name,
                "IDENTITY_REF": self.cfg.identity_ref,
            },
            sync=False,
        )

        job.wait_for_resource_creation()

        return job.resource_name
    
    def wait_training(self, job_id: str) -> dict:
        from google.cloud import aiplatform_v1
        from google.api_core.client_options import ClientOptions
        import time

        client = aiplatform_v1.PipelineServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.cfg.region}-aiplatform.googleapis.com"
            )
        )

        while True:
            pipeline = client.get_training_pipeline(name=job_id)
            state = pipeline.state.name

            if state in {
                "PIPELINE_STATE_SUCCEEDED",
                "PIPELINE_STATE_FAILED",
                "PIPELINE_STATE_CANCELLED",
                "PIPELINE_STATE_PAUSED",
            }:
                return {
                    "job_id": job_id,
                    "state": state,
                }

            time.sleep(15)

    def register_model(self, model_uri: str, name: str) -> str:
        aiplatform.init(
            project=self.cfg.project_id,
            location=self.cfg.region,
        )

        model = aiplatform.Model.upload(
            display_name=name,
            artifact_uri=model_uri,
            serving_container_image_uri=(
                "asia-docker.pkg.dev/vertex-ai/"
                "prediction/sklearn-cpu.1-6:latest"
            ),
            sync=True,
        )

        return model.version_id

    # submit_training / register_model  -> Lab 2 (Vertex custom training + Model Registry)
    # deploy / invoke                   -> Lab 3 (Vertex Endpoint)
    # emit_metric                       -> Lab 4 (Cloud Monitoring time series)
    # generate                          -> Lab 5 (managed LLM endpoint; read usageMetadata for tokens)
    # teardown                          -> Lab 5 (filter resources by label)
