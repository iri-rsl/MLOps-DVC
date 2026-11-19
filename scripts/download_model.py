import mlflow

mlflow.set_tracking_uri(
    "https://dagshub.com/iri-rsl/MLOps-DVC.mlflow"
)

MODEL_ID = "m-a36443290d8e45e1970c1edf7deb34e0"

model_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{MODEL_ID}",
    dst_path="model"
)

print(f"Model downloaded to: {model_path}")
