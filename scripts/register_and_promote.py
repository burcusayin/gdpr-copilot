import os, sys, pathlib, mlflow
from mlflow.tracking import MlflowClient

# Ensure repo root on sys.path so we can import our model class
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rag_mlflow.model import RagPyFunc  # noqa: E402

def main():
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking:
        raise SystemExit("Set MLFLOW_TRACKING_URI first, e.g. http://127.0.0.1:5001")
    client = MlflowClient(tracking_uri=tracking)
    model_name = "rag_pipeline"

    with mlflow.start_run(run_name="manual-register"):
        # Log the pyfunc model as an artifact in this run
        info = mlflow.pyfunc.log_model(
            artifact_path="rag_model",
            python_model=RagPyFunc(),
            code_paths=["models", "app", "pipelines"],
            pip_requirements="requirements.txt",
        )

        # Ensure the registered model exists
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        # Create a new model version from this run's artifact
        mv = client.create_model_version(
            name=model_name,
            source=info.model_uri,
            run_id=mlflow.active_run().info.run_id,
        )

        # Prefer aliases in new MLflow; still set stage for compatibility
        try:
            client.set_registered_model_alias(model_name, "prod", mv.version)
        except Exception:
            pass
        try:
            client.transition_model_version_stage(
                name=model_name, version=mv.version,
                stage="Production", archive_existing_versions=True
            )
        except Exception:
            pass

        print(f"[OK] Registered {model_name} v{mv.version}; alias=prod; stage=Production")
        print(f"[HINT] MODEL_URI can be 'models:/{model_name}@prod' or 'models:/{model_name}/{mv.version}'")

if __name__ == "__main__":
    main()
