"""Resolução do servidor de tracking do MLflow (local ou Azure ML).

Regra:
  - MLFLOW_TRACKING_URI definida (ambiente ou .env)  -> usa essa URI
    (ex.: azureml://... do workspace Azure ML; autenticação via DefaultAzureCredential / `az login`)
  - não definida                                      -> MLflow local em ./mlruns (offline)

Utilitário para descobrir a URI do workspace (requer requirements-azure.txt e `az login`):
    python -m src.tracking --azure
"""
from __future__ import annotations

import os
import sys
import time

import mlflow
from dotenv import load_dotenv

from src.config import ROOT

load_dotenv(ROOT / ".env", override=False)

# Local: pasta ./mlruns com backend SQLite (o backend de arquivos puro está em modo de
# manutenção no MLflow 3) e artefatos em ./mlruns/artifacts.
LOCAL_DIR = ROOT / "mlruns"
LOCAL_URI = f"sqlite:///{(LOCAL_DIR / 'mlflow.db').as_posix()}"
LOCAL_ARTIFACTS = (LOCAL_DIR / "artifacts").as_uri()


def resolve_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "").strip() or LOCAL_URI


def is_azure(uri: str | None = None) -> bool:
    return (uri or resolve_tracking_uri()).startswith("azureml://")


def backend_name(uri: str | None = None) -> str:
    return "azureml" if is_azure(uri) else "local"


def setup(experiment: str) -> str:
    """Configura URI + experimento. Falha cedo, com mensagem clara, se o Azure não estiver acessível."""
    uri = resolve_tracking_uri()
    if is_azure(uri):
        try:
            import azureml.mlflow  # noqa: F401  (registra o plugin azureml:// no MLflow)
        except ImportError as exc:
            raise RuntimeError("MLFLOW_TRACKING_URI aponta para o Azure ML, mas os pacotes não estão instalados: "
                               "pip install -r requirements-azure.txt") from exc
    else:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    if not is_azure(uri) and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"{LOCAL_ARTIFACTS}/{experiment}")
    for attempt in range(1, 4):  # o servidor remoto ocasionalmente responde 400/5xx transitórios
        try:
            mlflow.set_experiment(experiment)
            break
        except mlflow.exceptions.MlflowException:
            if attempt == 3 or not is_azure(uri):
                raise
            time.sleep(2 * attempt)
    return uri


def azure_uri_from_workspace() -> str:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    missing = [v for v in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE") if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Defina no .env: {', '.join(missing)}")
    client = MLClient(DefaultAzureCredential(), os.environ["AZURE_SUBSCRIPTION_ID"],
                      os.environ["AZURE_RESOURCE_GROUP"], os.environ["AZURE_ML_WORKSPACE"])
    return client.workspaces.get(os.environ["AZURE_ML_WORKSPACE"]).mlflow_tracking_uri


if __name__ == "__main__":
    if "--azure" in sys.argv:
        print(azure_uri_from_workspace())
    else:
        print(f"{backend_name()}: {resolve_tracking_uri()}")
