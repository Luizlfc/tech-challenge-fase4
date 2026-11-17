"""Testes da resolução do tracking URI (local offline x Azure ML)."""
from src import tracking


def test_default_is_local_mlruns(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = tracking.resolve_tracking_uri()
    assert uri.startswith("sqlite:///") and uri.endswith("mlruns/mlflow.db")
    assert tracking.backend_name(uri) == "local"


def test_env_var_overrides(monkeypatch):
    azure = ("azureml://brazilsouth.api.azureml.ms/mlflow/v1.0/subscriptions/x/resourceGroups/rg"
             "/providers/Microsoft.MachineLearningServices/workspaces/ws")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", azure)
    assert tracking.resolve_tracking_uri() == azure
    assert tracking.is_azure() and tracking.backend_name() == "azureml"


def test_blank_env_var_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "   ")
    assert tracking.backend_name() == "local"
