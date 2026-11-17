import pytest

from src import config, logging_utils


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    """Testes não escrevem em logs/ nem em data/quarantine/ do projeto."""
    monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logging_utils, "PIPELINE_LOG", tmp_path / "pipeline.jsonl")
    monkeypatch.setattr(logging_utils, "ALERTS_LOG", tmp_path / "alerts.jsonl")
    import src.validate as v
    monkeypatch.setattr(v, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path / "quarantine")
