"""Comando único do Tech Challenge — Fase 4.

    python main.py                # pipeline completo
    python main.py --skip-train   # reaproveita o modelo já treinado

Tracking do MLflow: MLFLOW_TRACKING_URI (ambiente ou .env) -> ex.: Azure ML;
sem a variável -> ./mlruns local (100% offline).

Etapas:
  1. data_prep        download, pseudonimização, separação referência x produção
  2. train            contrato no dataset de referência + XGBoost baseline + MLflow
  3. simulate_drift   6 lotes mensais com data/concept drift + 1 lote corrompido
  4. monitor          contrato, scoring, PSI/KS, Evidently, alertas, MLflow
  5. causal_analysis  intervenções contrafactuais + prescrição (retreino)
  6. dashboard        painel HTML + métricas Prometheus + cópia dos logs
  7. artefatos finais anexados ao run de monitoramento no MLflow
"""
from __future__ import annotations

import argparse
import os
import shutil
import warnings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

from src import causal_analysis, dashboard, data_prep, monitor, simulate_drift, tracking, train  # noqa: E402
from src.config import LOGS_DIR, PREDICTIONS_DIR, PRODUCTION_DIR, QUARANTINE_DIR, REPORTS_DIR  # noqa: E402
from src.logging_utils import RUN_ID, log_event, timed_stage  # noqa: E402


def reset_outputs() -> None:
    for d in (LOGS_DIR, PRODUCTION_DIR, QUARANTINE_DIR, PREDICTIONS_DIR,
              REPORTS_DIR / "evidently", REPORTS_DIR / "logs"):
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de monitoramento do modelo de crédito")
    parser.add_argument("--skip-train", action="store_true", help="reaproveita models/model.json")
    args = parser.parse_args()

    reset_outputs()
    uri = tracking.resolve_tracking_uri()
    log_event("pipeline", "started", run_id=RUN_ID, tracking_backend=tracking.backend_name(uri))
    with timed_stage("pipeline_total"):
        data_prep.main()
        if not args.skip_train:
            train.main()
        simulate_drift.main()
        records, mlflow_run_id = monitor.main()
        with timed_stage("causal_analysis"):
            causal_analysis.main()
        with timed_stage("dashboard"):
            dashboard.main()
        with timed_stage("mlflow_artifacts"):
            monitor.log_final_artifacts(mlflow_run_id)

    blocked = [r["batch"] for r in records if r["status"] == "BLOQUEADO"]
    alerting = [r["batch"] for r in records if r["status"] == "ALERTA"]
    print("\n" + "=" * 72)
    print(f"Lotes processados: {len(records) - len(blocked)} | bloqueados: {blocked}")
    print(f"Lotes em ALERTA:   {alerting}")
    print(f"Alertas:           {sum(r['alerts'] for r in records)}")
    print(f"Dashboard:         {REPORTS_DIR / 'dashboard.html'}")
    print(f"Evidently:         {REPORTS_DIR / 'evidently'}")
    print(f"MLflow ({tracking.backend_name(uri)}): {uri}")
    print(f"  run de monitoramento: {mlflow_run_id}")
    if not tracking.is_azure(uri):
        print("  visualizar: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db  ->  http://127.0.0.1:5000")
    print("=" * 72)


if __name__ == "__main__":
    main()
