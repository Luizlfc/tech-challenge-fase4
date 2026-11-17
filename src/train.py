"""Etapa 1 — Treino do modelo baseline (XGBoost) no Dataset de Referência.

- O dataset de referência passa pelo contrato ANTES do treino.
- Atributos protegidos (sexo, estado civil) NÃO entram como features.
- Divisão: 70% treino | 15% validação (escolha do limiar) | 15% holdout (métricas).
- Rastreamento completo no MLflow (parâmetros, métricas, modelo registrado).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import tracking
from src.config import (FEATURES, MLFLOW_EXPERIMENT_TRAIN, MODEL_META_PATH, MODEL_PATH, MODELS_DIR,
                        REFERENCE_PATH, REPORTS_DIR, SEED, TARGET)
from src.logging_utils import RUN_ID, log_event, timed_stage
from src.metrics import classification_metrics, fairness_report
from src.validate import validate_batch

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
              eval_metric="auc", random_state=SEED, n_jobs=-1)
MODEL_VERSION = "1.0.0"


def best_threshold(y, proba) -> float:
    grid = np.arange(0.10, 0.71, 0.01)
    scores = [f1_score(y, proba >= t) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def split_reference(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70% treino | 15% validação | 15% holdout (determinístico)."""
    train, rest = train_test_split(ref, test_size=0.30, stratify=ref[TARGET], random_state=SEED)
    val, holdout = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=SEED)
    return train, val, holdout.reset_index(drop=True)


def main() -> dict:
    ref = pd.read_csv(REFERENCE_PATH)
    res = validate_batch(ref, "reference")
    if not res.passed:
        raise RuntimeError("Dataset de referência violou o contrato — treino abortado.")
    ref = res.data

    train, val, holdout = split_reference(ref)

    uri = tracking.setup(MLFLOW_EXPERIMENT_TRAIN)
    with mlflow.start_run(run_name=f"baseline-xgb-v{MODEL_VERSION}") as run, timed_stage("training") as info:
        model = XGBClassifier(**PARAMS)
        model.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])], verbose=False)

        threshold = best_threshold(val[TARGET], model.predict_proba(val[FEATURES])[:, 1])
        proba = model.predict_proba(holdout[FEATURES])[:, 1]
        metrics = classification_metrics(holdout[TARGET], proba, threshold)
        hold = holdout.assign(approved=(proba < threshold).astype(int))
        fairness = fairness_report(hold)

        mlflow.log_params({**PARAMS, "threshold": threshold, "n_features": len(FEATURES),
                           "train_rows": len(train), "model_version": MODEL_VERSION})
        mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("holdout_di_sex", fairness["sex"]["disparate_impact"])
        mlflow.log_metric("holdout_di_age", fairness["age_band"]["disparate_impact"])
        # Resultado do contrato no dataset de referência (pré-condição do treino)
        mlflow.log_metrics({"reference_validation_passed": int(res.passed), "reference_rows": res.rows,
                            **{f"reference_contract_{r}_cases": n for r, n in res.cases_by_rule().items()}})
        mlflow.set_tags({"stage": "baseline", "dataset": "UCI default of credit card clients",
                         "protected_attrs_excluded": "sex,marriage", "tracking_backend": tracking.backend_name(uri),
                         "pipeline_run_id": RUN_ID})
        try:
            mlflow.xgboost.log_model(model, name="model", input_example=holdout[FEATURES].head(3),
                                     registered_model_name="credit-scoring-xgb")
        except Exception as exc:
            # O Azure ML ainda não implementa a API "logged-models" do MLflow 3. Fallback compatível:
            # salva no formato MLflow, envia a pasta como artefato e registra via runs:/<id>/model.
            log_event("training", "model_logging_fallback", level="WARNING", error=f"{type(exc).__name__}: {exc}")
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "model"
                mlflow.xgboost.save_model(model, str(local), input_example=holdout[FEATURES].head(3))
                mlflow.log_artifacts(str(local), "model")
            # API clássica do Model Registry (register_model do MLflow 3 também chama logged-models/search)
            client = mlflow.MlflowClient()
            try:
                client.get_registered_model("credit-scoring-xgb")
            except mlflow.exceptions.MlflowException:
                client.create_registered_model("credit-scoring-xgb")
            mv = client.create_model_version("credit-scoring-xgb", f"{run.info.artifact_uri}/model", run.info.run_id)
            log_event("training", "model_registered_fallback", version=mv.version)
        info.update(rows=len(train), auc=round(metrics["auc"], 4), threshold=threshold)

        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        meta = {
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": run.info.run_id,
            "features": FEATURES,
            "threshold": threshold,
            "decision_rule": "aprovar se P(default) < threshold",
            "baseline_metrics": metrics,
            "baseline_fairness": fairness,
            "feature_importance": importances.round(4).to_dict(),
            "reference_data_sha256": hashlib.sha256(pd.util.hash_pandas_object(ref).values.tobytes()).hexdigest(),
        }
        MODEL_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "baseline_model_card.json").write_text(
            json.dumps({k: v for k, v in meta.items() if k != "mlflow_run_id"}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        mlflow.log_artifact(str(MODEL_META_PATH))

    log_event("training", "model_registered", model_version=MODEL_VERSION, threshold=threshold,
              auc=round(metrics["auc"], 4), ks=round(metrics["ks_score"], 4))
    return meta


if __name__ == "__main__":
    m = main()
    print(json.dumps(m["baseline_metrics"], indent=2))
