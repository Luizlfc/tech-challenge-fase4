"""Etapas 2 e 3 — Monitoramento contínuo de cada lote de produção.

Para cada lote mensal:
  1. Ingestão  -> 2. Contrato de dados (bloqueia/quarentena) -> 3. Scoring
  4. Drift estatístico próprio (PSI + Kolmogorov-Smirnov por feature)
  5. Relatório Evidently (DataDrift PSI + KS + Classification quality)
  6. Desempenho com rótulos defasados (concept drift) + auditoria de viés
  7. Regras de alerta  -> logs/alerts.jsonl
  8. Rastreamento no MLflow (run pai com séries temporais + run filho por lote)

Referência de comparação: holdout do Dataset de Referência (mesma distribuição
do treino, nunca visto pelo modelo -> métricas sem viés otimista).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from xgboost import XGBClassifier

from src import tracking
from src.config import (APPROVAL_SHIFT_ALERT, AUC_DROP_ALERT, CATEGORICAL_FEATURES, DISPARATE_IMPACT_MIN,
                        DRIFT_SHARE_ALERT, FEATURES, ID_COL, KS_PVALUE, MLFLOW_EXPERIMENT_MONITOR, MODEL_META_PATH,
                        MODEL_PATH, NUMERICAL_FEATURES, PREDICTIONS_DIR, PRODUCTION_DIR, PSI_CRITICAL, PSI_WARNING,
                        REFERENCE_PATH, REPORTS_DIR, TARGET)
from src.logging_utils import RUN_ID, log_event, raise_alert, timed_stage
from src.metrics import classification_metrics, fairness_report, ks_test, psi
from src.train import split_reference
from src.validate import validate_batch

EVIDENTLY_DIR = REPORTS_DIR / "evidently"
DRIFT_KINDS = {"DATA_DRIFT", "DATASET_DRIFT", "PREDICTION_DRIFT", "CONCEPT_DRIFT"}
KS_FEATURES = ["limit_bal", "age", "bill_amt1", "pay_amt1"]  # KS explícito no Evidently (variáveis-chave)


def load_model() -> tuple[XGBClassifier, dict]:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))


def score(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["prob_default"] = model.predict_proba(out[FEATURES])[:, 1]
    out["prediction"] = (out["prob_default"] >= threshold).astype(int)
    out["approved"] = 1 - out["prediction"]
    return out


def feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> list[dict]:
    rows = []
    for col in FEATURES + ["prob_default"]:
        p = psi(ref[col], cur[col])
        ks_stat, ks_p = ks_test(ref[col], cur[col])
        if p >= PSI_CRITICAL:
            status = "drift"
        elif p >= PSI_WARNING or (ks_p < KS_PVALUE and ks_stat >= 0.10):
            status = "atenção"
        else:
            status = "estável"
        rows.append({"feature": col, "psi": round(p, 4), "ks_stat": round(ks_stat, 4),
                     "ks_pvalue": float(f"{ks_p:.3g}"), "status": status})
    return sorted(rows, key=lambda r: -r["psi"])


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, batch: str) -> tuple[Path, dict]:
    definition = DataDefinition(
        numerical_columns=NUMERICAL_FEATURES + ["prob_default"],
        categorical_columns=CATEGORICAL_FEATURES + [TARGET, "prediction"],
        classification=[BinaryClassification(target=TARGET, prediction_labels="prediction",
                                             prediction_probas="prob_default", pos_label=1)],
    )
    cols = FEATURES + ["prob_default"]
    report = Report([
        DataDriftPreset(columns=cols, method="psi", threshold=PSI_CRITICAL, drift_share=DRIFT_SHARE_ALERT),
        *[ValueDrift(column=c, method="ks", threshold=KS_PVALUE) for c in KS_FEATURES],
        ClassificationPreset(),
    ], metadata={"batch": batch}, tags=["credit-scoring", "monitoring"])
    snap = report.run(Dataset.from_pandas(cur[cols + [TARGET, "prediction"]], data_definition=definition),
                      Dataset.from_pandas(ref[cols + [TARGET, "prediction"]], data_definition=definition))
    EVIDENTLY_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENTLY_DIR / f"drift_report_{batch}.html"
    snap.save_html(str(path))

    summary = {"psi": {}, "ks_pvalue": {}}
    for m in snap.dict()["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            summary[cfg["method"] if cfg["method"] == "psi" else "ks_pvalue"][cfg["column"]] = float(m["value"])
        elif cfg.get("type", "").endswith("DriftedColumnsCount"):
            summary["drifted_columns"] = m["value"]
        elif cfg.get("type", "").endswith("RocAuc"):
            summary["roc_auc"] = float(m["value"])
    return path, summary


def evaluate_alerts(batch: str, rec: dict, meta: dict) -> list[dict]:
    alerts = []
    base = meta["baseline_metrics"]
    drifted = [f for f in rec["feature_drift"] if f["status"] == "drift" and f["feature"] != "prob_default"]
    if drifted:
        alerts.append(raise_alert("DATA_DRIFT", "CRITICAL",
                                  f"{batch}: {len(drifted)} feature(s) com PSI >= {PSI_CRITICAL}: "
                                  + ", ".join(f"{f['feature']} ({f['psi']:.2f})" for f in drifted[:6]),
                                  batch=batch, features=[f["feature"] for f in drifted]))
    if rec["drift_share"] >= DRIFT_SHARE_ALERT:
        alerts.append(raise_alert("DATASET_DRIFT", "CRITICAL",
                                  f"{batch}: {rec['drift_share']:.0%} das features com drift (limite {DRIFT_SHARE_ALERT:.0%})",
                                  batch=batch))
    pred = next(f for f in rec["feature_drift"] if f["feature"] == "prob_default")
    if pred["psi"] >= PSI_WARNING:
        alerts.append(raise_alert("PREDICTION_DRIFT", "WARNING",
                                  f"{batch}: distribuição do score mudou (PSI {pred['psi']:.2f})", batch=batch))
    drop = base["auc"] - rec["performance"]["auc"]
    if drop >= AUC_DROP_ALERT:
        alerts.append(raise_alert("CONCEPT_DRIFT", "CRITICAL",
                                  f"{batch}: AUC caiu {drop:.3f} (de {base['auc']:.3f} para {rec['performance']['auc']:.3f})",
                                  batch=batch, auc=rec["performance"]["auc"]))
    gap = rec["performance"]["default_rate_observed"] - base["default_rate_observed"]
    if gap >= 0.05:
        alerts.append(raise_alert("DEFAULT_RATE_SHIFT", "WARNING",
                                  f"{batch}: inadimplência observada {rec['performance']['default_rate_observed']:.1%} "
                                  f"vs {base['default_rate_observed']:.1%} no treino", batch=batch))
    if abs(rec["performance"]["approval_rate"] - base["approval_rate"]) >= APPROVAL_SHIFT_ALERT:
        alerts.append(raise_alert("APPROVAL_RATE_SHIFT", "WARNING",
                                  f"{batch}: taxa de aprovação {rec['performance']['approval_rate']:.1%} "
                                  f"vs {base['approval_rate']:.1%} no baseline", batch=batch))
    for group, fr in rec["fairness"].items():
        if fr["disparate_impact"] is not None and fr["disparate_impact"] < DISPARATE_IMPACT_MIN:
            alerts.append(raise_alert("FAIRNESS", "WARNING",
                                      f"{batch}: impacto desproporcional por {group} = {fr['disparate_impact']:.2f} (< 0,80)",
                                      batch=batch, group=group))
    return alerts


def process_batch(path: Path, model, meta: dict, ref: pd.DataFrame, step: int) -> dict:
    batch = path.stem
    rec: dict = {"batch": batch, "step": step, "stages": {}}
    t0 = time.perf_counter()

    with timed_stage("ingestion", batch=batch) as info:
        df = pd.read_csv(path)
        info["rows"] = len(df)
    rec["stages"]["ingestion"] = info["latency_s"]
    rec["rows_received"] = len(df)

    with timed_stage("validation", batch=batch) as info:
        res = validate_batch(df, batch)
        info["status"] = "success" if res.passed else "blocked"
    rec["stages"]["validation"] = info["latency_s"]
    rec["validation_passed"] = res.passed
    rec["violations_by_rule"] = res.cases_by_rule()
    if not res.passed:
        rec.update(status="BLOQUEADO", rows_scored=0, violations=res.failures, alerts=1,
                   alert_kinds=["DATA_CONTRACT_VIOLATION"], alerts_by_kind={"DATA_CONTRACT_VIOLATION": 1},
                   total_latency_s=round(time.perf_counter() - t0, 3))
        return rec

    with timed_stage("scoring", batch=batch) as info:
        cur = score(model, res.data, meta["threshold"])
        # Predições persistidas localmente (pseudonimizadas; não vão para o Git nem para a nuvem)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        cur[[ID_COL, "prob_default", "prediction", "approved"]].assign(
            model_version=meta["model_version"], batch=batch).to_csv(PREDICTIONS_DIR / f"{batch}.csv", index=False)
        info["rows"] = len(cur)
    rec["stages"]["scoring"] = info["latency_s"]
    rec["rows_scored"] = len(cur)
    rec["predictions_summary"] = {
        "approval_rate": round(float(cur["approved"].mean()), 4),
        "score_mean": round(float(cur["prob_default"].mean()), 4),
        "score_quantiles": {f"p{int(q * 100)}": round(float(cur["prob_default"].quantile(q)), 4)
                            for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
    }

    with timed_stage("drift_detection", batch=batch) as info:
        rec["feature_drift"] = feature_drift_table(ref, cur)
        feats = [f for f in rec["feature_drift"] if f["feature"] != "prob_default"]
        rec["n_drifted"] = sum(f["status"] == "drift" for f in feats)
        rec["n_warning"] = sum(f["status"] == "atenção" for f in feats)
        rec["drift_share"] = round(rec["n_drifted"] / len(feats), 4)
        rec["max_psi"] = max(f["psi"] for f in feats)
        info.update(n_drifted=rec["n_drifted"], max_psi=rec["max_psi"])
    rec["stages"]["drift_detection"] = info["latency_s"]

    with timed_stage("evidently_report", batch=batch) as info:
        html, ev = evidently_report(ref, cur, batch)
        rec["evidently_report"] = html.relative_to(REPORTS_DIR).as_posix()
        rec["evidently"] = ev
    rec["stages"]["evidently_report"] = info["latency_s"]

    with timed_stage("performance_monitoring", batch=batch) as info:
        rec["performance"] = classification_metrics(cur[TARGET], cur["prob_default"], meta["threshold"])
        rec["fairness"] = fairness_report(cur)
        info["auc"] = round(rec["performance"]["auc"], 4)
    rec["stages"]["performance_monitoring"] = info["latency_s"]

    alerts = evaluate_alerts(batch, rec, meta)
    rec["alerts"] = len(alerts)
    rec["alert_kinds"] = sorted({a["kind"] for a in alerts})
    rec["alerts_by_kind"] = {k: sum(a["kind"] == k for a in alerts) for k in rec["alert_kinds"]}
    rec["alerts_critical"] = sum(a["severity"] == "CRITICAL" for a in alerts)
    rec["status"] = "ALERTA" if any(a["severity"] == "CRITICAL" for a in alerts) else (
        "ATENÇÃO" if alerts else "SAUDÁVEL")
    rec["total_latency_s"] = round(time.perf_counter() - t0, 3)
    rec["throughput_rows_s"] = round(len(cur) / max(rec["stages"]["scoring"], 1e-3), 1)
    return rec


def batch_metrics(r: dict) -> dict:
    """Todas as métricas de um lote, prontas para o MLflow (nomes estáveis para comparar execuções)."""
    ok = r["status"] != "BLOQUEADO"
    kinds = r.get("alerts_by_kind", {})
    m = {
        # pipeline: status, volume, latência
        "pipeline_success": int(ok), "rows_received": r["rows_received"], "rows_scored": r["rows_scored"],
        "latency_total_s": r["total_latency_s"],
        **{f"latency_{stage}_s": v for stage, v in r["stages"].items()},
        # contrato de dados: passou/falhou + casos por regra
        "validation_passed": int(r["validation_passed"]),
        "contract_violations_total": sum(r["violations_by_rule"].values()),
        **{f"contract_{rule}_cases": n for rule, n in r["violations_by_rule"].items()},
        # alertas
        "alerts_total": r["alerts"],
        "alerts_contract": kinds.get("DATA_CONTRACT_VIOLATION", 0),
        "alerts_drift": sum(v for k, v in kinds.items() if k in DRIFT_KINDS),
        "alerts_other": sum(v for k, v in kinds.items() if k not in DRIFT_KINDS | {"DATA_CONTRACT_VIOLATION"}),
    }
    if ok:
        p = r["performance"]
        m.update({
            "auc": p["auc"], "f1": p["f1"], "precision": p["precision"], "recall": p["recall"],
            "ks_score": p["ks_score"], "approval_rate": p["approval_rate"],
            "default_rate_observed": p["default_rate_observed"],
            "drift_share": r["drift_share"], "n_drifted_features": r["n_drifted"],
            "n_warning_features": r["n_warning"], "max_psi": r["max_psi"],
            "throughput_rows_s": r["throughput_rows_s"], "alerts_critical": r["alerts_critical"],
            "evidently_drifted_columns": float(r["evidently"].get("drifted_columns", {}).get("count", 0)),
        })
        for f in r["feature_drift"]:
            name = f["feature"]
            m.update({f"psi_{name}": f["psi"], f"ks_stat_{name}": f["ks_stat"],
                      f"ks_pvalue_{name}": f["ks_pvalue"], f"drift_flag_{name}": int(f["status"] == "drift")})
    return m


def log_to_mlflow(records: list[dict], meta: dict) -> str:
    uri = tracking.setup(MLFLOW_EXPERIMENT_MONITOR)
    with mlflow.start_run(run_name="monitoramento-producao") as parent:
        mlflow.set_tags({"model_version": meta["model_version"], "model_run_id": meta["mlflow_run_id"],
                         "tracking_backend": tracking.backend_name(uri), "pipeline_run_id": RUN_ID})
        mlflow.log_params({"threshold": meta["threshold"], "psi_warning": PSI_WARNING, "psi_critical": PSI_CRITICAL,
                           "ks_pvalue": KS_PVALUE, "auc_drop_alert": AUC_DROP_ALERT,
                           "drift_share_alert": DRIFT_SHARE_ALERT, "n_batches": len(records)})
        for r in records:
            m = batch_metrics(r)
            # séries temporais no run pai: um step por lote (gráficos de evolução no MLflow/Azure ML)
            mlflow.log_metrics(m, step=r["step"])
            with mlflow.start_run(run_name=f"lote-{r['batch']}", nested=True):
                mlflow.set_tags({"batch": r["batch"], "status": r["status"], "pipeline_run_id": RUN_ID,
                                 "alert_kinds": ",".join(r.get("alert_kinds", [])) or "nenhum"})
                mlflow.log_metrics(m)
                if r["status"] != "BLOQUEADO":
                    mlflow.log_artifact(str(REPORTS_DIR / r["evidently_report"]), "evidently")
                    mlflow.log_dict(r["feature_drift"], "feature_drift.json")
                    mlflow.log_dict(r["predictions_summary"], "predictions_summary.json")
                else:
                    mlflow.log_dict(r["violations"], "contract_violations.json")
        # agregados da execução inteira
        mlflow.log_metrics({
            "run_batches_total": len(records),
            "run_batches_blocked": sum(r["status"] == "BLOQUEADO" for r in records),
            "run_batches_alerting": sum(r["status"] == "ALERTA" for r in records),
            "run_rows_scored": sum(r["rows_scored"] for r in records),
            "run_alerts_total": sum(r["alerts"] for r in records),
            "run_alerts_drift": sum(batch_metrics(r)["alerts_drift"] for r in records),
            "run_alerts_contract": sum(batch_metrics(r)["alerts_contract"] for r in records),
        })
        return parent.info.run_id


def log_final_artifacts(run_id: str) -> None:
    """Anexa dashboard, resumo, análise causal e logs ao run pai (chamado ao fim do pipeline).

    Os HTMLs do Evidently (~5 MB cada) já estão nos runs filhos de cada lote; não são
    reenviados aqui para não duplicar armazenamento (custo no Azure ML)."""
    tracking.setup(MLFLOW_EXPERIMENT_MONITOR)
    with mlflow.start_run(run_id=run_id):
        for name in ("dashboard.html", "monitoring_summary.json", "causal_analysis.json", "metrics.prom"):
            path = REPORTS_DIR / name
            if path.exists():
                mlflow.log_artifact(str(path), "reports")
        for path in sorted((REPORTS_DIR / "logs").glob("*.jsonl")):
            mlflow.log_artifact(str(path), "logs")


def main() -> tuple[list[dict], str]:
    model, meta = load_model()
    _, _, holdout = split_reference(pd.read_csv(REFERENCE_PATH))
    ref = score(model, holdout, meta["threshold"])
    log_event("monitoring", "started", reference_rows=len(ref))

    batches = sorted(PRODUCTION_DIR.glob("*.csv"))
    records = [process_batch(p, model, meta, ref, step) for step, p in enumerate(batches)]

    with timed_stage("mlflow_tracking", backend=tracking.backend_name()) as info:
        run_id = log_to_mlflow(records, meta)
        info.update(run_id=run_id, batches=len(records))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "monitoring_summary.json").write_text(
        json.dumps({"model": {k: meta[k] for k in ("model_version", "threshold", "baseline_metrics", "baseline_fairness")},
                    "batches": records}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return records, run_id


if __name__ == "__main__":
    main()
