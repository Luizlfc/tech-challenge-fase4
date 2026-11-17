"""Logs estruturados (JSON Lines) centralizados para pipeline, drift e alertas.

Cada evento vira uma linha em logs/pipeline.jsonl; alertas também vão para
logs/alerts.jsonl. Nenhum dado pessoal é registrado em log (apenas contagens,
métricas e hashes de lote).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import LOGS_DIR

PIPELINE_LOG = LOGS_DIR / "pipeline.jsonl"
ALERTS_LOG = LOGS_DIR / "alerts.jsonl"

RUN_ID = uuid.uuid4().hex[:12]

_console = logging.getLogger("credit-pipeline")
if not _console.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    _console.addHandler(h)
    _console.setLevel(logging.INFO)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path, record: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def log_event(stage: str, status: str, level: str = "INFO", **fields) -> dict:
    record = {"ts": _now(), "run_id": RUN_ID, "stage": stage, "status": status, "level": level, **fields}
    _append(PIPELINE_LOG, record)
    extra = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (dict, list)))
    getattr(_console, level.lower(), _console.info)(f"[{stage}] {status} {extra}")
    return record


def raise_alert(kind: str, severity: str, message: str, **fields) -> dict:
    record = {"ts": _now(), "run_id": RUN_ID, "kind": kind, "severity": severity, "message": message, **fields}
    _append(ALERTS_LOG, record)
    _console.warning(f"ALERTA [{severity}] {kind}: {message}")
    return record


@contextmanager
def timed_stage(stage: str, **fields):
    """Mede latência de uma etapa e registra sucesso/falha automaticamente."""
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    except Exception as exc:
        log_event(stage, "failed", level="ERROR", latency_s=round(time.perf_counter() - start, 3),
                  error=f"{type(exc).__name__}: {exc}", **fields)
        raise
    else:
        info["latency_s"] = round(time.perf_counter() - start, 3)
        log_event(stage, info.pop("status", "success"), latency_s=info["latency_s"], **fields,
                  **{k: v for k, v in info.items() if k != "latency_s"})


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
