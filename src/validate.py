"""Validação de lotes contra o contrato de dados, com bloqueio e quarentena.

Uso:
    python -m src.validate data/production/batch_corrupted.csv
Retorna código de saída 1 quando o lote é bloqueado (útil em CI/orquestradores).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.config import QUARANTINE_DIR
from src.contracts import RULES, CreditContract, rule_id
from src.logging_utils import log_event, raise_alert


@dataclass
class ValidationResult:
    batch: str
    passed: bool
    rows: int
    data: pd.DataFrame | None = None
    failures: list[dict] = field(default_factory=list)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    def cases_by_rule(self) -> dict[str, int]:
        """Nº de casos violados por regra (R1..R9), com zero para as regras respeitadas."""
        out = {r: 0 for r in RULES}
        for f in self.failures:
            out[f["rule"]] += f["n_cases"]
        return out


def _summarize(err: pa.errors.SchemaErrors) -> list[dict]:
    fc = err.failure_cases.copy()
    fc["failure_case"] = fc["failure_case"].astype(str)
    out = []
    for (col, check), grp in fc.groupby(["column", "check"], dropna=False):
        column = None if pd.isna(col) else str(col)
        out.append({
            "rule": rule_id(column, str(check)),
            "column": column,
            "check": str(check),
            "n_cases": int(len(grp)),
            "examples": grp["failure_case"].head(3).tolist(),
        })
    return sorted(out, key=lambda x: -x["n_cases"])


def validate_batch(df: pd.DataFrame, batch_name: str) -> ValidationResult:
    try:
        clean = CreditContract.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        failures = _summarize(err)
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(QUARANTINE_DIR / f"{batch_name}.csv", index=False)
        report = {"batch": batch_name, "rows": len(df), "contract": "CreditContract v1.0",
                  "status": "BLOCKED", "failures": failures}
        (QUARANTINE_DIR / f"{batch_name}_violations.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log_event("data_contract", "blocked", level="ERROR", batch=batch_name, rows=len(df),
                  n_violations=len(failures), violations=failures)
        raise_alert("DATA_CONTRACT_VIOLATION", "CRITICAL",
                    f"Lote '{batch_name}' bloqueado: {len(failures)} regra(s) violada(s)",
                    batch=batch_name, checks=[f"{f['column']}:{f['check']}" for f in failures])
        return ValidationResult(batch_name, False, len(df), None, failures)

    log_event("data_contract", "passed", batch=batch_name, rows=len(clean))
    return ValidationResult(batch_name, True, len(clean), clean)


def main(path: str) -> int:
    p = Path(path)
    res = validate_batch(pd.read_csv(p), p.stem)
    if res.passed:
        print(f"OK: lote '{p.stem}' aprovado no contrato ({res.rows} linhas).")
        return 0
    print(f"BLOQUEADO: lote '{p.stem}' violou o contrato:")
    for f in res.failures:
        print(f"  - {f['rule']} | {f['column'] or '<dataframe>'} | {f['check']} | {f['n_cases']} caso(s) | ex.: {f['examples']}")
    print(f"Lote movido para a quarentena: {QUARANTINE_DIR / (p.stem + '.csv')}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
