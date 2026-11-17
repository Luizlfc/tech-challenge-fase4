"""Etapa 2 — Simulação da passagem do tempo em produção (Data Drift + Concept Drift).

Cenário econômico (6 meses, intensidade crescente s = 0.0 → 1.0):

DATA DRIFT (mudança em P(X)):
  D1 Inflação + juros do rotativo: faturas (bill_amt1..6) sobem até +80%.
  D2 Aperto de renda: pagamentos (pay_amt1..6) caem até -40%.
  D3 Novo perfil de cliente: até 40% do lote passa a ser de jovens (21–27 anos)
     com limite de crédito baixo (R$ 10k–50k) — expansão para público digital.

CONCEPT DRIFT (mudança em P(y|X)):
  C1 Clientes com utilização alta do limite (fatura/limite > 0,5) passam a
     inadimplir mais, mesmo com histórico de pagamento "bom" (até +30 p.p.).
  C2 O novo perfil jovem tem inadimplência estruturalmente maior (até +25 p.p.).
  => o modelo continua "vendo" bons pagadores pelo histórico (pay_0..pay_6),
     mas a relação histórico → inadimplência mudou. É a degradação silenciosa.

O mês 1 (s=0) não tem drift e serve de controle. Um lote extra corrompido
("sistema legado") é gerado para demonstrar o contrato bloqueando a ingestão.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BILL_AMT, N_MONTHS, PAY_AMT, PRODUCTION_DIR, PRODUCTION_POOL_PATH, SEED, TARGET
from src.logging_utils import timed_stage

MONTHS = [f"2026-{m:02d}" for m in range(5, 5 + N_MONTHS)]   # mai/2026 → out/2026
CORRUPTED_BATCH = "2026-08_sistema_legado"

INFLATION_MAX = 0.80
INCOME_SQUEEZE_MAX = 0.40
NEW_PROFILE_MAX = 0.40
CONCEPT_UTIL_MAX = 0.30
CONCEPT_YOUNG_MAX = 0.25


def intensity(month_idx: int) -> float:
    return month_idx / (N_MONTHS - 1)


def apply_data_drift(df: pd.DataFrame, s: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df[BILL_AMT] = (df[BILL_AMT] * (1 + INFLATION_MAX * s)).round(0)
    df[PAY_AMT] = (df[PAY_AMT] * (1 - INCOME_SQUEEZE_MAX * s)).round(0)

    new_profile = rng.random(len(df)) < NEW_PROFILE_MAX * s
    df.loc[new_profile, "age"] = rng.integers(21, 28, new_profile.sum())
    df.loc[new_profile, "limit_bal"] = rng.choice([10_000.0, 20_000.0, 30_000.0, 50_000.0], new_profile.sum())
    df["_new_profile"] = new_profile
    return df


def apply_concept_drift(df: pd.DataFrame, s: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    utilization = df["bill_amt1"] / df["limit_bal"]
    good = df[TARGET] == 0
    flip_util = good & (utilization > 0.5) & (rng.random(len(df)) < CONCEPT_UTIL_MAX * s)
    flip_young = good & df.get("_new_profile", False) & (rng.random(len(df)) < CONCEPT_YOUNG_MAX * s)
    df.loc[flip_util | flip_young, TARGET] = 1
    return df


def build_month(pool_slice: pd.DataFrame, month_idx: int, data: bool = True, concept: bool = True) -> pd.DataFrame:
    s = intensity(month_idx)
    rng = np.random.default_rng(SEED + month_idx)
    df = apply_data_drift(pool_slice, s if data else 0.0, rng)
    df = apply_concept_drift(df, s if concept else 0.0, rng)
    return df.drop(columns="_new_profile")


def corrupt_batch(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Lote vindo de um sistema legado com bugs de integração."""
    bad = df.sample(500, random_state=SEED).copy()
    idx = bad.index
    bad.loc[idx[:25], "limit_bal"] = np.nan                     # renda/limite nulo
    bad.loc[idx[25:40], "age"] = rng.choice([0, 15, 17, 150], 15)  # idade impossível/menor
    bad.loc[idx[40:50], "pay_amt1"] = -1500.0                    # pagamento negativo
    bad.loc[idx[50:55], "education"] = 7                         # categoria inexistente
    bad.loc[idx[55:60], "pay_0"] = 99                            # código de status inválido
    bad = pd.concat([bad, bad.iloc[:20]])                        # 20 registros duplicados
    bad["cpf"] = "***"                                           # PII que não deveria trafegar
    return bad


def production_slices() -> list[pd.DataFrame]:
    pool = pd.read_csv(PRODUCTION_POOL_PATH)
    return np.array_split(pool.sample(frac=1, random_state=SEED).reset_index(drop=True), N_MONTHS)


def main() -> list[str]:
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    slices = production_slices()
    names = []
    for i, (month, part) in enumerate(zip(MONTHS, slices)):
        with timed_stage("simulate_production_batch", batch=month) as info:
            df = build_month(part, i)
            df.to_csv(PRODUCTION_DIR / f"{month}.csv", index=False)
            info.update(rows=len(df), drift_intensity=round(intensity(i), 2),
                        default_rate=round(float(df[TARGET].mean()), 4))
        names.append(month)

    with timed_stage("simulate_corrupted_batch", batch=CORRUPTED_BATCH) as info:
        bad = corrupt_batch(slices[3], np.random.default_rng(SEED))
        bad.to_csv(PRODUCTION_DIR / f"{CORRUPTED_BATCH}.csv", index=False)
        info["rows"] = len(bad)
    return names


if __name__ == "__main__":
    main()
