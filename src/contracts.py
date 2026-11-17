"""Etapa 1 — Contrato de dados (Pandera).

O contrato é a "porta de entrada" do modelo: qualquer lote que viole uma regra
rígida é BLOQUEADO e enviado para quarentena, sem gerar predições.

Regras rígidas (todas avaliadas de forma lazy, para reportar todas as violações):
 R1  client_hash: obrigatório, não nulo e ÚNICO (sem duplicatas)
 R2  limit_bal (limite/renda de crédito): não nulo, 0 < valor <= 1.000.000
 R3  age: não nulo, 18 <= idade <= 100 (somente maiores de idade)
 R4  education ∈ {1,2,3,4}; sex ∈ {1,2}; marriage ∈ {1,2,3}
 R5  pay_0..pay_6 (status de pagamento) ∈ [-2, 8]
 R6  bill_amt1..6 ∈ [-1.000.000, 5.000.000]; pay_amt1..6 ∈ [0, 5.000.000]
 R7  default (alvo, quando presente) ∈ {0,1}
 R8  Sem linhas duplicadas e esquema ESTRITO: colunas fora do contrato
     (ex.: CPF, nome, ID original) são rejeitadas — proteção de PII by design.
 R9  Lote não vazio.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from src.config import PAY_AMT, PAY_STATUS, BILL_AMT


class CreditContract(pa.DataFrameModel):
    client_hash: Series[str] = pa.Field(nullable=False, unique=True, str_length={"min_value": 16, "max_value": 16})
    limit_bal: Series[float] = pa.Field(nullable=False, gt=0, le=1_000_000)
    sex: Series[int] = pa.Field(nullable=False, isin=[1, 2])
    education: Series[int] = pa.Field(nullable=False, isin=[1, 2, 3, 4])
    marriage: Series[int] = pa.Field(nullable=False, isin=[1, 2, 3])
    age: Series[int] = pa.Field(nullable=False, ge=18, le=100)

    pay_0: Series[int] = pa.Field(nullable=False, ge=-2, le=8)
    pay_2: Series[int] = pa.Field(nullable=False, ge=-2, le=8)
    pay_3: Series[int] = pa.Field(nullable=False, ge=-2, le=8)
    pay_4: Series[int] = pa.Field(nullable=False, ge=-2, le=8)
    pay_5: Series[int] = pa.Field(nullable=False, ge=-2, le=8)
    pay_6: Series[int] = pa.Field(nullable=False, ge=-2, le=8)

    bill_amt1: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)
    bill_amt2: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)
    bill_amt3: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)
    bill_amt4: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)
    bill_amt5: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)
    bill_amt6: Series[float] = pa.Field(nullable=False, ge=-1_000_000, le=5_000_000)

    pay_amt1: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)
    pay_amt2: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)
    pay_amt3: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)
    pay_amt4: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)
    pay_amt5: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)
    pay_amt6: Series[float] = pa.Field(nullable=False, ge=0, le=5_000_000)

    default: Series[int] = pa.Field(nullable=False, isin=[0, 1], required=False)

    @pa.dataframe_check(error="lote vazio")
    def non_empty(cls, df) -> bool:
        return len(df) > 0

    @pa.dataframe_check(error="linhas duplicadas")
    def no_duplicate_rows(cls, df) -> bool:
        return not df.duplicated().any()

    class Config:
        strict = True   # rejeita colunas não previstas (ex.: PII como cpf/nome)
        coerce = True   # tenta converter tipos (ex.: '35' -> 35); falha se impossível
        name = "CreditContract v1.0"


# Garante que as listas da config e o contrato não divergem.
assert set(PAY_STATUS + BILL_AMT + PAY_AMT) <= set(CreditContract.to_schema().columns)


def rule_id(column: str | None, check: str) -> str:
    """Mapeia uma falha do Pandera (coluna, check) para o identificador da regra R1..R9."""
    col = column or ""
    if check in ("column_in_schema", "column_in_dataframe", "linhas duplicadas") or col.startswith("CreditContract"):
        return "R9" if check == "lote vazio" else "R8"
    if col == "client_hash":
        return "R1"
    if col == "limit_bal":
        return "R2"
    if col == "age":
        return "R3"
    if col in ("education", "sex", "marriage"):
        return "R4"
    if col.startswith("pay_") and not col.startswith("pay_amt"):
        return "R5"
    if col.startswith(("bill_amt", "pay_amt")):
        return "R6"
    if col == "default":
        return "R7"
    return "R8"


RULES = {
    "R1": "client_hash obrigatório e único",
    "R2": "limit_bal não nulo e em (0, 1.000.000]",
    "R3": "age entre 18 e 100",
    "R4": "categorias válidas (education/sex/marriage)",
    "R5": "status de pagamento em [-2, 8]",
    "R6": "valores monetários dentro dos limites",
    "R7": "alvo binário",
    "R8": "sem duplicatas e esquema estrito (sem PII extra)",
    "R9": "lote não vazio",
}
