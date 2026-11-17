"""Testes do contrato de dados: lote válido passa; cada regra rígida bloqueia sozinha."""
import numpy as np
import pandas as pd
import pytest

from src.validate import validate_batch


def make_valid(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "client_hash": [f"{i:016x}" for i in range(n)],
        "limit_bal": rng.integers(1, 50, n) * 10_000.0,
        "sex": rng.integers(1, 3, n), "education": rng.integers(1, 5, n),
        "marriage": rng.integers(1, 4, n), "age": rng.integers(21, 70, n),
    })
    for c in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        df[c] = rng.integers(-2, 3, n)
    for i in range(1, 7):
        df[f"bill_amt{i}"] = rng.integers(0, 100_000, n).astype(float)
        df[f"pay_amt{i}"] = rng.integers(0, 10_000, n).astype(float)
    df["default"] = rng.integers(0, 2, n)
    return df


def test_valid_batch_passes():
    assert validate_batch(make_valid(), "t_valid").passed


@pytest.mark.parametrize("mutate, expected_col", [
    (lambda d: d.assign(limit_bal=d.limit_bal.where(d.index != 0, np.nan)), "limit_bal"),   # renda nula
    (lambda d: d.assign(age=d.age.where(d.index != 0, 16)), "age"),                         # menor de idade
    (lambda d: d.assign(age=d.age.where(d.index != 0, 150)), "age"),                        # idade absurda
    (lambda d: d.assign(limit_bal=d.limit_bal.where(d.index != 0, -5.0)), "limit_bal"),     # limite negativo
    (lambda d: d.assign(pay_0=d.pay_0.where(d.index != 0, 99)), "pay_0"),                   # status inválido
    (lambda d: d.assign(education=d.education.where(d.index != 0, 7)), "education"),        # categoria inválida
    (lambda d: d.assign(pay_amt1=d.pay_amt1.where(d.index != 0, -1.0)), "pay_amt1"),        # pagamento negativo
    (lambda d: pd.concat([d, d.iloc[[0]]]), "client_hash"),                                 # duplicata
])
def test_each_rule_blocks(mutate, expected_col):
    res = validate_batch(mutate(make_valid()), "t_bad")
    assert not res.passed
    assert expected_col in {f["column"] for f in res.failures}


def test_unexpected_pii_column_blocked():
    res = validate_batch(make_valid().assign(cpf="123.456.789-00"), "t_pii")
    assert not res.passed
    assert any(f["check"] == "column_in_schema" for f in res.failures)


def test_missing_column_blocked():
    res = validate_batch(make_valid().drop(columns="age"), "t_missing")
    assert not res.passed


def test_empty_batch_blocked():
    assert not validate_batch(make_valid().iloc[0:0], "t_empty").passed
