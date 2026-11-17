"""Etapa 0 — Ingestão do dataset bruto, minimização/pseudonimização (LGPD) e
separação Referência x Produção.

- Baixa o dataset público UCI "Default of Credit Card Clients" (30.000 clientes).
- Substitui o ID original por um hash SHA-256 com salt (pseudonimização, LGPD art. 13 §4).
- Normaliza nomes de colunas e categorias inválidas documentadas.
- Separa 60% como Dataset de Referência (treino) e 40% como pool de Produção
  (dados que o modelo nunca viu, onde o drift será simulado).
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.config import (ID_COL, PRODUCTION_POOL_PATH, PROCESSED_DIR, RAW_DIR, RAW_URL, RAW_XLS,
                        REFERENCE_PATH, SEED, TARGET)
from src.logging_utils import timed_stage

# Em produção o salt viria de um cofre de segredos (ex.: Vault/KMS), nunca do código.
PSEUDO_SALT = os.environ.get("PSEUDO_SALT", "tc-fase4-demo-salt")

RENAME = {"ID": "id", "PAY_0": "pay_0", "default payment next month": TARGET}


def download_raw() -> None:
    if RAW_XLS.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(RAW_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(RAW_DIR)


def pseudonymize(ids: pd.Series) -> pd.Series:
    return ids.astype(str).map(lambda x: hashlib.sha256(f"{PSEUDO_SALT}:{x}".encode()).hexdigest()[:16])


def load_clean() -> pd.DataFrame:
    df = pd.read_excel(RAW_XLS, header=1)
    df = df.rename(columns=RENAME)
    df.columns = [c.lower() for c in df.columns]

    # Pseudonimização: o identificador direto sai do dataset de trabalho.
    df.insert(0, ID_COL, pseudonymize(df["id"]))
    df = df.drop(columns=["id"])

    # Categorias não documentadas no dicionário do UCI -> "outros".
    df["education"] = df["education"].where(df["education"].isin([1, 2, 3, 4]), 4)
    df["marriage"] = df["marriage"].where(df["marriage"].isin([1, 2, 3]), 3)

    money = [c for c in df.columns if c.startswith(("limit_bal", "bill_amt", "pay_amt"))]
    df[money] = df[money].astype(float)
    return df


def split_reference_production(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref, prod = train_test_split(df, test_size=0.40, stratify=df[TARGET], random_state=SEED)
    return ref.reset_index(drop=True), prod.reset_index(drop=True)


def main() -> None:
    with timed_stage("ingestion_raw") as info:
        download_raw()
        df = load_clean()
        info["rows"] = len(df)
    with timed_stage("split_reference_production") as info:
        ref, prod = split_reference_production(df)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ref.to_csv(REFERENCE_PATH, index=False)
        prod.to_csv(PRODUCTION_POOL_PATH, index=False)
        info.update(reference_rows=len(ref), production_rows=len(prod))


if __name__ == "__main__":
    main()
