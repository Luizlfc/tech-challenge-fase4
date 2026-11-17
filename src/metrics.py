"""Métricas reutilizadas no treino e no monitoramento: desempenho, PSI, KS e viés."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def classification_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks_score": float(np.max(tpr - fpr)),  # KS de separação bons x maus (padrão em crédito)
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "approval_rate": float(1 - pred.mean()),  # aprovado = previsto como bom pagador
        "default_rate_observed": float(y_true.mean()),
    }


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index com bins por quantis da referência.

    PSI < 0.10: estável | 0.10–0.20: alerta | > 0.20: mudança significativa.
    Variáveis discretas (poucos valores) usam as próprias categorias como bins.
    """
    reference = pd.Series(reference).dropna().to_numpy()
    current = pd.Series(current).dropna().to_numpy()
    uniq = np.unique(reference)
    if len(uniq) <= 15:
        cats = np.union1d(uniq, np.unique(current))
        ref_p = np.array([(reference == c).mean() for c in cats])
        cur_p = np.array([(current == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference, edges)[0] / len(reference)
        cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p = np.clip(ref_p, eps, None)
    cur_p = np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_test(reference, current) -> tuple[float, float]:
    stat, pvalue = ks_2samp(pd.Series(reference).dropna(), pd.Series(current).dropna())
    return float(stat), float(pvalue)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [17, 25, 35, 50, 120], labels=["18-25", "26-35", "36-50", "51+"]).astype(str)


def fairness_report(df: pd.DataFrame, approved_col: str = "approved") -> dict:
    """Taxa de aprovação por grupo e impacto desproporcional (min/max, regra dos 4/5)."""
    out = {}
    groups = {
        "sex": df["sex"].map({1: "masculino", 2: "feminino"}),
        "age_band": age_band(df["age"]),
    }
    for name, g in groups.items():
        rates = df.groupby(g)[approved_col].mean()
        out[name] = {
            "approval_rate_by_group": {k: round(float(v), 4) for k, v in rates.items()},
            "disparate_impact": round(float(rates.min() / rates.max()), 4) if rates.max() > 0 else None,
        }
    return out
