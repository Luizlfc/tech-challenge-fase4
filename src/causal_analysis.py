"""Análise causal da degradação (Etapa 4 — Causalidade).

Como a degradação foi simulada, podemos fazer INTERVENÇÕES (do-operator) no
mesmo lote de clientes do último mês e isolar a contribuição de cada causa:

  cenário A: sem drift                      (contrafactual "economia estável")
  cenário B: só data drift  — do(X')        (P(X) muda, P(y|X) mantida)
  cenário C: só concept drift — do(y|X)'    (P(X) mantida, P(y|X) muda)
  cenário D: data + concept drift           (o que aconteceu de fato)

Também mede a degradação por segmento (novo perfil x carteira antiga,
utilização alta x baixa) e testa a PRESCRIÇÃO: um modelo challenger retreinado
com dados recentes rotulados, comparado ao champion no último mês.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config import FEATURES, PRODUCTION_DIR, REFERENCE_PATH, REPORTS_DIR, SEED, TARGET
from src.logging_utils import timed_stage
from src.metrics import classification_metrics
from src.monitor import load_model
from src.simulate_drift import MONTHS, apply_concept_drift, apply_data_drift, intensity, production_slices
from src.train import PARAMS, split_reference


def scenario(part: pd.DataFrame, month_idx: int, data: bool, concept: bool) -> pd.DataFrame:
    s = intensity(month_idx)
    rng = np.random.default_rng(SEED + month_idx)
    df = apply_data_drift(part, s if data else 0.0, rng)
    return apply_concept_drift(df, s if concept else 0.0, rng)


def summarize(df: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict:
    m = classification_metrics(df[TARGET], proba, threshold)
    approved = proba < threshold
    m["bad_approved_share"] = float((approved & (df[TARGET] == 1)).sum() / approved.sum())
    return {k: round(v, 4) for k, v in m.items()}


def main() -> dict:
    model, meta = load_model()
    thr = meta["threshold"]
    last = len(MONTHS) - 1
    part = production_slices()[last]

    results = {}
    for name, (d, c) in {"A_sem_drift": (False, False), "B_somente_data_drift": (True, False),
                         "C_somente_concept_drift": (False, True), "D_data_e_concept_drift": (True, True)}.items():
        df = scenario(part, last, d, c)
        results[name] = summarize(df, model.predict_proba(df[FEATURES])[:, 1], thr)

    base_auc = results["A_sem_drift"]["auc"]
    total = base_auc - results["D_data_e_concept_drift"]["auc"]
    attribution = {
        "queda_auc_total": round(total, 4),
        "efeito_isolado_data_drift": round(base_auc - results["B_somente_data_drift"]["auc"], 4),
        "efeito_isolado_concept_drift": round(base_auc - results["C_somente_concept_drift"]["auc"], 4),
    }

    # Segmentos no cenário real (D)
    full = scenario(part, last, True, True)
    full["proba"] = model.predict_proba(full[FEATURES])[:, 1]
    util = full["bill_amt1"] / full["limit_bal"]
    segments = {}
    for seg, mask in {"novo_perfil_jovem_limite_baixo": (full["age"] <= 27) & (full["limit_bal"] <= 50_000),
                      "carteira_tradicional": ~((full["age"] <= 27) & (full["limit_bal"] <= 50_000)),
                      "utilizacao_alta_>50%": util > 0.5, "utilizacao_baixa_<=50%": util <= 0.5}.items():
        sub = full[mask]
        segments[seg] = {"n": int(len(sub)), "default_rate": round(float(sub[TARGET].mean()), 4),
                         "score_medio": round(float(sub["proba"].mean()), 4),
                         "auc": round(float(roc_auc_score(sub[TARGET], sub["proba"])), 4)}

    # Prescrição: challenger retreinado com referência + 2 meses recentes rotulados
    ref_train, _, _ = split_reference(pd.read_csv(REFERENCE_PATH))
    recent = pd.concat([pd.read_csv(PRODUCTION_DIR / f"{m}.csv") for m in MONTHS[-3:-1]])
    test = pd.read_csv(PRODUCTION_DIR / f"{MONTHS[-1]}.csv")
    challenger = XGBClassifier(**PARAMS)
    challenger.fit(pd.concat([ref_train, recent])[FEATURES], pd.concat([ref_train, recent])[TARGET], verbose=False)
    champ_p = model.predict_proba(test[FEATURES])[:, 1]
    chall_p = challenger.predict_proba(test[FEATURES])[:, 1]
    prescription = {
        "challenger_treino": f"referência (treino) + lotes rotulados {MONTHS[-3]} e {MONTHS[-2]}",
        "avaliado_em": MONTHS[-1],
        "champion_v1": summarize(test, champ_p, thr),
        "challenger_v2": summarize(test, chall_p, thr),
    }

    out = {"lote_analisado": MONTHS[-1], "intensidade_drift": 1.0, "cenarios": results,
           "atribuicao_queda_auc": attribution, "segmentos_cenario_real": segments, "prescricao_retreino": prescription}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "causal_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    with timed_stage("causal_analysis"):
        print(json.dumps(main(), indent=2, ensure_ascii=False))
