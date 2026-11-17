"""Etapa 3 — Dashboard de observabilidade (HTML estático, autocontido) + métricas Prometheus.

Lê: reports/monitoring_summary.json, reports/causal_analysis.json, logs/*.jsonl
Gera: reports/dashboard.html, reports/metrics.prom, reports/logs/*.jsonl (cópia sem PII)
"""
from __future__ import annotations

import html
import json
import shutil
from datetime import datetime

from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

from src.config import AUC_DROP_ALERT, PSI_CRITICAL, PSI_WARNING, REPORTS_DIR
from src.logging_utils import ALERTS_LOG, PIPELINE_LOG, read_jsonl

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STATUS = {"SAUDÁVEL": ("good", "✓"), "ATENÇÃO": ("warning", "!"), "ALERTA": ("critical", "▲"),
          "BLOQUEADO": ("critical", "⛔"), "CRITICAL": ("critical", "▲"), "WARNING": ("warning", "!")}
STAGES = ["ingestion", "validation", "scoring", "drift_detection", "evidently_report", "performance_monitoring"]
STAGE_PT = {"ingestion": "Ingestão", "validation": "Contrato", "scoring": "Scoring", "drift_detection": "PSI/KS",
            "evidently_report": "Evidently", "performance_monitoring": "Desempenho"}

e = html.escape


def pct(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def badge(status: str) -> str:
    cls, icon = STATUS.get(status, ("muted", "•"))
    return f'<span class="badge {cls}"><span aria-hidden="true">{icon}</span> {e(status.title() if len(status) > 3 else status)}</span>'


# ---------------------------------------------------------------- gráficos SVG
def line_chart(xs, series, y_min, y_max, fmt, refs=(), height=230, label="") -> str:
    """series: [(nome, valores, var_cor)] ; refs: [(valor, rótulo, classe)]"""
    W, H, L, R, T, B = 600, height, 52, 92, 14, 30
    pw, ph = W - L - R, H - T - B
    xp = lambda i: L + (pw * i / max(len(xs) - 1, 1))
    yp = lambda v: T + ph * (1 - (v - y_min) / (y_max - y_min))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(label)}" class="chart">']
    for k in range(5):
        v = y_min + (y_max - y_min) * k / 4
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{yp(v):.1f}" y2="{yp(v):.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{yp(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    for i, x in enumerate(xs):
        out.append(f'<text class="tick" x="{xp(i):.1f}" y="{H - 8}" text-anchor="middle">{e(x)}</text>')
    for v, lab, cls in refs:
        out.append(f'<line class="ref {cls}" x1="{L}" x2="{W - R}" y1="{yp(v):.1f}" y2="{yp(v):.1f}"/>'
                   f'<text class="reflabel" x="{W - R + 6}" y="{yp(v) + 4:.1f}">{e(lab)}</text>')
    for name, vals, color in series:
        pts = " ".join(f"{xp(i):.1f},{yp(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline class="series" style="stroke:var({color})" points="{pts}"/>')
        for i, v in enumerate(vals):
            out.append(f'<g class="pt"><circle cx="{xp(i):.1f}" cy="{yp(v):.1f}" r="10" class="hit"/>'
                       f'<circle cx="{xp(i):.1f}" cy="{yp(v):.1f}" r="4.5" style="fill:var({color})" class="dot"/>'
                       f'<title>{e(name)} — {e(xs[i])}: {fmt(v)}</title></g>')
        out.append(f'<text class="dlabel" x="{xp(len(vals) - 1) + 10:.1f}" y="{yp(vals[-1]) + 4:.1f}">{fmt(vals[-1])}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({c})"></i>{e(n)}</span>' for n, c in items) + "</div>"


def heat_color(v: float) -> tuple[str, str]:
    idx = min(len(BLUE_RAMP) - 1, int(v / 0.40 * (len(BLUE_RAMP) - 1)))
    return BLUE_RAMP[idx], ("#ffffff" if idx >= 7 else "#0b0b0b")


def heatmap(batches, features) -> str:
    head = "".join(f"<th>{e(b['batch'])}</th>" for b in batches)
    rows = []
    for f in features:
        cells = []
        for b in batches:
            d = next(x for x in b["feature_drift"] if x["feature"] == f)
            bg, fg = heat_color(d["psi"])
            flag = " crit" if d["psi"] >= PSI_CRITICAL else (" warn" if d["psi"] >= PSI_WARNING else "")
            mark = "▲ " if d["psi"] >= PSI_CRITICAL else ""
            tip = f"{f} — {b['batch']}: PSI {num(d['psi'])} | KS {num(d['ks_stat'])} (p={d['ks_pvalue']:.2g}) | {d['status']}"
            cells.append(f'<td class="hcell{flag}" style="background:{bg};color:{fg}" title="{e(tip)}">{mark}{num(d["psi"], 2)}</td>')
        name = "score do modelo (prob_default)" if f == "prob_default" else f
        rows.append(f"<tr><th class='feat'>{e(name)}</th>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table class="heat"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def bar_chart(items, y_max, fmt, label) -> str:
    W, H, L, T, B = 600, 220, 52, 14, 44
    pw, ph = W - L - 16, H - T - B
    n = len(items)
    bw = pw / n * 0.56
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(label)}" class="chart">']
    for k in range(5):
        v = y_max * k / 4
        y = T + ph * (1 - v / y_max)
        out.append(f'<line class="grid" x1="{L}" x2="{W - 16}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    for i, (name, v, color) in enumerate(items):
        cx = L + pw * (i + 0.5) / n
        h = ph * v / y_max
        y = T + ph - h
        out.append(f'<g class="bar"><path d="M{cx - bw / 2:.1f},{T + ph:.1f} V{y + 4:.1f} q0,-4 4,-4 H{cx + bw / 2 - 4:.1f} '
                   f'q4,0 4,4 V{T + ph:.1f} Z" style="fill:var({color})"/><title>{e(name)}: {fmt(v)}</title></g>'
                   f'<text class="dlabel" x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle">{fmt(v)}</text>')
        for j, part in enumerate(name.split("|")):
            out.append(f'<text class="tick" x="{cx:.1f}" y="{T + ph + 16 + 13 * j:.1f}" text-anchor="middle">{e(part)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- prometheus
def export_prometheus(batches) -> None:
    reg = CollectorRegistry()
    g = {k: Gauge(f"credit_{k}", d, ["batch"], registry=reg) for k, d in {
        "pipeline_success": "1 se o lote foi processado, 0 se bloqueado",
        "rows_received": "linhas recebidas no lote", "rows_scored": "linhas pontuadas",
        "latency_seconds": "latência total do lote", "alerts_total": "alertas gerados no lote",
        "model_auc": "AUC com rótulos defasados", "drift_share": "fração de features com drift",
        "max_psi": "maior PSI entre as features", "approval_rate": "taxa de aprovação"}.items()}
    for b in batches:
        ok = b["status"] != "BLOQUEADO"
        g["pipeline_success"].labels(b["batch"]).set(int(ok))
        g["rows_received"].labels(b["batch"]).set(b["rows_received"])
        g["rows_scored"].labels(b["batch"]).set(b["rows_scored"])
        g["latency_seconds"].labels(b["batch"]).set(b["total_latency_s"])
        g["alerts_total"].labels(b["batch"]).set(b["alerts"])
        if ok:
            g["model_auc"].labels(b["batch"]).set(b["performance"]["auc"])
            g["drift_share"].labels(b["batch"]).set(b["drift_share"])
            g["max_psi"].labels(b["batch"]).set(b["max_psi"])
            g["approval_rate"].labels(b["batch"]).set(b["performance"]["approval_rate"])
    write_to_textfile(str(REPORTS_DIR / "metrics.prom"), reg)


# ---------------------------------------------------------------- página
CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--goodtext:#006300;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#9a988f;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goodtext:#0ca30c;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a988f;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goodtext:#0ca30c;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 48px}
header h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}header p{margin:0;color:var(--ink2)}
h2{font-size:17px;margin:36px 0 4px}.sub{color:var(--ink2);margin:0 0 14px;max-width:80ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:22px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.kpi .l{color:var(--ink2);font-size:12.5px}.kpi .v{font-size:28px;font-weight:650;margin-top:2px;line-height:1.15}
.kpi .d{font-size:12.5px;color:var(--ink2);margin-top:4px}.kpi .d b{font-weight:600}
.kpi.alert{border-color:color-mix(in srgb,var(--critical) 55%,transparent)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;min-width:0}
.card h3{font-size:14px;margin:0 0 2px}.card .note{color:var(--ink2);font-size:12.5px;margin:0 0 8px}
.chart{width:100%;height:auto;display:block}.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.chart .series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{stroke:var(--surface);stroke-width:2}.chart .hit{fill:transparent}
.chart .pt:hover .dot{r:6}.chart .dlabel{fill:var(--ink);font-size:11.5px;font-weight:600}
.chart .ref{stroke-width:1.5;stroke-dasharray:5 4}.chart .ref.base{stroke:var(--ink2)}.chart .ref.crit{stroke:var(--critical)}
.chart .reflabel{fill:var(--ink2);font-size:11px}.chart .bar:hover path{opacity:.85}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink2);margin:2px 0 6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap;font-size:13px}
thead th{color:var(--ink2);font-weight:600;font-size:12px}td.r,th.r{text-align:right}
table.heat td.hcell{text-align:center;font-weight:600;border:2px solid var(--surface);border-radius:4px;min-width:74px}
table.heat td.crit{outline:2px solid var(--critical);outline-offset:-3px}table.heat th.feat{font-weight:500;color:var(--ink)}
.badge{display:inline-flex;gap:5px;align-items:center;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;
border:1px solid var(--border);color:var(--ink)}
.badge.good span{color:var(--good)}.badge.warning span{color:#c98500}.badge.critical span{color:var(--critical)}
.alerts{list-style:none;margin:0;padding:0}.alerts li{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--grid);align-items:baseline}
.alerts .k{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--ink2);min-width:190px}
a{color:var(--s1)}code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2);margin-top:10px;flex-wrap:wrap}
.scale .bar{display:flex;height:10px;width:220px;border-radius:3px;overflow:hidden}.scale .bar i{flex:1}
footer{margin-top:40px;color:var(--ink2);font-size:12.5px}
@media (max-width:640px){.alerts li{flex-direction:column;gap:2px}.alerts .k{min-width:0}}
"""


def build() -> str:
    summary = json.loads((REPORTS_DIR / "monitoring_summary.json").read_text(encoding="utf-8"))
    causal = json.loads((REPORTS_DIR / "causal_analysis.json").read_text(encoding="utf-8"))
    alerts = read_jsonl(ALERTS_LOG)
    model = summary["model"]
    base = model["baseline_metrics"]
    all_batches = summary["batches"]
    ok = [b for b in all_batches if b["status"] != "BLOQUEADO"]
    blocked = [b for b in all_batches if b["status"] == "BLOQUEADO"]
    last = ok[-1]
    xs = [b["batch"] for b in ok]

    n_crit = sum(a["severity"] == "CRITICAL" for a in alerts)
    feats_last = [f for f in last["feature_drift"] if f["feature"] != "prob_default"]
    auc_delta = last["performance"]["auc"] - base["auc"]
    kpi_items = [
        ("Lotes processados", f"{len(ok)}/{len(all_batches)}", f"{sum(b['rows_scored'] for b in ok):,} clientes pontuados".replace(",", "."), False),
        ("Lotes bloqueados pelo contrato", str(len(blocked)), f"{sum(b['rows_received'] for b in blocked)} registros em quarentena", bool(blocked)),
        ("Alertas críticos", str(n_crit), f"{len(alerts)} alertas no total", n_crit > 0),
        (f"AUC — último lote ({last['batch']})", num(last['performance']['auc']),
         f"<b>{num(auc_delta)}</b> vs baseline {num(base['auc'])}", auc_delta <= -AUC_DROP_ALERT),
        ("Inadimplência observada", pct(last['performance']['default_rate_observed']),
         f"vs {pct(base['default_rate_observed'])} no treino", True),
        (f"Features com drift (PSI ≥ {num(PSI_CRITICAL, 1)})", f"{last['n_drifted']}/{len(feats_last)}",
         f"+{last['n_warning']} em atenção (PSI ≥ {num(PSI_WARNING, 1)} ou KS)", last["n_drifted"] > 0),
    ]
    kpis = '<div class="kpis">' + "".join(
        f'<div class="kpi{" alert" if a else ""}"><div class="l">{e(l)}</div><div class="v">{v}</div><div class="d">{d}</div></div>'
        for l, v, d, a in kpi_items) + "</div>"

    auc_chart = line_chart(
        xs, [("AUC", [b["performance"]["auc"] for b in ok], "--s1")], 0.60, 0.82, lambda v: num(v, 2),
        refs=[(base["auc"], f"baseline {num(base['auc'], 3)}", "base"),
              (base["auc"] - AUC_DROP_ALERT, "limite de alerta", "crit")], label="AUC por lote")
    rate_chart = line_chart(
        xs, [("Inadimplência observada", [b["performance"]["default_rate_observed"] for b in ok], "--s2"),
             ("Taxa de aprovação", [b["performance"]["approval_rate"] for b in ok], "--s1")],
        0.0, 1.0, lambda v: pct(v, 0), refs=[(base["default_rate_observed"], "inadimpl. treino", "base")],
        label="Inadimplência e aprovação por lote")

    top = [f["feature"] for f in last["feature_drift"]][:12]
    if "prob_default" in top:
        top.remove("prob_default")
    top = ["prob_default"] + top
    heat = heatmap(ok, top)
    scale = '<div class="scale">PSI 0<span class="bar">' + "".join(
        f'<i style="background:{c}"></i>' for c in BLUE_RAMP) + f'</span>≥ 0,40 &nbsp;·&nbsp; ▲ e contorno vermelho = drift (PSI ≥ {num(PSI_CRITICAL, 2)})</div>'

    rows = []
    for b in all_batches:
        st = "".join(f'<td class="r">{num(b["stages"][s], 2) if s in b["stages"] else "—"}</td>' for s in STAGES)
        link = f'<a href="{e(b["evidently_report"])}">relatório</a>' if "evidently_report" in b else "quarentena"
        auc = num(b["performance"]["auc"]) if "performance" in b else "—"
        rows.append(f"<tr><td>{e(b['batch'])}</td><td>{badge(b['status'])}</td><td class='r'>{b['rows_received']}</td>"
                    f"<td class='r'>{b['rows_scored']}</td>{st}<td class='r'>{num(b['total_latency_s'], 2)}</td>"
                    f"<td class='r'>{auc}</td><td class='r'>{b['alerts']}</td><td>{link}</td></tr>")
    pipeline_table = (
        '<div class="scroll"><table><thead><tr><th>Lote</th><th>Status</th><th class="r">Recebidas</th><th class="r">Pontuadas</th>'
        + "".join(f'<th class="r">{STAGE_PT[s]} (s)</th>' for s in STAGES)
        + '<th class="r">Total (s)</th><th class="r">AUC</th><th class="r">Alertas</th><th>Evidently</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>")

    viol = ""
    for b in blocked:
        viol += f"<h3>Lote <code>{e(b['batch'])}</code> — {b['rows_received']} registros bloqueados</h3>"
        viol += ('<div class="scroll"><table><thead><tr><th>Coluna</th><th>Regra violada</th><th class="r">Casos</th>'
                 '<th>Exemplos</th></tr></thead><tbody>')
        for v in b["violations"]:
            col = v["column"] if v["column"] and not v["column"].startswith("CreditContract") else "(dataset)"
            viol += (f"<tr><td><code>{e(col)}</code></td><td>{e(v['check'])}</td><td class='r'>{v['n_cases']}</td>"
                     f"<td><code>{e(', '.join(v['examples'][:3]))}</code></td></tr>")
        viol += "</tbody></table></div>"

    alert_items = "".join(
        f'<li>{badge(a["severity"])}<span class="k">{e(a["kind"])}</span><span>{e(a["message"])}</span></li>' for a in alerts)

    sc = causal["cenarios"]
    labels = {"A_sem_drift": "A · sem drift", "B_somente_data_drift": "B · só data|drift",
              "C_somente_concept_drift": "C · só concept|drift", "D_data_e_concept_drift": "D · data +|concept"}
    causal_chart = bar_chart([(labels[k], v["auc"], "--s1") for k, v in sc.items()], 0.8, lambda v: num(v, 2),
                             "AUC por cenário contrafactual")
    bad_chart = bar_chart([(labels[k], v["bad_approved_share"], "--s2") for k, v in sc.items()], 0.4,
                          lambda v: pct(v, 0), "Inadimplentes entre aprovados por cenário")
    att = causal["atribuicao_queda_auc"]
    pr = causal["prescricao_retreino"]
    seg_rows = "".join(
        f"<tr><td>{e(k.replace('_', ' '))}</td><td class='r'>{v['n']}</td><td class='r'>{pct(v['default_rate'])}</td>"
        f"<td class='r'>{num(v['score_medio'])}</td><td class='r'>{num(v['auc'])}</td></tr>"
        for k, v in causal["segmentos_cenario_real"].items())
    pres_rows = "".join(
        f"<tr><td>{n}</td><td class='r'>{num(m['auc'])}</td><td class='r'>{num(m['recall'])}</td>"
        f"<td class='r'>{pct(m['approval_rate'])}</td><td class='r'>{pct(m['bad_approved_share'])}</td></tr>"
        for n, m in [("Champion v1 (em produção)", pr["champion_v1"]), ("Challenger v2 (retreinado)", pr["challenger_v2"])])

    generated = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Monitoramento Credit Scoring</title>
<style>{CSS}</style></head><body><main>
<header><h1>Monitoramento do modelo de Credit Scoring</h1>
<p>Modelo <b>v{e(model['model_version'])}</b> (XGBoost) · limiar de decisão {num(model['threshold'], 2)} · referência = holdout do dataset de treino · gerado em {generated}</p></header>
{kpis}

<h2>Saúde do modelo ao longo do tempo</h2>
<p class="sub">Com os rótulos de inadimplência que chegam com defasagem, a AUC cai abaixo do limite de alerta a partir de 2026-08,
enquanto a inadimplência real quase dobra. A taxa de aprovação mal se mexe: é a <b>degradação silenciosa</b>.</p>
<div class="grid2">
 <div class="card"><h3>AUC por lote</h3><p class="note">Linha tracejada vermelha = baseline − {num(AUC_DROP_ALERT, 2)} (alerta de concept drift)</p>{auc_chart}</div>
 <div class="card"><h3>Inadimplência observada × taxa de aprovação</h3>{legend([("Inadimplência observada", "--s2"), ("Taxa de aprovação", "--s1")])}{rate_chart}</div>
</div>

<h2>Drift por variável (PSI) — referência × produção</h2>
<p class="sub">PSI com bins por decis da referência; passe o mouse para ver também a estatística e o p-valor de Kolmogorov-Smirnov.
Variáveis ordenadas pelo PSI no último lote.</p>
<div class="card">{heat}{scale}</div>

<h2>Saúde do pipeline de dados</h2>
<p class="sub">Status, volume e latência de cada etapa por lote (logs estruturados em <code>reports/logs/pipeline.jsonl</code>;
séries também no MLflow e em <code>reports/metrics.prom</code>).</p>
<div class="card">{pipeline_table}</div>

<h2>Contrato de dados — ingestão bloqueada</h2>
<p class="sub">O lote do sistema legado violou o contrato Pandera e foi enviado para quarentena sem gerar nenhuma predição.</p>
<div class="card">{viol}</div>

<h2>Alertas gerados</h2>
<div class="card"><ul class="alerts">{alert_items}</ul></div>

<h2>Análise causal da degradação ({e(causal['lote_analisado'])})</h2>
<p class="sub">Intervenções no mesmo lote de clientes isolam cada causa. Data drift sozinho tira {num(att['efeito_isolado_data_drift'])} de AUC;
concept drift sozinho tira {num(att['efeito_isolado_concept_drift'])}; juntos, {num(att['queda_auc_total'])}.
O pior efeito de negócio vem do concept drift: clientes aprovados que não pagam.</p>
<div class="grid2">
 <div class="card"><h3>AUC por cenário</h3>{causal_chart}</div>
 <div class="card"><h3>Inadimplentes entre os aprovados</h3>{bad_chart}</div>
</div>
<div class="grid2" style="margin-top:16px">
 <div class="card"><h3>Segmentos no cenário real</h3><div class="scroll"><table><thead><tr><th>Segmento</th><th class="r">Clientes</th>
 <th class="r">Inadimpl.</th><th class="r">Score médio</th><th class="r">AUC</th></tr></thead><tbody>{seg_rows}</tbody></table></div></div>
 <div class="card"><h3>Prescrição: retreino (champion × challenger)</h3><p class="note">{e(pr['challenger_treino'])}; avaliado em {e(pr['avaliado_em'])}</p>
 <div class="scroll"><table><thead><tr><th>Modelo</th><th class="r">AUC</th><th class="r">Recall</th><th class="r">Aprovação</th>
 <th class="r">Inadimpl. aprovados</th></tr></thead><tbody>{pres_rows}</tbody></table></div></div>
</div>

<footer>Relatórios Evidently por lote em <code>reports/evidently/</code> · rastreamento completo no MLflow
(<code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>) · dados pseudonimizados, nenhum dado pessoal neste painel.</footer>
</main></body></html>"""


def main() -> None:
    summary = json.loads((REPORTS_DIR / "monitoring_summary.json").read_text(encoding="utf-8"))
    (REPORTS_DIR / "dashboard.html").write_text(build(), encoding="utf-8")
    export_prometheus(summary["batches"])
    logs_out = REPORTS_DIR / "logs"
    logs_out.mkdir(parents=True, exist_ok=True)
    for src in (PIPELINE_LOG, ALERTS_LOG):
        if src.exists():
            shutil.copy(src, logs_out / src.name)


if __name__ == "__main__":
    main()
