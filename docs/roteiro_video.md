# Roteiro do vídeo — método STAR (≤ 5 minutos)

**Antes de gravar:**
1. Rode `python main.py` para que todos os relatórios estejam atualizados.
2. Deixe abertos: o VS Code com o terminal, `reports/dashboard.html`, `reports/evidently/drift_report_2026-10.html` e o MLflow: local (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`, em http://127.0.0.1:5000) ou, se usar o Azure, o workspace em https://ml.azure.com → Jobs.
3. Grave a tela em 1080p (OBS ou Xbox Game Bar com Win+Alt+R).

---

## S — Situation (0:00 – 0:40)

**Tela:** README, aberto no topo.

> "Uma fintech tem um modelo de credit scoring em produção. A economia mudou: inflação, juros do rotativo e a entrada de um público mais jovem. O modelo continua respondendo, continua aprovando, nenhum erro aparece no log. Mesmo assim, a diretoria suspeita que ele está decidindo errado. Esse é o risco da **degradação silenciosa**: o modelo não quebra, ele só vai errando mais a cada mês, e cada erro é crédito concedido a quem não vai pagar."

## T — Task (0:40 – 1:10)

**Tela:** tabela de entregáveis no README.

> "Nossa tarefa foi construir a camada de sustentação: primeiro, **interceptar dados ruins** antes de chegarem ao modelo; segundo, **detectar drift** com testes estatísticos; terceiro, dar **visibilidade operacional** ao pipeline; e quarto, garantir **compliance com a LGPD** e explicar a causa do problema."

## A — Action (1:10 – 3:15)

**1. Contrato de dados (1:10 – 1:50)**
**Tela:** `src/contracts.py` e depois o terminal.
```
python -m src.validate "data/production/2026-08_sistema_legado.csv"
```
> "Usamos Pandera para criar um contrato com 9 regras rígidas: idade entre 18 e 100, renda não nula, sem duplicatas e um esquema estrito. Um lote vindo de um sistema legado com renda nula, menores de idade, pagamentos negativos e até uma coluna de CPF foi **bloqueado**: nenhuma predição foi gerada e o lote foi para a quarentena. Repare que o esquema estrito também protege a privacidade, porque o CPF nunca chega ao modelo."

**2. Modelo e simulação de drift (1:50 – 2:30)**
**Tela:** tabela "Cenário simulado" no README.
> "Treinamos um XGBoost só com o dataset de referência: 18 mil clientes, AUC de 0,78. Os outros 12 mil viraram seis meses de produção. Em cada mês aumentamos a inflação nas faturas, reduzimos os pagamentos e trouxemos um novo perfil jovem com limite baixo. Isso é **data drift**. Também fizemos clientes com alta utilização do limite inadimplirem mais, mesmo com bom histórico. Isso é **concept drift**."

**3. Detecção e observabilidade (2:30 – 3:15)**
**Tela:** relatório Evidently de 2026-10 (rolar a tabela de drift) e depois o MLflow, local ou no Azure ML Studio (run `monitoramento-producao` → aba de métricas: `auc`, `psi_age`, `alerts_total` por step; runs filhos `lote-AAAA-MM` com o HTML do Evidently como artefato).
> "Cada lote passa por PSI e Kolmogorov-Smirnov, e o Evidently gera um relatório HTML. Tudo é registrado em logs estruturados e no MLflow, que pode rodar local ou no Azure ML como servidor de tracking: status de cada etapa, latência, volume, violações por regra, PSI e KS por variável e AUC ao longo do tempo. Também exportamos métricas no formato Prometheus."

## R — Result (3:15 – 5:00)

**Tela:** `reports/dashboard.html`, rolando de cima para baixo.

**KPIs e AUC (3:15 – 3:45)**
> "O resultado: 6 lotes processados, 1 bloqueado, 14 alertas, 7 deles críticos. A AUC caiu de 0,78 para 0,69, e a inadimplência real quase dobrou, de 22% para 42%. Mas olhem a taxa de aprovação: ela mal se mexeu. Essa é a degradação silenciosa, finalmente visível."

**Heatmap de PSI (3:45 – 4:05)**
> "O heatmap mostra **quais variáveis** degradaram: idade, limite e pagamentos passaram de PSI 0,2. O histórico de pagamento, que é a variável mais importante do modelo, ficou estável, e é exatamente por isso que o problema é silencioso."

**Análise causal (4:05 – 4:35)**
> "Para entender a causa, fizemos intervenções no mesmo lote. Só o data drift tira 0,03 de AUC; só o concept drift tira 0,04 e faz a inadimplência entre os aprovados subir de 15% para 22%. Juntos, **um em cada três aprovados não paga**. A prescrição foi um challenger retreinado com dados recentes: a AUC volta para 0,72 e o recall sobe de 0,39 para 0,63."

**Governança e lições (4:35 – 5:00)**
**Tela:** seção 7 do README.
> "Na governança, o ID virou hash, sexo e estado civil ficaram fora do modelo, a base legal é a proteção do crédito do artigo 7º, e definimos prazos de retenção e revisão humana pelo artigo 20. A principal lição: **monitorar só os dados de entrada não basta**. É preciso olhar desempenho e impacto no negócio, e ter um contrato de dados como primeira linha de defesa."
