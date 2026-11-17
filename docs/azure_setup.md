# Azure ML como servidor de tracking do MLflow (custo mínimo)

O código **continua rodando na sua máquina**. O workspace do Azure ML serve **apenas** como servidor de tracking do MLflow: experimentos, runs, métricas e artefatos.

> ⚠️ **Não crie** Compute Instance, Compute Cluster nem Endpoint: são eles que geram custo relevante.
> O workspace em si não é cobrado. Os recursos que ele cria junto (Storage Account, Key Vault e Application Insights) custam centavos com o volume deste projeto.

Os comandos abaixo estão **documentados, não foram executados**. Rode-os você mesmo, na ordem.

---

## 0. Pré-requisitos

```powershell
winget install -e --id Microsoft.AzureCLI      # Azure CLI (reabra o terminal depois)
az login                                       # autenticação usada pelo DefaultAzureCredential
az extension add -n ml                         # comandos "az ml" (Azure ML CLI v2)
az account set --subscription "<ID-DA-ASSINATURA>"
```

O `DefaultAzureCredential` tenta, em ordem: variáveis de ambiente, identidade gerenciada, **Azure CLI (`az login`)**, Azure PowerShell e Azure Developer CLI. O caminho recomendado para rodar localmente é o `az login`.

## 1. Variáveis

```powershell
# PowerShell
$SUB  = az account show --query id -o tsv
$RG   = "rg-mlet-fase4"        # se o grupo já existir (ex.: criado pelo portal), use o nome dele e pule o passo 2
$WS   = "mlw-mlet-fase4"
$LOC  = "brazilsouth"          # ou eastus2 (geralmente mais barato)
$EMAIL = "<seu-email-para-alertas>"
```

```bash
# bash
SUB=$(az account show --query id -o tsv); RG=rg-mlet-fase4; WS=mlw-mlet-fase4; LOC=brazilsouth; EMAIL="<seu-email>"
```

## 2. Resource group

```bash
az group create --name $RG --location $LOC --tags projeto=mlet-fase4 finalidade=mlflow-tracking
```

## 3. Workspace do Azure ML (sem computação)

```bash
az ml workspace create --name $WS --resource-group $RG --location $LOC \
  --display-name "MLET Fase 4 - MLflow tracking" --tags projeto=mlet-fase4
```

Esse comando cria o workspace e os recursos associados: Storage Account (artefatos), Key Vault e Application Insights. Ele **não cria** nenhuma computação.

Para obter a URI de tracking do MLflow:

```bash
az ml workspace show --name $WS --resource-group $RG --query mlflow_tracking_uri -o tsv
# alternativa pelo Python (usa o .env): python -m src.tracking --azure
```

## 4. Orçamento (budget) com alerta de baixo valor

Crie um orçamento de **US$ 5/mês** no resource group, com e-mail em 50% (gasto real) e em 100% (gasto previsto). O `az consumption budget create` não configura notificações, por isso usamos a API REST via `az rest`:

```powershell
# PowerShell — salve o corpo do orçamento
@"
{
  "properties": {
    "category": "Cost",
    "amount": 5,
    "timeGrain": "Monthly",
    "timePeriod": { "startDate": "2026-11-01T00:00:00Z", "endDate": "2027-03-31T00:00:00Z" },
    "notifications": {
      "real_50pct":     { "enabled": true, "operator": "GreaterThan", "threshold": 50,  "thresholdType": "Actual",     "contactEmails": ["$EMAIL"] },
      "previsto_100pct":{ "enabled": true, "operator": "GreaterThan", "threshold": 100, "thresholdType": "Forecasted", "contactEmails": ["$EMAIL"] }
    }
  }
}
"@ | Set-Content -Encoding utf8 budget.json

az rest --method put `
  --url "https://management.azure.com/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Consumption/budgets/budget-mlet-fase4?api-version=2023-05-01" `
  --body "@budget.json"
```

A `startDate` precisa ser o primeiro dia do mês atual ou de um mês futuro. Para conferir: `az consumption budget list --resource-group $RG -o table`.

## 5. Apontar o projeto para o Azure ML

```powershell
pip install -r requirements.txt -r requirements-azure.txt
copy .env.example .env          # preencha AZURE_* e MLFLOW_TRACKING_URI (a URI do passo 3)
python -m src.tracking          # deve imprimir: azureml: azureml://...
python main.py                  # mesmo pipeline; o tracking agora vai para o workspace
```

Para ver os resultados: abra **https://ml.azure.com**, selecione o workspace e vá em **Jobs**. Lá estão os experimentos `credit-scoring-training` e `credit-scoring-monitoring`, com as métricas por lote (gráficos por step), os runs filhos `lote-AAAA-MM` e os artefatos (HTML do Evidently, dashboard, logs). O modelo fica em **Models → `credit-scoring-xgb`**.

Para voltar ao modo offline, basta apagar ou esvaziar `MLFLOW_TRACKING_URI` no `.env`.

**O que vai e o que não vai para a nuvem:**

| Vai para a nuvem | Não vai |
|---|---|
| Métricas agregadas, parâmetros, relatórios HTML, logs sem PII, modelo | Dados de clientes (raw, referência, produção), predições individuais. Esses dados ficam só na máquina local, por minimização (LGPD, art. 6º, III) |

## 6. Apagar tudo ao final

```bash
# 1) apaga o workspace de forma permanente (sem soft-delete) junto com os recursos associados
az ml workspace delete --name $WS --resource-group $RG --permanently-delete --all-resources --yes

# 2) apaga o resource group inteiro (budget incluído)
az group delete --name $RG --yes --no-wait

# 3) Key Vault fica em soft-delete por 90 dias (não cobra, mas pode bloquear o mesmo nome); para purgar:
az keyvault list-deleted --query "[?properties.tags.projeto=='mlet-fase4'].name" -o tsv
az keyvault purge --name <nome-do-keyvault>
```

Para confirmar que não sobrou nada: `az group exists --name $RG` deve retornar `false`.

## Custos esperados

| Recurso | Custo aproximado neste projeto |
|---|---|
| Workspace Azure ML | gratuito |
| Storage Account (≈ 70 MB de artefatos por execução) | centavos/mês |
| Key Vault (poucas operações) | centavos |
| Application Insights | dentro da franquia gratuita |
| Compute / Endpoints | **não criados → R$ 0** |
