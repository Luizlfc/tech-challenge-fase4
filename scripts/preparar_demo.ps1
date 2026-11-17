# Prepara a tela para gravar o vídeo STAR.
#   .\scripts\preparar_demo.ps1           -> só abre as abas (usa os relatórios já gerados)
#   .\scripts\preparar_demo.ps1 -Rodar    -> roda o pipeline completo antes (python main.py)
param([switch]$Rodar)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

if ($Rodar) {
    Write-Host "Rodando o pipeline completo (python main.py)..." -ForegroundColor Cyan
    & $py main.py
}

# 1) Abas no navegador, na ordem do vídeo
Start-Process "https://github.com/Luizlfc/tech-challenge-fase4"
Start-Sleep 1
Start-Process (Join-Path $root "reports\evidently\drift_report_2026-10.html")
Start-Sleep 1
Start-Process (Join-Path $root "reports\dashboard.html")

# 2) Azure ML Studio (lê o .env, se existir)
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $vars = @{}
    Get-Content $envFile | Where-Object { $_ -match "^\s*([A-Z_]+)\s*=\s*(.*)$" } | ForEach-Object { $vars[$Matches[1]] = $Matches[2].Trim() }
    if ($vars["AZURE_SUBSCRIPTION_ID"]) {
        $wsid = "/subscriptions/$($vars['AZURE_SUBSCRIPTION_ID'])/resourcegroups/$($vars['AZURE_RESOURCE_GROUP'])/providers/Microsoft.MachineLearningServices/workspaces/$($vars['AZURE_ML_WORKSPACE'])"
        Start-Sleep 1
        Start-Process "https://ml.azure.com/experiments?wsid=$wsid"
    }
}

Clear-Host
Write-Host ""
Write-Host "Tudo aberto. Comandos para digitar AO VIVO durante o video:" -ForegroundColor Green
Write-Host ""
Write-Host '  python -m src.validate "data/production/2026-08_sistema_legado.csv"' -ForegroundColor Yellow
Write-Host '  python -m src.validate "data/production/2026-05.csv"' -ForegroundColor Yellow
Write-Host ""
Write-Host "Roteiro: docs\roteiro_video.md" -ForegroundColor Gray
