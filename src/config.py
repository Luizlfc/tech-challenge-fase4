"""Configuração central do projeto: caminhos, features, limiares e sementes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTION_DIR = DATA_DIR / "production"
QUARANTINE_DIR = DATA_DIR / "quarantine"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

RAW_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_XLS = RAW_DIR / "default of credit card clients.xls"

REFERENCE_PATH = PROCESSED_DIR / "reference.csv"          # dados usados no treino
PRODUCTION_POOL_PATH = PROCESSED_DIR / "production_pool.csv"  # nunca vistos pelo modelo

MODEL_PATH = MODELS_DIR / "model.json"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT_TRAIN = "credit-scoring-training"
MLFLOW_EXPERIMENT_MONITOR = "credit-scoring-monitoring"

SEED = 42

TARGET = "default"
ID_COL = "client_hash"

# Atributos protegidos: mantidos SOMENTE para auditoria de viés, nunca usados como feature.
PROTECTED = ["sex", "marriage"]

PAY_STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_AMT = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT = [f"pay_amt{i}" for i in range(1, 7)]

FEATURES = ["limit_bal", "education", "age", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
CATEGORICAL_FEATURES = ["education", *PAY_STATUS]
NUMERICAL_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# Simulação de produção: 6 lotes mensais
N_MONTHS = 6

# Limiares de alerta (documentados no README)
PSI_WARNING = 0.10
PSI_CRITICAL = 0.20
KS_PVALUE = 0.05
DRIFT_SHARE_ALERT = 0.30   # fração de features com drift que caracteriza drift do dataset
AUC_DROP_ALERT = 0.05      # queda absoluta de AUC vs. baseline
APPROVAL_SHIFT_ALERT = 0.10  # variação absoluta da taxa de aprovação
DISPARATE_IMPACT_MIN = 0.80  # regra dos 4/5
