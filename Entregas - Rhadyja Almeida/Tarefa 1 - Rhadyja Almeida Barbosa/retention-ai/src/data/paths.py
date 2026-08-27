"""
Definição centralizada de caminhos do projeto.

Utilizar sempre caminhos relativos à raiz do projeto (PROJECT_ROOT),
nunca caminhos absolutos como "C:\\Users\\...".

Uso típico dentro do notebook (notebooks/01_churn_analysis.ipynb):

    import sys
    from pathlib import Path

    # Adiciona a raiz do projeto ao sys.path para permitir "from src..."
    sys.path.append(str(Path.cwd().parent))

    from src.data.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

    df = pd.read_csv(RAW_DATA_DIR / "nome_do_arquivo.csv")
"""

from pathlib import Path

# src/data/paths.py -> src/data -> src -> raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"


def ensure_dirs() -> None:
    """Garante que as pastas de saída existam antes de salvar arquivos."""
    for d in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
