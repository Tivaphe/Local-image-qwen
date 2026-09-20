from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BIN_DIR = ROOT / "bin"
UPLOADS_DIR = ROOT / "uploads"
CONFIG_FILE = ROOT / "config.json"
STATIC_DIR = ROOT / "app" / "static"

from .catalog import CATEGORIES, FAMILIES


def model_dir(family: str, category: str) -> Path:
    if family not in FAMILIES or category not in CATEGORIES:
        raise ValueError("famille ou catégorie inconnue")
    return MODELS_DIR / family / category


def ensure_dirs() -> None:
    for d in [MODELS_DIR, OUTPUTS_DIR, BIN_DIR, UPLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for fam in FAMILIES:
        for cat in CATEGORIES:
            (MODELS_DIR / fam / cat).mkdir(parents=True, exist_ok=True)
