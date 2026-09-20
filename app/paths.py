from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BIN_DIR = ROOT / "bin"
UPLOADS_DIR = ROOT / "uploads"
CONFIG_FILE = ROOT / "config.json"
STATIC_DIR = ROOT / "app" / "static"

MODEL_SUBDIRS = {
    "diffusion": MODELS_DIR / "diffusion",
    "text_encoder": MODELS_DIR / "text_encoder",
    "vision": MODELS_DIR / "vision",
    "vae": MODELS_DIR / "vae",
    "lora": MODELS_DIR / "lora",
}


def ensure_dirs() -> None:
    for d in [MODELS_DIR, OUTPUTS_DIR, BIN_DIR, UPLOADS_DIR, *MODEL_SUBDIRS.values()]:
        d.mkdir(parents=True, exist_ok=True)
