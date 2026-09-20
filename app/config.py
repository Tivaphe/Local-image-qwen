import json
from typing import Any

from .paths import CONFIG_FILE

DEFAULTS: dict[str, Any] = {
    # Fichiers sélectionnés (nom de fichier dans le sous-dossier models/*)
    "diffusion_model": "",
    "text_encoder": "",
    "vision_encoder": "",
    "vae": "",
    # Paramètres de génération par défaut
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg_scale": 6.0,
    "sampler": "euler",
    "seed": -1,
    "negative_prompt": "",
    # Performance
    "offload_to_cpu": True,
    "flash_attention": True,
    "vae_tiling": False,
    "threads": -1,
    "extra_args": "",
}


def load() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save(cfg: dict[str, Any]) -> None:
    merged = load()
    for k, v in cfg.items():
        if k in DEFAULTS:
            merged[k] = v
    CONFIG_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
