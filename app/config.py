import json
from typing import Any

from .paths import CONFIG_FILE

from .catalog import DEFAULT_FAMILY, FAMILIES

DEFAULTS: dict[str, Any] = {
    # Famille active + fichiers sélectionnés par famille
    "family": DEFAULT_FAMILY,
    "selections": {fam: {"diffusion": "", "text_encoder": "", "vision": "", "vae": ""} for fam in FAMILIES},
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
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            sels = data.pop("selections", {}) or {}
            cfg.update(data)
            for fam, s in sels.items():
                if fam in cfg["selections"] and isinstance(s, dict):
                    cfg["selections"][fam].update(s)
        except Exception:
            pass
    if cfg.get("family") not in FAMILIES:
        cfg["family"] = DEFAULT_FAMILY
    return cfg


def save(cfg: dict[str, Any]) -> None:
    merged = load()
    for k, v in cfg.items():
        if k == "selections" and isinstance(v, dict):
            for fam, s in v.items():
                if fam in merged["selections"] and isinstance(s, dict):
                    merged["selections"][fam].update(s)
        elif k in DEFAULTS:
            merged[k] = v
    CONFIG_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
