"""
Familles de modèles supportées et catalogue de fichiers téléchargeables en un clic.

Chaque famille a ses propres dossiers : models/<famille>/{diffusion,text_encoder,vae,vision,lora}
Tout fichier .gguf / .safetensors déposé manuellement dans ces dossiers est détecté
automatiquement, même s'il n'est pas listé ici (variantes, fine-tunes, autres quantifications…).
"""

HF = "https://huggingface.co"


def _f(repo: str, path: str, size_gb: float, label: str, recommended: bool = False) -> dict:
    return {
        "id": path.split("/")[-1],
        "label": f"{label} ({size_gb:g} Go)" + (" — recommandé" if recommended else ""),
        "url": f"{HF}/{repo}/resolve/main/{path}",
        "size_gb": size_gb,
        "recommended": recommended,
    }


# ------------------------------------------------------------ Qwen-Image-2.1
QWEN_IMAGE_21 = {
    "id": "qwen_image_2.1",
    "name": "Qwen‑Image‑2.1",
    "description": "Modèle Qwen 7B. Excellente qualité, très bon rendu du texte. ~30 étapes, plus lent.",
    "defaults": {"steps": 30, "cfg_scale": 6.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": True,
    "diffusion": [
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q2_K.gguf", 2.56, "Q2_K — très léger, qualité réduite"),
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q3_K.gguf", 3.27, "Q3_K — léger"),
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q4_K.gguf", 4.2, "Q4_K — bon équilibre", True),
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q5_0.gguf", 5.07, "Q5_0"),
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q6_K.gguf", 6.0, "Q6_K — haute qualité"),
        _f("leejet/Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q8_0.gguf", 7.69, "Q8_0 — quasi sans perte"),
    ],
    "text_encoder": [
        _f("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5.03, "Qwen3‑VL‑8B Q4_K_M", True),
        _f("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q8_0.gguf", 8.71, "Qwen3‑VL‑8B Q8_0"),
    ],
    "vision": [
        _f("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-F16.gguf", 1.16, "mmproj F16", True),
        _f("Qwen/Qwen3-VL-8B-Instruct-GGUF", "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", 0.75, "mmproj Q8_0"),
    ],
    "vae": [
        _f("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68, "VAE Qwen‑Image‑2.1 bf16", True),
    ],
}

# ------------------------------------------------------------ FLUX.2 klein
FLUX2_VAE = [_f("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors", 0.34, "VAE FLUX.2", True)]

FLUX2_KLEIN_4B = {
    "id": "flux2_klein_4b",
    "name": "FLUX.2 klein 4B",
    "description": "Modèle distillé 4B (Black Forest Labs). Très rapide : 4 étapes, CFG 1. Génération et édition.",
    "defaults": {"steps": 4, "cfg_scale": 1.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": False,
    "diffusion": [
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q3_K_M.gguf", 2.12, "Q3_K_M — léger"),
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf", 2.6, "Q4_K_M"),
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q5_K_M.gguf", 3.07, "Q5_K_M"),
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q6_K.gguf", 3.41, "Q6_K"),
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf", 4.3, "Q8_0 — quasi sans perte", True),
        _f("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-BF16.gguf", 7.75, "BF16 — original"),
    ],
    "text_encoder": [
        _f("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", 2.5, "Qwen3‑4B Q4_K_M"),
        _f("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q6_K.gguf", 3.31, "Qwen3‑4B Q6_K"),
        _f("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", 4.28, "Qwen3‑4B Q8_0", True),
    ],
    "vision": [],
    "vae": FLUX2_VAE,
}

FLUX2_KLEIN_9B = {
    "id": "flux2_klein_9b",
    "name": "FLUX.2 klein 9B",
    "description": "Modèle distillé 9B (Black Forest Labs). Meilleure qualité que le 4B, toujours 4 étapes, CFG 1.",
    "defaults": {"steps": 4, "cfg_scale": 1.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": False,
    "diffusion": [
        _f("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q3_K_M.gguf", 4.77, "Q3_K_M — léger"),
        _f("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf", 5.91, "Q4_K_M"),
        _f("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q5_K_M.gguf", 7.02, "Q5_K_M"),
        _f("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q6_K.gguf", 7.87, "Q6_K — bon équilibre", True),
        _f("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf", 9.98, "Q8_0 — quasi sans perte"),
    ],
    "text_encoder": [
        _f("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", 5.03, "Qwen3‑8B Q4_K_M", True),
        _f("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q6_K.gguf", 6.73, "Qwen3‑8B Q6_K"),
        _f("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", 8.71, "Qwen3‑8B Q8_0"),
    ],
    "vision": [],
    "vae": FLUX2_VAE,
}

FAMILIES = {f["id"]: f for f in (QWEN_IMAGE_21, FLUX2_KLEIN_4B, FLUX2_KLEIN_9B)}
DEFAULT_FAMILY = "qwen_image_2.1"
CATEGORIES = ("diffusion", "text_encoder", "vae", "vision", "lora")
MODEL_EXTENSIONS = (".gguf", ".safetensors")


def recommended_pack(family: str) -> list[dict]:
    fam = FAMILIES.get(family)
    if not fam:
        return []
    items = []
    for cat in ("diffusion", "text_encoder", "vae", "vision"):
        if cat == "vision" and not fam.get("edit_requires_vision"):
            continue
        cat_items = fam.get(cat, [])
        rec = next((x for x in cat_items if x.get("recommended")), None)
        if rec:
            items.append({"category": cat, **rec})
    return items


def recommended_size_gb(family: str) -> float:
    return round(sum(item["size_gb"] for item in recommended_pack(family)), 2)
