"""
Familles de modèles supportées et catalogue de fichiers téléchargeables en un clic.

Chaque famille a ses propres dossiers : models/<famille>/{diffusion,text_encoder,vae,vision,lora}
Tout fichier .gguf / .safetensors déposé manuellement dans ces dossiers est détecté
automatiquement, même s'il n'est pas listé ici (variantes, fine-tunes, autres quantifications…).

Pour chaque famille, « pack » décrit l'installation complète en un clic
(génération **et** édition d'image) : quel fichier de diffusion, quel encodeur de
texte, quel VAE et, quand c'est nécessaire, quel encodeur de vision (mmproj).

Références :
  - Qwen‑Image‑2.1 : encodeur de texte Qwen3‑VL‑8B + VAE dédié, mmproj obligatoire
    pour l'édition d'image (--llm_vision).
  - FLUX.2 klein 4B / 9B : distillé, 4 étapes, CFG 1, édition d'image native
    (images de référence passées avec -r, aucun fichier supplémentaire).
"""

HF = "https://huggingface.co"

# Catégories de fichiers, dans l'ordre où elles apparaissent dans l'interface
CATEGORIES = ("diffusion", "text_encoder", "vae", "vision", "controlnet", "pose_detector", "lora")
MODEL_EXTENSIONS = (".gguf", ".safetensors", ".onnx")   # .onnx : détecteur de pose (YOLOv8)


def _f(repo: str, path: str, size_gb: float, label: str, recommended: bool = False,
       dest: str = "", **extra) -> dict:
    """Fiche catalogue : ``id`` = nom du fichier écrit sur le disque (``dest`` si fourni)."""
    return {
        "id": dest or path.split("/")[-1],
        "label": f"{label} ({size_gb:g} Go)" + (" — recommandé" if recommended else ""),
        "url": f"{HF}/{repo}/resolve/main/{path}",
        "size_gb": size_gb,
        "recommended": recommended,
        "repo": repo,
        "path": path,
        **extra,
    }


# ------------------------------------------------------------ Qwen-Image-2.1
QWEN_IMAGE_21 = {
    "id": "qwen_image_2.1",
    "name": "Qwen‑Image‑2.1",
    "description": "Modèle Qwen 7B. Excellente qualité, très bon rendu du texte. ~30 étapes, plus lent.",
    "defaults": {"steps": 30, "cfg_scale": 6.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": True,
    "edit_info": "Édition d'image : images de référence + instruction, avec l'encodeur de vision (mmproj) à télécharger.",
    "supports_controlnet": False,
    "bundled": False,
    "uses_init_image": False,
    "control_modes": [],
    "control_reason": "Qwen‑Image‑2.1 est un modèle « DiT » : stable-diffusion.cpp ne gère ControlNet que pour les modèles UNet (SD 1.5 / SDXL). La famille « SD 1.5 + ControlNet » sert à piloter la pose.",
    "required_selections": ["diffusion", "text_encoder", "vae"],
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
    "pack": {
        "diffusion": "qwen_image_2.1-Q4_K.gguf",
        "text_encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
        "vision": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "vision_optional": False,
        "hint": "Pack complet : diffusion + encodeur Qwen3‑VL‑8B + VAE + mmproj (édition d'image).",
    },
}

# ------------------------------------------------------------ FLUX.2 klein
FLUX2_VAE = [_f("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors", 0.34, "VAE FLUX.2", True)]

FLUX2_KLEIN_4B = {
    "id": "flux2_klein_4b",
    "name": "FLUX.2 klein 4B",
    "description": "Modèle distillé 4B (Black Forest Labs). Très rapide : 4 étapes, CFG 1. Génération et édition.",
    "defaults": {"steps": 4, "cfg_scale": 1.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": False,
    "edit_info": "Édition d'image : native (aucun fichier supplémentaire), les images de référence sont passées directement au modèle.",
    "supports_controlnet": False,
    "bundled": False,
    "uses_init_image": False,
    "control_modes": [],
    "control_reason": "FLUX.2 klein est un modèle « DiT » : stable-diffusion.cpp ne gère ControlNet que pour les modèles UNet (SD 1.5 / SDXL). La famille « SD 1.5 + ControlNet » sert à piloter la pose.",
    "required_selections": ["diffusion", "text_encoder", "vae"],
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
    "pack": {
        "diffusion": "flux-2-klein-4b-Q8_0.gguf",
        "text_encoder": "Qwen3-4B-Q8_0.gguf",
        "vae": "flux2-vae.safetensors",
        "vision": "",
        "vision_optional": True,
        "hint": "Pack complet : diffusion + encodeur Qwen3‑4B + VAE. L'édition fonctionne sans fichier supplémentaire.",
    },
}

FLUX2_KLEIN_9B = {
    "id": "flux2_klein_9b",
    "name": "FLUX.2 klein 9B",
    "description": "Modèle distillé 9B (Black Forest Labs). Meilleure qualité que le 4B, toujours 4 étapes, CFG 1.",
    "defaults": {"steps": 4, "cfg_scale": 1.0, "sampler": "euler", "width": 1024, "height": 1024},
    "edit_requires_vision": False,
    "edit_info": "Édition d'image : native (aucun fichier supplémentaire), les images de référence sont passées directement au modèle.",
    "supports_controlnet": False,
    "bundled": False,
    "uses_init_image": False,
    "control_modes": [],
    "control_reason": "FLUX.2 klein est un modèle « DiT » : stable-diffusion.cpp ne gère ControlNet que pour les modèles UNet (SD 1.5 / SDXL). La famille « SD 1.5 + ControlNet » sert à piloter la pose.",
    "required_selections": ["diffusion", "text_encoder", "vae"],
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
    "pack": {
        "diffusion": "flux-2-klein-9b-Q6_K.gguf",
        "text_encoder": "Qwen3-8B-Q4_K_M.gguf",
        "vae": "flux2-vae.safetensors",
        "vision": "",
        "vision_optional": True,
        "hint": "Pack complet : diffusion + encodeur Qwen3‑8B + VAE. L'édition fonctionne sans fichier supplémentaire.",
    },
}


# ---------------------------------------------- SD 1.5 + ControlNet (pose)
# stable-diffusion.cpp ne prend en charge ControlNet que pour les architectures UNet
# (SD 1.5 / SDXL). C'est donc la seule famille de l'application qui permet d'imposer
# ou d'ajuster la pose des personnages, y compris en repartant d'une photo (img2img).
CONTROLNET_MODELS = [
    _f("lllyasviel/control_v11p_sd15_openpose", "diffusion_pytorch_model.fp16.safetensors", 0.72,
       "OpenPose — squelette des personnages", recommended=True,
       dest="control_v11p_sd15_openpose.safetensors", mode="pose"),
    _f("lllyasviel/control_v11p_sd15_canny", "diffusion_pytorch_model.fp16.safetensors", 0.72,
       "Canny — contours et composition", recommended=True,
       dest="control_v11p_sd15_canny.safetensors", mode="canny"),
]

POSE_MODELS = [
    _f("Xenova/yolov8n-pose", "onnx/model.onnx", 0.014,
       "Détecteur de personnages et de pose (YOLOv8n-pose, ONNX)", recommended=True,
       dest="yolov8n-pose.onnx"),
]

SD15_CONTROL = {
    "id": "sd15_control",
    "name": "SD 1.5 + ControlNet (pose)",
    "description": "Seul modèle du moteur à accepter ControlNet : imposez ou ajustez la pose des personnages, "
                   "à partir d'une photo (pose détectée automatiquement, retouchable) ou d'un squelette de référence. "
                   "Qualité plus modeste que Qwen/FLUX.2 : ~25 étapes, CFG 7.",
    "defaults": {"steps": 25, "cfg_scale": 7.0, "sampler": "euler_a", "width": 512, "height": 512},
    "edit_requires_vision": False,
    "edit_info": "Édition d'image : repartez de la photo (img2img) et gardez la pose détectée — le personnage garde "
                 "sa posture pendant que le prompt change l'apparence, la tenue ou le décor.",
    "supports_controlnet": True,
    "control_reason": "",
    "control_modes": ["pose", "canny"],
    "uses_init_image": True,
    "bundled": True,
    "required_selections": ["diffusion"],
    "diffusion": [
        _f("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", 4.27,
           "SD 1.5 — checkpoint complet (UNet + CLIP + VAE)", recommended=True),
    ],
    "text_encoder": [],
    "vision": [],
    "vae": [],
    "controlnet": CONTROLNET_MODELS,
    "pose_detector": POSE_MODELS,
    "pack": {
        "diffusion": "v1-5-pruned-emaonly.safetensors",
        "text_encoder": "",
        "vae": "",
        "vision": "",
        "controlnet": "control_v11p_sd15_openpose.safetensors",
        "pose_detector": "yolov8n-pose.onnx",
        "hint": "Pack complet : SD 1.5 + ControlNet OpenPose + détecteur de pose automatique.",
    },
}

FAMILIES = {f["id"]: f for f in (QWEN_IMAGE_21, FLUX2_KLEIN_4B, FLUX2_KLEIN_9B, SD15_CONTROL)}
DEFAULT_FAMILY = "qwen_image_2.1"


# ------------------------------------------------------------------ outils
def entry(family: str, category: str, file_id: str) -> dict | None:
    """Fiche catalogue d'un fichier précis (ou None)."""
    fam = FAMILIES.get(family)
    if not fam or not file_id:
        return None
    return next((e for e in fam.get(category, []) if e["id"] == file_id), None)


def categories_for(family: str, include_vision: bool = True) -> list[str]:
    fam = FAMILIES[family]
    cats = ["diffusion", "text_encoder", "vae"]
    if include_vision and fam.get("vision"):
        cats.append("vision")
    return cats


def install_plan(family: str, diffusion_id: str = "", text_encoder_id: str = "", include_vision: bool = True) -> list[dict]:
    """
    Liste ordonnée des fichiers à télécharger pour installer une famille complète
    (génération + édition si include_vision). Les identifiants vides retombent sur
    les fichiers recommandés du pack.
    """
    fam = FAMILIES[family]
    pack = fam.get("pack", {})
    wanted = {
        "diffusion": diffusion_id or pack.get("diffusion", ""),
        "text_encoder": text_encoder_id or pack.get("text_encoder", ""),
        "vae": pack.get("vae", ""),
        "vision": pack.get("vision", "") if (include_vision and fam.get("edit_requires_vision")) else "",
        # ControlNet : le modèle de contrôle et le détecteur de pose font partie du pack
        "controlnet": pack.get("controlnet", "") if fam.get("controlnet") else "",
        "pose_detector": pack.get("pose_detector", "") if fam.get("controlnet") else "",
    }
    plan: list[dict] = []
    for cat in ("diffusion", "text_encoder", "vae", "vision", "controlnet", "pose_detector"):
        fid = wanted.get(cat, "")
        if not fid:
            continue
        e = entry(family, cat, fid)
        if not e:
            # identifiant inconnu : on prend le premier recommandé de la catégorie
            e = next((x for x in fam.get(cat, []) if x["recommended"]), None) or next(iter(fam.get(cat, [])), None)
        if e:
            plan.append({**e, "category": cat})
    return plan


def controlnet_files(families: dict | None = None) -> list[dict]:
    """Tous les modèles ControlNet du catalogue (dédupliqués par nom de fichier)."""
    seen: dict[str, dict] = {}
    for fam in (families or FAMILIES).values():
        for e in fam.get("controlnet", []):
            seen.setdefault(e["id"], e)
    return list(seen.values())


def pose_models() -> list[dict]:
    seen: dict[str, dict] = {}
    for fam in FAMILIES.values():
        for e in fam.get("pose_detector", []):
            seen.setdefault(e["id"], e)
    return list(seen.values())


def family_for_control(kind: str = "") -> str:
    """Famille à utiliser pour ControlNet (celle qui gère ``kind`` : « pose » ou « canny »)."""
    for fid, fam in FAMILIES.items():
        if not fam.get("controlnet"):
            continue
        if not kind or kind in (fam.get("control_modes") or []):
            return fid
    return DEFAULT_FAMILY


def category_of_file(family: str, name: str) -> str | None:
    """Catégorie d'un fichier présent sur le disque (par son nom)."""
    fam = FAMILIES.get(family) or {}
    for cat in CATEGORIES:
        if any(e["id"] == name for e in fam.get(cat, []) if isinstance(e, dict)):
            return cat
    return None


def pack_total_gb(family: str, diffusion_id: str = "", text_encoder_id: str = "", include_vision: bool = True) -> float:
    return round(sum(e["size_gb"] for e in install_plan(family, diffusion_id, text_encoder_id, include_vision)), 2)
