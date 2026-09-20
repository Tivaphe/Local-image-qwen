"""
Catalogue des fichiers "officiels" proposés au téléchargement en un clic.

Tout fichier .gguf / .safetensors déposé manuellement dans les dossiers
`models/diffusion`, `models/text_encoder`, `models/vae`, `models/vision`
est détecté automatiquement, même s'il n'est pas listé ici.
"""

HF = "https://huggingface.co"

# --- Modèle de diffusion (Qwen-Image-2.1, GGUF) ---------------------------
DIFFUSION_MODELS = [
    {
        "id": "qwen_image_2.1-Q2_K.gguf",
        "label": "Q2_K  (2.6 Go) — très léger, qualité réduite",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q2_K.gguf",
        "size_gb": 2.56,
    },
    {
        "id": "qwen_image_2.1-Q3_K.gguf",
        "label": "Q3_K  (3.3 Go) — léger",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q3_K.gguf",
        "size_gb": 3.27,
    },
    {
        "id": "qwen_image_2.1-Q4_K.gguf",
        "label": "Q4_K  (4.2 Go) — recommandé (bon équilibre)",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q4_K.gguf",
        "size_gb": 4.2,
        "recommended": True,
    },
    {
        "id": "qwen_image_2.1-Q5_0.gguf",
        "label": "Q5_0  (5.1 Go)",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q5_0.gguf",
        "size_gb": 5.07,
    },
    {
        "id": "qwen_image_2.1-Q6_K.gguf",
        "label": "Q6_K  (6.0 Go) — haute qualité",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q6_K.gguf",
        "size_gb": 6.0,
    },
    {
        "id": "qwen_image_2.1-Q8_0.gguf",
        "label": "Q8_0  (7.7 Go) — quasi sans perte",
        "url": f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q8_0.gguf",
        "size_gb": 7.69,
    },
]

# --- Encodeur de texte (Qwen3-VL-8B-Instruct, GGUF) -----------------------
TEXT_ENCODERS = [
    {
        "id": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "label": "Qwen3-VL-8B Q4_K_M (5.0 Go) — recommandé",
        "url": f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 5.03,
        "recommended": True,
    },
    {
        "id": "Qwen3VL-8B-Instruct-Q8_0.gguf",
        "label": "Qwen3-VL-8B Q8_0 (8.7 Go)",
        "url": f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q8_0.gguf",
        "size_gb": 8.71,
    },
]

# --- Encodeur de vision (nécessaire uniquement pour l'édition d'image) ----
VISION_ENCODERS = [
    {
        "id": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "label": "mmproj F16 (1.2 Go) — recommandé",
        "url": f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "size_gb": 1.16,
        "recommended": True,
    },
    {
        "id": "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        "label": "mmproj Q8_0 (0.75 Go)",
        "url": f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        "size_gb": 0.75,
    },
]

# --- VAE (spécifique à Qwen-Image-2.1, non interchangeable) ---------------
VAES = [
    {
        "id": "qwen_image_2.1_vae_bf16.safetensors",
        "label": "VAE Qwen-Image-2.1 bf16 (0.68 Go) — obligatoire",
        "url": f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
        "size_gb": 0.68,
        "recommended": True,
    },
]

CATALOG = {
    "diffusion": DIFFUSION_MODELS,
    "text_encoder": TEXT_ENCODERS,
    "vision": VISION_ENCODERS,
    "vae": VAES,
}

MODEL_EXTENSIONS = (".gguf", ".safetensors")
