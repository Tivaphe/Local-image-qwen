# Local Image Qwen

Interface web **simple, 100 % locale** pour générer et éditer des images au format **GGUF**,
propulsée par [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Modèles pris en charge (sélectionnables dans un menu déroulant) :

| Modèle | Points forts | Réglages par défaut |
|---|---|---|
| [Qwen‑Image‑2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (7B) | Qualité maximale, texte dans l'image | 30 étapes, CFG 6 |
| [FLUX.2 klein 4B](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) | Très rapide, léger | 4 étapes, CFG 1 |
| [FLUX.2 klein 9B](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) | Rapide, meilleure qualité que le 4B | 4 étapes, CFG 1 |

Les trois font de la génération **et** de l'édition d'image (image(s) de référence + instruction).

- Zéro dépendance lourde : pas de PyTorch, pas de CUDA toolkit à installer.
- Le moteur (binaire précompilé) et les modèles se téléchargent **en un clic depuis l'interface**.
- Texte → image et **édition d'image** (glisser-déposer de références, conservation automatique ou manuelle du format et du ratio de l'image source, choix de ratios étendus 1:1, 4:3, 3:2, 16:9, 21:9…, verrouillage de ratio et inversion rapide).
- Galerie avec seed, réglages, réutilisation en un clic.
- Fonctionne sous **Windows / Linux / macOS**, GPU NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel (Vulkan) ou CPU.
- Tout fichier `.gguf` déposé dans `models/<modèle>/diffusion/` est utilisable : vous choisissez librement votre variante (fine‑tune, autre quantification, etc.) via un bouton radio.

## Prérequis

- [Python 3.10+](https://www.python.org/downloads/) (Windows : cocher **"Add python.exe to PATH"**).
- ~12 Go d'espace disque pour l'installation recommandée.
- 8 Go de VRAM minimum conseillés (16 Go confortable) ; le déchargement en RAM est activé par défaut.

## Démarrage rapide

```text
Windows : double-cliquer sur  start.bat
Linux/macOS :                 ./start.sh
```

Le navigateur s'ouvre sur `http://127.0.0.1:7860`. Puis, dans l'onglet **Configuration** :

1. **Moteur** → « Installer » (la variante est détectée automatiquement : `cuda` pour NVIDIA, sinon `vulkan`).
2. **Modèles** → déplier le modèle voulu et cliquer « Télécharger » sur les fichiers *recommandés*
   (diffusion + encodeur de texte + VAE ; pour Qwen‑Image‑2.1 ajoutez le mmproj si vous voulez l'édition d'image).
3. Revenir dans **Générer**, choisir le modèle dans le menu, écrire un prompt, cliquer **Générer**.

## Dossiers

```
bin/                                  moteur sd-cli (téléchargé automatiquement)
models/qwen_image_2.1/diffusion/      qwen_image_2.1-*.gguf   (n'importe quelle variante/quantification)
models/qwen_image_2.1/text_encoder/   Qwen3-VL-8B-Instruct *.gguf
models/qwen_image_2.1/vae/            qwen_image_2.1_vae_bf16.safetensors
models/qwen_image_2.1/vision/         mmproj-Qwen3VL-8B-Instruct-*.gguf (édition d'image)
models/flux2_klein_4b/diffusion/      flux-2-klein-4b-*.gguf
models/flux2_klein_4b/text_encoder/   Qwen3-4B-*.gguf
models/flux2_klein_4b/vae/            flux2-vae.safetensors
models/flux2_klein_9b/…               idem avec flux-2-klein-9b-*.gguf et Qwen3-8B-*.gguf
models/<modèle>/lora/                 LoRA optionnels (syntaxe <lora:nom:0.8> dans le prompt)
outputs/                              images générées + .json de métadonnées
```

Vous pouvez copier manuellement des fichiers dans ces dossiers : ils sont détectés
automatiquement et sélectionnables via un bouton radio dans **Configuration**.
Une URL directe peut aussi être collée dans le champ prévu pour télécharger n'importe quel fichier.

## Réglages conseillés

| Machine | Modèle diffusion | Encodeur texte | Options |
|---|---|---|---|
| GPU 16 Go (ex. RTX 2000 Ada / 4060 Ti 16G) | Q4_K ou Q6_K | Q4_K_M | offload CPU ✔, flash attention ✔ |
| GPU 8–12 Go | Q4_K ou Q3_K | Q4_K_M | offload CPU ✔, flash attention ✔, VAE tiling ✔ |
| GPU 6 Go / CPU seul | Q2_K–Q3_K, 768×768 | Q4_K_M | offload CPU ✔, VAE tiling ✔, patience… |
| GPU ≥ 24 Go | Q8_0 | Q8_0 | offload CPU ✘ |

Les réglages par défaut (étapes, CFG, sampler) sont appliqués automatiquement à chaque changement de modèle.
FLUX.2 klein est distillé : gardez CFG = 1 et 4 étapes (jusqu'à 8 pour un peu plus de finesse).
Dimensions toujours arrondies au multiple de 32.

Tailles indicatives des installations recommandées : Qwen‑Image‑2.1 ≈ 11 Go · FLUX.2 klein 4B ≈ 9 Go · FLUX.2 klein 9B ≈ 13 Go.

## Accès depuis un autre appareil du réseau local

```
python run.py --host 0.0.0.0 --port 7860
```

## Problèmes fréquents

- **« Moteur non installé »** : onglet Configuration → Installer. Si GitHub est inaccessible, téléchargez manuellement une
  [release](https://github.com/leejet/stable-diffusion.cpp/releases) et dézippez-la dans `bin/`
  (sous Windows avec NVIDIA prenez `sd-…-win-cuda12-x64.zip` **et** `cudart-sd-bin-win-cu12-x64.zip`).
- **Manque de mémoire (CUDA out of memory)** : activer *VAE tiling*, réduire la résolution, ou ajouter `--max-vram 14` dans *Arguments supplémentaires*.
- **Très lent** : vérifier que la variante `cuda` (et non `cpu`/`vulkan`) est installée, et que *flash attention* est cochée.
- **Édition d'image refusée** : téléchargez l'encodeur de vision (mmproj).
- Le journal complet de `sd-cli` est visible sous la barre de progression.

## Licence

Code de cette application : MIT. Les modèles ont leur propre licence (Qwen Research License pour Qwen‑Image‑2.1).
