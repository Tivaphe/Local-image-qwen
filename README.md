# Local Image Qwen

Interface web **simple, 100 % locale** pour générer et éditer des images au format **GGUF**,
propulsée par [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Les trois modèles se téléchargent **en un clic depuis l'interface** (onglet **Modèles**), avec les
fichiers nécessaires à la génération **et** à l'édition d'image :

| Modèle | Points forts | Réglages par défaut | Pack complet |
|---|---|---|---|
| [Qwen‑Image‑2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (7B) | Qualité maximale, texte dans l'image | 30 étapes, CFG 6 | ≈ 11,1 Go<br>diffusion Q4_K + Qwen3‑VL‑8B Q4_K_M + VAE + mmproj (édition) |
| [FLUX.2 klein 9B](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) | Rapide, meilleure qualité que le 4B | 4 étapes, CFG 1 | ≈ 13,2 Go<br>diffusion Q6_K + Qwen3‑8B Q4_K_M + VAE |
| [FLUX.2 klein 4B](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) | Très rapide, léger | 4 étapes, CFG 1 | ≈ 8,9 Go<br>diffusion Q8_0 + Qwen3‑4B Q8_0 + VAE |

- Zéro dépendance lourde : pas de PyTorch, pas de CUDA toolkit à installer.
- Le moteur (binaire précompilé) et les modèles se téléchargent **en un clic** depuis l'interface.
- **Édition d'image** : image(s) de référence + instruction dans le prompt.
  Qwen‑Image‑2.1 utilise l'encodeur de vision (`mmproj`, inclus dans son pack) ;
  FLUX.2 klein édite **sans fichier supplémentaire**.
- Choix de la quantification (Q2_K → Q8_0/BF16) pour chaque modèle, taille affichée avant téléchargement.
- Téléchargements repris après coupure, progression **fichier par fichier**, annulation possible.
- Galerie avec seed, réglages, réutilisation en un clic.
- Fonctionne sous **Windows / Linux / macOS**, GPU NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel (Vulkan) ou CPU.
- Tout fichier `.gguf` / `.safetensors` déposé dans `models/<modèle>/<catégorie>/` est utilisable :
  vous choisissez librement votre variante (fine‑tune, autre quantification…) via un bouton radio.

## Prérequis

- [Python 3.10+](https://www.python.org/downloads/) (Windows : cocher **"Add python.exe to PATH"**).
- ~12 Go d'espace disque pour l'installation recommandée (jusqu'à ~25 Go si vous prenez les trois modèles).
- 8 Go de VRAM minimum conseillés (16 Go confortable) ; le déchargement en RAM est activé par défaut.

## Démarrage rapide

```text
Windows : double-cliquer sur  start.bat
Linux/macOS :                 ./start.sh
```

Le navigateur s'ouvre sur `http://127.0.0.1:7860`. Ensuite :

1. Onglet **Modèles** → bloc **1. Moteur** → « Installer » (la variante est détectée automatiquement :
   `cuda` pour NVIDIA, sinon `vulkan`).
2. Toujours dans **Modèles**, pour le modèle voulu : choisir la **quantification** puis cliquer sur
   **« ⬇ Tout télécharger (~11 Go) »**. Le pack (génération + édition) est téléchargé à la suite ;
   la progression s'affiche fichier par fichier et le modèle devient actif à la fin.
3. Onglet **Générer** : choisir le modèle, écrire un prompt, cliquer **Générer**.
   Pour éditer une image : passer en mode **✏️ Éditer une image**, ajouter les images de référence
   et décrire la modification dans le prompt.

## Générer ou éditer (onglet Générer)

Deux modes, en haut du formulaire :

- **🎨 Générer une image** : texte → image.
- **✏️ Éditer une image** : les images de référence passent en tête du formulaire (bordure violette),
  le prompt devient l'instruction de modification (`ex : remplace le fond par une plage`), et le bouton
  devient « Appliquer la modification ». Le bouton vérifie qu'une image de référence est bien présente
  et, pour Qwen‑Image‑2.1, que l'encodeur de vision est installé — sinon il propose de le télécharger
  en un clic.
  Qwen‑Image‑2.1 utilise `--llm_vision` (mmproj) ; FLUX.2 klein édite nativement avec `-r`.

## Télécharger les modèles (onglet Modèles)

- **Installation en un clic** : pour un modèle, le pack contient le fichier de diffusion choisi,
  l'encodeur de texte, le VAE et — pour Qwen‑Image‑2.1 — l'encodeur de vision (`mmproj`) indispensable
  à l'édition. Les fichiers déjà présents sont simplement ignorés (« déjà présent »).
- **Édition incluse d'office** : la case « Télécharger aussi l'encodeur de vision (mmproj) » est cochée
  par défaut pour Qwen‑Image‑2.1. FLUX.2 klein n'a besoin d'aucun fichier supplémentaire pour éditer.
- **Fichiers à l'unité** : dépliez « Fichiers installés, autres variantes et URL directe » pour
  télécharger une variante précise, supprimer un fichier, coller une **URL directe**
  (`.gguf` / `.safetensors`) ou choisir le fichier utilisé (bouton radio).
- **Reprise et annulation** : un téléchargement interrompu reprend où il s'est arrêté
  (fichiers `.part`) ; le bouton « annuler » de la tâche ou « Tout annuler » interrompt la file.
- **Sélection automatique** : après un téléchargement, le fichier devient le fichier actif du modèle
  (sauf si vous en avez déjà choisi un autre).
- **Reprise du modèle existant** : si vous avez déjà une variante installée, le pack s'aligne dessus
  et ne télécharge que les fichiers réellement manquants.

## Dossiers

```
bin/                                  moteur sd-cli (téléchargé automatiquement)
models/qwen_image_2.1/diffusion/      qwen_image_2.1-*.gguf   (n'importe quelle variante/quantification)
models/qwen_image_2.1/text_encoder/   Qwen3VL-8B-Instruct-*.gguf
models/qwen_image_2.1/vae/            qwen_image_2.1_vae_bf16.safetensors
models/qwen_image_2.1/vision/         mmproj-Qwen3VL-8B-Instruct-*.gguf (édition d'image)
models/flux2_klein_4b/diffusion/      flux-2-klein-4b-*.gguf
models/flux2_klein_4b/text_encoder/   Qwen3-4B-*.gguf
models/flux2_klein_4b/vae/            flux2-vae.safetensors
models/flux2_klein_9b/…               idem avec flux-2-klein-9b-*.gguf et Qwen3-8B-*.gguf
models/<modèle>/lora/                 LoRA optionnels (syntaxe <lora:nom:0.8> dans le prompt)
outputs/                              images générées + .json de métadonnées
```

Une URL directe peut aussi être collée dans le champ prévu pour télécharger n'importe quel fichier.
Le dossier et les chemins utilisés sont rappelés dans **Configuration**.

## Réglages conseillés

| Machine | Modèle diffusion | Encodeur texte | Options |
|---|---|---|---|
| GPU 16 Go (ex. RTX 2000 Ada / 4060 Ti 16G) | Q4_K ou Q6_K | Q4_K_M | offload CPU ✔, flash attention ✔ |
| GPU 8–12 Go | Q4_K ou Q3_K | Q4_K_M | offload CPU ✔, flash attention ✔, VAE tiling ✔ |
| GPU 6 Go / CPU seul | Q2_K–Q3_K, 768×768 | Q4_K_M | offload CPU ✔, VAE tiling ✔, patience… |
| GPU ≥ 24 Go | Q8_0 | Q8_0 | offload CPU ✘ |

Les réglages par défaut (étapes, CFG, sampler) sont appliqués automatiquement à chaque changement de modèle.
FLUX.2 klein est distillé : gardez CFG = 1 et 4 étapes (jusqu'à 8 pour un peu plus de finesse).
Pour les deux modèles FLUX.2 klein, l'édition fonctionne directement : passez les images de référence
et décrivez la modification dans le prompt.
Dimensions toujours arrondies au multiple de 32.

## Tests

```bash
pip install -r requirements-dev.txt     # pytest + httpx
python -m pytest -q                     # 24 tests : catalogue, téléchargements, API, commande sd-cli

npm install jsdom                       # une seule fois, pour les tests d'interface
node tests/ui_render.mjs                # rejoue app.js sur un vrai /api/status (serveur lancé)
node tests/ui_render.mjs status.json    # …ou sur une réponse enregistrée
```

`tests/ui_render.mjs` vérifie que l'onglet Modèles affiche bien les trois modèles, le bouton
« Tout télécharger », la présence de l'édition d'image, et que le clic déclenche exactement
`POST /api/install` (avec la quantification choisie), puis le suivi fichier par fichier et l'annulation.

## Accès depuis un autre appareil du réseau local

```
python run.py --host 0.0.0.0 --port 7860
```

## Problèmes fréquents

- **« Moteur non installé »** : onglet Modèles → bloc 1 → Installer. Si GitHub est inaccessible, téléchargez
  manuellement une [release](https://github.com/leejet/stable-diffusion.cpp/releases) et dézippez-la dans `bin/`
  (sous Windows avec NVIDIA prenez `sd-…-win-cuda12-x64.zip` **et** `cudart-sd-bin-win-cu12-x64.zip`).
- **Le téléchargement d'un fichier échoue** : le détail (HTTP 403/404, coupure réseau, URL) est affiché
  sous la tâche ; relancez, le téléchargement reprend là où il s'était arrêté. Un miroir peut être utilisé
  via le champ **URL directe**.
- **Manque de mémoire (CUDA out of memory)** : activer *VAE tiling*, réduire la résolution, prendre une
  quantification plus légère (Q3_K, Q2_K) ou ajouter `--max-vram 14` dans *Arguments supplémentaires*.
- **Très lent** : vérifier que la variante `cuda` (et non `cpu`/`vulkan`) est installée, et que *flash attention* est cochée.
- **Édition d'image refusée (Qwen‑Image‑2.1)** : téléchargez l'encodeur de vision (mmproj) —
  bouton dédié dans l'onglet Générer ou case à cocher du pack dans l'onglet Modèles.
- **Une nouveauté de l'interface n'apparaît pas** : rechargez avec **Ctrl+F5** (les URL des scripts sont
  horodatées, mais un vieux cache de navigateur peut persister).
- Le journal complet de `sd-cli` est visible sous la barre de progression.

## Licence

Code de cette application : MIT. Les modèles ont leur propre licence (Qwen Research License pour Qwen‑Image‑2.1,
FLUX Non‑Commercial License pour FLUX.2 klein 9B, Apache‑2.0 pour FLUX.2 klein 4B).
