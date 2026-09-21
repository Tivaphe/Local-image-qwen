# Local Image Qwen

Interface web **simple, 100 % locale** pour générer et éditer des images au format **GGUF**,
propulsée par [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Les modèles se téléchargent **en un clic depuis l'interface** (onglet **Modèles**), avec les
fichiers nécessaires à la génération **et** à l'édition d'image :

| Modèle | Points forts | Réglages par défaut | Pack complet |
|---|---|---|---|
| [Qwen‑Image‑2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (7B) | Qualité maximale, texte dans l'image | 30 étapes, CFG 6 | ≈ 11,1 Go<br>diffusion Q4_K + Qwen3‑VL‑8B Q4_K_M + VAE + mmproj (édition) |
| [FLUX.2 klein 9B](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF) | Rapide, meilleure qualité que le 4B | 4 étapes, CFG 1 | ≈ 13,2 Go<br>diffusion Q6_K + Qwen3‑8B Q4_K_M + VAE |
| [FLUX.2 klein 4B](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF) | Très rapide, léger | 4 étapes, CFG 1 | ≈ 8,9 Go<br>diffusion Q8_0 + Qwen3‑4B Q8_0 + VAE |
| **SD 1.5 + ControlNet (pose)** | **Pilotage de la pose des personnages**, détection automatique, retouche d'une photo | 25 étapes, CFG 7 | ≈ 5,0 Go<br>SD 1.5 + ControlNet OpenPose + ControlNet Canny + détecteur de pose (13 Mo) |

- Zéro dépendance lourde : pas de PyTorch, pas de CUDA toolkit à installer.
- Le moteur (binaire précompilé) et les modèles se téléchargent **en un clic** depuis l'interface.
- **Édition d'image** : image(s) de référence + instruction dans le prompt.
  Qwen‑Image‑2.1 utilise l'encodeur de vision (`mmproj`, inclus dans son pack) ;
  FLUX.2 klein édite **sans fichier supplémentaire**.
- **ControlNet** (famille *SD 1.5 + ControlNet* — les modèles Qwen/FLUX, de type « DiT », ne le gèrent pas
  dans stable-diffusion.cpp) : détection **automatique** des personnages sur une photo, squelette de pose
  modifiable à la souris, contours (Canny), et génération ou **modification d'image** (img2img) qui
  conserve la posture.
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
- Pour ControlNet : `numpy` et `opencv-python-headless` (installés avec `pip install -r requirements.txt`) ;
  `onnxruntime` est optionnel (accélère la détection, repli automatique sur OpenCV DNN).

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
4. Pour **imposer ou corriger une pose** : installez la famille **SD 1.5 + ControlNet** (≈ 5 Go),
   puis utilisez la carte « 🧍 Personnages, pose et composition » (voir ci‑dessous).

## Générer ou éditer (onglet Générer)

Deux modes, en haut du formulaire :

- **🎨 Générer une image** : texte → image.
- **✏️ Éditer une image** : les images de référence passent en tête du formulaire (bordure violette),
  le prompt devient l'instruction de modification (`ex : remplace le fond par une plage`), et le bouton
  devient « Appliquer la modification ». Le bouton vérifie qu'une image de référence est bien présente
  et, pour Qwen‑Image‑2.1, que l'encodeur de vision est installé — sinon il propose de le télécharger
  en un clic.
  Qwen‑Image‑2.1 utilise `--llm_vision` (mmproj) ; FLUX.2 klein édite nativement avec `-r`.

## Personnages, pose et composition — ControlNet (onglet Générer)

La carte **🧍 Personnages, pose et composition** permet de **placer ou corriger la pose** d'un ou
plusieurs personnages, et de **conserver cette pose pendant la génération ou la modification d'image**.

1. Modèle : choisissez **« SD 1.5 + ControlNet (pose) »** (onglet Modèles → ⬇ Tout télécharger, ≈ 5 Go ;
   le pack contient SD 1.5, les deux modèles ControlNet — OpenPose et Canny — et le détecteur de
   personnages `yolov8n-pose.onnx`). Les modèles Qwen‑Image‑2.1 et FLUX.2 klein sont des modèles
   « DiT » : stable-diffusion.cpp ne sait pas leur appliquer ControlNet, l'interface le signale et
   propose la bascule vers cette famille en un clic.
2. Type de contrôle :
   - **Pose des personnages** : cliquez sur **🔍 Détecter les personnages** → chaque personne de la
     photo est trouvée automatiquement (YOLOv8‑pose, 17 points par personne) et un squelette est dessiné.
     Le détecteur est local, aucune connexion n'est nécessaire après le téléchargement (~13 Mo).
   - **Contours / composition (Canny)** : pour suivre le contour d'un objet, d'un décor ou d'un logo
     plutôt qu'une posture.
3. **Ajustez la pose** : dans l'éditeur, faites glisser une articulation (points jaunes = déplacés,
   « ↺ Réinitialiser » revient à la détection). L'aperçu « Image de contrôle envoyée au modèle » est
   mis à jour, et c'est **cette** image qui part dans `--control-image`.
4. **Force du contrôle** : 0,3 = simple suggestion, 0,8–1,0 = pose respectée fermement. Au‑delà de 1,0,
   le squelette peut figer le rendu (tenue, décor).
5. **Modifier une photo en gardant la pose** : cochez **« Repartir de la photo (img2img) »** (force 0,45
   par défaut = la photo reste très présente, 0,7 = plus de liberté) et décrivez la transformation :
   `« la même personne, en armure dorée, dans un décor de neige »`. La génération part de votre photo,
   guidée par le prompt **et** par le squelette.
6. Sources possibles pour la détection : l'image de référence ajoutée en mode Édition, un autre fichier
   (`Autre fichier…`), ou la dernière image générée (`Dernière image générée`) — pratique pour
   « corriger » une pose obtenue par hasard.

Sous le capot, la commande envoyée à `sd-cli` est
`--control-net <modèle> --control-image <squelette|contours> --control-strength <force>`
(+ `-i <photo> --strength <force img2img>` en modification d'image).

## Mannequin articulé — poser un personnage sans photo (onglet Générer)

Quand aucune image de référence ne convient, la carte **🤸 Mannequin articulé** permet de **poser un
personnage à la main** puis de s'en servir comme référence de pose. Tout est dessiné par l'application :
**aucun fichier à télécharger**, aucune connexion (le module fait ~1100 lignes de JavaScript et une
projection 3D maison).

1. **Poser** : faites glisser une articulation sur le canvas. Les os ont une **longueur fixe** : tirer le
   poignet plie le coude (IK 2 os), tirer la cheville plie le genou — impossible d'étirer un membre.
   - glisser dans le vide : tourner autour du personnage · **molette** : zoomer · **flèches** du clavier :
     déplacer finement le point sélectionné · curseur **« Avancer / reculer »** : placer le membre en
     profondeur (avant/arrière).
   - seuls **poignet** et **cheville** déclenchent l'IK ; la **main** pivote autour du poignet, le **pied**
     reste rigide (la pointe levée fait tourner le talon) et les points du visage suivent la tête.
   - **↶ Annuler**, **⇄ Miroir**, **Face / Profil**, **⤢ Recadrer**, poses types (debout, marche, course,
     assis, accroupi, danse, main levée).
2. **Dimensions** (dépliant « 📐 Tableau des dimensions ») : **une ligne par segment du corps**
   (bassin, colonne, poitrine, cou, tête, épaules, bras, avant-bras, main, cuisse, jambe, pied…), avec sa
   **longueur** et son **épaisseur** en centimètres. Modifiez une valeur : le pantin garde **exactement**
   ces dimensions, quelle que soit la pose (l'épaisseur pilote les capsules et le tronc). Le bouton ⇄
   recopie une valeur sur le côté opposé (ou décochez « Appliquer aux deux côtés » pour un corps
   asymétrique). Le menu **Morphologie de départ** (neutre, fine, athlétique, forte, féminine) remplit tout
   le tableau d'un coup, et « ↺ Dimensions par défaut » y revient — la pose déjà réglée est conservée.
3. **Image à produire** :
   - **Squelette OpenPose** → pour ControlNet (pose exacte, 18 points aux couleurs canoniques) ;
   - **Mannequin ombré** ou **filaire** → image de référence pour Qwen‑Image‑2.1 / FLUX.2 klein ;
   - **Carte de profondeur**, **Silhouette (masque)** → autres contrôles.
4. **🎛️ Utiliser cette pose comme image de contrôle** : le rendu est envoyé au serveur et devient
   l'image de contrôle de la carte ControlNet (le type est réglé automatiquement : « pose » pour le
   squelette, « contours » pour les autres rendus). Il ne reste qu'à régler la force et à générer.
5. **🎯 Générer une image de référence dans cette pose** : choisissez le moteur — c'est le même bouton
   pour les deux voies :
   - **SD 1.5 + ControlNet** : la pose est **imposée** (résolution ramenée à 768 px max pour ce modèle) ;
   - **Qwen‑Image‑2.1 / FLUX.2 klein** : le mannequin part comme **image de référence** (`-r`) avec
     l'instruction « Reproduis exactement la pose du personnage de l'image de référence ».
6. **📎 Ajouter le dernier rendu aux références** : l'image obtenue est ajoutée au champ
   « Édition d'image » pour être réutilisée en mode ✏️ Modification (décor, tenue, style…), sans
   repasser par un fichier.

Côté serveur, `app/mannequin.py` redessine la pose reçue (mêmes longueurs d'os, même projection que le
navigateur) via `POST /api/mannequin/render` et l'enregistre dans `uploads/controls/`, donc l'image
produite est une image de contrôle comme une autre (utilisable, réutilisable, supprimable).

## Télécharger les modèles (onglet Modèles)

- **Installation en un clic** : pour un modèle, le pack contient le fichier de diffusion choisi,
  l'encodeur de texte, le VAE et — pour Qwen‑Image‑2.1 — l'encodeur de vision (`mmproj`) indispensable
  à l'édition. Pour **SD 1.5 + ControlNet**, il ajoute les modèles ControlNet (OpenPose + Canny) et le
  détecteur de personnages. Les fichiers déjà présents sont simplement ignorés (« déjà présent »).
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
models/sd15_control/diffusion/        v1-5-pruned-emaonly.safetensors (UNet + CLIP + VAE)
models/sd15_control/controlnet/       control_v11p_sd15_openpose.safetensors, …_canny.safetensors
models/sd15_control/pose_detector/    yolov8n-pose.onnx (détection automatique des personnages)
models/<modèle>/lora/                 LoRA optionnels (syntaxe <lora:nom:0.8> dans le prompt)
uploads/                              images téléversées + uploads/controls/ (squelettes, contours)
outputs/                              images générées + .json de métadonnées (pose/force utilisées)
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

Pour la famille **SD 1.5 + ControlNet**, restez en 512×512 / 512×768 : le modèle et les ControlNet
sont entraînés pour cette résolution (au‑delà, la pose est moins bien suivie). 25 étapes et CFG 7 sont
de bons réglages de départ.

## Tests

```bash
pip install -r requirements-dev.txt     # pytest + httpx
python -m pytest -q                     # 89 tests : catalogue, téléchargements, API, pose, mannequin, commande sd-cli

npm install jsdom                       # une seule fois, pour les tests d'interface
node tests/ui_render.mjs                # rejoue app.js sur un vrai /api/status (serveur lancé)
node tests/ui_render.mjs status.json    # …ou sur une réponse enregistrée
```

`tests/ui_render.mjs` vérifie que l'onglet Modèles affiche bien les quatre modèles, le bouton
« Tout télécharger », la présence de l'édition d'image, et que le clic déclenche exactement
`POST /api/install` (avec la quantification choisie), puis le suivi fichier par fichier et l'annulation.
Il rejoue aussi la carte ControlNet : refus expliqué sur les modèles « DiT », bascule en un clic vers
SD 1.5 + ControlNet, détection automatique de la pose, glisser‑déposer d'une articulation
(`POST /api/control/pose`) et envoi des paramètres de contrôle à `POST /api/generate`.

Il vérifie enfin le **mannequin articulé** : dessin du pantin, pose à la souris (le membre suit et le
coude se plie), **longueurs d'os conservées au 1e‑9 près** après chaque manipulation, tableau des
dimensions (une ligne par segment, symétrie gauche/droite, morphologies), miroir, annulation, export (`POST /api/mannequin/render`) et les deux voies de génération
(ControlNet avec `control_id`, ou modèle d'édition avec `ref_id`). `tests/canvas_shim.mjs` fournit un
canvas 2D minimal (chemins, dégradés, transformations) et un encodeur PNG : les rendus du mannequin
sont donc réellement rasterisés dans les tests Python (`tests/test_mannequin.py`), sans navigateur.

`tests/data/fake-yolov8n-pose.onnx` est un **faux** réseau YOLOv8‑pose de 1,3 ko (généré par
`tests/data/make_fake_onnx.py`) qui encode une pose connue : les tests valident ainsi le chargement ONNX,
l'inférence, le décodage des 17 points et le rendu du squelette **sans télécharger** les 13 Mo du vrai
modèle. Les fonctions pures de `app/pose.py` (décodage, NMS, squelette, Canny) sont testées séparément.

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
- **« Ce modèle ne peut pas utiliser ControlNet »** : normal pour Qwen‑Image‑2.1 et FLUX.2 klein
  (modèles « DiT »). Cliquez sur le bouton de bascule vers **SD 1.5 + ControlNet**, ou retirez le contrôle.
- **« Aucun personnage détecté »** : cadrez la photo de plus près, évitez les silhouettes minuscules ou
  trop sombres ; vous pouvez aussi importer un squelette de référence (bouton « Choisir un squelette de
  référence… ») ou utiliser le contrôle Canny.
- **« Détecteur de personnages absent »** : onglet Modèles → famille **SD 1.5 + ControlNet** →
  « Tout télécharger » (ou le bouton « Télécharger maintenant » affiché dans la carte).
- **La pose détectée n'est pas suivie** : augmentez la force du contrôle (0,9–1,1), restez en 512×768,
  et vérifiez que le squelette affiché correspond bien à la posture voulue.
- **Une nouveauté de l'interface n'apparaît pas** : rechargez avec **Ctrl+F5** (les URL des scripts sont
  horodatées, mais un vieux cache de navigateur peut persister).
- Le journal complet de `sd-cli` est visible sous la barre de progression.

## Licence

Code de cette application : MIT. Les modèles ont leur propre licence (Qwen Research License pour Qwen‑Image‑2.1,
FLUX Non‑Commercial License pour FLUX.2 klein 9B, Apache‑2.0 pour FLUX.2 klein 4B,
CreativeML OpenRAIL‑M pour SD 1.5 et les ControlNet de lllyasviel).
