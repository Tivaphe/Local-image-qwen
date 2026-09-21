"""Construit les assets du mannequin à partir des données MakeHuman (CC0).

Le mannequin n'est plus dessiné à la main : il utilise un vrai maillage anatomique
libre de droits (base MakeHuman, publiée en CC0 1.0 par Data Collection AB), ses
cibles de morphologie (morphs) et son squelette avec les poids de peau.

    python tools/build_mannequin_assets.py /chemin/vers/makehuman/data

Produit dans `app/assets/mannequin/` :

- `maillage.npz`  : sommets + faces du corps (sans les helpers de rig)
- `morphes.npz`   : deltas de chaque cible (creuser la taille, les hanches, le sexe…)
- `squelette.json`: os, hiérarchie, positions de repos, poids de peau
- `LISEZ-MOI.md`  : provenance et licence des données

Les données source ne sont PAS copiées dans le dépôt : seuls ces fichiers
dérivés le sont (ils restent CC0, aucune attribution obligatoire).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

# --- notre rig : un os par segment anatomique, comme le reste de l'application ----
# nom applicatif → os MakeHuman fusionnés (du plus proximal au plus distal)
RIG = {
    "hips": ("pelvis.L", "pelvis.R"),
    "spine": ("spine05", "spine04"),
    "chest": ("spine03", "spine02", "spine01", "clavicle.L", "clavicle.R"),
    "neck": ("neck01", "neck02", "neck03"),
    "head": ("head",),
    # l'articulation est à la TÊTE de l'os : « shoulder » est donc le bras (la
    # clavicule reste au thorax), « elbow » l'avant-bras, « wrist » la main.
    "shoulder_l": ("upperarm01.L", "upperarm02.L"),
    "elbow_l": ("lowerarm01.L", "lowerarm02.L"),
    "wrist_l": ("wrist.L",),
    "hand_l": ("metacarpal1.L", "metacarpal2.L", "metacarpal3.L", "metacarpal4.L"),
    "shoulder_r": ("upperarm01.R", "upperarm02.R"),
    "elbow_r": ("lowerarm01.R", "lowerarm02.R"),
    "wrist_r": ("wrist.R",),
    "hand_r": ("metacarpal1.R", "metacarpal2.R", "metacarpal3.R", "metacarpal4.R"),
    "hip_l": ("upperleg01.L", "upperleg02.L"),
    "knee_l": ("lowerleg01.L", "lowerleg02.L"),
    "ankle_l": ("foot.L",),
    "toe_l": ("toe1-1.L", "toe1-2.L", "toe2-1.L", "toe2-2.L", "toe3-1.L", "toe3-2.L",
              "toe4-1.L", "toe4-2.L", "toe5-1.L", "toe5-2.L"),
    "hip_r": ("upperleg01.R", "upperleg02.R"),
    "knee_r": ("lowerleg01.R", "lowerleg02.R"),
    "ankle_r": ("foot.R",),
    "toe_r": ("toe1-1.R", "toe1-2.R", "toe2-1.R", "toe2-2.R", "toe3-1.R", "toe3-2.R",
              "toe4-1.R", "toe4-2.R", "toe5-1.R", "toe5-2.R"),
}
PARENT = {
    "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "shoulder_l": "chest", "elbow_l": "shoulder_l", "wrist_l": "elbow_l", "hand_l": "wrist_l",
    "shoulder_r": "chest", "elbow_r": "shoulder_r", "wrist_r": "elbow_r", "hand_r": "wrist_r",
    "hip_l": "hips", "knee_l": "hip_l", "ankle_l": "knee_l", "toe_l": "ankle_l",
    "hip_r": "hips", "knee_r": "hip_r", "ankle_r": "knee_r", "toe_r": "ankle_r",
}

# --- cibles de morphologie livrées ----------------------------------------------
# (nom applicatif, fichier .target, étiquette courte)
MORPHES = {
    "genre_femme": ("macrodetails/caucasian-female-young.target", "corps féminin"),
    "genre_homme": ("macrodetails/caucasian-male-young.target", "corps masculin"),
    # diamètres / tours (curseurs « épaisseur » par partie du corps)
    "hanches": ("measure/measure-hips-circ-incr.target", "tour de hanches"),
    "hanches_moins": ("measure/measure-hips-circ-decr.target", "tour de hanches (-)"),
    "tour_de_taille": ("measure/measure-waist-circ-incr.target", "tour de taille (-)"),
    "tour_de_taille_moins": ("measure/measure-waist-circ-decr.target", "tour de taille (+)"),
    "poitrine": ("measure/measure-bust-circ-incr.target", "tour de poitrine"),
    "poitrine_moins": ("measure/measure-bust-circ-decr.target", "tour de poitrine (-)"),
    "sous_poitrine": ("measure/measure-underbust-circ-incr.target", "sous-poitrine"),
    "sous_poitrine_moins": ("measure/measure-underbust-circ-decr.target", "sous-poitrine (-)"),
    "cuisse": ("measure/measure-thigh-circ-incr.target", "tour de cuisse"),
    "cuisse_moins": ("measure/measure-thigh-circ-decr.target", "tour de cuisse (-)"),
    "mollet": ("measure/measure-calf-circ-incr.target", "tour de mollet"),
    "mollet_moins": ("measure/measure-calf-circ-decr.target", "tour de mollet (-)"),
    "genou": ("measure/measure-knee-circ-incr.target", "tour de genou"),
    "genou_moins": ("measure/measure-knee-circ-decr.target", "tour de genou (-)"),
    "cheville": ("measure/measure-ankle-circ-incr.target", "tour de cheville"),
    "cheville_moins": ("measure/measure-ankle-circ-decr.target", "tour de cheville (-)"),
    "bras": ("measure/measure-upperarm-circ-incr.target", "tour de bras"),
    "bras_moins": ("measure/measure-upperarm-circ-decr.target", "tour de bras (-)"),
    "poignet": ("measure/measure-wrist-circ-incr.target", "tour de poignet"),
    "poignet_moins": ("measure/measure-wrist-circ-decr.target", "tour de poignet (-)"),
    "cou": ("measure/measure-neck-circ-incr.target", "tour de cou"),
    "cou_moins": ("measure/measure-neck-circ-decr.target", "tour de cou (-)"),
    # longueurs
    "carure": ("measure/measure-shoulder-dist-incr.target", "largeur d'épaules"),
    "carure_moins": ("measure/measure-shoulder-dist-decr.target", "largeur d'épaules (-)"),
    "taille_hanches": ("measure/measure-waisttohip-dist-incr.target", "taille → hanches"),
    "taille_hanches_moins": ("measure/measure-waisttohip-dist-decr.target", "taille → hanches (-)"),
    "hauteur_cou": ("measure/measure-neck-height-incr.target", "hauteur de cou"),
    "hauteur_cou_moins": ("measure/measure-neck-height-decr.target", "hauteur de cou (-)"),
    # formes locales
    "tronc_largeur": ("torso/torso-scale-horiz-incr.target", "largeur du tronc"),
    "tronc_largeur_moins": ("torso/torso-scale-horiz-decr.target", "largeur du tronc (-)"),
    "tronc_profondeur": ("torso/torso-scale-depth-incr.target", "profondeur du tronc"),
    "tronc_profondeur_moins": ("torso/torso-scale-depth-decr.target", "profondeur du tronc (-)"),
    "fessiers": ("buttocks/buttocks-volume-incr.target", "volume des fessiers"),
    "fessiers_moins": ("buttocks/buttocks-volume-decr.target", "volume des fessiers (-)"),
    "ventre": ("stomach/stomach-pregnant-incr.target", "ventre"),
    "ventre_moins": ("stomach/stomach-pregnant-decr.target", "ventre (-)"),
    "poitrine_volume": ("breast/breast-volume-vert-up.target", "volume de poitrine"),
    "poitrine_volume_moins": ("breast/breast-volume-vert-down.target", "volume de poitrine (-)"),
}


def charge_corps(chemin: pathlib.Path, groupe: str = "body"):
    """Sommets du fichier OBJ et faces du seul groupe du corps (les helpers servent au rig)."""
    sommets, faces, courant = [], [], None
    for ligne in open(chemin, encoding="utf-8", errors="replace"):
        if ligne.startswith("v "):
            _, x, y, z = ligne.split()
            sommets.append((float(x), float(y), float(z)))
        elif ligne.startswith("g "):
            courant = ligne[2:].strip()
        elif ligne.startswith("f ") and courant == groupe:
            idx = [int(p.split("/")[0]) - 1 for p in ligne.split()[1:]]
            for k in range(1, len(idx) - 1):        # le maillage mélange triangles et quads
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(sommets, dtype="float32"), np.asarray(faces, dtype="int32")


def charge_cible(chemin: pathlib.Path):
    """Deltas (indices, dx, dy, dz) d'une cible de morphologie."""
    idx, deltas = [], []
    for ligne in open(chemin, encoding="utf-8", errors="replace"):
        if ligne.startswith("#"):
            continue
        parts = ligne.split()
        if len(parts) == 4:
            idx.append(int(parts[0]))
            deltas.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(idx, dtype="int32"), np.asarray(deltas, dtype="float32")


def charge_squelette(chemin: pathlib.Path, sommets: np.ndarray):
    """Os MakeHuman : positions de repos (moyenne des sommets d'articulation) et parents."""
    donnees = json.load(open(chemin, encoding="utf-8"))
    positions = {}
    for nom, indices in donnees["joints"].items():
        utiles = [i for i in indices if i < len(sommets)]
        if utiles:
            positions[nom] = sommets[utiles].mean(axis=0)
    return donnees["bones"], positions


def charge_poids(chemin: pathlib.Path):
    """Poids de peau : os MakeHuman → liste (sommet, poids)."""
    return json.load(open(chemin, encoding="utf-8"))["weights"]


def fusionne_rig(bones, positions, poids, sommets: np.ndarray):
    """Fusionne les os MakeHuman en un os par segment et récupère les poids correspondants."""
    nb_sommets = len(sommets)
    os_rig, poids_rig = [], {}
    for nom, membres in RIG.items():
        existants = [m for m in membres if m in bones and bones[m]["head"] in positions]
        if not existants:
            raise SystemExit(f"os manquant dans le squelette : {nom} ({membres})")
        tete = positions[bones[existants[0]]["head"]]
        queue = positions[bones[existants[-1]]["tail"]] if bones[existants[-1]]["tail"] in positions else tete
        os_rig.append({
            "nom": nom,
            "parent": PARENT.get(nom),
            "tete": [round(float(v), 5) for v in tete],
            "queue": [round(float(v), 5) for v in queue],
            "makehuman": existants,
        })
        cumul = {}
        for m in existants:
            for sommet, poids_v in poids.get(m, ()):
                cumul[sommet] = cumul.get(sommet, 0.0) + float(poids_v)
        poids_rig[nom] = cumul

    # un sommet sans poids (doigts, visage, orteils…) rejoint l'os le plus proche :
    # sans cela, les doigts se retrouveraient soudés au bassin
    assigne = np.zeros(nb_sommets, dtype="float32")
    for nom, cumul in poids_rig.items():
        for sommet, poids_v in cumul.items():
            assigne[sommet] += poids_v
    orphelins = np.nonzero(assigne < 1e-6)[0]
    if len(orphelins):
        tete = np.array([os["tete"] for os in os_rig], dtype="float32")
        queue = np.array([os["queue"] for os in os_rig], dtype="float32")
        noms = [os["nom"] for os in os_rig]
        for sommet in orphelins:
            p = np.asarray(sommets[sommet], dtype="float32")
            seg = queue - tete
            long = np.maximum((seg * seg).sum(axis=1), 1e-9)
            t = np.clip(((p - tete) * seg).sum(axis=1) / long, 0.0, 1.0)
            proj = tete + seg * t[:, None]
            distance = np.linalg.norm(proj - p, axis=1)
            poids_rig[noms[int(distance.argmin())]][int(sommet)] = 1.0
    # normalisation à 1 par sommet
    for nom, cumul in poids_rig.items():
        for sommet in list(cumul):
            total = assigne[sommet] if assigne[sommet] > 1e-6 else 1.0
            cumul[sommet] = round(cumul[sommet] / total, 5)
    return os_rig, poids_rig, orphelins


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = pathlib.Path(sys.argv[1])
    cible = pathlib.Path(__file__).resolve().parent.parent / "app" / "assets" / "mannequin"
    cible.mkdir(parents=True, exist_ok=True)

    sommets, faces = charge_corps(source / "3dobjs" / "base.obj")
    print(f"corps : {len(sommets)} sommets, {len(faces)} facettes")
    bones, positions = charge_squelette(source / "rigs" / "default.mhskel", sommets)
    poids = charge_poids(source / "rigs" / "default_weights.mhw")
    os_rig, poids_rig, orphelins = fusionne_rig(bones, positions, poids, sommets)
    print(f"rig : {len(os_rig)} os ({orphelins} sommets sans poids, rattachés à l'os le plus proche)")

    np.savez_compressed(cible / "maillage.npz", sommets=sommets, faces=faces)

    morphs, manquants = {}, []
    for nom, (fichier, _) in MORPHES.items():
        chemin = source / "targets" / fichier
        if not chemin.exists():
            manquants.append(fichier)
            continue
        idx, deltas = charge_cible(chemin)
        morphs["idx_" + nom] = idx
        morphs["delta_" + nom] = deltas
    if manquants:
        print("cibles absentes (ignorées) :", ", ".join(manquants))
    np.savez_compressed(cible / "morphes.npz", **morphs)
    print(f"morphes : {len(morphs) // 2} cibles")

    poids_json = {nom: [[int(s), float(w)] for s, w in sorted(cumul.items())]
                  for nom, cumul in poids_rig.items()}
    (cible / "squelette.json").write_text(json.dumps({
        "os": os_rig,
        "poids": poids_json,
        "etiquettes": {nom: etiquette for nom, (_, etiquette) in MORPHES.items()},
    }, ensure_ascii=False), encoding="utf-8")

    (cible / "LISEZ-MOI.md").write_text(
        """# Assets du mannequin

Ces fichiers sont **dérivés** du maillage anatomique libre de MakeHuman :

- maillage de base et cibles de morphologie : *MakeHuman*, publiés en **CC0 1.0**
  par Data Collection AB (https://github.com/makehumancommunity/makehuman,
  fichiers `LICENSE.ASSETS.md` et `data/3dobjs/base.obj`) ;
- squelette et poids de peau : `data/rigs/default.mhskel` et `default_weights.mhw`,
  également **CC0** (`© 2021 Data Collection AB, Joel Palmius, Jonas Hauquier`).

Le CC0 ne demande aucune attribution : elle est ici par courtoisie, et pour que
l'origine des données reste claire. Régénérer ces fichiers :

    python tools/build_mannequin_assets.py /chemin/vers/makehuman/data
""", encoding="utf-8")

    for fichier in ("maillage.npz", "morphes.npz", "squelette.json"):
        taille = (cible / fichier).stat().st_size
        print(f"  {fichier:16} {taille / 1024:8.1f} Kio")


if __name__ == "__main__":
    main()
