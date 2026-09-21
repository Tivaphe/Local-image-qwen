"""Mannequin anatomique : maillage MakeHuman (CC0), morphologies et squelette.

Ce module remplace le bonhomme dessiné à la main par un vrai corps :

- le maillage vient de `app/assets/mannequin/` (base CC0 de MakeHuman, voir son
  `LISEZ-MOI.md`) ; il est déformé par des **cibles de morphologie** : sexe,
  tours de taille/hanches/cuisses/bras/poignets, largeur d'épaules, ventre…
- le squelette fait 21 os (un par segment) et la peau est **pondérée** (chaque
  sommet suit plusieurs os), donc les articulations s'arrondissent au lieu de
  casser ;
- les poses sont des **angles** par articulation (flexion, abduction, torsion),
  bornés par des butées anatomiques — c'est ce qui rend les curseurs naturels.

Tout est calculé ici, sans dépendance externe : le rendu est un rasteriseur
numpy (z-buffer, normales lissées, lumière douce).
"""
from __future__ import annotations

import json
import math
import pathlib
import threading

import numpy as np

ASSETS = pathlib.Path(__file__).resolve().parent / "assets" / "mannequin"

# Le maillage MakeHuman est en décimètres : 1 unité = 10 cm.
ECHELLE = 0.1

# Ordre de construction : parent avant enfant.
ORDRE = [
    "hips", "spine", "chest", "neck", "head",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "hip_l", "knee_l", "ankle_l", "toe_l",
    "hip_r", "knee_r", "ankle_r", "toe_r",
]
PARENT = {
    "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "shoulder_l": "chest", "elbow_l": "shoulder_l", "wrist_l": "elbow_l", "hand_l": "wrist_l",
    "shoulder_r": "chest", "elbow_r": "shoulder_r", "wrist_r": "elbow_r", "hand_r": "wrist_r",
    "hip_l": "hips", "knee_l": "hip_l", "ankle_l": "knee_l", "toe_l": "ankle_l",
    "hip_r": "hips", "knee_r": "hip_r", "ankle_r": "knee_r", "toe_r": "ankle_r",
}
# Noms lisibles (interface, messages d'erreur).
ETIQUETTES = {
    "hips": "bassin", "spine": "bas du dos", "chest": "poitrine", "neck": "cou", "head": "tête",
    "shoulder_l": "épaule gauche", "elbow_l": "coude gauche", "wrist_l": "poignet gauche", "hand_l": "main gauche",
    "shoulder_r": "épaule droite", "elbow_r": "coude droit", "wrist_r": "poignet droit", "hand_r": "main droite",
    "hip_l": "hanche gauche", "knee_l": "genou gauche", "ankle_l": "cheville gauche", "toe_l": "orteils gauches",
    "hip_r": "hanche droite", "knee_r": "genou droit", "ankle_r": "cheville droite", "toe_r": "orteils droits",
}

# Mouvement obtenu quand l'angle « plier » (ou « écarter ») augmente : direction vers
# laquelle part le bout de l'os, exprimée sur le corps debout, paumes vers le sol.
#   - « plier »  : le geste naturel de l'articulation (lever le bras, plier le coude,
#                  avancer la cuisse, plier le genou) ;
#   - « écarter » : le mouvement latéral ou de balancement.
MOUVEMENTS = {
    "hips": ((0, 0, 1), (1, 0, 0)),        # basculer le bassin vers l'avant / sur le côté
    "spine": ((0, 0, 1), (1, 0, 0)),       # pencher le buste en avant / sur le côté
    "chest": ((0, 0, 1), (1, 0, 0)),
    "neck": ((0, 0, 1), (1, 0, 0)),        # hocher / pencher la tête
    "head": ((0, 0, 1), (1, 0, 0)),
    "shoulder_l": ((0, 1, 0), (0, 0, -1)),   # lever le bras / le balancer vers l'avant
    "shoulder_r": ((0, 1, 0), (0, 0, -1)),
    "elbow_l": ((0, 0, 1), (0, 1, 0)),       # plier le coude (main vers l'avant) / vers le haut
    "elbow_r": ((0, 0, 1), (0, 1, 0)),
    "wrist_l": ((0, 0, 1), (0, 1, 0)),       # fléchir le poignet
    "wrist_r": ((0, 0, 1), (0, 1, 0)),
    "hand_l": ((0, 1, 0), (0, 0, 1)),        # incliner la main
    "hand_r": ((0, 1, 0), (0, 0, 1)),
    "hip_l": ((0, 0, 1), (1, 0, 0)),         # avancer la cuisse / l'écarter
    "hip_r": ((0, 0, 1), (-1, 0, 0)),
    "knee_l": ((0, 0, -1), (1, 0, 0)),       # plier le genou (talon vers l'arrière)
    "knee_r": ((0, 0, -1), (-1, 0, 0)),
    "ankle_l": ((0, 1, 0), (1, 0, 0)),       # relever la pointe du pied
    "ankle_r": ((0, 1, 0), (-1, 0, 0)),
    "toe_l": ((0, 1, 0), (1, 0, 0)),         # relever les orteils
    "toe_r": ((0, 1, 0), (-1, 0, 0)),
}

# Butées anatomiques, en degrés : (flexion mini, flexion maxi, abduction maxi, torsion maxi).
# La flexion est le mouvement naturel de l'articulation (plier/déplier), l'abduction
# l'écartement latéral, la torsion la rotation autour de l'os.
LIMITES = {
    "hips": (-45, 45, 35, 35),           # le bassin peut s'incliner (rotation du tronc)
    "spine": (-25, 25, 18, 25),
    "chest": (-20, 25, 16, 20),
    "neck": (-40, 45, 30, 35),
    "head": (-25, 25, 20, 30),
    "shoulder_l": (-95, 100, 80, 30),     # bras le long du corps (-) ou levé (+)
    "elbow_l": (0, 150, 8, 60),
    "wrist_l": (-70, 75, 25, 0),
    "hand_l": (-35, 35, 25, 0),
    "shoulder_r": (-95, 100, 80, 30),
    "elbow_r": (0, 150, 8, 60),
    "wrist_r": (-70, 75, 25, 0),
    "hand_r": (-35, 35, 25, 0),
    "hip_l": (-35, 120, 45, 40),
    "knee_l": (0, 150, 5, 10),
    "ankle_l": (-45, 35, 25, 0),
    "toe_l": (0, 55, 0, 0),
    "hip_r": (-35, 120, 45, 40),
    "knee_r": (0, 150, 5, 10),
    "ankle_r": (-45, 35, 25, 0),
    "toe_r": (0, 55, 0, 0),
}

# Décalage du bassin pour les poses où le corps n'est pas debout (mètres).
POSES_ORIGINE = {"assis": (0.0, -0.44, 0.02), "accroupi": (0.0, -0.50, 0.05)}

# Poses proposées : angles (flexion, abduction, torsion) par os, en degrés.
# Pose de repos de l'utilisateur : le maillage est modélisé en T, on ramène donc les
# bras le long du corps pour toutes les poses « debout ».
BRAS_LE_LONG_DU_CORPS = -38

POSES = {
    "debout": {
        "shoulder_l": (BRAS_LE_LONG_DU_CORPS, 6, 0), "shoulder_r": (BRAS_LE_LONG_DU_CORPS, 6, 0),
        "elbow_l": (12, 0, 0), "elbow_r": (12, 0, 0),
    },
    "bras_leves": {
        "shoulder_l": (78, 18, 0), "elbow_l": (18, 0, 0),
        "shoulder_r": (78, 18, 0), "elbow_r": (18, 0, 0),
        "spine": (-6, 0, 0),
    },
    "marche": {
        "hip_l": (34, 2, 0), "knee_l": (16, 0, 0), "ankle_l": (-10, 0, 0),
        "hip_r": (-20, 2, 0), "knee_r": (46, 0, 0), "ankle_r": (14, 0, 0),
        "shoulder_l": (BRAS_LE_LONG_DU_CORPS, 26, 0), "elbow_l": (30, 0, 0),
        "shoulder_r": (BRAS_LE_LONG_DU_CORPS, -24, 0), "elbow_r": (22, 0, 0),
        "spine": (4, 0, 5),
    },
    "assis": {
        "hip_l": (86, 8, 0), "hip_r": (86, 8, 0),
        "knee_l": (88, 0, 0), "knee_r": (88, 0, 0),
        "ankle_l": (6, 0, 0), "ankle_r": (6, 0, 0),
        "spine": (-4, 0, 0), "chest": (-3, 0, 0),
        "shoulder_l": (-24, 18, 0), "elbow_l": (58, 0, 0),
        "shoulder_r": (-24, 18, 0), "elbow_r": (58, 0, 0),
    },
    "accroupi": {
        "hip_l": (100, 16, 0), "hip_r": (100, 16, 0),
        "knee_l": (118, 0, 0), "knee_r": (118, 0, 0),
        "ankle_l": (24, 0, 0), "ankle_r": (24, 0, 0),
        "spine": (16, 0, 0), "chest": (8, 0, 0),
        "shoulder_l": (-6, 26, 0), "elbow_l": (76, 0, 0),
        "shoulder_r": (-6, 26, 0), "elbow_r": (76, 0, 0),
    },
    "saut": {
        "shoulder_l": (95, 20, 0), "elbow_l": (22, 0, 0),
        "shoulder_r": (95, 20, 0), "elbow_r": (22, 0, 0),
        "hip_l": (58, 14, 0), "hip_r": (58, 14, 0),
        "knee_l": (78, 0, 0), "knee_r": (78, 0, 0),
        "ankle_l": (24, 0, 0), "ankle_r": (24, 0, 0),
        "spine": (-8, 0, 0),
    },
    "salut": {
        "shoulder_l": (0, 78, 55), "elbow_l": (142, 0, 0), "wrist_l": (14, 0, 0),
        "shoulder_r": (BRAS_LE_LONG_DU_CORPS, 6, 0), "elbow_r": (16, 0, 0),
        "neck": (8, 0, 6), "head": (4, 0, 5),
    },
}

# Groupes de morphologies proposés dans l'interface : nom → poids des cibles.
MORPHOLOGIES = {
    "neutre": {"genre_femme": 0.5, "genre_homme": 0.5},
    "femme": {"genre_femme": 1.0},
    "homme": {"genre_homme": 1.0},
    "fine": {"genre_femme": 0.6, "genre_homme": 0.4, "tour_de_taille": 0.45, "hanches_moins": 0.35,
             "cuisse_moins": 0.3, "bras_moins": 0.3},
    "athletique": {"genre_homme": 0.75, "tronc_largeur": 0.45, "carure": 0.4, "bras": 0.35, "ventre_moins": 0.4},
    "forte": {"genre_homme": 0.35, "genre_femme": 0.65, "ventre": 0.5, "hanches": 0.3, "cuisse": 0.25,
              "bras": 0.2},
}

_LOCK = threading.Lock()
_CACHE: dict = {}


class MannequinError(ValueError):
    """Erreur montrée à l'utilisateur (message en français)."""


def _charge():
    """Charge maillage, morphs et squelette (une seule fois par processus)."""
    with _LOCK:
        if _CACHE:
            return _CACHE
        if not (ASSETS / "maillage.npz").exists():
            raise MannequinError(
                "assets du mannequin absents — lancez : python tools/build_mannequin_assets.py "
                "/chemin/vers/makehuman/data")
        maille = np.load(ASSETS / "maillage.npz")
        morphs_f = np.load(ASSETS / "morphes.npz")
        rig = json.loads((ASSETS / "squelette.json").read_text(encoding="utf-8"))
        sommets = (maille["sommets"].astype("float32") * ECHELLE)
        faces = maille["faces"].astype("int32")
        os_rig = rig["os"]
        tete = np.array([o["tete"] for o in os_rig], dtype="float32") * ECHELLE
        queue = np.array([o["queue"] for o in os_rig], dtype="float32") * ECHELLE
        indice = {o["nom"]: i for i, o in enumerate(os_rig)}

        # poids de peau à plat (sommet, os, poids) pour un skinning vectorisé
        lignes, colonnes, valeurs = [], [], []
        for nom, liste in rig["poids"].items():
            if nom not in indice:
                continue
            for sommet, poids in liste:
                lignes.append(sommet); colonnes.append(indice[nom]); valeurs.append(poids)
        poids_sommets = (np.array(lignes, dtype="int32"), np.array(colonnes, dtype="int32"),
                         np.array(valeurs, dtype="float32"))

        morphs = {}
        for cle in morphs_f.files:
            if cle.startswith("idx_"):
                nom = cle[4:]
                morphs[nom] = (morphs_f["idx_" + nom].astype("int32"),
                               morphs_f["delta_" + nom].astype("float32") * ECHELLE)

        # orientation cohérente des facettes + normales lissées : calculées une fois
        faces = _oriente(sommets, faces)
        normales = _normales(sommets, faces)

        _CACHE.update({
            "sommets": sommets, "faces": faces, "normales": normales,
            "os": [o["nom"] for o in os_rig], "indice": indice,
            "tete": tete, "queue": queue, "parents": [PARENT.get(o["nom"]) for o in os_rig],
            "poids": poids_sommets, "morphs": morphs,
            "etiquettes_morphs": rig.get("etiquettes", {}),
            "reperes": _reperes(tete, queue, [o["nom"] for o in os_rig]),
        })
        return _CACHE


def _reperes(tete: np.ndarray, queue: np.ndarray, os: list[str]):
    """Repère de rotation de chaque os, déduit du mouvement anatomique attendu.

    Plutôt que de deviner un axe, on déclare **où doit aller le bout de l'os** quand
    l'angle augmente (« plier » et « écarter », table MOUVEMENTS). L'axe s'en déduit :
    faire tourner l'os autour de `axe` déplace son bout dans la direction `mouvement`,
    donc `axe = bout × mouvement`.
    """
    bout = queue - tete
    bout /= np.maximum(np.linalg.norm(bout, axis=1, keepdims=True), 1e-9)

    def axe_pour(position, direction):
        d = np.array(direction, dtype="float32")
        a = np.cross(bout[position], d)
        norme = np.linalg.norm(a)
        if norme < 1e-6:                       # mouvement déjà dans l'axe de l'os
            a = np.cross(bout[position], np.array([0.0, 1.0, 0.0], dtype="float32"))
            norme = np.linalg.norm(a)
            if norme < 1e-6:
                a = np.cross(bout[position], np.array([0.0, 0.0, 1.0], dtype="float32"))
                norme = max(np.linalg.norm(a), 1e-6)
        return a / norme

    flexion, abduction = [], []
    for i, nom in enumerate(os):
        d_flex, d_ecart = MOUVEMENTS.get(nom, ((0, 0, 1), (1, 0, 0)))
        flexion.append(axe_pour(i, d_flex))
        abduction.append(axe_pour(i, d_ecart))
    return (bout.astype("float32"), np.array(flexion, dtype="float32"),
            np.array(abduction, dtype="float32"))


def _oriente(sommets: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Rend le sens de parcours des facettes cohérent (sinon l'éclairage mouchette)."""
    faces = faces.copy()
    aretes: dict = {}
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            aretes.setdefault((min(u, v), max(u, v)), []).append((i, u < v))
    vus = np.zeros(len(faces), dtype=bool)
    for depart in range(len(faces)):
        if vus[depart]:
            continue
        vus[depart] = True
        pile = [depart]
        while pile:
            i = pile.pop()
            a, b, c = faces[i]
            for u, v in ((a, b), (b, c), (c, a)):
                for j, _ in aretes[(min(u, v), max(u, v))]:
                    if j == i or vus[j]:
                        continue
                    jv = faces[j]
                    if not any((jv[k], jv[(k + 1) % 3]) == (v, u) for k in range(3)):
                        faces[j] = faces[j][[0, 2, 1]]
                    vus[j] = True
                    pile.append(j)
    a, b, c = sommets[faces[:, 0]], sommets[faces[:, 1]], sommets[faces[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        faces = faces[:, [0, 2, 1]]
    return faces.astype("int32")


def _normales(sommets: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normales lissées par sommet (moyenne des facettes) : la peau paraît lisse."""
    a, b, c = sommets[faces[:, 0]], sommets[faces[:, 1]], sommets[faces[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    normales = np.zeros_like(sommets)
    for k in range(3):
        np.add.at(normales, faces[:, k], n)
    return normales / np.maximum(np.linalg.norm(normales, axis=1, keepdims=True), 1e-9)


# --------------------------------------------------------------- morphologies

def applique_morphs(poids: dict) -> np.ndarray:
    """Applique les cibles de morphologie et renvoie les sommets déformés (mètres)."""
    donnees = _charge()
    sommets = donnees["sommets"].copy()
    for nom, coefficient in (poids or {}).items():
        cible = donnees["morphs"].get(nom)
        if cible is None:
            raise MannequinError(f"morphologie inconnue : « {nom} »")
        try:
            coefficient = float(coefficient)
        except (TypeError, ValueError):
            raise MannequinError(f"poids invalide pour « {nom} »")
        if abs(coefficient) < 1e-6:
            continue
        idx, deltas = cible
        sommets[idx] += deltas * coefficient
    return sommets


def dimensions(poids: dict | None = None) -> dict:
    """Quelques mesures du corps (en mètres) pour l'interface."""
    donnees = _charge()
    sommets = applique_morphs(poids or {})
    return {
        "stature": float(np.ptp(sommets[:, 1])),
        "carure": float(np.ptp(sommets[:, 0])),
        "os": len(donnees["os"]),
        "sommets": int(len(sommets)),
    }


def morphologies() -> dict:
    """Catalogue des morphologies proposées à l'utilisateur."""
    donnees = _charge()
    return {
        "morphologies": MORPHOLOGIES,
        "morphs": {nom: donnees["etiquettes_morphs"].get(nom, nom) for nom in donnees["morphs"]},
    }


# ------------------------------------------------------------------- squelette

def _matrice_rotation(axe, angle):
    """Matrice 3×3 de rotation autour d'un axe unitaire (Rodrigues)."""
    x, y, z = axe
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype="float32")


def _borne(valeurs, limites):
    """Ramène (flexion, abduction, torsion) dans les butées anatomiques."""
    flexion, abduction, torsion = valeurs
    mini, maxi, abd_max, tors_max = limites
    return (max(mini, min(maxi, flexion)),
            max(-abd_max, min(abd_max, abduction)),
            max(-tors_max, min(tors_max, torsion)))


def matrices(angles: dict | None = None, bornes: bool = True, origine=None) -> dict:
    """Matrices monde de chaque os pour une pose donnée (angles en degrés).

    `origine` décale tout le corps (s'asseoir, s'accroupir : le bassin descend).
    """
    donnees = _charge()
    angles = angles or {}
    inconnus = [n for n in angles if n not in donnees["indice"]]
    if inconnus:
        raise MannequinError("articulation inconnue : " + ", ".join(sorted(inconnus)))
    tete = donnees["tete"]
    axes, flexion_axe, abduction_axe = donnees["reperes"]
    monde = {}
    for nom in ORDRE:
        i = donnees["indice"][nom]
        valeurs = angles.get(nom, (0.0, 0.0, 0.0))
        if isinstance(valeurs, dict):
            valeurs = (valeurs.get("flexion", 0.0), valeurs.get("abduction", 0.0), valeurs.get("torsion", 0.0))
        try:
            flexion, abduction, torsion = (float(v) for v in valeurs)
        except (TypeError, ValueError):
            raise MannequinError(f"angles invalides pour « {ETIQUETTES.get(nom, nom)} »")
        if bornes:
            flexion, abduction, torsion = _borne((flexion, abduction, torsion), LIMITES[nom])
        r = (_matrice_rotation(abduction_axe[i], math.radians(abduction))
             @ _matrice_rotation(flexion_axe[i], math.radians(flexion))
             @ _matrice_rotation(axes[i], math.radians(torsion)))
        local = np.eye(4, dtype="float32")
        local[:3, :3] = r
        if nom == "hips":
            local[:3, 3] = tete[i] + np.asarray(origine or (0.0, 0.0, 0.0), dtype="float32")
        else:
            parent = PARENT[nom]
            local[:3, 3] = tete[i] - tete[donnees["indice"][parent]]
        monde[nom] = (monde[PARENT[nom]] @ local) if nom in PARENT else local
    return monde


def articulations(angles: dict | None = None, origine=None) -> dict:
    """Position monde de chaque articulation (pour les poignées et OpenPose)."""
    positions = {}
    for nom, matrice in matrices(angles, origine=origine).items():
        positions[nom] = [float(v) for v in matrice[:3, 3]]
    return positions


def _skinnage(sommets: np.ndarray, monde: dict) -> np.ndarray:
    """Peau pondérée : chaque sommet suit ses os (transformation linéaire)."""
    donnees = _charge()
    lignes, colonnes, valeurs = donnees["poids"]
    os_noms = donnees["os"]
    sortie = np.zeros_like(sommets)
    for position, nom in enumerate(os_noms):
        selection = colonnes == position
        if not selection.any():
            continue
        sommets_vises = lignes[selection]
        poids = valeurs[selection][:, None]
        m = monde[nom]
        local = sommets[sommets_vises] - donnees["tete"][position]
        sortie[sommets_vises] += (local @ m[:3, :3].T + m[:3, 3]) * poids
    return sortie


# ---------------------------------------------------------------------- rendu

PEAU = np.array([198.0, 176.0, 234.0])      # BGR
LUMIERE = np.array([-0.36, 0.78, -0.51], dtype="float32")
LUMIERE /= np.linalg.norm(LUMIERE)
VERS_CAMERA = np.array([0.0, 0.0, -1.0], dtype="float32")
SPECULAIRE = (LUMIERE + VERS_CAMERA) / np.linalg.norm(LUMIERE + VERS_CAMERA)


def _camera(angle: float, hauteur: float, largeur: int, hauteur_img: int, marge: float):
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype="float32")


def rasterise(sommets, faces, normales, largeur, hauteur, yaw, pitch, marge=0.92,
              couleur=PEAU, fond=26.0, masque_seulement=False):
    """Rasterise un maillage et renvoie (image BGR, masque) — z-buffer et normales lissées."""
    sommets = np.asarray(sommets, dtype="float32")
    y_bas = sommets[:, 1].min()
    centre_x = (sommets[:, 0].min() + sommets[:, 0].max()) / 2
    centre_z = (sommets[:, 2].min() + sommets[:, 2].max()) / 2
    local = sommets - np.array([centre_x, y_bas, centre_z], dtype="float32")
    taille = float(local[:, 1].max())
    echelle = hauteur * marge / max(taille, 1e-6)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_x, sin_x = math.cos(pitch), math.sin(pitch)
    rotation = np.array([
        [cos_y, 0, sin_y],
        [sin_x * sin_y, cos_x, -sin_x * cos_y],
        [-cos_x * sin_y, sin_x, cos_x * cos_y],
    ], dtype="float32")
    points = local @ rotation.T
    normales = normales @ rotation.T
    points[:, 0] -= (points[:, 0].min() + points[:, 0].max()) / 2
    focale = 3.8 * taille
    z = points[:, 2] + focale
    perspective = focale / np.maximum(z, 1e-6)
    px = largeur / 2 + points[:, 0] * echelle * perspective
    py = hauteur - points[:, 1] * echelle * perspective - hauteur * 0.02

    image = np.full((hauteur, largeur, 3), fond, dtype="float32")
    zbuffer = np.full((hauteur, largeur), 1e12, dtype="float32")
    # faces vues de dos : inutiles (le corps est fermé) → moitié du travail en moins
    normales_faces = normales[faces]
    face_avant = (normales_faces[:, :, 2].mean(axis=1) < 2e-3)
    ordre = np.argsort(-z[faces].mean(axis=1))
    masque_total = np.zeros((hauteur, largeur), dtype=bool)
    for t in ordre:
        if not face_avant[t]:
            continue
        i0, i1, i2 = faces[t]
        xs, ys, zs = px[[i0, i1, i2]], py[[i0, i1, i2]], z[[i0, i1, i2]]
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        x0, x1 = max(x0, 0), min(x1, largeur - 1)
        y0, y1 = max(y0, 0), min(y1, hauteur - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        det = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(det) < 1e-12:
            continue
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / det
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / det
        l2 = 1 - l0 - l1
        dedans = (l0 >= -0.004) & (l1 >= -0.004) & (l2 >= -0.004)
        if not dedans.any():
            continue
        zz = l0 * zs[0] + l1 * zs[1] + l2 * zs[2]
        zone = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        garde = dedans & (zz < zbuffer[zone])
        if not garde.any():
            continue
        zbuffer[zone] = np.where(garde, zz, zbuffer[zone])
        masque_total[zone] |= garde
        if masque_seulement:
            continue
        n = (l0[:, :, None] * normales[i0] + l1[:, :, None] * normales[i1] + l2[:, :, None] * normales[i2])
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)
        diffus = np.clip(n @ LUMIERE, 0, 1)
        speculaire = np.clip(n @ SPECULAIRE, 0, 1) ** 30
        nuance = couleur[None, None, :] * (0.20 + 0.88 * diffus)[:, :, None] + speculaire[:, :, None] * 46
        cible = image[zone]
        cible[garde] = nuance[garde]
    return np.clip(image, 0, 255).astype("uint8"), masque_total


def rend(payload: dict, mode: str = "volume", width: int = 512, height: int = 768) -> np.ndarray:
    """Rendu dans le mode demandé (volume, silhouette, depth, openpose)."""
    if mode not in ("volume", "silhouette", "depth", "openpose"):
        raise MannequinError(f"rendu inconnu : « {mode} »")
    donnees = _charge()
    poids_morphs = payload.get("morphs") or _morphs_du_payload(payload)
    angles = payload.get("pose") or {}
    origine = payload.get("origine")
    sommets = applique_morphs(poids_morphs)
    monde = matrices(angles, origine=origine)
    deformes = _skinnage(sommets, monde)
    normales = _normales(deformes, donnees["faces"])
    # le corps regarde vers +z : la caméra se place devant, en trois-quarts
    yaw = float(payload.get("yaw") if payload.get("yaw") is not None else math.pi - 0.42)
    pitch = float(payload.get("pitch") if payload.get("pitch") is not None else 0.04)
    largeur, hauteur = max(64, int(width)), max(64, int(height))
    ss = 2 if mode in ("volume", "silhouette", "depth") else 1
    image, masque = rasterise(deformes, donnees["faces"], normales, largeur * ss, hauteur * ss,
                              yaw, pitch, masque_seulement=(mode == "silhouette"))
    if mode == "openpose":
        return _openpose(deformes, monde, donnees, largeur, hauteur, yaw, pitch)
    if mode == "silhouette":
        import cv2
        img = np.where(np.repeat(masque[:, :, None], 3, axis=2), 255, 0).astype("uint8")
        return cv2.resize(img, (largeur, hauteur), interpolation=cv2.INTER_AREA) if ss > 1 else img
    if ss > 1:
        import cv2
        image = cv2.resize(image, (largeur, hauteur), interpolation=cv2.INTER_AREA)
    return image


def _morphs_du_payload(payload: dict) -> dict:
    """Morphologie demandée : soit un nom connu, soit un jeu de poids de cibles."""
    nom = payload.get("morphology")
    if not nom:
        return {}
    if isinstance(nom, dict):
        return nom
    if nom not in MORPHOLOGIES:
        raise MannequinError("morphologie inconnue : « %s » (essayez : %s)"
                             % (nom, ", ".join(MORPHOLOGIES)))
    return MORPHOLOGIES[nom]


def _openpose(deformes, monde, donnees, largeur, hauteur, yaw, pitch):
    """Squelette OpenPose (18 points) à partir des articulations du mannequin."""
    import cv2
    positions = {nom: m[:3, 3] for nom, m in monde.items()}
    sommets = np.array(list(positions.values()), dtype="float32")
    y_bas = deformes[:, 1].min()
    taille = float(deformes[:, 1].max() - y_bas)
    echelle = hauteur * 0.92 / max(taille, 1e-6)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_x, sin_x = math.cos(pitch), math.sin(pitch)
    rotation = np.array([[cos_y, 0, sin_y], [sin_x * sin_y, cos_x, -sin_x * cos_y],
                         [-cos_x * sin_y, sin_x, cos_x * cos_y]], dtype="float32")
    points = (sommets - np.array([0, y_bas, 0], dtype="float32")) @ rotation.T
    focale = 3.8 * taille
    z = points[:, 2] + focale
    persp = focale / np.maximum(z, 1e-6)
    px = largeur / 2 + points[:, 0] * echelle * persp
    py = hauteur - points[:, 1] * echelle * persp - hauteur * 0.02
    ecran = {nom: (float(px[i]), float(py[i])) for i, nom in enumerate(donnees["os"])}

    def milieu(a, b):
        return ((ecran[a][0] + ecran[b][0]) / 2, (ecran[a][1] + ecran[b][1]) / 2)

    # 18 points OpenPose (BODY_25 non : le format standard 18 points des ControlNet)
    tete = ecran["head"]
    nuque = ecran["neck"]
    os = {
        "nose": (tete[0], tete[1] - (nuque[1] - tete[1]) * 0.35),
        "neck": milieu("neck", "chest"),
        "shoulder_r": ecran["shoulder_r"], "elbow_r": ecran["elbow_r"], "wrist_r": ecran["wrist_r"],
        "shoulder_l": ecran["shoulder_l"], "elbow_l": ecran["elbow_l"], "wrist_l": ecran["wrist_l"],
        "hip_r": ecran["hip_r"], "knee_r": ecran["knee_r"], "ankle_r": ecran["ankle_r"],
        "hip_l": ecran["hip_l"], "knee_l": ecran["knee_l"], "ankle_l": ecran["ankle_l"],
        "eye_r": (tete[0] - 6, tete[1] - 4), "eye_l": (tete[0] + 6, tete[1] - 4),
        "ear_r": (tete[0] - 12, tete[1] + 2), "ear_l": (tete[0] + 12, tete[1] + 2),
    }
    noms = ["nose", "neck", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
            "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l", "eye_r", "eye_l", "ear_r", "ear_l"]
    liens = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
             (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)]
    couleurs = [(0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255), (0, 255, 170), (0, 255, 85),
                (0, 255, 0), (85, 255, 0), (170, 255, 0), (255, 255, 0), (255, 170, 0), (255, 85, 0),
                (255, 0, 0), (255, 0, 85), (255, 0, 170), (255, 0, 255), (170, 0, 255), (85, 0, 255)]
    image = np.zeros((hauteur, largeur, 3), dtype="uint8")
    epaisseur = max(2, int(min(largeur, hauteur) / 130))
    for i, (a, b) in enumerate(liens):
        cv2.line(image, (int(os[noms[a]][0]), int(os[noms[a]][1])),
                 (int(os[noms[b]][0]), int(os[noms[b]][1])), couleurs[i % len(couleurs)], epaisseur, cv2.LINE_AA)
    for i, nom in enumerate(noms):
        cv2.circle(image, (int(os[nom][0]), int(os[nom][1])), max(2, epaisseur // 2),
                   couleurs[i % len(couleurs)], -1, cv2.LINE_AA)
    return image
