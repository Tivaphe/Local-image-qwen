"""Mannequin articulé — rendu serveur (géométrie identique au moteur JavaScript).

Le navigateur envoie la pose (articulations 3D + morphologie + caméra) ; ici on
dessine l'image finale, sans dépendre du navigateur :

* ``openpose`` : squelette 18 points aux couleurs canoniques (ControlNet SD 1.5/SDXL) ;
* ``volume``   : mannequin ombré (image de référence pour Qwen‑Image / FLUX.2 klein) ;
* ``wireframe``: filaire plus lisible en contrôle de contour ;
* ``depth``    : carte de profondeur (0 = loin, 255 = proche) ;
* ``silhouette``: masse blanche sur fond noir (masque).

Toutes les longueurs d'os sont fixes : on ne peut pas étirer un membre, seulement
le plier (voir ``solve_two_bone_ik`` côté navigateur, qui produit la pose envoyée).
"""

from __future__ import annotations

import math
from pathlib import Path

from . import pose as _pose

# --------------------------------------------------------------- squelette
# x = droite du personnage, y = haut, z = vers l'avant de la caméra.
JOINTS = [
    "hips", "spine", "chest", "neck", "head", "head_top",
    "nose", "eye_l", "eye_r", "ear_l", "ear_r",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "hip_l", "knee_l", "ankle_l", "toe_l", "heel_l",
    "hip_r", "knee_r", "ankle_r", "toe_r", "heel_r",
]

# (parent, enfant, rayon départ, rayon arrivée, type)
BONES = [
    ("hips", "spine", 0.112, 0.150, "torso"),
    ("spine", "chest", 0.150, 0.165, "torso"),
    ("chest", "neck", 0.120, 0.048, "neck"),
    ("neck", "head", 0.072, 0.100, "head"),
    ("head", "head_top", 0.100, 0.055, "head"),
    ("neck", "shoulder_l", 0.062, 0.066, "shoulder"),
    ("shoulder_l", "elbow_l", 0.062, 0.048, "arm"),
    ("elbow_l", "wrist_l", 0.048, 0.036, "forearm"),
    ("wrist_l", "hand_l", 0.036, 0.044, "hand"),
    ("neck", "shoulder_r", 0.062, 0.066, "shoulder"),
    ("shoulder_r", "elbow_r", 0.062, 0.048, "arm"),
    ("elbow_r", "wrist_r", 0.048, 0.036, "forearm"),
    ("wrist_r", "hand_r", 0.036, 0.044, "hand"),
    ("hips", "hip_l", 0.082, 0.098, "pelvis"),
    ("hip_l", "knee_l", 0.088, 0.062, "thigh"),
    ("knee_l", "ankle_l", 0.062, 0.040, "shin"),
    ("ankle_l", "toe_l", 0.042, 0.038, "foot"),
    ("ankle_l", "heel_l", 0.040, 0.034, "foot"),
    ("hips", "hip_r", 0.082, 0.098, "pelvis"),
    ("hip_r", "knee_r", 0.088, 0.062, "thigh"),
    ("knee_r", "ankle_r", 0.062, 0.040, "shin"),
    ("ankle_r", "toe_r", 0.042, 0.038, "foot"),
    ("ankle_r", "heel_r", 0.040, 0.034, "foot"),
]

REST_DIRS = {
    "spine": (0, 1, 0), "chest": (0, 1, 0), "neck": (0, 1, 0),
    "head": (0, 0.99, 0.12), "head_top": (0, 0.98, 0.18),
    "shoulder_l": (0.94, -0.34, 0), "elbow_l": (0.06, -0.995, 0.08),
    "wrist_l": (0.03, -0.998, 0.05), "hand_l": (0, -1, 0.05),
    "shoulder_r": (-0.94, -0.34, 0), "elbow_r": (-0.06, -0.995, 0.08),
    "wrist_r": (-0.03, -0.998, 0.05), "hand_r": (0, -1, 0.05),
    "hip_l": (0.66, -0.75, 0), "knee_l": (0.09, -0.99, 0.09),
    "ankle_l": (0, -1, -0.02), "toe_l": (0, -0.18, 0.98), "heel_l": (0, -0.32, -0.95),
    "hip_r": (-0.66, -0.75, 0), "knee_r": (-0.09, -0.99, 0.09),
    "ankle_r": (0, -1, -0.02), "toe_r": (0, -0.18, 0.98), "heel_r": (0, -0.32, -0.95),
    "nose": (0, -0.10, 0.98), "eye_l": (0.28, 0.24, 0.86), "eye_r": (-0.28, 0.24, 0.86),
    "ear_l": (0.62, 0.06, 0.05), "ear_r": (-0.62, 0.06, 0.05),
}
PARENT = {child: parent for parent, child, *_ in BONES}
FACE = {"nose": "head", "eye_l": "head", "eye_r": "head", "ear_l": "head", "ear_r": "head"}
PARENT.update(FACE)
PARENT["hips"] = None
BUILD_ORDER = ["hips", "spine", "chest", "neck", "head", "head_top",
               "nose", "eye_l", "eye_r", "ear_l", "ear_r",
               "shoulder_l", "elbow_l", "wrist_l", "hand_l",
               "shoulder_r", "elbow_r", "wrist_r", "hand_r",
               "hip_l", "knee_l", "ankle_l", "toe_l", "heel_l",
               "hip_r", "knee_r", "ankle_r", "toe_r", "heel_r"]

# ------------------------------------------------------- butées articulaires
# « cone » : écart maximal de l'os par rapport à sa direction de repos (degrés).
# « pli »  : flexion autorisée par rapport au segment parent (sens unique : pas
#            d'hyperextension pour un coude ou un genou).
LIMITS = {
    "neck": {"cone": 45, "pli": (-40, 45)}, "head": {"cone": 38, "pli": (-35, 40)},
    "head_top": {"cone": 18},
    "shoulder_l": {"cone": 28},
    "elbow_l": {"cone": 170},                         # épaule : cône, l'anti-collision suffit
    "wrist_l": {"cone": 180, "pli": (0, 150)},        # coude : flexion seulement
    "hand_l": {"cone": 100, "pli": (0, 85)},          # poignet
    "hip_l": {"cone": 30},
    "knee_l": {"cone": 160, "pli": (-25, 115)},       # hanche : extension limitée
    "ankle_l": {"cone": 170, "pli": (0, 145)},        # genou : flexion seulement
    "toe_l": {"cone": 110, "pli": (-75, 40)},         # cheville
    "heel_l": {"cone": 110},                          # talon : solidaire du pied
}
for _cote in ("l", "r"):
    for _base in ("shoulder", "elbow", "wrist", "hand", "hip", "knee", "ankle", "toe", "heel"):
        _lim = LIMITS.get(_base + "_l")
        if _lim:
            LIMITS[_base + "_" + _cote] = _lim


# Articulations qui ne doivent jamais entrer dans le volume du tronc (coude dans le cou…).
MEMBRES_TESTES = ("elbow_l", "wrist_l", "hand_l", "elbow_r", "wrist_r", "hand_r")

# Marge acceptée sur les butées : l'IK du navigateur place les articulations au degré près.
TOLERANCE_BUTEE = 5.0

# Tendons/élastiques : (articulation, os parent, os enfant, axe de pliage préféré).
TENDONS = []
for _cote in ("l", "r"):
    TENDONS.append(("shoulder_" + _cote, "shoulder_" + _cote, "elbow_" + _cote, (-1, 0, 0)))
    TENDONS.append(("elbow_" + _cote, "elbow_" + _cote, "wrist_" + _cote, (1, 0, 0)))
    TENDONS.append(("hip_" + _cote, "hip_" + _cote, "knee_" + _cote, (1, 0, 0)))
    TENDONS.append(("knee_" + _cote, "knee_" + _cote, "ankle_" + _cote, (-1, 0, 0)))


def _composantes(v):
    """Un point de pose accepte un tuple, une liste ou un objet {x, y, z}."""
    if isinstance(v, dict):
        return (float(v["x"]), float(v["y"]), float(v["z"]))
    return (float(v[0]), float(v[1]), float(v[2]))


def rest_bend(nom: str) -> float:
    """Pli de repos (degrés) entre un os et son parent, dans la pose debout."""
    parent = PARENT.get(nom)
    d, dp = REST_DIRS.get(nom), (REST_DIRS.get(parent) if parent else None)
    if not d or not dp:
        return 0.0
    nd, ndp = _norm(tuple(d)), _norm(tuple(dp))
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(nd, ndp))))))


def _soustraction(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _somme(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _produit(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _angle(a, b):
    na, nb = _norm(a), _norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(na, nb))))))


def _interp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _coord(pose, nom):
    p = pose.get(nom)
    if p is None:
        return None
    return _composantes(p)


def _croix(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rotation(vector, axe, angle):
    """Rotation d'un vecteur autour d'un axe (formule de Rodrigues)."""
    c, sn = math.cos(angle), math.sin(angle)
    k = _norm(axe)
    kxv = _croix(k, vector)
    kv = sum(x * y for x, y in zip(k, vector))
    return tuple(vector[i] * c + kxv[i] * sn + k[i] * kv * (1 - c) for i in range(3))


def _ecrire(pose: dict, nom: str, point) -> None:
    """Réécrit un point de pose en conservant son type (liste, tuple ou objet)."""
    actuel = pose.get(nom)
    if isinstance(actuel, dict):
        actuel["x"], actuel["y"], actuel["z"] = point[0], point[1], point[2]
    elif isinstance(actuel, list):
        pose[nom] = [point[0], point[1], point[2]]
    else:
        pose[nom] = tuple(point)


def clamp_pose(pose: dict, lengths: dict | None = None) -> dict:
    """Ramène chaque os dans ses butées (cône autour du repos + charnière du parent).

    Même règle que le moteur du navigateur : une pose bricolée hors butées est
    redressée au lieu d'être refusée, comme lorsque l'utilisateur tire un membre.
    """
    lengths = lengths or bone_lengths({})
    for nom in BUILD_ORDER:
        parent = PARENT.get(nom)
        if not parent:
            continue
        lim = LIMITS.get(nom)
        p, q = _coord(pose, parent), _coord(pose, nom)
        if not lim or p is None or q is None:
            continue
        long = lengths.get(nom) or math.dist(p, q)
        if long <= 0:
            continue
        grand = PARENT.get(parent)
        r = _coord(pose, grand) if grand else None
        dir_parent = _norm(_soustraction(p, r)) if r else (0.0, 1.0, 0.0)
        out = _norm(_soustraction(q, p))
        if sum(v * v for v in out) < 1e-12:
            continue
        if "pli" in lim:
            pli = _angle(out, dir_parent) - rest_bend(nom)
            mini, maxi = lim["pli"]
            cible = max(float(mini), min(float(maxi), pli))
            if abs(cible - pli) > 1e-6:
                out = _rotation(out, _norm(_croix(out, dir_parent)), math.radians(pli - cible))
        dir_repos = REST_DIRS.get(nom)
        if "cone" in lim and dir_repos:
            repos = _norm(tuple(dir_repos))
            ecart = _angle(out, repos)
            if ecart > lim["cone"]:
                out = _rotation(out, _norm(_croix(out, repos)), math.radians(ecart - lim["cone"]))
        _ecrire(pose, nom, _somme(p, _produit(_norm(out), long)))
    return pose


def _buts_violees(pose: dict) -> list[tuple[str, float, float]]:
    """Articulations hors butées : (nom, angle mesuré, maximum autorisé), en degrés."""
    fautes = []
    for nom in BUILD_ORDER:
        parent = PARENT.get(nom)
        if not parent:
            continue
        lim = LIMITS.get(nom)
        if not lim:
            continue
        p, q = _coord(pose, nom), _coord(pose, parent)
        if p is None or q is None:
            continue
        dir_os = _norm(_soustraction(p, q))
        grand = PARENT.get(parent)
        r = _coord(pose, grand) if grand else None
        dir_parent = _norm(_soustraction(q, r)) if r else (0.0, 1.0, 0.0)
        pli = _angle(dir_os, dir_parent) - rest_bend(nom)
        # tolérance : l'IK du navigateur place le coude/genou au degré près.
        # Sans « pli » déclaré (épaule, hanche…), seule la position de repos compte.
        if "pli" in lim:
            mini, maxi = lim["pli"]
            if pli > maxi + TOLERANCE_BUTEE:
                fautes.append((nom, pli, maxi))
            elif pli < mini - TOLERANCE_BUTEE:
                fautes.append((nom, pli, mini))
        dir_repos = REST_DIRS.get(nom)
        if "cone" in lim and dir_repos:
            ecart = _angle(dir_os, _norm(dir_repos))
            if ecart > lim["cone"] + TOLERANCE_BUTEE:
                fautes.append((nom, ecart, float(lim["cone"])))
    return fautes


def _collisions(pose: dict, lengths: dict, thickness: dict) -> list[str]:
    """Articulations du bras entrées dans le volume du tronc (le mannequin se traverse)."""
    haut, cou = pose.get("hips"), pose.get("neck")
    if haut is None or cou is None:
        return []
    haut, cou = _composantes(haut), _composantes(cou)
    ecart = max(1e-3, cou[1] - haut[1])
    fautes = []
    for nom in MEMBRES_TESTES:
        p = _coord(pose, nom)
        if p is None or p[1] > cou[1] + 0.02:
            continue
        f = max(0.0, min(1.0, (p[1] - haut[1]) / ecart))
        if f <= 0.03:
            continue
        largeur, profondeur = _torso_size(f, thickness, pose)
        centre = _torso_frame(pose, lengths, f)["centre"]
        dx = (p[0] - centre[0]) / max(1e-3, largeur / 2)
        dz = (p[2] - centre[2]) / max(1e-3, profondeur / 2)
        if dx * dx + dz * dz < 0.98:
            fautes.append(nom)
    return fautes


def _tendon_points(pose: dict, lengths: dict, thickness: dict, tend: tuple) -> dict | None:
    """Géométrie d'un tendon : attaches de part et d'autre, du côté extérieur au pli."""
    joint, parent_os, enfant, axe_prefere = tend
    grand = PARENT.get(parent_os)
    a = _coord(pose, grand) if grand else None
    if a is None:
        a = _coord(pose, parent_os)
    j, b = _coord(pose, joint), _coord(pose, enfant)
    if a is None or j is None or b is None:
        return None
    da, db = _norm(_soustraction(j, a)), _norm(_soustraction(b, j))
    if _norm(da) == (0.0, 0.0, 0.0) or _norm(db) == (0.0, 0.0, 0.0):
        return None
    maxi = float(LIMITS.get(enfant, {}).get("pli", (0, 120))[1] or 120)
    pli = abs(_angle(da, db) - rest_bend(enfant))
    tension = max(0.0, min(1.0, pli / max(1.0, maxi)))
    rayon = max(0.02, _mean_radius(enfant, thickness) * profile_at(enfant, 0.06))
    dehors = _norm(_soustraction(da, db))
    if _norm(dehors) == (0.0, 0.0, 0.0):
        dehors = _norm((axe_prefere[1] * da[2] - axe_prefere[2] * da[1],
                        axe_prefere[2] * da[0] - axe_prefere[0] * da[2],
                        axe_prefere[0] * da[1] - axe_prefere[1] * da[0]))
    if _norm(dehors) == (0.0, 0.0, 0.0):
        dehors = (0.0, 0.0, -1.0)
    pA = _somme(_interp(a, j, 0.86), _produit(dehors, rayon * 0.34))
    pB = _somme(_interp(j, b, 0.18), _produit(dehors, rayon * 0.34))
    ctrl = _somme(_interp(pA, pB, 0.5), _produit(dehors, rayon * (0.06 + tension * 0.55)))
    return {"A": pA, "B": pB, "centre": ctrl, "tension": tension, "rayon": rayon}


def _draw_tendon(canvas, item: dict, camera, mode: str) -> None:
    """Trace un tendon : ombre douce puis cordon clair, tension selon le pli."""
    import cv2
    import numpy as np

    if mode != "volume":
        return
    hauteur, largeur = canvas.shape[:2]
    A = project(item["A"], camera, largeur, hauteur)
    B = project(item["B"], camera, largeur, hauteur)
    C = project(item["centre"], camera, largeur, hauteur)
    pts = []
    for i in range(13):
        t = i / 12
        x = (1 - t) ** 2 * A["x"] + 2 * (1 - t) * t * C["x"] + t * t * B["x"]
        y = (1 - t) ** 2 * A["y"] + 2 * (1 - t) * t * C["y"] + t * t * B["y"]
        pts.append([int(round(x)), int(round(y))])
    trace = np.array(pts, dtype="int32")
    echelle = (A["scale"] + B["scale"]) / 2
    ep = max(1, int(round(0.0095 * echelle)))
    for couleur, epaisseur, alpha in (((38, 50, 90), max(1, int(round(ep * 1.6))), 0.06 + 0.14 * item["tension"]),
                                      ((205, 221, 242), max(1, int(round(ep * 0.9))), 0.14 + 0.44 * item["tension"])):
        couche = np.zeros_like(canvas)
        cv2.polylines(couche, [trace], False, couleur, epaisseur, cv2.LINE_AA)
        m = couche.any(axis=2)
        canvas[m] = (canvas[m] * (1 - alpha) + couche[m] * alpha).astype("uint8")


# Longueurs de référence de chaque segment (mètres) — valeurs du tableau de l'interface.
BASE_LENGTHS = {
    "spine": 0.16, "chest": 0.20, "neck": 0.115, "head": 0.13, "head_top": 0.10,
    "shoulder_l": 0.19, "elbow_l": 0.28, "wrist_l": 0.25, "hand_l": 0.10,
    "shoulder_r": 0.19, "elbow_r": 0.28, "wrist_r": 0.25, "hand_r": 0.10,
    "hip_l": 0.11, "knee_l": 0.44, "ankle_l": 0.42, "toe_l": 0.17, "heel_l": 0.08,
    "hip_r": 0.11, "knee_r": 0.44, "ankle_r": 0.42, "toe_r": 0.17, "heel_r": 0.08,
}
# Épaisseurs de référence (diamètre au milieu du segment, mètres).
BASE_THICK = {
    "spine": 0.262, "chest": 0.315, "neck": 0.116, "head": 0.172, "head_top": 0.150,
    "shoulder_l": 0.108, "elbow_l": 0.110, "wrist_l": 0.088, "hand_l": 0.075,
    "shoulder_r": 0.108, "elbow_r": 0.110, "wrist_r": 0.088, "hand_r": 0.075,
    "hip_l": 0.180, "knee_l": 0.150, "ankle_l": 0.102, "toe_l": 0.080, "heel_l": 0.074,
    "hip_r": 0.180, "knee_r": 0.150, "ankle_r": 0.102, "toe_r": 0.080, "heel_r": 0.074,
}
MORPHOLOGIES = {
    "neutre": {"stature": 1, "shoulders": 1, "legs": 1, "arms": 1, "girth": 1},
    "fin": {"stature": 1, "shoulders": 0.88, "legs": 1.04, "arms": 0.97, "girth": 0.84},
    "athletique": {"stature": 1.01, "shoulders": 1.12, "legs": 1.0, "arms": 1.0, "girth": 1.16},
    "fort": {"stature": 1, "shoulders": 1.18, "legs": 0.97, "arms": 0.99, "girth": 1.34},
    "femme": {"stature": 0.97, "shoulders": 0.88, "legs": 1.02, "arms": 0.96, "girth": 0.92},
    "homme": {"stature": 1.03, "shoulders": 1.10, "legs": 1.0, "arms": 1.03, "girth": 1.12},
}
BUILDS = MORPHOLOGIES                     # ancien nom conservé

# Tronc : coupes elliptiques le long de la colonne (largeur/profondeur en mètres)
TORSO_PROFILE = [
    {"at": -0.14, "width": 0.158, "depth": 0.168},   # entrejambe
    {"at": 0.00, "width": 0.180, "depth": 0.175},    # sous le bassin
    {"at": 0.16, "width": 0.205, "depth": 0.205},    # bassin / hanches
    {"at": 0.45, "width": 0.166, "depth": 0.178},    # taille
    {"at": 0.70, "width": 0.238, "depth": 0.196},    # bas des côtes
    {"at": 0.86, "width": 0.302, "depth": 0.202},    # haut du thorax
    {"at": 0.94, "width": 0.330, "depth": 0.192},    # épaules
    {"at": 1.00, "width": 0.150, "depth": 0.150},    # base du cou
]
TORSO_BUMPS = [
    {"centre": 0.16, "sigma": 0.14, "scale": 1.02},   # hanches
    {"centre": 0.93, "sigma": 0.09, "scale": 0.99},   # épaules
]
TORSO_SLICES = 19
TORSO_BAS = -0.12                 # le tronc descend jusqu'à l'entrejambe

# ControlNet / OpenPose : 18 points, 17 membres, couleurs canoniques
OPENPOSE_18 = ["nose", "neck", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
               "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l",
               "eye_r", "eye_l", "ear_r", "ear_l"]
OPENPOSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                  (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
OPENPOSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
                   (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
                   (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
                   (255, 0, 255), (255, 0, 170), (255, 0, 85)]

SKIN = (188, 193, 198)          # BGR, teinte chair neutre
SKIN_DARK = (154, 160, 168)
LIMB_COLOR = {
    "torso": (214, 107, 91), "neck": (214, 107, 91), "head": (224, 122, 107),
    "shoulder": (214, 107, 91), "pelvis": (214, 107, 91), "arm": (196, 92, 75),
    "forearm": (196, 92, 75), "hand": (232, 147, 139),
    "thigh": (196, 92, 75), "shin": (196, 92, 75), "foot": (232, 147, 139),
}

MODES = ("volume", "wireframe", "openpose", "depth", "silhouette")

# écart maximal toléré entre la longueur d'un os envoyée et celle du gabarit (2 %)
BONE_TOLERANCE = 0.02


class MannequinError(RuntimeError):
    """Erreur de rendu du mannequin, affichée telle quelle dans l'interface."""


def _require_cv():
    _pose._require_cv()


def prepare(payload: dict, width: int, height: int) -> tuple[dict, dict, dict]:
    """Valide la charge utile et applique le cadrage automatique si la caméra est muette."""
    build, pose, camera = validate(payload)
    if "distance" not in (payload.get("camera") or {}):
        camera = fit_camera(pose, camera, build, width, height)
    return build, pose, camera


# --------------------------------------------------------------- géométrie
def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-9 else (0.0, 0.0, 0.0)


# Galbe de chaque segment (facteur du rayon de référence, 0 = parent, 1 = extrémité).
# Chaque profil est normalisé : l'épaisseur réglée reste le diamètre MOYEN du segment.
LIMB_PROFILE = {
    "neck": [[0, 1.04], [1, 0.88]],
    "shoulder_l": [[0, 0.98], [1, 0.9]], "shoulder_r": [[0, 0.98], [1, 0.9]],
    "elbow_l": [[0, 0.62], [0.14, 1.34], [0.42, 1.26], [0.72, 1.0], [1, 0.72]],
    "elbow_r": [[0, 0.62], [0.14, 1.34], [0.42, 1.26], [0.72, 1.0], [1, 0.72]],
    "wrist_l": [[0, 0.7], [0.18, 1.18], [0.5, 1.06], [0.85, 0.7], [1, 0.56]],
    "wrist_r": [[0, 0.7], [0.18, 1.18], [0.5, 1.06], [0.85, 0.7], [1, 0.56]],
    "hand_l": [[0, 0.9], [0.35, 1.06], [0.7, 0.94], [1, 0.6]],
    "hand_r": [[0, 0.9], [0.35, 1.06], [0.7, 0.94], [1, 0.6]],
    "knee_l": [[0, 0.7], [0.14, 1.24], [0.46, 1.16], [0.78, 0.9], [1, 0.66]],
    "knee_r": [[0, 0.7], [0.14, 1.24], [0.46, 1.16], [0.78, 0.9], [1, 0.66]],
    "ankle_l": [[0, 0.72], [0.22, 1.22], [0.5, 1.04], [0.8, 0.68], [1, 0.5]],
    "ankle_r": [[0, 0.72], [0.22, 1.22], [0.5, 1.04], [0.8, 0.68], [1, 0.5]],
    "toe_l": [[0, 1.0], [0.6, 0.94], [1, 0.78]], "toe_r": [[0, 1.0], [0.6, 0.94], [1, 0.78]],
    "heel_l": [[0, 1.0], [1, 0.88]], "heel_r": [[0, 1.0], [1, 0.88]],
}
for _nom, _profil in LIMB_PROFILE.items():
    _aire, _span = 0.0, 0.0
    for _i in range(len(_profil) - 1):
        _d = _profil[_i + 1][0] - _profil[_i][0]
        _aire += (_profil[_i][1] + _profil[_i + 1][1]) / 2 * _d
        _span += _d
    _moyenne = (_aire / _span) if _span > 0 else 1.0
    if _moyenne > 0:
        for _point in _profil:
            _point[1] /= _moyenne


def profile_at(child: str, u: float) -> float:
    """Facteur de galbe d'un segment à la fraction u (0 = parent, 1 = extrémité)."""
    profil = LIMB_PROFILE.get(child)
    if not profil:
        return 1.0
    x = max(0.0, min(1.0, u))
    for i in range(len(profil) - 1):
        if profil[i][0] <= x <= profil[i + 1][0]:
            d = profil[i + 1][0] - profil[i][0] or 1.0
            t = (x - profil[i][0]) / d
            return profil[i][1] + (profil[i + 1][1] - profil[i][1]) * t
    return profil[-1][1]


# Éclairage de la peau : lumière de haut-gauche, légèrement devant le personnage.
SKIN = (152, 176, 214)                 # albédo BGR (peau claire, mat)
SKIN_CHAUD = (104, 128, 226)           # liseré chaud (lumière qui traverse la peau)
LIGHT = (0.38, -0.82, -0.42)           # x = droite, y = haut (négatif dans l'image), z = avant
SKIN_AMBIENT, SKIN_DIFFUSE, SKIN_SPEC, SKIN_SHIN = 0.52, 0.56, 0.22, 26


def morphed_dimensions(morphology: str = "neutre") -> tuple[dict, dict]:
    """Longueurs et épaisseurs (mètres) correspondant à une morphologie type."""
    o = dict(MORPHOLOGIES["neutre"])
    o.update(MORPHOLOGIES.get(morphology) or {})
    lengths, thickness = {}, {}
    for bone, base in BASE_LENGTHS.items():
        f = float(o["stature"])               # tout le squelette suit la stature
        if bone.endswith("_l") or bone.endswith("_r"):
            if bone.startswith("shoulder"):
                f *= float(o["shoulders"])
            if bone.startswith(("elbow", "wrist")):
                f *= float(o["arms"])
            if bone.startswith(("knee", "ankle")):
                f *= float(o["legs"])
        lengths[bone] = base * f
        thickness[bone] = BASE_THICK[bone] * float(o["girth"])
    h = lengths["head"]
    lengths["nose"], lengths["eye_l"] = 0.985 * h, 0.930 * h
    lengths["eye_r"], lengths["ear_l"], lengths["ear_r"] = 0.930 * h, 0.625 * h, 0.625 * h
    thickness["nose"] = thickness["eye_l"] = thickness["eye_r"] = 0.06
    thickness["ear_l"] = thickness["ear_r"] = 0.05
    return lengths, thickness


def _with_face(lengths: dict) -> dict:
    h = lengths.get("head", 0.13)
    lengths["nose"] = 0.985 * h
    lengths["eye_l"] = lengths["eye_r"] = 0.930 * h
    lengths["ear_l"] = lengths["ear_r"] = 0.625 * h
    return lengths


def bone_lengths(build: dict | None = None) -> dict:
    """Longueurs d'os (mètres) — par morphologie (compatibilité) ou valeurs explicites."""
    if isinstance(build, dict) and "lengths" in build:
        # résultat de validate() : le tableau des dimensions fait foi
        return _with_face(dict(build["lengths"]))
    if build and any(k in build for k in BASE_LENGTHS):
        out = {k: float(v) for k, v in build.items() if k in BASE_LENGTHS}
        for bone, base in BASE_LENGTHS.items():
            out.setdefault(bone, base)
        return _with_face(out)
    o = dict(MORPHOLOGIES["neutre"])
    o.update(build or {})
    lengths = {k: v * float(o["stature"]) for k, v in BASE_LENGTHS.items()}
    for side in ("l", "r"):
        lengths["shoulder_" + side] *= float(o["shoulders"])
        lengths["elbow_" + side] *= float(o["arms"])
        lengths["wrist_" + side] *= float(o["arms"])
        lengths["knee_" + side] *= float(o["legs"])
        lengths["ankle_" + side] *= float(o["legs"])
    return _with_face(lengths)


# rapport de conicité de chaque segment, dérivé de la table de capsules
BONE_TAPER = {child: ((r1 + r2) / 2 and (r1 / ((r1 + r2) / 2), r2 / ((r1 + r2) / 2)))
              for _, child, r1, r2, _ in BONES}


def bone_radii(thickness: dict | None = None) -> list[tuple[float, float]]:
    """Rayons des capsules à partir des épaisseurs (diamètre au milieu)."""
    t = thickness or BASE_THICK
    out = []
    for _, child, *_ in BONES:
        dia = float(t.get(child, BASE_THICK[child]))
        k1, k2 = BONE_TAPER[child]
        out.append((dia / 2 * k1, dia / 2 * k2))
    return out


def default_pose(lengths: dict | None = None) -> dict:
    """Pose debout construite à partir des longueurs d'os (proportions exactes)."""
    if lengths is None or "spine" not in lengths:
        lengths = bone_lengths(lengths if isinstance(lengths, dict) else None)
    lengths = _with_face(dict(lengths))
    p = {"hips": (0.0, 0.0, 0.0)}
    for name in BUILD_ORDER:
        if name == "hips":
            continue
        parent = PARENT[name]
        d = _norm(REST_DIRS[name])
        rest = lengths.get(name, 0.0)
        pp = p[parent]
        p[name] = (pp[0] + d[0] * rest, pp[1] + d[1] * rest, pp[2] + d[2] * rest)
    # les pieds reposent sur le sol
    sol = min(p[j][1] for j in ("ankle_l", "ankle_r", "toe_l", "toe_r", "heel_l", "heel_r"))
    return {name: (q[0], q[1] - sol + 0.015, q[2]) for name, q in p.items()}


def rotation(yaw: float, pitch: float) -> tuple[tuple[float, float, float], ...]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    return ((cy, 0.0, sy),
            (sy * sx, cx, -cy * sx),
            (-sy * cx, sx, cy * cx))


def project(point, camera: dict, width: int, height: int) -> dict:
    """Projection perspective identique à celle du navigateur."""
    yaw = float(camera.get("yaw", 0.42))
    pitch = float(camera.get("pitch", 0.10))
    distance = float(camera.get("distance", 4.2))
    target = tuple(camera.get("target") or (0.0, 0.92, 0.0))
    if camera.get("focal"):
        focal = float(camera["focal"])
    else:
        focal = min(width, height) * 1.25
    r = rotation(yaw, pitch)
    px, py, pz = point[0] - target[0], point[1] - target[1], point[2] - target[2]
    cam = (r[0][0] * px + r[0][1] * py + r[0][2] * pz,
           r[1][0] * px + r[1][1] * py + r[1][2] * pz,
           r[2][0] * px + r[2][1] * py + r[2][2] * pz)
    depth = max(0.15, distance - cam[2])
    scale = focal / depth
    return {"x": width / 2 + cam[0] * scale, "y": height / 2 - cam[1] * scale,
            "depth": depth, "scale": scale}


def fit_camera(pose: dict, camera: dict, build: dict, width: int, height: int, margin: float = 1.22) -> dict:
    """Cadre la pose (équivalent de ``fitCamera`` côté navigateur)."""
    camera = dict(camera)
    ys_all = [q[1] for q in pose.values()]
    xs = [q[0] for q in pose.values()]
    ankles = [pose[n][1] for n in ("ankle_l", "ankle_r") if n in pose]
    center = (max(ys_all) + (min(ankles) if ankles else 0.0)) / 2
    camera["target"] = [0.0, center, 0.0]
    ys = ys_all
    span_y = max(0.4, max(ys) - min(ys)) * margin
    span_x = max(0.4, max(xs) - min(xs)) * margin
    focal = min(width, height) * 1.25
    camera["distance"] = max(1.4, min(14.0, max(span_y * focal / height, span_x * focal / width)))
    return camera


def check_bone_lengths(pose: dict, build: dict) -> list[str]:
    """Vérifie que la pose reçue conserve les longueurs d'os (proportions du gabarit).

    Le navigateur les garantit (IK 2 os + remise à longueur) ; cette vérification évite
    qu'une pose bricolée à la main produise un pantin étiré.
    """
    lengths = build.get("lengths") if isinstance(build, dict) and "lengths" in build else bone_lengths(build)
    abîmes = []
    for parent, child, *_ in BONES:
        if parent not in pose or child not in pose:
            continue
        mesure = math.dist(pose[parent], pose[child])
        attendu = lengths[child]
        if attendu > 0 and abs(mesure - attendu) / attendu > BONE_TOLERANCE:
            abîmes.append(child)
    return abîmes


def validate(payload: dict) -> tuple[dict, dict, dict]:
    """Vérifie la charge utile envoyée par le navigateur et renvoie (build, pose, camera)."""
    if not isinstance(payload, dict):
        raise MannequinError("pose invalide")
    # dimensions : valeurs explicites du tableau (mètres), sinon morphologie type
    morphology = str(payload.get("morphology") or "")
    lengths, thickness = morphed_dimensions(morphology)
    for bone, value in (payload.get("lengths") or {}).items():
        if bone not in BASE_LENGTHS:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise MannequinError(f"longueur invalide pour « {bone} »")
        if not 0.005 <= v <= 3:
            raise MannequinError(f"longueur hors limites pour « {bone} » : {v:g} m")
        lengths[bone] = v
    for bone, value in (payload.get("thickness") or {}).items():
        if bone not in BASE_THICK:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise MannequinError(f"épaisseur invalide pour « {bone} »")
        if not 0.005 <= v <= 1:
            raise MannequinError(f"épaisseur hors limites pour « {bone} » : {v:g} m")
        thickness[bone] = v
    _with_face(lengths)
    build = {"lengths": lengths, "thickness": thickness, "morphology": morphology or "neutre"}
    raw = payload.get("pose") or {}
    if not raw:
        # pose absente : on part du gabarit debout (utile pour les tests / l'API)
        base = default_pose(lengths)
        pose = {k: [round(c, 6) for c in v] for k, v in base.items()}
    else:
        if len(raw) < len(BUILD_ORDER):
            raise MannequinError(
                f"pose incomplète : {len(raw)} articulation(s) reçue(s), {len(BUILD_ORDER)} attendues.")
        pose = {}
        for name in JOINTS:
            v = raw.get(name)
            if not v:
                continue
            try:
                pose[name] = [float(v[0]), float(v[1]), float(v[2])]
            except (TypeError, ValueError, IndexError):
                raise MannequinError(f"coordonnées invalides pour « {name} »")
        if "hips" not in pose or "neck" not in pose:
            raise MannequinError("pose incomplète : « hips » et « neck » sont requis")
        deformes = check_bone_lengths(pose, build)
        if deformes:
            raise MannequinError(
                "pose incohérente : les os gardent une longueur fixe (le mannequin ne s'étire pas). "
                "Os hors gabarit : " + ", ".join(deformes[:6])
                + (f" (+{len(deformes) - 6})" if len(deformes) > 6 else "")
                + ". Déplacez les articulations depuis l'application (le coude et le genou se plient par IK).")
        buts = _buts_violees(pose)
        if buts:
            nom, mesure, maximum = buts[0]
            raise MannequinError(
                f"pose hors butées : « {nom} » à {mesure:.0f}° (maximum {maximum:.0f}°). "
                "Les articulations du mannequin s'arrêtent avant l'hyperextension.")
        travers = _collisions(pose, build["lengths"], build["thickness"])
        if travers:
            raise MannequinError(
                "pose impossible : le mannequin se traverse (« " + " », « ".join(travers[:3])
                + " » entre dans le tronc). Éloignez le bras du corps pour que le membre butte sur la peau.")
    camera = dict(payload.get("camera") or {})
    camera.setdefault("yaw", 0.42)
    camera.setdefault("pitch", 0.12)
    camera.setdefault("target", [0.0, 0.85, 0.0])
    return build, pose, camera


# ------------------------------------------------------------------- dessin
def _view(camera: dict) -> dict:
    """Repère de vue : matrice de rotation et axe caméra (vers l'objectif)."""
    r = rotation(float(camera.get("yaw", 0.42)), float(camera.get("pitch", 0.12)))
    return {"r": r, "eye": (r[2][0], r[2][1], r[2][2])}


def _skin_bgr(n, eye) -> tuple[int, int, int]:
    """Couleur de peau pour une normale unitaire : Lambert + reflet + liseré chaud."""
    diffuse = max(0.0, n[0] * LIGHT[0] + n[1] * LIGHT[1] + n[2] * LIGHT[2])
    h = (LIGHT[0] + eye[0], LIGHT[1] + eye[1], LIGHT[2] + eye[2])
    hl = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2) or 1.0
    nh = max(0.0, (n[0] * h[0] + n[1] * h[1] + n[2] * h[2]) / hl)
    spec = (nh ** SKIN_SHIN) * SKIN_SPEC
    face = min(1.0, abs(n[0] * eye[0] + n[1] * eye[1] + n[2] * eye[2]))
    bord = (1.0 - face) ** 2.2
    sss = bord * 0.38
    k = SKIN_AMBIENT + SKIN_DIFFUSE * diffuse
    return tuple(int(min(255.0, max(0.0, SKIN[i] * k + SKIN_CHAUD[i] * sss + 255 * spec)))
                 for i in range(3))


def _limb_frame(a3, b3, eye):
    """Repère d'ombrage d'un segment : axe, normale écran et axe caméra."""
    w = _norm((b3[0] - a3[0], b3[1] - a3[1], b3[2] - a3[2]))
    u = _norm((w[1] * eye[2] - w[2] * eye[1], w[2] * eye[0] - w[0] * eye[2], w[0] * eye[1] - w[1] * eye[0]))
    if _norm(u) == (0.0, 0.0, 0.0):
        u = (1.0, 0.0, 0.0)
    z = _norm((u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]))
    if z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2] < 0:
        z = (-z[0], -z[1], -z[2])
    return w, u, z


def _limb_mask(p0, p1, profil, mean_r, camera, shape, points: int = 20):
    """Masque du segment (silhouette musclée) + paramètres par pixel (u, t)."""
    import cv2
    import numpy as np

    dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
    long = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / long, dx / long
    gauche, droite = [], []
    for i in range(points + 1):
        u = i / points
        cx, cy = p0["x"] + dx * u, p0["y"] + dy * u
        echelle = p0["scale"] + (p1["scale"] - p0["scale"]) * u
        r = max(1.2, mean_r * _profil_valeur(profil, u) * echelle)
        gauche.append((cx + nx * r, cy + ny * r))
        droite.append((cx - nx * r, cy - ny * r))
    poly = np.array([[int(round(x)), int(round(y))] for x, y in droite + list(reversed(gauche))], dtype="int32")
    mask = np.zeros(shape, dtype="uint8")
    cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
    # bouts arrondis (sphères aux extrémités), comme les capsules du navigateur
    r0 = max(1, int(round(mean_r * _profil_valeur(profil, 0.0) * p0["scale"])))
    r1 = max(1, int(round(mean_r * _profil_valeur(profil, 1.0) * p1["scale"])))
    cv2.circle(mask, (int(round(p0["x"])), int(round(p0["y"]))), r0, 255, -1, cv2.LINE_AA)
    cv2.circle(mask, (int(round(p1["x"])), int(round(p1["y"]))), r1, 255, -1, cv2.LINE_AA)
    return mask, dx, dy, long, nx, ny


def _profil_valeur(profil, u: float) -> float:
    """Valeur d'un profil [(u, facteur), …] (même interpolation que le navigateur)."""
    if not profil:
        return 1.0
    x = max(profil[0][0], min(profil[-1][0], u))
    us = [pt[0] for pt in profil]
    fs = [pt[1] for pt in profil]
    for i in range(len(profil) - 1):
        if us[i] <= x <= us[i + 1]:
            d = us[i + 1] - us[i]
            t = 0.0 if d <= 0 else (x - us[i]) / d
            return fs[i] + (fs[i + 1] - fs[i]) * t
    return fs[-1]


def _draw_limb(canvas, p0, p1, profil, mean_r, a3, b3, camera, mode, fill=None) -> None:
    """Dessine un segment de chair : silhouette musclée, peau ombrée (vectorisé)."""
    import numpy as np

    view = _view(camera)
    mask, dx, dy, long, nx, ny = _limb_mask(p0, p1, profil, mean_r, camera, canvas.shape[:2])
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    alpha = (mask[ys, xs].astype("float64") / 255.0)[:, None]
    if mode == "silhouette":
        canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + 255 * alpha, 0, 255).astype("uint8")
        return
    if mode == "depth":
        g = float(fill if fill is not None else 128)
        canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + g * alpha, 0, 255).astype("uint8")
        return
    w, u_axe, z_axe = _limb_frame(a3, b3, view["eye"])
    px = (xs - p0["x"]).astype("float64")
    py = (ys - p0["y"]).astype("float64")
    u = np.clip((px * dx + py * dy) / (long * long), 0.0, 1.0)
    travers = px * nx + py * ny
    rayons = mean_r * np.interp(u, [pt[0] for pt in profil], [pt[1] for pt in profil]) * (
        p0["scale"] + (p1["scale"] - p0["scale"]) * u)
    t = np.clip(travers / np.maximum(1e-6, rayons), -1.0, 1.0)
    theta = math.pi * (1.0 - t) / 2.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    n0 = u_axe[0] * cos_t + z_axe[0] * sin_t
    n1 = u_axe[1] * cos_t + z_axe[1] * sin_t
    n2 = u_axe[2] * cos_t + z_axe[2] * sin_t
    eye = view["eye"]
    diffuse = np.clip(n0 * LIGHT[0] + n1 * LIGHT[1] + n2 * LIGHT[2], 0, None)
    hx, hy, hz = LIGHT[0] + eye[0], LIGHT[1] + eye[1], LIGHT[2] + eye[2]
    hl = math.sqrt(hx * hx + hy * hy + hz * hz) or 1.0
    nh = np.clip((n0 * hx + n1 * hy + n2 * hz) / hl, 0, None)
    spec = (nh ** SKIN_SHIN) * SKIN_SPEC
    face = np.abs(n0 * eye[0] + n1 * eye[1] + n2 * eye[2])
    sss = ((1.0 - face) ** 2.2) * 0.38
    k = SKIN_AMBIENT + SKIN_DIFFUSE * diffuse
    couleurs = np.empty((len(xs), 3), dtype="float64")
    for i in range(3):
        couleurs[:, i] = SKIN[i] * k + SKIN_CHAUD[i] * sss + 255.0 * spec
    canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + couleurs * alpha, 0, 255).astype("uint8")


def _draw_sphere(canvas, point3, radius, camera, mode, profondeur, d_min, d_max, gris=None) -> None:
    """Rotule ombrée d'une articulation (rayon en mètres)."""
    import cv2
    import numpy as np

    view = _view(camera)
    p = project(point3, camera, canvas.shape[1], canvas.shape[0])
    r = max(1, int(round(radius * p["scale"])))
    mask = np.zeros(canvas.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(p["x"]), int(p["y"])), r, 255, -1, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    alpha = (mask[ys, xs].astype("float64") / 255.0)[:, None]
    if mode == "silhouette":
        canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + 255 * alpha, 0, 255).astype("uint8")
        return
    if mode == "depth":
        g = float(gris if gris is not None else 128)
        canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + g * alpha, 0, 255).astype("uint8")
        return
    # normale approchée : la sphère est vue de face, décalée vers la lumière
    lx, ly = LIGHT[0] * view["r"][0][0] + LIGHT[1] * view["r"][0][1] + LIGHT[2] * view["r"][0][2], -(
        LIGHT[0] * view["r"][1][0] + LIGHT[1] * view["r"][1][1] + LIGHT[2] * view["r"][1][2])
    nl = math.hypot(lx, ly) or 1.0
    ex = (xs - p["x"]) / max(1.0, float(r))
    ey = (ys - p["y"]) / max(1.0, float(r))
    ep = np.clip(ex * ex + ey * ey, 0, 1)
    ez = np.sqrt(np.maximum(0.0, 1.0 - ep))
    n0 = ex * (lx / nl) * 0.55 + view["eye"][0] * ez
    n1 = -ey * (ly / nl) * 0.55 + view["eye"][1] * ez
    n2 = view["eye"][2] * ez + (lx / nl) * 0.55 * 0.2
    ln = np.sqrt(n0 ** 2 + n1 ** 2 + n2 ** 2)
    ln[ln == 0] = 1.0
    n0, n1, n2 = n0 / ln, n1 / ln, n2 / ln
    eye = view["eye"]
    diffuse = np.clip(n0 * LIGHT[0] + n1 * LIGHT[1] + n2 * LIGHT[2], 0, None)
    hx, hy, hz = LIGHT[0] + eye[0], LIGHT[1] + eye[1], LIGHT[2] + eye[2]
    hl = math.sqrt(hx * hx + hy * hy + hz * hz) or 1.0
    nh = np.clip((n0 * hx + n1 * hy + n2 * hz) / hl, 0, None)
    spec = (nh ** SKIN_SHIN) * SKIN_SPEC
    face = np.abs(n0 * eye[0] + n1 * eye[1] + n2 * eye[2])
    sss = ((1.0 - face) ** 2.2) * 0.30
    k = SKIN_AMBIENT + SKIN_DIFFUSE * diffuse
    couleurs = np.empty((len(xs), 3), dtype="float64")
    for i in range(3):
        couleurs[:, i] = SKIN[i] * k + SKIN_CHAUD[i] * sss + 255.0 * spec
    canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + couleurs * alpha, 0, 255).astype("uint8")


def _torso_frame(pose: dict, lengths: dict, f: float) -> dict:
    """Repère local du tronc à la fraction f (0 = bassin, 1 = cou)."""
    chain = ["hips", "spine", "chest", "neck"]
    weights = [max(1e-6, lengths.get("spine", 0.16)), max(1e-6, lengths.get("chest", 0.2)),
               max(1e-6, lengths.get("neck", 0.1))]
    total = sum(weights)
    reste = max(0.0, min(1.0, f)) * total
    i = 0
    while i < len(weights) - 1 and reste > weights[i]:
        reste -= weights[i]
        i += 1
    t = max(0.0, min(1.0, reste / weights[i]))
    A, B = pose[chain[i]], pose[chain[i + 1]]
    centre = tuple(A[k] + (B[k] - A[k]) * t for k in range(3))
    axis = _norm(tuple(B[k] - A[k] for k in range(3)))
    bas = _norm(tuple(pose["hip_l"][k] - pose["hip_r"][k] for k in range(3)))
    haut = _norm(tuple(pose["shoulder_l"][k] - pose["shoulder_r"][k] for k in range(3)))
    melange = tuple(bas[k] * (1 - f) + haut[k] * f for k in range(3))
    proj = tuple(melange[k] - axis[k] * sum(melange[j] * axis[j] for j in range(3)) for k in range(3))
    x = _norm(proj)
    if sum(v * v for v in x) < 1e-12:
        x = bas
    z = _norm((x[1] * axis[2] - x[2] * axis[1], x[2] * axis[0] - x[0] * axis[2], x[0] * axis[1] - x[1] * axis[0]))
    if f < 0:
        centre = tuple(centre[k] - axis[k] * (-f * (weights[0] + weights[1])) for k in range(3))
    return {"centre": centre, "axis": axis, "x": x, "z": z}


def _torso_size(f: float, thickness: dict, pose: dict) -> tuple[float, float]:
    """Largeur et profondeur du tronc à la fraction f, d'après les épaisseurs réglées."""
    profil = TORSO_PROFILE
    a, b = profil[0], profil[-1]
    for i in range(len(profil) - 1):
        if profil[i]["at"] <= f <= profil[i + 1]["at"]:
            a, b = profil[i], profil[i + 1]
            break
    t = 0.0 if b["at"] == a["at"] else (f - a["at"]) / (b["at"] - a["at"])
    interp = lambda k: a[k] + (b[k] - a[k]) * t            # noqa: E731
    girth = (thickness.get("spine", BASE_THICK["spine"]) + thickness.get("chest", BASE_THICK["chest"])) / (
        BASE_THICK["spine"] + BASE_THICK["chest"])
    width = interp("width") * girth
    depth = interp("depth") * girth
    epaules = math.dist(pose["shoulder_l"], pose["shoulder_r"]) + 0.05
    hanches = math.dist(pose["hip_l"], pose["hip_r"]) + 0.09
    for bump, ref in zip(TORSO_BUMPS, (hanches, epaules)):
        g = math.exp(-((f - bump["centre"]) / bump["sigma"]) ** 2)
        width = max(width, ref * bump["scale"] * max(0.12, g))
    if f < 0:
        k = max(0.0, min(1.0, (0 - f) / 0.14))
        ferme = 1 - 0.42 * k
        return width * ferme, depth * ferme
    retrecissement = 1 - 0.25 * max(0.0, min(1.0, (f - 0.95) / 0.05))
    return width * retrecissement, depth * retrecissement


def _torso_ring(pose: dict, lengths: dict, thickness: dict, f: float, ring_points: int = 16):
    """Coupe du tronc à la fraction f : points 3D + repère."""
    frame = _torso_frame(pose, lengths, f)
    width, depth = _torso_size(f, thickness, pose)
    pts = []
    for i in range(ring_points):
        th = (i / ring_points) * 2 * math.pi
        pts.append(tuple(frame["centre"][k] + frame["x"][k] * math.cos(th) * width / 2
                         + frame["z"][k] * math.sin(th) * depth / 2 for k in range(3)))
    return pts, frame, (width, depth)


def _surface_tronc(pose, lengths, thickness, f, phi):
    """Point de la surface du tronc (f = hauteur, phi = angle autour du volume)."""
    frame = _torso_frame(pose, lengths, f)
    width, depth = _torso_size(f, thickness, pose)
    return tuple(frame["centre"][k] + frame["x"][k] * math.cos(phi) * width / 2
                 + frame["z"][k] * math.sin(phi) * depth / 2 for k in range(3))


def _couleur_anneau(view, x_axis, z_axis, t):
    """Couleur de peau à la position t (0 = bord gauche, 1 = bord droit) d'un anneau."""

    theta = math.pi * t
    c, sn = math.cos(theta), math.sin(theta)
    normale = tuple(x_axis[k] * c + z_axis[k] * sn for k in range(3))
    return _skin_bgr(normale, view["eye"])


def _draw_rings(canvas, anneaux, camera, mode, profondeur, d_min, d_max, cues=None, gris=None):
    """Dessine un volume d'anneaux (tronc, tête) : peau ombrée continue, sans couture."""
    import cv2
    import numpy as np

    view = _view(camera)
    hauteur, largeur = canvas.shape[:2]
    proj = []
    for pts, frame, taille in anneaux:
        p = [project(q, camera, largeur, hauteur) for q in pts]
        proj.append({"gauche": min(p, key=lambda q: q["x"]), "droite": max(p, key=lambda q: q["x"]),
                     "centre": project(frame["centre"], camera, largeur, hauteur),
                     "x": frame["x"], "z": frame["z"]})
    poly = np.array([[int(round(q["x"])), int(round(q["y"]))] for q in
                     [a["droite"] for a in proj] + [a["gauche"] for a in reversed(proj)]], dtype="int32")
    mask = np.zeros((hauteur, largeur), dtype="uint8")
    cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
    if mode in ("silhouette", "depth"):
        g = 255.0 if mode == "silhouette" else float(gris if gris is not None else 128)
        alpha = (mask.astype("float64") / 255.0)[:, :, None]
        canvas[:, :] = np.clip(canvas * (1 - alpha) + g * alpha, 0, 255).astype("uint8")
        return
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    # repères de référence pour orienter les normales de la même façon partout
    ref = proj[len(proj) // 2]
    axes = []
    for a in proj:
        axe_x, axe_z = a["x"], a["z"]
        if sum(axe_x[k] * ref["x"][k] for k in range(3)) < 0:
            axe_x = tuple(-v for v in axe_x)
        if sum(axe_z[k] * ref["z"][k] for k in range(3)) < 0:
            axe_z = tuple(-v for v in axe_z)
        axes.append((np.array(axe_x), np.array(axe_z)))
    ordre = np.argsort([a["centre"]["y"] for a in proj])
    cy = np.array([proj[i]["centre"]["y"] for i in ordre])
    cx = np.array([proj[i]["centre"]["x"] for i in ordre])
    demi = np.array([max(1.0, (proj[i]["droite"]["x"] - proj[i]["gauche"]["x"]) / 2) for i in ordre])
    ax = np.array([axes[i][0] for i in ordre])          # (n, 3)
    az = np.array([axes[i][1] for i in ordre])
    bande = np.clip(np.searchsorted(cy, ys) - 1, 0, len(cy) - 2)
    suit = np.clip(bande + 1, 0, len(cy) - 1)
    span = np.maximum(1e-6, cy[suit] - cy[bande])
    melange = np.clip((ys - cy[bande]) / span, 0, 1)[:, None]
    centre_x = cx[bande] + (cx[suit] - cx[bande]) * melange[:, 0]
    demi_x = demi[bande] + (demi[suit] - demi[bande]) * melange[:, 0]
    n0 = (ax[bande] * (1 - melange) + ax[suit] * melange)
    n2 = (az[bande] * (1 - melange) + az[suit] * melange)
    t = np.clip((xs - centre_x) / (2 * demi_x) + 0.5, 0, 1)
    theta = math.pi * t
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    vx = n0[:, 0] * cos_t + n2[:, 0] * sin_t
    vy = n0[:, 1] * cos_t + n2[:, 1] * sin_t
    vz = n0[:, 2] * cos_t + n2[:, 2] * sin_t
    eye = view["eye"]
    diffuse = np.clip(vx * LIGHT[0] + vy * LIGHT[1] + vz * LIGHT[2], 0, None)
    hx, hy, hz = LIGHT[0] + eye[0], LIGHT[1] + eye[1], LIGHT[2] + eye[2]
    hl = math.sqrt(hx * hx + hy * hy + hz * hz) or 1.0
    nh = np.clip((vx * hx + vy * hy + vz * hz) / hl, 0, None)
    spec = (nh ** SKIN_SHIN) * SKIN_SPEC
    face = np.abs(vx * eye[0] + vy * eye[1] + vz * eye[2])
    sss = ((1.0 - face) ** 2.2) * 0.38
    k = SKIN_AMBIENT + SKIN_DIFFUSE * diffuse
    couleurs = np.empty((len(xs), 3), dtype="float64")
    for c in range(3):
        couleurs[:, c] = SKIN[c] * k + SKIN_CHAUD[c] * sss + 255.0 * spec
    alpha = (mask[ys, xs].astype("float64") / 255.0)[:, None]
    canvas[ys, xs] = np.clip(canvas[ys, xs] * (1 - alpha) + couleurs * alpha, 0, 255).astype("uint8")


def _head_frame(pose: dict, thickness: dict) -> dict:
    """Repère de la tête : axe cou → sommet, largeur et profondeur réglées."""
    bas, haut = pose["neck"], pose["head_top"]
    axis = _norm(tuple(haut[k] - bas[k] for k in range(3)))
    epaules = _norm(tuple(pose["shoulder_l"][k] - pose["shoulder_r"][k] for k in range(3)))
    proj = tuple(epaules[k] - axis[k] * sum(epaules[j] * axis[j] for j in range(3)) for k in range(3))
    x = _norm(proj)
    if sum(v * v for v in x) < 1e-12:
        x = (1.0, 0.0, 0.0)
    z = _norm((x[1] * axis[2] - x[2] * axis[1], x[2] * axis[0] - x[0] * axis[2], x[0] * axis[1] - x[1] * axis[0]))
    avant = _norm(tuple(pose["nose"][k] - bas[k] for k in range(3))) if "nose" in pose else z
    if sum(z[k] * avant[k] for k in range(3)) < 0:
        z = tuple(-v for v in z)
    largeur = thickness.get("head", 0.176)
    return {"bas": bas, "haut": haut, "axis": axis, "x": x, "z": z,
            "longueur": math.dist(bas, haut), "largeur": largeur, "profondeur": largeur * 1.18}


def _head_profile(u: float) -> float:
    """Galbe de la tête (0 = menton, 1 = sommet)."""
    profil = [(0, 0.26), (0.06, 0.46), (0.14, 0.62), (0.26, 0.80), (0.40, 0.92),
              (0.56, 0.99), (0.72, 1.0), (0.85, 0.94), (0.94, 0.78), (1, 0.40)]
    return _profil_valeur(profil, u)


def _head_face(u: float) -> float:
    """Avancée du visage (fraction de la profondeur) selon la hauteur."""
    return 0.16 * math.exp(-((u - 0.22) / 0.20) ** 2) + 0.10 * math.exp(-((u - 0.48) / 0.30) ** 2)


def _head_rings(pose: dict, thickness: dict, slices: int = 11, ring_points: int = 14):
    """Anneaux de la tête du menton au sommet."""
    fr = _head_frame(pose, thickness)
    anneaux = []
    for i in range(slices):
        u = i / (slices - 1)
        centre = tuple(fr["bas"][k] + fr["axis"][k] * fr["longueur"] * (u * 0.96 - 0.10)
                       + fr["z"][k] * fr["profondeur"] * _head_face(u) for k in range(3))
        r = _head_profile(u)
        pts = []
        for j in range(ring_points):
            th = (j / ring_points) * 2 * math.pi
            pts.append(tuple(centre[k] + fr["x"][k] * math.cos(th) * fr["largeur"] / 2 * r * 1.02
                             + fr["z"][k] * math.sin(th) * fr["profondeur"] / 2 * r for k in range(3)))
        anneaux.append((pts, {"centre": centre, "x": fr["x"], "z": fr["z"], "axis": fr["axis"]}, (0.0, 0.0)))
    return anneaux, fr


def _head_surface(fr, u, phi):
    r = _head_profile(u)
    centre = tuple(fr["bas"][k] + fr["axis"][k] * fr["longueur"] * (u * 0.96 - 0.10)
                   + fr["z"][k] * fr["profondeur"] * _head_face(u) for k in range(3))
    return tuple(centre[k] + fr["x"][k] * math.cos(phi) * fr["largeur"] / 2 * r * 1.02
                 + fr["z"][k] * math.sin(phi) * fr["profondeur"] / 2 * r for k in range(3))


def _tache(canvas, p3, rx, ry, couleur, alpha, camera) -> None:
    """Tache douce (repère anatomique, œil, bouche…) dessinée sur la peau."""
    import cv2
    import numpy as np

    p = project(p3, camera, canvas.shape[1], canvas.shape[0])
    r = max(1, int(round(max(rx, ry) * p["scale"])))
    if r < 1:
        return
    mask = np.zeros(canvas.shape[:2], dtype="uint8")
    cv2.ellipse(mask, (int(p["x"]), int(p["y"])), (max(1, int(rx * p["scale"])), max(1, int(ry * p["scale"]))),
                0, 0, 360, 255, -1, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    a = max(0.0, min(1.0, alpha))
    canvas[ys, xs] = (canvas[ys, xs] * (1 - a) + np.array(couleur, dtype="float64") * a).astype("uint8")


JOINT_BALLS: tuple = ()                # rotules à dessiner en plus des segments (aucune par défaut)


def _joint_radius(joint: str, thickness: dict) -> float:
    """Rayon de la rotule : celui des segments qui s'y raccordent."""
    r = 0.02
    for parent, child, *_ in BONES:
        if child == joint:
            r = max(r, _mean_radius(child, thickness) * profile_at(child, 0.0))
        if parent == joint:
            r = max(r, _mean_radius(child, thickness) * profile_at(child, 1.0),
                    _mean_radius(child, thickness) * profile_at(child, 0.0) * 0.92)
    if joint in ("hip_l", "hip_r"):
        r = min(r, 0.052)
    if joint in ("shoulder_l", "shoulder_r"):
        r = min(r, 0.046)
    if joint in ("elbow_l", "elbow_r", "knee_l", "knee_r"):
        r *= 0.94
    return r


def _grid(canvas, camera, width, height):
    """Sol quadrillé (repère de profondeur) sous le personnage."""
    import cv2
    for i in range(-6, 7):
        a = project((i * 0.35, 0.0, -2.2), camera, width, height)
        b = project((i * 0.35, 0.0, 3.4), camera, width, height)
        cv2.line(canvas, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (78, 88, 96), 1, cv2.LINE_AA)
        c = project((-2.2, 0.0, i * 0.35), camera, width, height)
        d = project((3.4, 0.0, i * 0.35), camera, width, height)
        cv2.line(canvas, (int(c["x"]), int(c["y"])), (int(d["x"]), int(d["y"])), (78, 88, 96), 1, cv2.LINE_AA)


def _mean_radius(child: str, thickness: dict) -> float:
    """Rayon moyen d'un segment (moitié de l'épaisseur réglée)."""
    t = thickness.get(child)
    if t:
        return t / 2
    return BASE_THICK.get(child, 0.08) / 2


# Repères anatomiques du tronc (f = hauteur, phi = angle autour du volume)
CUES_TRONC = [
    {"f": 0.90, "phi": 0.55, "rx": 0.055, "ry": 0.020, "alpha": 0.045, "clair": True},
    {"f": 0.90, "phi": 2.59, "rx": 0.055, "ry": 0.020, "alpha": 0.045, "clair": True},
    {"f": 0.755, "phi": 1.10, "rx": 0.042, "ry": 0.030, "alpha": 0.05},
    {"f": 0.755, "phi": 2.04, "rx": 0.042, "ry": 0.030, "alpha": 0.05},
    {"f": 0.50, "phi": 1.57, "rx": 0.013, "ry": 0.011, "alpha": 0.22},
    {"f": 0.26, "phi": 4.71, "rx": 0.055, "ry": 0.024, "alpha": 0.07},
]


def render(payload: dict, mode: str = "volume", width: int = 768, height: int = 1024) -> "object":
    """Rend le mannequin dans le mode demandé et renvoie une image BGR (uint8).

    Le personnage est dessiné comme un corps : tronc et membres galbés, peau ombrée
    (lumière de haut-gauche, reflet, liseré chaud), tête et pieds reconnaissables.
    """
    _require_cv()
    import cv2
    import numpy as np

    if mode not in MODES:
        raise MannequinError(f"rendu inconnu : {mode}")
    width, height = max(128, int(width)), max(128, int(height))
    build, pose, camera = prepare(payload, width, height)
    lengths, thickness = build["lengths"], build["thickness"]

    ss = 2 if mode in ("volume", "wireframe") else 1     # anticrénelage
    W, H = width * ss, height * ss
    background = (28, 24, 20) if mode == "volume" else (16, 12, 10)
    canvas = np.full((H, W, 3), background, dtype="uint8")
    if mode == "openpose":
        canvas = np.zeros((H, W, 3), dtype="uint8")

    projected = {name: project(pose[name], camera, W, H) for name in pose}

    if mode == "openpose":
        pts = [projected[n] for n in OPENPOSE_18 if n in projected]
        if len(pts) < len(OPENPOSE_18):
            raise MannequinError("pose incomplète pour un squelette OpenPose")
        stick = max(2, int(min(W, H) / 130))
        for i, (a, b) in enumerate(OPENPOSE_LIMBS):
            color = OPENPOSE_COLORS[i % len(OPENPOSE_COLORS)]
            pa, pb = projected[OPENPOSE_18[a]], projected[OPENPOSE_18[b]]
            cv2.line(canvas, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), color, stick, cv2.LINE_AA)
            cv2.circle(canvas, (int(pa["x"]), int(pa["y"])), max(2, stick // 2), color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(pb["x"]), int(pb["y"])), max(2, stick // 2), color, -1, cv2.LINE_AA)
        return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

    if mode == "depth":
        canvas = np.zeros((H, W, 3), dtype="uint8")
    if mode == "volume" and payload.get("ground"):
        _grid(canvas, camera, W, H)
    if mode in ("volume", "wireframe"):
        hips = pose.get("hips", (0, 0, 0))
        sol = project((hips[0], 0.002, hips[2]), camera, W, H)
        shadow = np.zeros_like(canvas)
        cv2.ellipse(shadow, (int(sol["x"]), int(sol["y"])),
                    (int(0.60 * sol["scale"]), int(0.17 * sol["scale"])), 0, 0, 360, (52, 40, 34), -1, cv2.LINE_AA)
        np.copyto(canvas, (canvas * 0.68 + shadow * 0.32).astype("uint8"), where=shadow > 0)

    anatomique = mode in ("volume", "silhouette", "depth")
    items = []
    if anatomique:
        anneaux = []
        for i in range(TORSO_SLICES):
            f = TORSO_BAS + (i / (TORSO_SLICES - 1)) * (1 - TORSO_BAS)
            pts, frame, taille = _torso_ring(pose, lengths, thickness, f)
            anneaux.append((pts, frame, taille))
        profondeur = sum(project(a[1]["centre"], camera, W, H)["depth"] for a in anneaux) / len(anneaux)
        items.append({"type": "torso", "anneaux": anneaux, "depth": profondeur})
        tete, fr = _head_rings(pose, thickness)
        items.append({"type": "head", "anneaux": tete, "frame": fr,
                      "depth": project(fr["bas"], camera, W, H)["depth"] - 0.01})
        for cote in ("l", "r"):
            ankle = pose["ankle_" + cote]
            cheville = thickness.get("ankle_" + cote, 0.102)
            doigts = thickness.get("toe_" + cote, 0.08)
            cote3 = cheville * 0.44
            lame = doigts * 0.46
            bas = (ankle[0], 0.012, ankle[2])
            basT = (pose["toe_" + cote][0], 0.016, pose["toe_" + cote][2])
            basH = (pose["heel_" + cote][0], 0.014, pose["heel_" + cote][2])
            items.append({"type": "foot", "a3": basH, "b3": bas, "c3": basT,
                          "rTal": cote3 * 1.02, "rAvant": cote3 * 0.98, "rPointe": lame * 0.95,
                          "depth": max(project(bas, camera, W, H)["depth"], project(basT, camera, W, H)["depth"]) + 0.01})

    for parent, child, *_ in BONES:
        if anatomique and (child in ("hip_l", "hip_r", "shoulder_l", "shoulder_r", "spine", "chest", "head", "head_top",
                                     "nose", "toe_l", "toe_r", "heel_l", "heel_r")):
            continue
        pa, pb = projected.get(parent), projected.get(child)
        if pa is None or pb is None:
            continue
        items.append({"type": "limb", "a": pa, "b": pb, "a3": pose[parent], "b3": pose[child],
                      "child": child, "profil": LIMB_PROFILE.get(child, [(0, 1.0), (1, 1.0)]),
                      "mean_r": _mean_radius(child, thickness), "depth": (pa["depth"] + pb["depth"]) / 2})

    if anatomique:
        for joint in JOINT_BALLS:            # aucune rotule : le galbe des segments suffit
            items.append({"type": "joint", "joint": joint, "r": _joint_radius(joint, thickness) * 0.98,
                          "depth": projected[joint]["depth"] - 0.005})

    items.sort(key=lambda it: -it["depth"])
    profondeurs = [it["depth"] for it in items] or [camera.get("distance", 3.0)]
    d_min, d_max = min(profondeurs), max(profondeurs)

    def gris(depth: float) -> int:
        t = 0.5 if d_max <= d_min else (depth - d_min) / (d_max - d_min)
        return int(round(255 - 205 * t))

    for it in items:
        if it["type"] == "torso":
            _draw_rings(canvas, it["anneaux"], camera, mode, it["depth"], d_min, d_max,
                        gris=gris(it["depth"]))
        elif it["type"] == "head":
            _draw_rings(canvas, it["anneaux"], camera, mode, it["depth"], d_min, d_max,
                        gris=gris(it["depth"]))
        elif it["type"] == "foot":
            profil_talon = [(0, it["rTal"]), (1, it["rTal"] * 0.88)]
            profil_avant = [(0, it["rAvant"]), (1, it["rAvant"] * 0.7 + it["rPointe"] * 0.3)]
            _draw_limb(canvas, project(it["a3"], camera, W, H), project(it["b3"], camera, W, H),
                       profil_talon, 1.0, it["a3"], it["b3"], camera, mode, gris(it["depth"]))
            _draw_limb(canvas, project(it["b3"], camera, W, H), project(it["c3"], camera, W, H),
                       profil_avant, 1.0, it["b3"], it["c3"], camera, mode, gris(it["depth"]))
        elif it["type"] == "joint":
            _draw_sphere(canvas, pose[it["joint"]], it["r"], camera, mode, it["depth"], d_min, d_max,
                         gris=gris(it["depth"]))
        else:
            _draw_limb(canvas, it["a"], it["b"], it["profil"], it["mean_r"], it["a3"], it["b3"],
                       camera, mode, gris(it["depth"]))

    if mode == "volume":
        for tend in TENDONS:
            item = _tendon_points(pose, lengths, thickness, tend)
            if item:
                _draw_tendon(canvas, item, camera, mode)

    if mode == "volume" and payload.get("anatomy") is not False:
        for cue in CUES_TRONC:
            p3 = _surface_tronc(pose, lengths, thickness, cue["f"], cue["phi"])
            couleur = (255, 246, 239) if cue.get("clair") else (18, 26, 43)
            _tache(canvas, p3, cue["rx"], cue["ry"], couleur, cue["alpha"], camera)
        fr_tete = _head_rings(pose, thickness)[1]
        lg = fr_tete["largeur"]
        surface_tete = lambda u, phi: _head_surface(fr_tete, u, phi)     # noqa: E731
        for cote in (1, -1):                       # sourcils, yeux, orbites
            _tache(canvas, surface_tete(0.67, math.pi / 2 + cote * 0.42), lg * 0.105, lg * 0.022,
                   (43, 26, 20), 0.22, camera)
            _tache(canvas, surface_tete(0.60, math.pi / 2 + cote * 0.42), lg * 0.10, lg * 0.052,
                   (41, 50, 74), 0.50, camera)
            _tache(canvas, surface_tete(0.65, math.pi / 2 + cote * 0.42), lg * 0.13, lg * 0.040,
                   (20, 26, 43), 0.08, camera)
        _tache(canvas, surface_tete(0.58, math.pi / 2), lg * 0.028, lg * 0.075,       # arete du nez
               (232, 242, 255), 0.30, camera)
        _tache(canvas, surface_tete(0.55, math.pi / 2 + 0.24), lg * 0.045, lg * 0.07,  # flanc ombre
               (74, 44, 34), 0.16, camera)
        for cote in (1, -1):                                                          # narines
            _tache(canvas, surface_tete(0.475, math.pi / 2 + cote * 0.10), lg * 0.016, lg * 0.012,
                   (68, 42, 44), 0.34, camera)
        _tache(canvas, surface_tete(0.34, math.pi / 2), lg * 0.13, lg * 0.035,        # bouche
               (52, 63, 109), 0.42, camera)
        _tache(canvas, surface_tete(0.20, math.pi / 2), lg * 0.065, lg * 0.032,       # menton
               (216, 231, 255), 0.12, camera)
        for cote in (1, -1):                       # oreilles
            _tache(canvas, surface_tete(0.55, cote * 0.02), lg * 0.045, lg * 0.085,
                   (143, 169, 211), 0.85, camera)

    if ss > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    if mode == "silhouette":
        blanc = np.where(canvas.max(axis=2) > 40, 255, 0).astype("uint8")
        canvas = np.repeat(blanc[:, :, None], 3, axis=2)
    return canvas


def openpose_points(payload: dict, width: int, height: int) -> dict[str, list[float]]:
    """Points 2D du squelette (repère image), pour vérifier la pose ou la réutiliser."""
    _build, pose, camera = prepare(payload, width, height)
    points = {}
    for name in OPENPOSE_18:
        if name not in pose:
            continue
        p = project(pose[name], camera, width, height)
        points[name] = [round(p["x"], 2), round(p["y"], 2)]
    return points


def save(payload: dict, dst: str | Path, mode: str = "volume", width: int = 768, height: int = 1024) -> Path:
    """Rend et enregistre l'image (PNG)."""
    img = render(payload, mode, width, height)
    return _pose.save_image(img, dst)
