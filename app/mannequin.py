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

# Longueurs de référence de chaque segment (mètres) — valeurs du tableau de l'interface.
BASE_LENGTHS = {
    "spine": 0.16, "chest": 0.20, "neck": 0.10, "head": 0.13, "head_top": 0.10,
    "shoulder_l": 0.19, "elbow_l": 0.28, "wrist_l": 0.25, "hand_l": 0.10,
    "shoulder_r": 0.19, "elbow_r": 0.28, "wrist_r": 0.25, "hand_r": 0.10,
    "hip_l": 0.11, "knee_l": 0.44, "ankle_l": 0.42, "toe_l": 0.17, "heel_l": 0.08,
    "hip_r": 0.11, "knee_r": 0.44, "ankle_r": 0.42, "toe_r": 0.17, "heel_r": 0.08,
}
# Épaisseurs de référence (diamètre au milieu du segment, mètres).
BASE_THICK = {
    "spine": 0.262, "chest": 0.315, "neck": 0.168, "head": 0.176, "head_top": 0.155,
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
}
BUILDS = MORPHOLOGIES                     # ancien nom conservé

# Tronc : coupes elliptiques le long de la colonne (largeur/profondeur en mètres)
TORSO_PROFILE = [
    {"at": 0.00, "width": 0.170, "depth": 0.165},
    {"at": 0.16, "width": 0.190, "depth": 0.198},
    {"at": 0.45, "width": 0.158, "depth": 0.172},
    {"at": 0.70, "width": 0.245, "depth": 0.200},
    {"at": 0.88, "width": 0.215, "depth": 0.195},
    {"at": 1.00, "width": 0.145, "depth": 0.145},
]
TORSO_BUMPS = [
    {"centre": 0.16, "sigma": 0.15, "scale": 1.05},
    {"centre": 0.90, "sigma": 0.10, "scale": 0.93},
]
TORSO_SLICES = 13

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
    camera = dict(payload.get("camera") or {})
    camera.setdefault("yaw", 0.42)
    camera.setdefault("pitch", 0.12)
    camera.setdefault("target", [0.0, 0.85, 0.0])
    return build, pose, camera


# ------------------------------------------------------------------- dessin
def _chain_point(pose, chain, weights, f):
    """Point du tronc à la fraction f (0 = bassin, 1 = cou), et direction de la colonne."""
    total = sum(weights) or 1.0
    reste = max(0.0, min(1.0, f)) * total
    i = 0
    while i < len(weights) - 1 and reste > weights[i]:
        reste -= weights[i]
        i += 1
    t = max(0.0, min(1.0, reste / weights[i])) if weights[i] > 0 else 0.0
    A, B = pose[chain[i]], pose[chain[i + 1]]
    centre = tuple(A[k] + (B[k] - A[k]) * t for k in range(3))
    axe = _norm((B[0] - A[0], B[1] - A[1], B[2] - A[2]))
    return centre, axe


def _torso_size(f, thickness, pose):
    """Largeur/profondeur du tronc à la fraction f, d'après les épaisseurs réglées."""
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
    epaules = math.dist(pose["shoulder_l"], pose["shoulder_r"]) + 0.05
    hanches = math.dist(pose["hip_l"], pose["hip_r"]) + 0.09
    refs = (hanches, epaules)
    for bump, ref in zip(TORSO_BUMPS, refs):
        g = math.exp(-((f - bump["centre"]) / bump["sigma"]) ** 2)
        width = max(width, ref * bump["scale"] * max(0.12, g))
    retrecissement = 1 - 0.25 * max(0.0, min(1.0, (f - 0.95) / 0.05))
    return width * retrecissement, interp("depth") * girth * retrecissement


def _torso_rings(pose, thickness, slices: int = TORSO_SLICES, ring_points: int = 14):
    """Coupes du tronc : liste de (points 3D, centre 3D) du bassin vers le cou."""
    chain = ["hips", "spine", "chest", "neck"]
    weights = [max(1e-6, math.dist(pose[chain[i]], pose[chain[i + 1]])) for i in range(3)]
    rings = []
    for i in range(slices):
        f = i / (slices - 1)
        centre, axe = _chain_point(pose, chain, weights, f)
        bas = _norm(tuple(pose["hip_l"][k] - pose["hip_r"][k] for k in range(3)))
        haut = _norm(tuple(pose["shoulder_l"][k] - pose["shoulder_r"][k] for k in range(3)))
        melange = tuple(bas[k] * (1 - f) + haut[k] * f for k in range(3))
        proj = tuple(melange[k] - axe[k] * sum(melange[j] * axe[j] for j in range(3)) for k in range(3))
        x = _norm(proj)
        if sum(v * v for v in x) < 1e-12:
            x = bas
        z = _norm((x[1] * axe[2] - x[2] * axe[1], x[2] * axe[0] - x[0] * axe[2], x[0] * axe[1] - x[1] * axe[0]))
        width, depth = _torso_size(f, thickness, pose)
        pts = []
        for k in range(ring_points):
            th = (k / ring_points) * 2 * math.pi
            pts.append(tuple(centre[j] + x[j] * math.cos(th) * width / 2 + z[j] * math.sin(th) * depth / 2
                             for j in range(3)))
        rings.append((pts, centre))
    return rings


def _draw_torso(canvas, rings, camera, width, height, mode):
    """Dessine le tronc : silhouette lissée remplie (dégradé horizontal en mode volume)."""
    import cv2
    import numpy as np

    projected = [[project(pt, camera, width, height) for pt in pts] for pts, _ in rings]
    centres = [project(c, camera, width, height) for _, c in rings]
    gauche, droite = [], []
    for ring in projected:
        gauche.append(min(ring, key=lambda p: p["x"]))
        droite.append(max(ring, key=lambda p: p["x"]))
    poly = np.array([[int(round(p["x"])), int(round(p["y"]))] for p in droite + list(reversed(gauche))], dtype="int32")
    profondeur = sum(c["depth"] for c in centres) / max(1, len(centres))
    mask = np.zeros(canvas.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA)
    if mode == "silhouette":
        couche = np.full_like(canvas, 255)
    elif mode == "depth":
        t = max(0.0, min(1.0, (profondeur - (camera.get("distance", 3.0) - 0.7)) / 1.4))
        g = int(round(255 - 205 * (1 - t)))
        couche = np.full(canvas.shape, g, dtype="uint8")
    else:
        min_x = min(int(p["x"]) for p in droite + gauche)
        max_x = max(int(p["x"]) for p in droite + gauche)
        couche = np.zeros_like(canvas)
        span = max(1, max_x - min_x)
        for x in range(max(0, min_x), min(width, max_x + 1)):
            t = (x - min_x) / span
            if t < 0.42:
                k = t / 0.42
                color = [SKIN[i] * (1.02 - 0.07 * k) for i in range(3)]
            else:
                k = (t - 0.42) / 0.58
                color = [SKIN[i] * 0.95 * (1 - k) + SKIN_DARK[i] * 0.92 * k for i in range(3)]
            couche[:, x] = [min(255, max(0, int(round(c)))) for c in color]
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return profondeur
    alpha = (mask[ys, xs].astype("float32") / 255.0)[:, None]
    region = canvas[ys, xs].astype("float32")
    canvas[ys, xs] = np.clip(region * (1 - alpha) + couche[ys, xs].astype("float32") * alpha, 0, 255).astype("uint8")
    return profondeur


def _draw_cap(canvas, point, radius, camera, mode, profondeur, d_min, d_max):
    """Rotule : disque de la même teinte que les segments (silhouette, profondeur ou volume)."""
    import cv2
    import numpy as np

    p = project(point, camera, canvas.shape[1], canvas.shape[0])
    r = max(1, int(round(radius * p["scale"])))
    mask = np.zeros(canvas.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(p["x"]), int(p["y"])), r, 255, -1, cv2.LINE_AA)
    if mode == "silhouette":
        couche = np.full_like(canvas, 255)
    elif mode == "depth":
        t = 0.5 if d_max <= d_min else (profondeur - d_min) / (d_max - d_min)
        g = int(round(255 - 205 * t))
        couche = np.full(canvas.shape, g, dtype="uint8")
    else:
        couche = np.zeros_like(canvas)
        x0, x1 = int(p["x"] - r), int(p["x"] + r)
        span = max(1, x1 - x0)
        for x in range(max(0, x0), min(canvas.shape[1], x1 + 1)):
            t = (x - x0) / span
            if t < 0.45:
                k = t / 0.45
                color = [SKIN[i] * (1.01 - 0.06 * k) for i in range(3)]
            else:
                k = (t - 0.45) / 0.55
                color = [SKIN[i] * 0.95 * (1 - k) + SKIN_DARK[i] * 0.94 * k for i in range(3)]
            couche[:, x] = [min(255, max(0, int(round(c)))) for c in color]
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    alpha = (mask[ys, xs].astype("float32") / 255.0)[:, None]
    region = canvas[ys, xs].astype("float32")
    canvas[ys, xs] = np.clip(region * (1 - alpha) + couche[ys, xs].astype("float32") * alpha, 0, 255).astype("uint8")


def _capsule(canvas, p0, p1, r0, r1, color, alpha=1.0):
    """Dessine une capsule (segment à bouts ronds) sur un calque BGR."""
    import cv2
    import numpy as np

    overlay = np.zeros_like(canvas)
    a = (int(round(p0["x"])), int(round(p0["y"])))
    b = (int(round(p1["x"])), int(round(p1["y"])))
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-3:
        cv2.circle(overlay, a, max(1, int(round(r0))), color, -1)
    else:
        nx, ny = -dy / length, dx / length
        quad = np.array([
            [a[0] + nx * r0, a[1] + ny * r0], [b[0] + nx * r1, b[1] + ny * r1],
            [b[0] - nx * r1, b[1] - ny * r1], [a[0] - nx * r0, a[1] - ny * r0],
        ], dtype="int32")
        cv2.fillConvexPoly(overlay, quad, color, cv2.LINE_AA)
        cv2.circle(overlay, a, max(1, int(round(r0))), color, -1, cv2.LINE_AA)
        cv2.circle(overlay, b, max(1, int(round(r1))), color, -1, cv2.LINE_AA)
    if alpha >= 1.0:
        cv2.copyTo(overlay, overlay, canvas)
        return
    np.copyto(canvas, (canvas * (1 - alpha) + overlay * alpha).astype("uint8"), where=overlay > 0)


def _shaded_capsule(canvas, p0, p1, r0, r1, base, dark, depth_t):
    """Capsule ombrée : dégradé perpendiculaire à l'os (lumière en haut à gauche)."""
    import cv2
    import numpy as np

    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    a = (int(round(p0["x"])), int(round(p0["y"])))
    b = (int(round(p1["x"])), int(round(p1["y"])))
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    rr = [max(1, int(round(r0))), max(1, int(round(r1)))]
    if length < 1e-3:
        cv2.circle(mask, a, rr[0], 255, -1, cv2.LINE_AA)
    else:
        nx, ny = -dy / length, dx / length
        quad = np.array([
            [a[0] + nx * rr[0], a[1] + ny * rr[0]], [b[0] + nx * rr[1], b[1] + ny * rr[1]],
            [b[0] - nx * rr[1], b[1] - ny * rr[1]], [a[0] - nx * rr[0], a[1] - ny * rr[0]],
        ], dtype="int32")
        cv2.fillConvexPoly(mask, quad, 255, cv2.LINE_AA)
        cv2.circle(mask, a, rr[0], 255, -1, cv2.LINE_AA)
        cv2.circle(mask, b, rr[1], 255, -1, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    light = 0.98 + 0.10 * (1.0 - depth_t)
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    nx = -(dy / length) if length > 1e-3 else 0.0
    ny = (dx / length) if length > 1e-3 else 0.0
    half = max(rr) or 1
    t = (((xs - cx) * nx + (ys - cy) * ny) / half + 1) / 2      # 0 = côté éclairé
    t = np.clip(t, 0, 1).astype("float32")
    lit = np.array([min(255.0, c * light * 1.04) for c in base], dtype="float32")
    mid = np.array([min(255.0, c * light * 0.95) for c in base], dtype="float32")
    shade = np.array([min(255.0, c * light * 0.88) for c in dark], dtype="float32")
    k = np.clip((t - 0.45) / 0.55, 0, 1)[:, None]
    k0 = np.clip(t / 0.45, 0, 1)[:, None]
    color = lit[None, :] * (1 - k0) + mid[None, :] * k0
    color = color * (1 - k) + shade[None, :] * k
    layer = np.zeros_like(canvas)
    layer[ys, xs] = np.clip(color, 0, 255).astype("uint8")
    alpha = (mask[ys, xs].astype("float32") / 255.0)[:, None]
    region = canvas[ys, xs].astype("float32")
    canvas[ys, xs] = np.clip(region * (1 - alpha) + layer[ys, xs].astype("float32") * alpha, 0, 255).astype("uint8")


def _grid(canvas, camera, width, height):
    import cv2
    for i in range(-6, 7):
        a = project((i * 0.35, 0.0, -2.2), camera, width, height)
        b = project((i * 0.35, 0.0, 3.4), camera, width, height)
        cv2.line(canvas, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (96, 88, 78), 1, cv2.LINE_AA)
        c = project((-2.2, 0.0, i * 0.35), camera, width, height)
        d = project((3.4, 0.0, i * 0.35), camera, width, height)
        cv2.line(canvas, (int(c["x"]), int(c["y"])), (int(d["x"]), int(d["y"])), (96, 88, 78), 1, cv2.LINE_AA)


def render(payload: dict, mode: str = "volume", width: int = 768, height: int = 1024) -> "object":
    """Rend le mannequin dans le mode demandé et renvoie une image BGR (uint8).

    L'image produite ne contient que le personnage (fond sombre, ombre au sol discrète) :
    elle sert directement d'image de contrôle ou d'image de référence.
    """
    _require_cv()
    import cv2
    import numpy as np

    if mode not in MODES:
        raise MannequinError(f"rendu inconnu : {mode}")
    width, height = max(128, int(width)), max(128, int(height))
    build, pose, camera = prepare(payload, width, height)

    ss = 2 if mode in ("volume", "wireframe") else 1     # anticrénelage
    W, H = width * ss, height * ss
    background = (28, 24, 20) if mode == "volume" else (16, 12, 10)
    canvas = np.full((H, W, 3), background, dtype="uint8")
    if mode == "openpose":
        canvas = np.zeros((H, W, 3), dtype="uint8")

    # projection de toutes les articulations
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

    # l'image exportée reste propre : pas de grille d'édition, seulement l'ombre au sol
    if mode == "volume" and payload.get("ground"):
        _grid(canvas, camera, W, H)

    # ombre portée au sol : dessinée AVANT le corps (elle reste derrière lui)
    if mode in ("volume", "wireframe"):
        hips = pose.get("hips", [0, 0, 0])
        sol = project((hips[0], 0.002, hips[2]), camera, W, H)
        shadow = np.zeros_like(canvas)
        cv2.ellipse(shadow, (int(sol["x"]), int(sol["y"])),
                    (int(0.60 * sol["scale"]), int(0.17 * sol["scale"])), 0, 0, 360, (52, 40, 34), -1, cv2.LINE_AA)
        np.copyto(canvas, (canvas * 0.68 + shadow * 0.32).astype("uint8"), where=shadow > 0)

    thickness = build["thickness"]
    radii = bone_radii(thickness)
    skip = {"hip_l", "hip_r"} if mode == "volume" else set()
    caps = ("elbow_l", "elbow_r", "knee_l", "knee_r", "wrist_l", "wrist_r", "ankle_l", "ankle_r")

    # éléments : tronc, rotules, os — triés du plus lointain au plus proche
    elements = []
    if mode in ("volume", "silhouette", "depth"):
        rings = _torso_rings(pose, thickness)
        centres = [project(c, camera, W, H) for _, c in rings]
        elements.append({"type": "torso", "rings": rings,
                         "depth": sum(c["depth"] for c in centres) / max(1, len(centres))})
        for joint in caps:
            if joint not in pose:
                continue
            best = 0.0
            for (a, b, _r1, _r2, _k), (r1, r2) in zip(BONES, radii):
                if a == joint:
                    best = max(best, min(r1, r2))
            p_cap = project(pose[joint], camera, W, H)
            elements.append({"type": "cap", "joint": joint, "radius": best * 0.99,
                             "depth": p_cap["depth"] - 0.002, "point": pose[joint]})
    for (a, b, _r1, _r2, kind), (r1, r2) in zip(BONES, radii):
        if b in skip or a not in pose or b not in pose:
            continue
        p0 = project(pose[a], camera, W, H)
        p1 = project(pose[b], camera, W, H)
        elements.append({"type": "bone", "p0": p0, "p1": p1, "r0": r1, "r1": r2, "part": kind,
                         "depth": (p0["depth"] + p1["depth"]) / 2})
    elements.sort(key=lambda e: -e["depth"])
    depths = [e["depth"] for e in elements] or [camera.get("distance", 3.0)]
    d_min, d_max = min(depths), max(depths)

    for el in elements:
        if el["type"] == "torso":
            el["depth"] = _draw_torso(canvas, el["rings"], camera, W, H, mode)
            continue
        if el["type"] == "cap":
            _draw_cap(canvas, el["point"], el["radius"], camera, mode, el["depth"], d_min, d_max)
            continue
        p0, p1, r0, r1 = el["p0"], el["p1"], el["r0"] * el["p0"]["scale"], el["r1"] * el["p1"]["scale"]
        depth_t = 0.5 if d_max <= d_min else (el["depth"] - d_min) / (d_max - d_min)
        if mode == "silhouette":
            _capsule(canvas, p0, p1, r0, r1, (255, 255, 255))
            continue
        if mode == "depth":
            grey = int(round(255 - 205 * depth_t))
            _capsule(canvas, p0, p1, r0, r1, (grey, grey, grey))
            continue
        if mode == "wireframe":
            _capsule(canvas, p0, p1, max(2.0, r0 * 0.42), max(2.0, r1 * 0.42),
                     LIMB_COLOR.get(el["part"], (232, 147, 139)))
            continue
        _shaded_capsule(canvas, p0, p1, r0, r1, SKIN, SKIN_DARK, depth_t)

    # ombre portée au sol : dessinée AVANT le corps (elle reste derrière lui)
    if mode in ("volume", "wireframe"):
        hips = pose.get("hips", [0, 0, 0])
        sol = project((hips[0], 0.002, hips[2]), camera, W, H)
        shadow = np.zeros_like(canvas)
        cv2.ellipse(shadow, (int(sol["x"]), int(sol["y"])),
                    (int(0.60 * sol["scale"]), int(0.17 * sol["scale"])), 0, 0, 360, (52, 40, 34), -1, cv2.LINE_AA)
        np.copyto(canvas, (canvas * 0.68 + shadow * 0.32).astype("uint8"), where=shadow > 0)

    radii = bone_radii(build)
    parts = []
    for (a, b, _r1, _r2, kind), (r1, r2) in zip(BONES, radii):
        if a not in projected or b not in projected:
            continue
        parts.append((projected[a], projected[b], r1, r2, kind,
                      (projected[a]["depth"] + projected[b]["depth"]) / 2))
    parts.sort(key=lambda p: -p[5])                          # peintre : loin → proche
    depths = [p[5] for p in parts] or [camera.get("distance", 4.2)]
    d_min, d_max = min(depths), max(depths)

    for p0, p1, r1, r2, kind, depth in parts:
        depth_t = 0.5 if d_max <= d_min else (depth - d_min) / (d_max - d_min)
        if mode == "silhouette":
            _capsule(canvas, p0, p1, r1 * p0["scale"], r2 * p1["scale"], (255, 255, 255))
            continue
        if mode == "depth":
            grey = int(round(255 - 205 * depth_t))   # proche = blanc, loin = sombre
            _capsule(canvas, p0, p1, r1 * p0["scale"], r2 * p1["scale"], (grey, grey, grey))
            continue
        if mode == "wireframe":
            _capsule(canvas, p0, p1, max(2.0, r1 * p0["scale"] * 0.42), max(2.0, r2 * p1["scale"] * 0.42),
                     LIMB_COLOR.get(kind, (232, 147, 139)))
            continue
        _shaded_capsule(canvas, p0, p1, r1 * p0["scale"], r2 * p1["scale"], SKIN, SKIN_DARK, depth_t)

    if mode == "silhouette":
        # masque net : uniquement du blanc sur du noir (utilisable comme masque de composition)
        canvas = np.where(canvas > 127, 255, 0).astype("uint8")
    if ss > 1:
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    if mode == "silhouette":
        # le masque exporté est binaire : les jointures entre segments sont comblées
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
