"""
Détection des personnages (et de leur pose) et fabrication des images de contrôle.

Deux usages :
  - **détection automatique** : un modèle YOLOv8-pose (ONNX, ~13 Mo, téléchargeable
    dans l'application) repère chaque personnage et ses 17 points articulaires ;
  - **images de contrôle** pour ControlNet : squelette OpenPose, contours (Canny) ou
    silhouette — fichiers passés à sd-cli via ``--control-image``.

Le décodage des sorties du réseau (``decode_yolo_pose``), la suppression des
doublons (``nms``), le rendu du squelette (``render_skeleton``) et le prétraitement
Canny sont des fonctions pures, testables sans modèle ni carte graphique.
"""
from __future__ import annotations

from pathlib import Path

# --- OpenCV / NumPy sont optionnels : sans eux, seules la détection et le rendu
# --- du squelette (et les contours) sont indisponibles, le reste de l'application
# --- continue de fonctionner normalement.
try:  # pragma: no cover - dépend de l'installation
    import numpy as np
    import cv2
    CV_AVAILABLE = True
    CV_ERROR = ""
except Exception as e:  # pragma: no cover
    np = None
    cv2 = None
    CV_AVAILABLE = False
    CV_ERROR = str(e)

# --- Points du squelette COCO 17 (ordre des sorties YOLOv8-pose)
KEYPOINT_NAMES = [
    "nez", "œil_gauche", "œil_droit", "oreille_gauche", "oreille_droite",
    "épaule_gauche", "épaule_droite", "coude_gauche", "coude_droit",
    "poignet_gauche", "poignet_droit", "hanche_gauche", "hanche_droite",
    "genou_gauche", "genou_droit", "cheville_gauche", "cheville_droite",
]

# Segments du squelette (paires de points) — rendu OpenPose classique
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Couleurs (B, G, R) OpenPose : visage, torse, bras gauche/droit, jambes
COLORS = [
    (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),
    (0, 255, 255), (0, 128, 255), (0, 255, 0), (0, 128, 255),
    (0, 255, 0), (0, 128, 255), (0, 255, 0), (0, 128, 255),
    (255, 128, 0), (255, 128, 0), (255, 255, 0), (255, 255, 0),
]

# Couleur par segment (indices dans SKELETON)
LIMB_COLORS = [
    (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),
    (0, 255, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255),
    (0, 255, 0), (0, 255, 0), (255, 255, 0),
    (255, 128, 0), (255, 255, 0), (255, 128, 0), (255, 255, 0),
]

CONF_THRESHOLD = 0.25    # seuil de confiance « c'est un personnage »
KPT_THRESHOLD = 0.30     # seuil de confiance d'un point articulaire
NMS_IOU = 0.45


class PoseError(RuntimeError):
    """Erreur explicite, affichée telle quelle dans l'interface."""


# --------------------------------------------------------------- utilitaires
def _require_cv() -> None:
    if not CV_AVAILABLE:
        raise PoseError(
            "Détection et rendu impossibles : NumPy/OpenCV ne sont pas installés. "
            "Lancez « pip install -r requirements.txt » (numpy + opencv-python-headless) puis redémarrez. "
            f"Détail : {CV_ERROR}"
        )


def letterbox(width: int, height: int, size: int = 640) -> tuple[float, int, int]:
    """Échelle et marges appliquées par un redimensionnement « letterbox » (YOLO)."""
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    return scale, (size - new_w) // 2, (size - new_h) // 2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def nms(boxes: list[tuple[float, float, float, float]], scores: list[float], thr: float = NMS_IOU) -> list[int]:
    """Indices conservés après suppression des recouvrements (ordre = score décroissant)."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= thr]
    return kept


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# ------------------------------------------------------- décodage YOLOv8-pose
def decode_yolo_pose(output, kpt_count: int = 17, conf: float = CONF_THRESHOLD,
                     kpt_conf: float = KPT_THRESHOLD, nms_thr: float = NMS_IOU,
                     pad: tuple[int, int] = (0, 0), scale: float = 1.0,
                     size: tuple[int, int] | None = None) -> list[dict]:
    """
    Transforme la sortie brute du réseau ([1, 4+1+3*K, N] ou [N, 4+1+3*K]) en
    détections : ``{'box': (x1, y1, x2, y2), 'conf': float, 'keypoints': [(x, y, c), …]}``.

    ``pad``/``scale``/``size`` permettent de revenir aux coordonnées de l'image
    d'origine après le letterbox (``size`` = largeur, hauteur d'origine).
    """
    _require_cv()
    arr = np.asarray(output, dtype="float32")
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise PoseError(f"Sortie de détection inattendue (forme {arr.shape}).")
    # colonnes = prédictions (56, N) → on transpose ; sinon on garde tel quel
    if arr.shape[0] == 4 + 1 + 3 * kpt_count and arr.shape[1] != arr.shape[0]:
        arr = arr.T
    if arr.shape[-1] < 4 + 1 + 3 * kpt_count:
        raise PoseError(f"Sortie de détection incompatible ({arr.shape[-1]} valeurs par prédiction au lieu de "
                        f"{4 + 1 + 3 * kpt_count}). Le fichier ONNX n'est pas un YOLOv8-pose.")

    boxes_raw = arr[:, :4]
    scores = arr[:, 4]
    kpts_raw = arr[:, 5:5 + 3 * kpt_count].reshape(-1, kpt_count, 3)

    keep = scores >= conf
    if not np.any(keep):
        return []
    boxes_raw, scores, kpts_raw = boxes_raw[keep], scores[keep], kpts_raw[keep]

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # retour aux coordonnées d'origine : on retire les marges du letterbox puis on divise par l'échelle
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    kpts_raw[:, :, 0] = (kpts_raw[:, :, 0] - pad_x) / scale
    kpts_raw[:, :, 1] = (kpts_raw[:, :, 1] - pad_y) / scale

    if size:
        width, height = size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    idx = nms([tuple(b) for b in boxes.tolist()], scores.tolist(), nms_thr)
    persons = []
    for i in idx:
        kps = []
        for k in range(kpt_count):
            x, y, c = (float(v) for v in kpts_raw[i, k])
            kps.append({"x": round(x, 1), "y": round(y, 1), "c": round(c, 3),
                        "visible": bool(c >= kpt_conf), "name": KEYPOINT_NAMES[k] if k < len(KEYPOINT_NAMES) else f"pt{k}"})
        x1, y1, x2, y2 = (round(float(v), 1) for v in boxes[i])
        persons.append({"box": [x1, y1, x2, y2], "conf": round(float(scores[i]), 3), "keypoints": kps})
    persons.sort(key=lambda p: p["conf"], reverse=True)
    return persons


# ----------------------------------------------------------------- rendu
def render_skeleton(persons: list[dict], width: int, height: int, kind: str = "pose",
                    line_width: int = 6, dot_radius: int = 5, background=None) -> "np.ndarray":
    """Image de contrôle : squelette OpenPose (fond noir) ou silhouette (fond noir).

    ``background`` : image d'origine (tableau BGR) pour dessiner le squelette par-dessus.
    """
    _require_cv()
    if background is not None:
        bg = cv2.resize(background, (width, height)) if background.shape[1] != width or background.shape[0] != height else background
        canvas = (bg.astype("float32") * 0.35).astype("uint8")  # image assombrie
    else:
        canvas = np.zeros((height, width, 3), dtype="uint8")
    if kind == "silhouette":
        for p in persons:
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            cv2.rectangle(canvas, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)), (255, 255, 255), -1)
        return canvas
    for p in persons:
        kps = p["keypoints"]
        for li, (a, b) in enumerate(SKELETON):
            if a >= len(kps) or b >= len(kps):
                continue
            pa, pb = kps[a], kps[b]
            if not (pa["visible"] and pb["visible"]):
                continue
            color = LIMB_COLORS[li % len(LIMB_COLORS)]
            cv2.line(canvas, (int(pa["x"]), int(pa["y"])), (int(pb["x"]), int(pb["y"])), color, line_width, cv2.LINE_AA)
        for ki, kp in enumerate(kps):
            if not kp["visible"]:
                continue
            cv2.circle(canvas, (int(kp["x"]), int(kp["y"])), dot_radius, COLORS[ki % len(COLORS)], -1, cv2.LINE_AA)
    return canvas


def read_image(image_path: str | Path):
    """Lit une image BGR (lève une erreur explicite si illisible)."""
    _require_cv()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise PoseError(f"Image illisible : {image_path}")
    return img


def canny_control(image_path: str | Path, low: int = 100, high: int = 200) -> "np.ndarray":
    """Carte de contours (entrée du ControlNet Canny)."""
    _require_cv()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise PoseError(f"Image illisible : {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def image_size(image_path: str | Path) -> tuple[int, int]:
    _require_cv()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise PoseError(f"Image illisible : {image_path}")
    h, w = img.shape[:2]
    return w, h


# ------------------------------------------------------------- détecteur
class PoseDetector:
    """Détecteur YOLOv8-pose ONNX (onnxruntime si présent, sinon OpenCV DNN)."""

    def __init__(self, model_path: str | Path, size: int = 640):
        self.path = Path(model_path)
        self.size = size
        self._session = None
        self._net = None
        self._backend = ""

    # ------------------------------------------------------------ chargement
    def backend(self) -> str:
        return self._backend

    def load(self) -> str:
        _require_cv()
        if not self.path.exists():
            raise PoseError(
                "Détecteur de pose absent : téléchargez « yolov8n-pose.onnx » (~13 Mo) dans l'onglet Modèles "
                "(Tout télécharger de la famille SD 1.5 + ControlNet)."
            )
        if self._session is not None or self._net is not None:
            return self._backend
        try:  # 1) onnxruntime, le plus fiable
            import onnxruntime as ort  # type: ignore
            self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            self._backend = f"onnxruntime {ort.__version__}"
            return self._backend
        except ImportError:
            pass
        except Exception as e:
            raise PoseError(
                f"Impossible de charger {self.path.name} avec onnxruntime ({e}). "
                "Le fichier est peut-être incomplet (.part) : retéléchargez-le."
            ) from e
        try:  # 2) OpenCV DNN
            self._net = cv2.dnn.readNetFromONNX(str(self.path))
            self._backend = f"OpenCV {cv2.__version__} (dnn)"
            return self._backend
        except Exception as e:
            raise PoseError(
                f"Impossible de charger {self.path.name} ({e}). "
                "Retéléchargez le fichier, ou installez onnxruntime (« pip install onnxruntime ») pour un moteur plus robuste."
            ) from e

    # ------------------------------------------------------------- inférence
    def _infer(self, blob) -> "np.ndarray":
        if self._session is not None:
            name = self._session.get_inputs()[0].name
            out = self._session.run(None, {name: blob})
        else:
            self._net.setInput(blob)
            out = self._net.forward()
        return np.asarray(out[0] if isinstance(out, (list, tuple)) else out)

    def detect(self, image_path: str | Path) -> dict:
        """Détecte les personnages d'une image ; renvoie aussi le rendu du squelette."""
        _require_cv()
        backend = self.load()
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise PoseError(f"Image illisible : {image_path}")
        height, width = img.shape[:2]
        scale, pad_x, pad_y = letterbox(width, height, self.size)
        resized = cv2.resize(img, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype="uint8")
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        raw = self._infer(blob)
        persons = decode_yolo_pose(raw, pad=(pad_x, pad_y), scale=scale, size=(width, height))
        skeleton = render_skeleton(persons, width, height)
        return {
            "width": width, "height": height, "persons": persons,
            "count": len(persons), "backend": backend, "skeleton": skeleton,
        }

    def detect_file(self, image_path: str | Path, out_path: str | Path) -> dict:
        """Détecte puis écrit le squelette à l'emplacement ``out_path``."""
        result = self.detect(image_path)
        _require_cv()
        if not cv2.imwrite(str(out_path), result.pop("skeleton")):
            raise PoseError(f"Impossible d'écrire {out_path}")
        return result


# --------------------------------------------------------------- commodités
def available() -> dict:
    """État des dépendances de détection (affiché dans l'interface)."""
    info = {"cv": CV_AVAILABLE, "cv_version": "", "onnxruntime": False, "error": CV_ERROR}
    if CV_AVAILABLE:
        info["cv_version"] = getattr(cv2, "__version__", "")
        try:
            import onnxruntime  # type: ignore
            info["onnxruntime"] = onnxruntime.__version__
        except Exception:
            pass
    return info


def save_image(img, dst: str | Path) -> Path:
    """Écrit une image (BGR) sur le disque."""
    _require_cv()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), img):
        raise PoseError(f"Impossible d'écrire l'image : {dst}")
    return dst


def prepare_control(src, dst: str | Path, width: int, height: int, kind: str = "pose") -> Path:
    """Recadre/redimensionne l'image de contrôle exactement à la taille de génération.

    ``src`` : chemin d'un fichier ou tableau BGR déjà en mémoire.
    """
    _require_cv()
    img = src if hasattr(src, "shape") else read_image(src)
    if img.shape[1] != width or img.shape[0] != height:
        interp = cv2.INTER_NEAREST if kind == "canny" else cv2.INTER_LINEAR
        img = cv2.resize(img, (width, height), interpolation=interp)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), img):
        raise PoseError(f"Impossible d'écrire l'image de contrôle : {dst}")
    return dst
