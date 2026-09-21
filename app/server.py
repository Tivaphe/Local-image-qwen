from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import backend, config, downloads, generator, mannequin, mannequin_mesh, pose
from .catalog import (
    CATEGORIES,
    DEFAULT_FAMILY,
    FAMILIES,
    MODEL_EXTENSIONS,
    controlnet_files,
    family_for_control,
    pack_total_gb,
    pose_models,
)
from .paths import CONTROLS_DIR, MODELS_DIR, OUTPUTS_DIR, STATIC_DIR, UPLOADS_DIR, ensure_dirs, model_dir
from .catalog import entry as catalog_entry
from . import paths as _paths

ensure_dirs()
app = FastAPI(title="Local Image Qwen", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

_ASSET_RE = re.compile(r'(?P<url>/static/(?:app\.js|mannequin\.js|style\.css))(?P<q>["\'])')


def required_selections(family: str) -> list[str]:
    return generator.required_selections(family)


@app.get("/", response_class=HTMLResponse)
def index():
    """Page principale. Les URL des assets sont horodatées pour éviter qu'un
    ancien app.js/style.css resté en cache ne masque des fonctions (onglet Modèles…)."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    stamps = {}
    for name in ("app.js", "mannequin.js", "style.css"):
        p = STATIC_DIR / name
        stamps[f"/static/{name}"] = int(p.stat().st_mtime) if p.exists() else 0
    html = _ASSET_RE.sub(lambda m: f"{m.group('url')}?v={stamps.get(m.group('url'), 0)}{m.group('q')}", html)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ status
def _list_models(family: str, category: str) -> list[dict]:
    d = model_dir(family, category)
    out = []
    for p in sorted(d.iterdir()) if d.exists() else []:
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
            out.append({"name": p.name, "size_gb": round(p.stat().st_size / 1e9, 2)})
    return out


def _installed_gb(local: dict[str, list[dict]]) -> float:
    return round(sum(m["size_gb"] for models in local.values() for m in models), 2)


@app.get("/api/status")
def status():
    cfg = config.load()
    local = {fam: {c: _list_models(fam, c) for c in CATEGORIES} for fam in FAMILIES}
    # auto-sélection : si le fichier configuré n'existe plus, prendre le premier disponible
    changed = False
    for fam in FAMILIES:
        for cat in ("diffusion", "text_encoder", "vae", "vision"):
            names = [m["name"] for m in local[fam][cat]]
            if cfg["selections"][fam].get(cat) not in names:
                cfg["selections"][fam][cat] = names[0] if names else ""
                changed = True
    if changed:
        config.save(cfg)
    fam = cfg["family"]
    sel = cfg["selections"][fam]
    ready = all(sel.get(k) for k in required_selections(fam))
    families_status = {
        f: {
            "ready": all(cfg["selections"][f].get(k) for k in required_selections(f)),
            "edit_ready": (not FAMILIES[f]["edit_requires_vision"]) or bool(cfg["selections"][f].get("vision")),
            "control_ready": bool(FAMILIES[f].get("supports_controlnet")) and bool(local[f].get("controlnet")),
            "installed_gb": _installed_gb(local[f]),
            "file_count": sum(len(v) for v in local[f].values()),
        }
        for f in FAMILIES
    }
    pose_detectors = local.get(family_for_control(), {}).get("pose_detector", [])
    families = []
    for f in FAMILIES:
        data = dict(FAMILIES[f])
        data["pack_total_gb"] = pack_total_gb(f)
        data["installed_gb"] = families_status[f]["installed_gb"]
        families.append(data)
    return {
        "engine": backend.installed_info(),
        "models": local,
        "families": families,
        "families_status": families_status,
        "config": cfg,
        "ready": ready and backend.find_binary() is not None,
        "edit_ready": families_status[fam]["edit_ready"],
        "samplers": generator.SAMPLERS,
        "disk_free_gb": round(shutil.disk_usage(str(MODELS_DIR)).free / 1e9, 1),
        "control": {
            "types": list(generator.CONTROL_TYPES),
            "mannequin": {"modes": list(mannequin.MODES), "default_size": [768, 1024]},
            "mannequin_mesh": {"disponible": (mannequin_mesh.ASSETS / "maillage.npz").exists(),
                               "morphologies": list(mannequin_mesh.MORPHOLOGIES),
                               "poses": list(mannequin_mesh.POSES)},
            "pose": pose.available(),
            "pose_models": pose_models(),
            "pose_model_present": bool(pose_detectors),
            "controlnets": controlnet_files(),
            "control_family": family_for_control(),
            "selected_controlnet": cfg["selections"].get(family_for_control(), {}).get("controlnet", ""),
            "selected_pose_model": cfg["selections"].get(family_for_control(), {}).get("pose_detector", ""),
        },
        "paths": {"models": str(MODELS_DIR), "outputs": str(OUTPUTS_DIR)},
        "jobs": downloads.list_jobs(),
        "generation": generator.state(),
    }


# ------------------------------------------------------------------ config
class ConfigIn(BaseModel):
    family: Optional[str] = None
    selections: Optional[dict] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    sampler: Optional[str] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    offload_to_cpu: Optional[bool] = None
    flash_attention: Optional[bool] = None
    vae_tiling: Optional[bool] = None
    threads: Optional[int] = None
    extra_args: Optional[str] = None


@app.post("/api/config")
def set_config(c: ConfigIn):
    data = {k: v for k, v in c.model_dump().items() if v is not None}
    if "family" in data and data["family"] not in FAMILIES:
        raise HTTPException(400, "famille inconnue")
    config.save(data)
    return {"ok": True, "config": config.load()}


# --------------------------------------------------------------- downloads
class DownloadIn(BaseModel):
    family: str = DEFAULT_FAMILY
    category: str
    file_id: str = ""
    url: Optional[str] = None


@app.post("/api/download")
def download(d: DownloadIn):
    """Télécharge un fichier précis (catalogue ou URL personnalisée)."""
    if d.family not in FAMILIES or d.category not in CATEGORIES:
        raise HTTPException(400, "famille ou catégorie inconnue")
    try:
        jid = downloads.start_model_download(d.family, d.category, d.file_id, d.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"job": jid}


class InstallIn(BaseModel):
    family: str = DEFAULT_FAMILY
    diffusion: str = ""          # identifiant du fichier de diffusion (quantification)
    text_encoder: str = ""       # identifiant de l'encodeur de texte
    include_vision: bool = True  # télécharger aussi l'encodeur de vision (édition d'image)


@app.post("/api/install")
def install(i: InstallIn):
    """Installation complète en un clic : génération **et** édition d'image."""
    if i.family not in FAMILIES:
        raise HTTPException(400, "famille inconnue")
    try:
        jid = downloads.start_family_install(i.family, i.diffusion, i.text_encoder, i.include_vision)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"job": jid, "jobs": downloads.list_jobs()}


class EngineIn(BaseModel):
    flavor: Optional[str] = None


@app.post("/api/engine/install")
def engine_install(e: EngineIn):
    return {"job": downloads.start_engine_install(e.flavor)}


@app.get("/api/jobs")
def jobs():
    return downloads.list_jobs()


@app.post("/api/jobs/clear")
def jobs_clear():
    downloads.clear_finished()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return {"cancelled": downloads.cancel(job_id)}


@app.post("/api/jobs/cancel-all")
def jobs_cancel_all():
    return {"cancelled": downloads.cancel_all()}


@app.delete("/api/models/{family}/{category}/{name}")
def delete_model(family: str, category: str, name: str):
    if family not in FAMILIES or category not in CATEGORIES or "/" in name or "\\" in name:
        raise HTTPException(400, "requête invalide")
    p = model_dir(family, category) / name
    if p.exists():
        p.unlink()
    p.with_suffix(p.suffix + ".part").unlink(missing_ok=True)
    return {"ok": True}


# ------------------------------------------------------- personnages / contrôle
def _safe_upload(rel: str) -> Path:
    """Chemin d'un fichier déjà téléversé (protège contre les remontées de dossier)."""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(400, "fichier invalide")
    p = (UPLOADS_DIR / rel).resolve()
    if UPLOADS_DIR.resolve() not in p.parents:
        raise HTTPException(400, "fichier invalide")
    if not p.exists():
        raise HTTPException(404, f"fichier introuvable : {rel}")
    return p


async def _save_upload(f: UploadFile) -> Path:
    ext = Path(f.filename or "image.png").suffix.lower() or ".png"
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    dest.write_bytes(await f.read())
    return dest


def _pose_model_path() -> Path | None:
    """Détecteur de pose choisi, sinon le premier présent sur le disque."""
    cfg = config.load()
    fam = family_for_control()
    chosen = cfg["selections"].get(fam, {}).get("pose_detector", "")
    for candidate in (chosen, *[m["id"] for m in pose_models()]):
        p = paths_controlnet(candidate)
        if p is not None:
            return p
    return None


def paths_controlnet(name: str):
    return _paths.controlnet_path(name)


def _control_url(name: str) -> str:
    return f"/uploads/controls/{name}"


class PoseIn(BaseModel):
    persons: list[dict]
    width: int
    height: int
    kind: str = "pose"


@app.post("/api/control/detect")
async def control_detect(
    image: UploadFile = File(...),
    kind: str = Form("pose"),
    background: str = Form("black"),
    low: int = Form(100),
    high: int = Form(200),
):
    """Détecte automatiquement les personnages (pose) ou calcule les contours de l'image."""
    if kind not in generator.CONTROL_TYPES:
        raise HTTPException(400, f"type de contrôle inconnu : {kind}")
    src = await _save_upload(image)
    try:
        if kind == "canny":
            edges = pose.canny_control(src, low, high)
            name = f"{uuid.uuid4().hex}-canny.png"
            pose.save_image(edges, CONTROLS_DIR / name)
            height, width = edges.shape[:2]
            return {
                "ok": True, "kind": "canny", "count": 1, "persons": [],
                "control": {"id": f"controls/{name}", "url": _control_url(name), "width": width, "height": height},
                "source": {"id": src.name, "url": f"/uploads/{src.name}"},
                "image": {"width": width, "height": height},
                "message": "Contours calculés. Ajustez la force du contrôle puis lancez la génération.",
            }

        model = _pose_model_path()
        if model is None:
            raise HTTPException(
                400,
                "Détecteur de personnages absent : téléchargez « yolov8n-pose.onnx » (~13 Mo) dans l'onglet Modèles "
                "(famille « SD 1.5 + ControlNet »), puis relancez la détection.",
            )
        detector = pose.PoseDetector(model)
        result = detector.detect(src)
        persons = result["persons"]
        if not persons:
            raise HTTPException(
                400,
                "Aucun personnage détecté sur cette image. Essayez une photo où les personnes sont plus grandes ou "
                "mieux éclairées, ou importez un squelette de référence à la place.",
            )
        base = pose.read_image(src) if background == "image" else None
        skeleton = pose.render_skeleton(persons, result["width"], result["height"], background=base)
        name = f"{uuid.uuid4().hex}-pose.png"
        pose.save_image(skeleton, CONTROLS_DIR / name)
        return {
            "ok": True, "kind": "pose", "count": len(persons), "persons": persons,
            "backend": result.get("backend", ""),
            "control": {"id": f"controls/{name}", "url": _control_url(name),
                        "width": result["width"], "height": result["height"]},
            "source": {"id": src.name, "url": f"/uploads/{src.name}"},
            "image": {"width": result["width"], "height": result["height"]},
            "message": f"{len(persons)} personnage(s) détecté(s) — vous pouvez ajuster le squelette dans l'éditeur.",
        }
    except pose.PoseError as e:
        raise HTTPException(400, str(e))


@app.post("/api/control/pose")
def control_pose(p: PoseIn):
    """Rend un squelette (ou une silhouette) à partir des points ajustés dans l'éditeur."""
    if not p.persons:
        raise HTTPException(400, "aucun squelette à dessiner")
    if p.width < 64 or p.height < 64:
        raise HTTPException(400, "dimensions invalides")
    kind = "silhouette" if p.kind == "silhouette" else "pose"
    try:
        img = pose.render_skeleton(p.persons, int(p.width), int(p.height), kind=kind)
    except pose.PoseError as e:
        raise HTTPException(400, str(e))
    name = f"{uuid.uuid4().hex}-{kind}-edit.png"
    pose.save_image(img, CONTROLS_DIR / name)
    return {"ok": True, "kind": kind,
            "control": {"id": f"controls/{name}", "url": _control_url(name),
                        "width": int(p.width), "height": int(p.height)}}


# --------------------------------------------------------------- mannequin
class MannequinIn(BaseModel):
    pose: dict = {}              # {articulation: [x, y, z]} en mètres
    lengths: dict = {}           # {segment: longueur en mètres} — tableau des dimensions
    thickness: dict = {}         # {segment: épaisseur en mètres} — diamètre au milieu
    morphology: str = ""         # morphologie de départ : neutre|fin|athletique|fort|femme
    build: dict = {}             # ancien nom des proportions (conservé)
    camera: dict = {}            # {yaw, pitch, distance, target}
    mode: str = "volume"         # volume | wireframe | openpose | depth | silhouette
    width: int = 768
    height: int = 1024


def _render_mannequin(payload: dict, mode: str, width: int, height: int, tag: str) -> dict:
    """Rend la pose du mannequin et renvoie le fichier de contrôle créé."""
    try:
        img = mannequin.render(payload, mode, width, height)
    except mannequin.MannequinError as e:
        raise HTTPException(400, str(e))
    except pose.PoseError as e:
        raise HTTPException(400, str(e))
    name = f"{uuid.uuid4().hex}-mannequin-{tag}.png"
    pose.save_image(img, CONTROLS_DIR / name)
    return {"ok": True, "mode": mode, "kind": "pose" if mode == "openpose" else mode,
            "control": {"id": f"controls/{name}", "url": _control_url(name),
                        "width": int(width), "height": int(height)}}


@app.post("/api/mannequin/render")
def mannequin_render(m: MannequinIn):
    """Rend le mannequin dans le mode demandé (volume, filaire, squelette, profondeur…)."""
    width, height = max(256, m.width // 32 * 32), max(256, m.height // 32 * 32)
    return _render_mannequin(m.model_dump(), m.mode, width, height, m.mode)


@app.post("/api/mannequin/pose")
def mannequin_pose(m: MannequinIn):
    """Renvoie les longueurs d'os et les points 2D (diagnostic / partage de pose)."""
    build, pose_data, _camera = mannequin.validate(m.model_dump())
    return {"ok": True,
            "lengths": {k: round(v, 4) for k, v in mannequin.bone_lengths(build).items()},
            "points": mannequin.openpose_points(m.model_dump(), max(256, m.width), max(256, m.height)),
            "modes": list(mannequin.MODES)}


# ------------------------------------------------- mannequin anatomique (maillage)
class MannequinMeshIn(BaseModel):
    """Requête du mannequin anatomique : morphologie + angles d'articulation."""
    pose: dict = {}              # {articulation: [flexion, abduction, torsion]} en degrés
    morphology: str = ""         # neutre | femme | homme | fine | athletique | forte
    morphs: dict = {}            # poids bruts des cibles de morphologie (curseurs fins)
    origine: list = []           # décalage du bassin (assis, accroupi) en mètres
    yaw: float | None = None     # orientation de la caméra (radians)
    pitch: float | None = None
    mode: str = "volume"         # volume | openpose | depth | silhouette
    width: int = 768
    height: int = 1024


@app.get("/api/mannequin/model")
def mannequin_model():
    """Catalogue du mannequin anatomique : morphologies, poses, articulations et butées."""
    return {
        "ok": True,
        "morphologies": mannequin_mesh.MORPHOLOGIES,
        "poses": {nom: angles for nom, angles in mannequin_mesh.POSES.items()},
        "poses_origine": mannequin_mesh.POSES_ORIGINE,
        "articulations": [
            {"nom": nom, "etiquette": mannequin_mesh.ETIQUETTES[nom],
             "parent": mannequin_mesh.PARENT.get(nom),
             "limites": mannequin_mesh.LIMITES[nom]}
            for nom in mannequin_mesh.ORDRE
        ],
        "modes": ["volume", "openpose", "depth", "silhouette"],
        "dimensions": mannequin_mesh.dimensions(),
    }


@app.post("/api/mannequin/mesh/pose")
def mannequin_mesh_pose(m: MannequinMeshIn):
    """Positions des 21 articulations pour une pose donnée (poignées de l'éditeur)."""
    try:
        points = mannequin_mesh.articulations(m.pose or None, origine=m.origine or None)
    except mannequin_mesh.MannequinError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "articulations": points, "etiquettes": mannequin_mesh.ETIQUETTES,
            "limites": mannequin_mesh.LIMITES}


@app.post("/api/mannequin/mesh/render")
def mannequin_mesh_render(m: MannequinMeshIn):
    """Rend le mannequin anatomique (volumes, squelette OpenPose, profondeur, silhouette)."""
    donnees = m.model_dump()
    width, height = max(128, m.width // 32 * 32), max(128, m.height // 32 * 32)
    try:
        img = mannequin_mesh.rend(donnees, m.mode, width, height)
    except mannequin_mesh.MannequinError as e:
        raise HTTPException(400, str(e))
    name = f"{uuid.uuid4().hex}-mannequin-{m.mode}.png"
    pose.save_image(img, CONTROLS_DIR / name)
    return {"ok": True, "mode": m.mode, "kind": "pose" if m.mode == "openpose" else m.mode,
            "control": {"id": f"controls/{name}", "url": _control_url(name),
                        "width": int(width), "height": int(height)}}


# -------------------------------------------------------------- generation
@app.post("/api/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(30),
    cfg_scale: float = Form(6.0),
    sampler: str = Form("euler"),
    seed: int = Form(-1),
    mode: str = Form("generate"),
    control_type: str = Form(""),
    control_id: str = Form(""),
    control_strength: float = Form(0.9),
    control_net: str = Form(""),
    use_init: bool = Form(False),
    strength: float = Form(0.45),
    init_id: str = Form(""),
    ref_id: str = Form(""),          # image déjà présente (rendu du mannequin, image de la galerie)
    ref_images: list[UploadFile] = File(default=[]),
    init_image: UploadFile | None = File(default=None),
):
    if not prompt.strip():
        raise HTTPException(400, "Le prompt est vide.")
    width, height = max(256, width // 32 * 32), max(256, height // 32 * 32)
    refs: list[str] = []
    for f in ref_images:
        if not f.filename:
            continue
        refs.append(str(await _save_upload(f)))
    if ref_id:
        refs.append(str(_safe_upload(ref_id)))

    # ControlNet : image de contrôle déjà préparée (pose détectée / contours) recadrée
    # exactement à la taille de génération
    control_image = ""
    if control_type:
        if control_type not in generator.CONTROL_TYPES:
            raise HTTPException(400, f"type de contrôle inconnu : {control_type}")
        if not control_id:
            raise HTTPException(400, "ControlNet activé mais aucune image de contrôle : lancez d'abord "
                                     "« Détecter les personnages » ou « Calculer les contours ».")
        src = _safe_upload(control_id)
        dest = CONTROLS_DIR / f"{Path(control_id).stem}-{width}x{height}.png"
        try:
            pose.prepare_control(src, dest, width, height, control_type)
        except pose.PoseError as e:
            raise HTTPException(400, str(e))
        control_image = str(dest)

    # img2img : repartir d'une photo (référence téléversée ou fichier déjà présent)
    init_path = ""
    if init_image is not None and init_image.filename:
        init_path = str(await _save_upload(init_image))
    elif init_id:
        init_path = str(_safe_upload(init_id))
    elif use_init and refs:
        init_path = refs[0]

    params = {
        "prompt": prompt, "negative_prompt": negative_prompt, "width": width, "height": height,
        "steps": steps, "cfg_scale": cfg_scale, "sampler": sampler, "seed": seed, "ref_images": refs,
        "mode": mode, "control_type": control_type, "control_image": control_image,
        "control_strength": control_strength, "control_net": control_net,
        "init_image": init_path, "strength": strength,
    }
    cfg = config.load()
    config.save({k: params[k] for k in ("width", "height", "steps", "cfg_scale", "sampler", "negative_prompt")})
    try:
        generator.start(cfg, params)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "seed": params["seed"]}


@app.get("/api/generation")
def generation():
    return generator.state()


@app.post("/api/cancel")
def cancel():
    return {"cancelled": generator.cancel()}


@app.get("/api/gallery")
def gallery():
    return generator.gallery()


@app.delete("/api/gallery/{name}")
def gallery_delete(name: str):
    if "/" in name or "\\" in name or not name.endswith(".png"):
        raise HTTPException(400, "nom invalide")
    p = OUTPUTS_DIR / name
    p.unlink(missing_ok=True)
    p.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/open-folder")
def open_folder(which: str = Form("outputs")):
    target = OUTPUTS_DIR if which == "outputs" else MODELS_DIR
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif shutil.which("xdg-open"):
            os.spawnlp(os.P_NOWAIT, "xdg-open", "xdg-open", str(target))
        elif shutil.which("open"):
            os.spawnlp(os.P_NOWAIT, "open", "open", str(target))
    except Exception:
        pass
    return {"path": str(target)}
