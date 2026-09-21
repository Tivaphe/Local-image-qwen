from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import backend, catalog, config, downloads, generator
from .catalog import CATEGORIES, DEFAULT_FAMILY, FAMILIES, MODEL_EXTENSIONS
from .paths import MODELS_DIR, OUTPUTS_DIR, STATIC_DIR, UPLOADS_DIR, ensure_dirs, model_dir

ensure_dirs()
app = FastAPI(title="Local Image Qwen", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ status
def _list_models(family: str, category: str) -> list[dict]:
    d = model_dir(family, category)
    out = []
    for p in sorted(d.iterdir()) if d.exists() else []:
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
            out.append({"name": p.name, "size_gb": round(p.stat().st_size / 1e9, 2)})
    return out


@app.get("/api/status")
def status():
    cfg = config.load()
    local = {fam: {c: _list_models(fam, c) for c in CATEGORIES} for fam in FAMILIES}
    
    # Auto-sélection : si le fichier configuré n'existe plus ou est vide, prendre le premier disponible
    changed = False
    for fam in FAMILIES:
        for cat in ("diffusion", "text_encoder", "vae", "vision"):
            names = [m["name"] for m in local[fam][cat]]
            current_sel = cfg["selections"][fam].get(cat)
            if not current_sel or current_sel not in names:
                cfg["selections"][fam][cat] = names[0] if names else ""
                changed = True
    if changed:
        config.save(cfg)

    fam = cfg["family"]
    sel = cfg["selections"][fam]
    engine_bin = backend.find_binary()
    all_jobs = downloads.list_jobs()

    families_status = {}
    for f in FAMILIES:
        fam_def = FAMILIES[f]
        rec_items = []
        for cat in ("diffusion", "text_encoder", "vae", "vision"):
            if cat == "vision" and not fam_def.get("edit_requires_vision"):
                continue
            items = fam_def.get(cat, [])
            rec = next((x for x in items if x.get("recommended")), None)
            if rec:
                rec_items.append({"category": cat, **rec})
        
        rec_total_gb = round(sum(x["size_gb"] for x in rec_items), 2)
        rec_installed = [
            x for x in rec_items
            if any(m["name"] == x["id"] for m in local[f][x["category"]])
        ]
        
        running_fam_jobs = [
            j for j in all_jobs
            if j["status"] == "running" and (j.get("family") == f or f in j.get("label", ""))
        ]
        
        has_diffusion = bool(cfg["selections"][f].get("diffusion"))
        has_text_encoder = bool(cfg["selections"][f].get("text_encoder"))
        has_vae = bool(cfg["selections"][f].get("vae"))
        is_ready = has_diffusion and has_text_encoder and has_vae
        edit_ready = (not fam_def.get("edit_requires_vision")) or bool(cfg["selections"][f].get("vision"))

        total_done = sum(j["done"] for j in running_fam_jobs)
        total_size = sum(j["total"] for j in running_fam_jobs)
        dl_progress = (total_done / total_size) if total_size > 0 else 0.0

        families_status[f] = {
            "ready": is_ready,
            "edit_ready": edit_ready,
            "has_recommended": len(rec_installed) == len(rec_items) and len(rec_items) > 0,
            "recommended_total_gb": rec_total_gb,
            "recommended_installed_count": len(rec_installed),
            "recommended_total_count": len(rec_items),
            "is_downloading": len(running_fam_jobs) > 0,
            "download_progress": round(dl_progress, 3),
            "running_jobs": running_fam_jobs,
            "missing": [
                cat_name for cat_name, ok in [
                    ("diffusion", has_diffusion),
                    ("text_encoder", has_text_encoder),
                    ("vae", has_vae),
                    ("vision (édition)", edit_ready or not fam_def.get("edit_requires_vision")),
                ] if not ok
            ]
        }

    return {
        "engine": backend.installed_info(),
        "models": local,
        "families": [{k: v for k, v in FAMILIES[f].items()} for f in FAMILIES],
        "families_status": families_status,
        "config": cfg,
        "ready": families_status[fam]["ready"] and engine_bin is not None,
        "edit_ready": families_status[fam]["edit_ready"],
        "samplers": generator.SAMPLERS,
        "disk_free_gb": round(shutil.disk_usage(str(MODELS_DIR)).free / 1e9, 1),
        "jobs": all_jobs,
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
        raise HTTPException(400, "Famille inconnue")
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
    try:
        jid = downloads.start_model_download(d.family, d.category, d.file_id, d.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"job": jid}


class DownloadFamilyIn(BaseModel):
    family: str
    include_vision: bool = True


@app.post("/api/download-family")
def download_family(d: DownloadFamilyIn):
    if d.family not in FAMILIES:
        raise HTTPException(400, f"Famille inconnue : {d.family}")
    try:
        jids = downloads.start_family_download(d.family, include_vision=d.include_vision)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"jobs": jids, "count": len(jids)}


class EngineIn(BaseModel):
    flavor: Optional[str] = None


@app.post("/api/engine/install")
def engine_install(e: EngineIn):
    return {"job": downloads.start_engine_install(e.flavor)}


@app.get("/api/jobs")
def jobs():
    return downloads.list_jobs()


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    ok = downloads.cancel_job(jid)
    if not ok:
        raise HTTPException(404, "Tâche introuvable ou déjà terminée")
    return {"ok": True}


@app.post("/api/jobs/clear")
def jobs_clear():
    downloads.clear_finished()
    return {"ok": True}


@app.delete("/api/models/{family}/{category}/{name}")
def delete_model(family: str, category: str, name: str):
    if family not in FAMILIES or category not in CATEGORIES or "/" in name or "\\" in name:
        raise HTTPException(400, "requête invalide")
    p = model_dir(family, category) / name
    if p.exists():
        p.unlink()
    return {"ok": True}


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
    ref_images: list[UploadFile] = File(default=[]),
):
    if not prompt.strip():
        raise HTTPException(400, "Le prompt est vide.")
    width, height = max(256, width // 32 * 32), max(256, height // 32 * 32)
    refs: list[str] = []
    for f in ref_images:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower() or ".png"
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
        dest.write_bytes(await f.read())
        refs.append(str(dest))
    params = {
        "prompt": prompt, "negative_prompt": negative_prompt, "width": width, "height": height,
        "steps": steps, "cfg_scale": cfg_scale, "sampler": sampler, "seed": seed, "ref_images": refs,
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
