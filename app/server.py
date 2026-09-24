from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import backend, config, downloads, generator
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
    ready = all(sel.get(k) for k in ("diffusion", "text_encoder", "vae"))
    families_status = {
        f: {"ready": all(cfg["selections"][f].get(k) for k in ("diffusion", "text_encoder", "vae")),
            "edit_ready": (not FAMILIES[f]["edit_requires_vision"]) or bool(cfg["selections"][f].get("vision"))}
        for f in FAMILIES
    }
    return {
        "engine": backend.installed_info(),
        "models": local,
        "families": [{k: v for k, v in FAMILIES[f].items()} for f in FAMILIES],
        "families_status": families_status,
        "config": cfg,
        "ready": ready and backend.find_binary() is not None,
        "edit_ready": families_status[fam]["edit_ready"],
        "samplers": generator.SAMPLERS,
        "disk_free_gb": round(shutil.disk_usage(str(MODELS_DIR)).free / 1e9, 1),
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
    try:
        jid = downloads.start_model_download(d.family, d.category, d.file_id, d.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"job": jid}


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
    width = min(2048, max(256, round(width / 32) * 32))
    height = min(2048, max(256, round(height / 32) * 32))
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
