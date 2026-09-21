"""Téléchargements en arrière-plan (modèles et moteur) avec suivi de progression.

Trois types de tâches :
  - ``model``  : un fichier précis du catalogue (ou une URL collée à la main) ;
  - ``bundle`` : installation complète d'une famille en un clic
                 (diffusion + encodeur de texte + VAE [+ mmproj pour l'édition]) ;
  - ``engine`` : moteur stable-diffusion.cpp (sd-cli).

Chaque tâche expose ``files`` (une entrée par fichier) pour afficher la progression
fichier par fichier dans l'interface, et peut être annulée.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import backend, config
from .catalog import FAMILIES, entry, install_plan
from .paths import model_dir

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

CATEGORY_LABELS = {
    "diffusion": "modèle de diffusion",
    "text_encoder": "encodeur de texte",
    "vae": "VAE",
    "vision": "encodeur de vision (mmproj)",
    "lora": "LoRA",
}


def _new_job(kind: str, label: str, family: str = "", files: list[dict] | None = None) -> str:
    jid = uuid.uuid4().hex[:8]
    with _lock:
        _jobs[jid] = {
            "id": jid, "kind": kind, "label": label, "family": family, "status": "running",
            "done": 0, "total": 0, "message": "", "started": time.time(),
            "files": files or [], "cancel": threading.Event(),
        }
    return jid


def _update(jid: str, **kw):
    with _lock:
        job = _jobs.get(jid)
        if job:
            job.update(kw)


def _update_file(jid: str, index: int, **kw):
    with _lock:
        job = _jobs.get(jid)
        if job and 0 <= index < len(job["files"]):
            job["files"][index].update(kw)


def _cancelled(jid: str) -> bool:
    with _lock:
        job = _jobs.get(jid)
        return bool(job and job["cancel"].is_set())


def _overall(jid: str) -> None:
    """Recalcule l'avancement global d'une tâche à partir de ses fichiers."""
    with _lock:
        job = _jobs.get(jid)
        if not job:
            return
        done = sum(f.get("done", 0) for f in job["files"])
        total = sum(f.get("total", 0) for f in job["files"])
        job["done"], job["total"] = done, total
        running = next((f for f in job["files"] if f.get("status") in ("running", "pending")), None)
        if running:
            job["message"] = f"{running['label']}"


def list_jobs() -> list[dict]:
    with _lock:
        out = []
        for j in sorted(_jobs.values(), key=lambda j: j["started"], reverse=True):
            j = {k: v for k, v in j.items() if k != "cancel"}
            j["files"] = [dict(f) for f in j["files"]]
            out.append(j)
        return out


def clear_finished():
    with _lock:
        for k in [k for k, j in _jobs.items() if j["status"] != "running"]:
            del _jobs[k]


def cancel(jid: str) -> bool:
    with _lock:
        job = _jobs.get(jid)
        if not job or job["status"] != "running":
            return False
        job["cancel"].set()
    return True


def cancel_all() -> int:
    with _lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
    return sum(1 for j in running if cancel(j["id"]))


# ---------------------------------------------------------------- fichiers
def _download_file(url: str, dest: Path, jid: str, index: int | None = None):
    """Téléchargement reprenable. index = position du fichier dans la tâche (progression fine)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "local-image-qwen"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status == 206:
            total = existing + int(r.headers.get("Content-Length", 0))
            mode = "ab"
        else:
            total = int(r.headers.get("Content-Length", 0))
            existing = 0
            mode = "wb"
        done = existing
        if index is None:
            _update(jid, total=total, done=done)
        else:
            _update_file(jid, index, total=total, done=done, status="running")
            _overall(jid)
        with open(tmp, mode) as f:
            while True:
                if _cancelled(jid):
                    raise InterruptedError("Téléchargement annulé")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if index is None:
                    _update(jid, done=done)
                else:
                    _update_file(jid, index, done=done)
                    _overall(jid)
    tmp.replace(dest)


def _auto_select(family: str, category: str, name: str) -> None:
    """Après un téléchargement : sélectionne le fichier si la case est vide ou périmée."""
    if not name:
        return
    cfg = config.load()
    sel = cfg.get("selections", {}).get(family, {})
    current = sel.get(category, "")
    if current and (model_dir(family, category) / current).exists():
        return
    config.save({"selections": {family: {category: name}}})


def _run(job_id: str, tasks: list[dict]) -> None:
    """Exécute une liste de tâches {url, dest, family, category, name} dans l'ordre."""
    try:
        for i, t in enumerate(tasks):
            if _cancelled(job_id):
                raise InterruptedError("Téléchargement annulé")
            dest: Path = t["dest"]
            if dest.exists():
                size = dest.stat().st_size
                _update_file(job_id, i, status="done", done=size, total=size, message="déjà présent")
                _overall(job_id)
                _auto_select(t["family"], t["category"], t["name"])
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            _update_file(job_id, i, status="running")
            _overall(job_id)
            try:
                _download_file(t["url"], dest, job_id, i)
            except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                raise RuntimeError(f"{t['name']} : HTTP {e.code} — {e.reason} (URL : {t['url']})") from e
            except urllib.error.URLError as e:  # type: ignore[attr-defined]
                raise RuntimeError(f"{t['name']} : impossible de télécharger ({e.reason}). URL : {t['url']}") from e
            _update_file(job_id, i, status="done", done=dest.stat().st_size, message="terminé")
            _overall(job_id)
            _auto_select(t["family"], t["category"], t["name"])
        _update(job_id, status="done", message="Terminé")
    except InterruptedError as e:
        _update(job_id, status="cancelled", message=str(e))
    except Exception as e:  # pragma: no cover - dépend du réseau
        _update(job_id, status="error", message=str(e))


# ---------------------------------------------------------------- modèles
def start_model_download(family: str, category: str, file_id: str, custom_url: str | None = None) -> str:
    """Télécharge un seul fichier (catalogue ou URL personnalisée)."""
    dest_dir = model_dir(family, category)
    url = custom_url
    if not url:
        e = entry(family, category, file_id)
        if not e:
            raise ValueError("fichier inconnu dans le catalogue")
        url = e["url"]
    if not file_id:
        file_id = url.split("?")[0].rstrip("/").split("/")[-1]
    if not file_id or "/" in file_id or "\\" in file_id:
        raise ValueError("nom de fichier invalide")
    dest = dest_dir / file_id
    if dest.exists():
        raise ValueError("fichier déjà présent")
    catalog_entry = entry(family, category, file_id)
    size = int(catalog_entry["size_gb"] * 1e9) if catalog_entry else 0
    jid = _new_job("model", f"{FAMILIES[family]['name']} · {CATEGORY_LABELS.get(category, category)} · {file_id}",
                   family=family,
                   files=[{"label": file_id, "category": category,
                           "size_gb": catalog_entry["size_gb"] if catalog_entry else 0,
                           "status": "pending", "done": 0, "total": size, "message": ""}])
    threading.Thread(target=_run, args=(jid, [{"url": url, "dest": dest, "family": family, "category": category, "name": file_id}]),
                     daemon=True).start()
    return jid


def start_family_install(family: str, diffusion_id: str = "", text_encoder_id: str = "",
                         include_vision: bool = True, include_existing: bool = True) -> str:
    """
    Installe une famille complète en un clic : diffusion + encodeur de texte + VAE
    (+ encodeur de vision/mmproj si le modèle en a besoin pour l'édition d'image).
    Les fichiers déjà présents sont simplement marqués « déjà présent ».
    """
    if family not in FAMILIES:
        raise ValueError("famille inconnue")
    plan = install_plan(family, diffusion_id, text_encoder_id, include_vision)
    if not plan:
        raise ValueError("aucun fichier à télécharger pour ce modèle")

    tasks, files = [], []
    for e in plan:
        dest = model_dir(family, e["category"]) / e["id"]
        files.append({
            "label": e["id"], "category": e["category"], "size_gb": e["size_gb"],
            "status": "done" if dest.exists() else "pending",
            "done": dest.stat().st_size if dest.exists() else 0,
            "total": dest.stat().st_size if dest.exists() else int(e["size_gb"] * 1e9),
            "message": "déjà présent" if dest.exists() else "",
        })
        if include_existing or not dest.exists():
            tasks.append({"url": e["url"], "dest": dest, "family": family, "category": e["category"], "name": e["id"]})

    to_fetch = [f for f in files if f["status"] != "done"]
    label = f"{FAMILIES[family]['name']} — installation complète"
    if not to_fetch:
        jid = _new_job("bundle", label, family=family, files=files)
        _update(jid, status="done", message="Tous les fichiers sont déjà présents")
        _overall(jid)
        for f in files:
            _auto_select(family, f["category"], f["label"])
        return jid

    jid = _new_job("bundle", label, family=family, files=files)
    _overall(jid)
    threading.Thread(target=_run, args=(jid, tasks), daemon=True).start()
    return jid


# ---------------------------------------------------------------- moteur
def start_engine_install(flavor: str | None) -> str:
    jid = _new_job("engine", f"stable-diffusion.cpp ({flavor or backend.detect_flavor()})",
                   files=[{"label": f"sd-cli ({flavor or backend.detect_flavor()})", "category": "engine",
                           "status": "pending", "done": 0, "total": 0, "message": ""}])

    def prog(msg, done, total):
        _update(jid, message=msg, done=done, total=total)
        _update_file(jid, 0, status="running", done=done, total=total, message=msg)
        _overall(jid)

    def run():
        try:
            exe = backend.install(flavor, prog)
            _update(jid, status="done", message=f"Installé : {exe}")
            size = exe.stat().st_size
            _update_file(jid, 0, status="done", done=size, total=size, message="installé")
            _overall(jid)
        except Exception as e:
            _update(jid, status="error", message=str(e))
            _update_file(jid, 0, status="error", message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jid
