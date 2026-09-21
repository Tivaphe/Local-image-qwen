"""Téléchargements en arrière-plan (modèles et moteur) avec suivi de progression, vitesse et annulation."""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import backend
from .catalog import CATEGORIES, FAMILIES
from .paths import model_dir

_jobs: dict[str, dict] = {}
_cancel_flags: dict[str, bool] = {}
_lock = threading.Lock()


def _new_job(kind: str, label: str, family: str = "", category: str = "", file_id: str = "") -> str:
    import uuid
    jid = uuid.uuid4().hex[:8]
    with _lock:
        _jobs[jid] = {
            "id": jid,
            "kind": kind,
            "label": label,
            "family": family,
            "category": category,
            "file_id": file_id,
            "status": "running",
            "done": 0,
            "total": 0,
            "speed": 0.0,  # bytes/s
            "eta": 0.0,    # seconds
            "message": "",
            "started": time.time(),
        }
        _cancel_flags[jid] = False
    return jid


def _update(jid: str, **kw):
    with _lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def is_cancelled(jid: str) -> bool:
    with _lock:
        return _cancel_flags.get(jid, False)


def cancel_job(jid: str) -> bool:
    with _lock:
        if jid in _jobs and _jobs[jid]["status"] == "running":
            _cancel_flags[jid] = True
            _jobs[jid]["status"] = "cancelled"
            _jobs[jid]["message"] = "Annulé par l'utilisateur"
            return True
    return False


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)


def clear_finished():
    with _lock:
        for k in [k for k, j in _jobs.items() if j["status"] != "running"]:
            del _jobs[k]
            _cancel_flags.pop(k, None)


def normalize_url(url: str) -> str:
    url = url.strip()
    # Hugging Face: convertir les liens d'interface web /blob/ en liens directs /resolve/
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    return url


# ---------------------------------------------------------------- modèles
def _download_file(url: str, dest: Path, jid: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Local-image-qwen"
    }
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if is_cancelled(jid):
                tmp.unlink(missing_ok=True)
                return

            if r.status == 206:
                content_len = int(r.headers.get("Content-Length", 0) or 0)
                total = existing + content_len
                mode = "ab"
            else:
                total = int(r.headers.get("Content-Length", 0) or 0)
                existing = 0
                mode = "wb"

            done = existing
            _update(jid, total=total, done=done)

            start_t = time.time()
            last_t = start_t
            last_done = done
            speed = 0.0

            with open(tmp, mode) as f:
                while True:
                    if is_cancelled(jid):
                        tmp.unlink(missing_ok=True)
                        _update(jid, status="cancelled", message="Téléchargement annulé")
                        return

                    chunk = r.read(1 << 20)  # 1 Mo
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)

                    now = time.time()
                    dt = now - last_t
                    if dt >= 0.5:
                        speed = (done - last_done) / dt
                        eta = (total - done) / speed if (speed > 0 and total > done) else 0.0
                        last_t = now
                        last_done = done
                        _update(jid, done=done, total=total, speed=speed, eta=eta)

        if is_cancelled(jid):
            tmp.unlink(missing_ok=True)
            return

        tmp.replace(dest)
        _update(jid, status="done", message="Terminé", done=total if total else done, total=total if total else done, speed=0.0, eta=0.0)

    except Exception as e:
        if is_cancelled(jid):
            tmp.unlink(missing_ok=True)
            _update(jid, status="cancelled", message="Téléchargement annulé")
        else:
            tmp.unlink(missing_ok=True)
            _update(jid, status="error", message=str(e), speed=0.0, eta=0.0)
            raise


def start_model_download(family: str, category: str, file_id: str, custom_url: str | None = None) -> str:
    if family not in FAMILIES:
        raise ValueError("Famille de modèle inconnue")
    if category not in CATEGORIES:
        raise ValueError("Catégorie inconnue")

    dest_dir = model_dir(family, category)
    url = normalize_url(custom_url) if custom_url else ""
    
    if not url:
        entry = next((e for e in FAMILIES[family].get(category, []) if e["id"] == file_id), None)
        if not entry:
            raise ValueError(f"Fichier '{file_id}' inconnu dans le catalogue de {family}/{category}")
        url = entry["url"]
        
    if not file_id:
        parsed_path = urllib.parse.urlparse(url).path
        raw_name = parsed_path.rstrip("/").split("/")[-1]
        file_id = urllib.parse.unquote(raw_name)
        if not file_id:
            file_id = "model.gguf"

    dest = dest_dir / file_id
    if dest.exists():
        raise ValueError(f"Le fichier '{file_id}' est déjà présent dans {family}/{category}")

    label = f"{family}/{category}/{file_id}"
    jid = _new_job("model", label, family=family, category=category, file_id=file_id)

    def run():
        try:
            _download_file(url, dest, jid)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()
    return jid


def start_family_download(family: str, include_vision: bool = True) -> list[str]:
    """Télécharge l'ensemble des fichiers recommandés pour une famille."""
    if family not in FAMILIES:
        raise ValueError(f"Famille inconnue : {family}")
    fam = FAMILIES[family]
    started_jobs: list[str] = []
    
    cats = ["diffusion", "text_encoder", "vae"]
    if fam.get("edit_requires_vision") and include_vision:
        cats.append("vision")
        
    for cat in cats:
        items = fam.get(cat, [])
        if not items:
            continue
        rec = next((x for x in items if x.get("recommended")), items[0])
        file_id = rec["id"]
        dest = model_dir(family, cat) / file_id
        if dest.exists():
            continue  # déjà téléchargé
        
        # Vérifier si déjà en cours de téléchargement
        is_running = any(
            j["status"] == "running" and j["family"] == family and j["category"] == cat and j["file_id"] == file_id
            for j in list_jobs()
        )
        if is_running:
            continue
            
        jid = start_model_download(family, cat, file_id, rec["url"])
        started_jobs.append(jid)

    return started_jobs


# ---------------------------------------------------------------- moteur
def start_engine_install(flavor: str | None) -> str:
    target_flavor = flavor or backend.detect_flavor()
    jid = _new_job("engine", f"stable-diffusion.cpp ({target_flavor})")

    def prog(msg, done, total):
        _update(jid, message=msg, done=done, total=total)

    def run():
        try:
            exe = backend.install(flavor, prog)
            _update(jid, status="done", message=f"Installé : {exe}")
        except Exception as e:
            _update(jid, status="error", message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jid
