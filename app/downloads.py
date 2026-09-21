"""Téléchargements en arrière-plan (modèles, modèles complets et moteur) avec suivi de progression."""
from __future__ import annotations

import shutil
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from . import backend
from .catalog import DEFAULT_TIER, FAMILIES, TIER_SHORT, bundle_plan
from . import paths

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _new_job(kind: str, label: str, **extra) -> str:
    jid = uuid.uuid4().hex[:8]
    with _lock:
        _jobs[jid] = {
            "id": jid, "kind": kind, "label": label, "status": "running",
            "done": 0, "total": 0, "message": "", "started": time.time(),
            "family": None, "tier": None, "files": [],
        }
        _jobs[jid].update(extra)
    return jid


def _update(jid: str, **kw):
    with _lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def list_jobs() -> list[dict]:
    with _lock:
        out = [{**j, "files": [dict(f) for f in j["files"]]} for j in _jobs.values()]
    return sorted(out, key=lambda j: j["started"], reverse=True)


def clear_finished():
    with _lock:
        for k in [k for k, j in _jobs.items() if j["status"] != "running"]:
            del _jobs[k]


# ---------------------------------------------------------------- téléchargement
def _download_file(url: str, dest: Path, jid: str | None = None, on_progress=None) -> int:
    """Télécharge `url` vers `dest` (reprise sur fichier .part). Retourne le nombre d'octets écrits."""
    dest.parent.mkdir(parents=True, exist_ok=True)
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

        def report():
            if jid:
                _update(jid, total=total, done=done)
            if on_progress:
                on_progress(done, total)

        report()
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                report()
    tmp.replace(dest)
    return done


# ---------------------------------------------------------------- un fichier
def start_model_download(family: str, category: str, file_id: str, custom_url: str | None = None) -> str:
    dest_dir = paths.model_dir(family, category)
    url = custom_url
    if not url:
        entry = next((e for e in FAMILIES[family].get(category, []) if e["id"] == file_id), None)
        if not entry:
            raise ValueError("fichier inconnu dans le catalogue")
        url = entry["url"]
    if not file_id:
        file_id = url.split("?")[0].rstrip("/").split("/")[-1]
    dest = dest_dir / file_id
    if dest.exists():
        raise ValueError("fichier déjà présent")
    jid = _new_job("model", f"{family}/{category}/{file_id}", family=family)

    def run():
        try:
            size = _download_file(url, dest, jid)
            _update(jid, status="done", done=size, message="Terminé")
        except Exception as e:
            _update(jid, status="error", message=str(e))

    threading.Thread(target=run, daemon=True).start()
    return jid


# ---------------------------------------------------------------- modèle complet
def start_bundle_download(family: str, tier: str = DEFAULT_TIER, include_vision: bool | None = None) -> str:
    """Télécharge tous les fichiers d'un modèle (diffusion + encodeur de texte + VAE [+ vision]).

    Les fichiers déjà présents sont ignorés : on peut relancer pour compléter/reprendre.
    """
    fam = FAMILIES.get(family)
    if fam is None:
        raise ValueError(f"modèle inconnu : {family}")
    plan = bundle_plan(family, tier, include_vision=include_vision)

    todo = []
    for p in plan:
        dest = paths.model_dir(family, p["category"]) / p["id"]
        if not dest.exists():
            todo.append({**p, "dest": dest})
    if not todo:
        raise ValueError(f"{fam['name']} : tous les fichiers sont déjà téléchargés.")

    need = sum(p["size_bytes"] for p in todo)
    free = shutil.disk_usage(str(paths.MODELS_DIR)).free
    if free < need:
        raise ValueError(f"Espace disque insuffisant : {need / 1e9:.1f} Go nécessaires, {free / 1e9:.1f} Go libres.")

    jid = _new_job(
        "bundle", f"{fam['name']} — {TIER_SHORT.get(tier, tier)}",
        family=family, tier=tier, total=need,
        message=f"{len(todo)} fichier(s) à télécharger",
        files=[{"category": p["category"], "id": p["id"], "size_gb": p["size_gb"],
                "status": "pending", "done": 0, "total": p["size_bytes"]} for p in todo],
    )

    def run():
        n = len(todo)
        written = 0        # octets réellement écrits (fichiers terminés)
        remaining = need   # estimation des fichiers pas encore terminés
        try:
            for i, p in enumerate(todo):
                est = p["size_bytes"]

                def sync(downloaded: int, real_total: int, i: int = i, est: int = est):
                    adj = (real_total - est) if real_total else 0
                    with _lock:
                        j = _jobs[jid]
                        j["done"] = written + downloaded
                        j["total"] = max(1, written + remaining + adj)
                        j["files"][i].update(done=downloaded, total=real_total or est)

                with _lock:
                    _jobs[jid]["files"][i]["status"] = "running"
                _update(jid, message=f"({i + 1}/{n}) {p['id']}")
                size = _download_file(p["url"], p["dest"], on_progress=sync)
                with _lock:
                    _jobs[jid]["files"][i].update(status="done", done=size, total=size)
                written += size
                remaining -= est
            _update(jid, status="done", done=written, total=max(1, written),
                    message=f"Terminé — {n} fichier(s), {written / 1e9:.2f} Go")
        except Exception as e:
            with _lock:
                for f in _jobs[jid]["files"]:
                    if f["status"] == "running":
                        f["status"] = "error"
            _update(jid, status="error", message=f"{e} — relancez le téléchargement pour reprendre.")

    threading.Thread(target=run, daemon=True).start()
    return jid


# ---------------------------------------------------------------- moteur
def start_engine_install(flavor: str | None) -> str:
    jid = _new_job("engine", f"stable-diffusion.cpp ({flavor or backend.detect_flavor()})")

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
