"""Téléchargements en arrière-plan (modèles et moteur) avec suivi de progression."""
from __future__ import annotations

import threading
import time
import urllib.request
import uuid
from pathlib import Path

from . import backend
from .catalog import FAMILIES
from .paths import model_dir

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _new_job(kind: str, label: str) -> str:
    jid = uuid.uuid4().hex[:8]
    with _lock:
        _jobs[jid] = {
            "id": jid, "kind": kind, "label": label, "status": "running",
            "done": 0, "total": 0, "message": "", "started": time.time(),
        }
    return jid


def _update(jid: str, **kw):
    with _lock:
        _jobs[jid].update(kw)


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)


def clear_finished():
    with _lock:
        for k in [k for k, j in _jobs.items() if j["status"] != "running"]:
            del _jobs[k]


# ---------------------------------------------------------------- modèles
def _download_file(url: str, dest: Path, jid: str):
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
        _update(jid, total=total, done=done)
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                _update(jid, done=done)
    tmp.replace(dest)


def start_model_download(family: str, category: str, file_id: str, custom_url: str | None = None) -> str:
    dest_dir = model_dir(family, category)
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
    jid = _new_job("model", f"{family}/{category}/{file_id}")

    def run():
        try:
            _download_file(url, dest, jid)
            _update(jid, status="done", message="Terminé")
        except Exception as e:
            _update(jid, status="error", message=str(e))

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
