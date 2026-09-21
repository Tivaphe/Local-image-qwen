"""Fixtures communes : dossiers modèles/config isolés + petit serveur HTTP local.

Aucun accès à Internet n'est nécessaire : les téléchargements de test se font
depuis un serveur HTTP local lancé par les fixtures.
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import paths  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirige models/ et config.json vers un dossier temporaire."""
    from app import catalog

    models = tmp_path / "models"
    for fam in catalog.FAMILIES:
        for cat in catalog.CATEGORIES:
            (models / fam / cat).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "MODELS_DIR", models)
    monkeypatch.setattr(paths, "CONFIG_FILE", tmp_path / "config.json")
    return models


@pytest.fixture()
def fake_engine(tmp_path, monkeypatch):
    """Faux binaire sd-cli : permet de tester la ligne de commande sans moteur installé."""
    import os

    from app import backend

    exe = tmp_path / ("sd-cli.exe" if os.name == "nt" else "sd-cli")
    exe.write_bytes(b"#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(backend, "find_binary", lambda: exe)
    return exe


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def file_server(tmp_path):
    """Sert un dossier via HTTP local (comme Hugging Face, en minuscule)."""
    payload = b"x" * (512 * 1024)
    served = tmp_path / "served"
    served.mkdir()
    (served / "model-test.gguf").write_bytes(payload)
    (served / "encoder-test.gguf").write_bytes(payload)
    (served / "vae-test.safetensors").write_bytes(payload)
    (served / "vision-test.gguf").write_bytes(payload)
    (served / "pose-test.onnx").write_bytes(payload)
    (served / "control-test.safetensors").write_bytes(payload)

    handler = functools.partial(_Quiet, directory=str(served))
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as httpd:
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}", served
        httpd.shutdown()


def wait_job(job_id: str, timeout: float = 30.0) -> dict:
    """Attend la fin d'une tâche et la renvoie."""
    import time

    from app import downloads

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = next((j for j in downloads.list_jobs() if j["id"] == job_id), None)
        if job and job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"tâche {job_id} non terminée")
