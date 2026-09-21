"""Tests du téléchargement des modèles : catalogue, téléchargement réel (HTTP local) et API.

Lancement :  python -m pytest tests -q
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import time

import pytest

from app import catalog, config, downloads, paths, server


# ------------------------------------------------------------------ miroir HTTP
@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """Sert les fichiers du catalogue en local et redirige models/ + config.json vers tmp_path."""
    serve_dir = tmp_path / "mirror"
    serve_dir.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # silencieux
            pass

    httpd = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(serve_dir))
    )
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # chaque fichier du catalogue -> un petit fichier local homonyme, servi en http
    for fam in catalog.FAMILIES.values():
        for cat in catalog.CATEGORIES:
            for e in fam.get(cat, []):
                f = serve_dir / e["id"]
                if not f.exists():
                    f.write_bytes(b"GGUF" + e["id"].encode() + b"\0" * 512)
                monkeypatch.setitem(e, "url", f"http://127.0.0.1:{port}/{e['id']}")

    models = tmp_path / "models"
    monkeypatch.setattr(paths, "MODELS_DIR", models)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    for fam in catalog.FAMILIES:
        for cat in catalog.CATEGORIES:
            (models / fam / cat).mkdir(parents=True, exist_ok=True)

    yield models
    httpd.shutdown()
    httpd.server_close()


def _wait(jid: str, timeout: float = 30.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = next(x for x in downloads.list_jobs() if x["id"] == jid)
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError(f"téléchargement {jid} toujours en cours après {timeout} s")


# -------------------------------------------------------------------- catalogue
@pytest.mark.parametrize("fam_id", sorted(catalog.FAMILIES))
@pytest.mark.parametrize("tier", catalog.TIERS)
def test_bundle_complet_et_coherent(fam_id, tier):
    plan = catalog.bundle_plan(fam_id, tier)
    cats = [p["category"] for p in plan]
    assert set(catalog.REQUIRED_CATEGORIES) <= set(cats)
    for p in plan:
        assert p["id"] in [e["id"] for e in catalog.FAMILIES[fam_id][p["category"]]]
        assert p["url"].startswith("https://huggingface.co/")
        assert p["size_bytes"] > 0
    # l'encodeur de vision n'est embarqué que si le modèle en a besoin
    assert ("vision" in cats) == bool(catalog.FAMILIES[fam_id]["edit_requires_vision"])


def test_bundle_sans_vision_sur_demande():
    assert len(catalog.bundle_plan("qwen_image_2.1")) == 4
    assert len(catalog.bundle_plan("qwen_image_2.1", include_vision=False)) == 3


def test_bundle_qualite_inconnue():
    with pytest.raises(ValueError, match="qualité"):
        catalog.bundle_plan("qwen_image_2.1", "ultra")


def test_bundle_modele_inconnu():
    with pytest.raises(ValueError, match="inconnu"):
        catalog.bundle_plan("sdxl", "balanced")


def test_les_tailles_du_catalogue_sont_coherentes():
    for fam_id, fam in catalog.FAMILIES.items():
        for tier in catalog.TIERS:
            b = catalog.bundle_status(fam_id, tier, {})
            assert b["total_gb"] == round(sum(f["size_gb"] for f in b["files"]), 2)
            assert b["missing_gb"] == b["total_gb"]
            assert not b["complete"]


# ------------------------------------------------------------- téléchargement réel
def test_telechargement_du_modele_complet(mirror):
    plan = catalog.bundle_plan("flux2_klein_4b", "balanced")
    jid = downloads.start_bundle_download("flux2_klein_4b", "balanced")
    j = _wait(jid)

    assert j["status"] == "done", j["message"]
    assert j["done"] == j["total"] > 0
    assert [f["status"] for f in j["files"]] == ["done"] * len(plan)
    for p in plan:
        f = mirror / "flux2_klein_4b" / p["category"] / p["id"]
        assert f.exists() and f.stat().st_size > 0
        assert not (f.parent / (f.name + ".part")).exists()


def test_les_fichiers_deja_presents_sont_ignores(mirror):
    assert _wait(downloads.start_bundle_download("flux2_klein_9b", "light"))["status"] == "done"
    with pytest.raises(ValueError, match="déjà"):
        downloads.start_bundle_download("flux2_klein_9b", "light")
    # autre qualité : seuls les fichiers manquants sont re-téléchargés (la VAE est déjà là)
    j = _wait(downloads.start_bundle_download("flux2_klein_9b", "quality"))
    assert j["status"] == "done"
    assert len(j["files"]) == 2


def test_un_fichier_partiel_ne_bloque_pas(mirror):
    p = catalog.bundle_plan("flux2_klein_4b", "light")[0]
    part = mirror / "flux2_klein_4b" / p["category"] / (p["id"] + ".part")
    part.write_bytes(b"X" * 100)
    j = _wait(downloads.start_bundle_download("flux2_klein_4b", "light"))
    assert j["status"] == "done"
    assert (mirror / "flux2_klein_4b" / p["category"] / p["id"]).exists()


def test_erreur_reseau_remontee(mirror, monkeypatch):
    for fam in catalog.FAMILIES.values():
        for cat in catalog.CATEGORIES:
            for e in fam.get(cat, []):
                monkeypatch.setitem(e, "url", "http://127.0.0.1:1/introuvable.gguf")
    j = _wait(downloads.start_bundle_download("flux2_klein_4b", "balanced"))
    assert j["status"] == "error"
    assert "reprendre" in j["message"]
    assert any(f["status"] == "error" for f in j["files"])


def test_espace_disque_insuffisant(mirror, monkeypatch):
    monkeypatch.setattr(
        downloads.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024 * 1024})()
    )
    with pytest.raises(ValueError, match="insuffisant"):
        downloads.start_bundle_download("qwen_image_2.1", "quality")


# ------------------------------------------------------------------------- API
def test_api_status_expose_les_bundles(mirror):
    from fastapi.testclient import TestClient

    st = TestClient(server.app).get("/api/status").json()
    assert set(st["bundles"]) == set(catalog.FAMILIES)
    assert [t["id"] for t in st["tiers"]] == list(catalog.TIERS)
    assert st["default_tier"] in catalog.TIERS
    assert st["bundles"]["qwen_image_2.1"]["balanced"]["total_gb"] > 10
    assert st["bundles"]["flux2_klein_4b"]["light"]["missing_gb"] > 4


def test_api_telechargement_en_un_clic_rend_le_modele_pret(mirror):
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    r = c.post("/api/download/bundle", json={"family": "qwen_image_2.1", "tier": "light"})
    assert r.status_code == 200, r.text
    j = _wait(r.json()["job"])
    assert j["status"] == "done", j["message"]

    st = c.get("/api/status").json()
    assert st["bundles"]["qwen_image_2.1"]["light"]["complete"] is True
    assert st["families_status"]["qwen_image_2.1"]["ready"] is True
    assert st["families_status"]["qwen_image_2.1"]["edit_ready"] is True
    sel = st["config"]["selections"]["qwen_image_2.1"]
    assert sel["diffusion"] == "qwen_image_2.1-Q3_K.gguf"
    assert sel["text_encoder"] == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    assert sel["vae"] == "qwen_image_2.1_vae_bf16.safetensors"
    assert sel["vision"] == "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"


def test_api_refuse_les_modeles_et_qualites_inconnus(mirror):
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    assert c.post("/api/download/bundle", json={"family": "sdxl"}).status_code == 400
    assert c.post("/api/download/bundle", json={"family": "flux2_klein_4b", "tier": "ultra"}).status_code == 400


def test_interface_propose_bien_le_telechargement(mirror):
    from fastapi.testclient import TestClient

    c = TestClient(server.app)
    html = c.get("/").text
    assert 'id="bundleCards"' in html
    assert 'id="btnQuickDl"' in html
    js = c.get("/static/app.js").text
    assert "renderBundles" in js
    assert "/api/download/bundle" in js
