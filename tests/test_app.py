"""Tests de l'application : catalogue, téléchargements, API, ligne de commande sd-cli."""
from __future__ import annotations

import json

import pytest

from conftest import wait_job

from app import catalog, config, downloads, generator
from app.paths import model_dir


# ------------------------------------------------------------------ catalogue
def test_quatre_familles_avec_pack_complet():
    assert set(catalog.FAMILIES) == {"qwen_image_2.1", "flux2_klein_9b", "flux2_klein_4b", "sd15_control"}
    for fam in catalog.FAMILIES.values():
        pack = fam["pack"]
        # génération toujours complète
        for cat in ("diffusion", "text_encoder", "vae"):
            if not fam["bundled"]:                      # SD 1.5 : un seul fichier (UNet + CLIP + VAE)
                assert catalog.entry(fam["id"], cat, pack[cat]), f"{fam['id']}/{cat} introuvable"
        assert catalog.entry(fam["id"], "diffusion", pack["diffusion"])
        # édition : mmproj obligatoire pour Qwen, natif pour FLUX.2 klein
        if fam["edit_requires_vision"]:
            assert catalog.entry(fam["id"], "vision", pack["vision"])
        else:
            assert pack["vision"] == ""
        # ControlNet : uniquement sur la famille SD 1.5 (UNet), avec une explication ailleurs
        if fam["supports_controlnet"]:
            assert fam["control_modes"] and fam["control_reason"] == ""
            for cat in ("controlnet", "pose_detector"):
                assert catalog.entry(fam["id"], cat, pack[cat]), f"{fam['id']}/{cat} introuvable"
        else:
            assert fam["control_reason"]
        assert fam["edit_info"]


def test_plan_installation_et_total():
    plan = catalog.install_plan("qwen_image_2.1")
    assert [e["category"] for e in plan] == ["diffusion", "text_encoder", "vae", "vision"]
    assert sum(e["size_gb"] for e in plan) == pytest.approx(catalog.pack_total_gb("qwen_image_2.1"), abs=0.05)
    assert all(e["url"].startswith("https://huggingface.co/") for e in plan)

    sans_edition = catalog.install_plan("qwen_image_2.1", include_vision=False)
    assert "vision" not in [e["category"] for e in sans_edition]

    # FLUX.2 klein : édition native, pas de fichier supplémentaire à télécharger
    for fam in ("flux2_klein_4b", "flux2_klein_9b"):
        assert [e["category"] for e in catalog.install_plan(fam)] == ["diffusion", "text_encoder", "vae"]


def test_quantification_choisie_et_variante_inconnue():
    plan = catalog.install_plan("flux2_klein_9b", diffusion_id="flux-2-klein-9b-Q4_K_M.gguf")
    assert plan[0]["id"] == "flux-2-klein-9b-Q4_K_M.gguf"
    # identifiant inconnu : on retombe sur le fichier recommandé, jamais sur une erreur
    assert catalog.install_plan("flux2_klein_9b", diffusion_id="inexistant.gguf")[0]["recommended"] is True


# ------------------------------------------------------------- téléchargements
def test_installation_complete_telecharge_et_selectionne(sandbox, file_server, monkeypatch):
    base, _ = file_server
    monkeypatch.setattr(downloads, "install_plan", lambda *a, **k: [
        {"category": "diffusion", "id": "model-test.gguf", "size_gb": 0.001, "url": f"{base}/model-test.gguf"},
        {"category": "text_encoder", "id": "encoder-test.gguf", "size_gb": 0.001, "url": f"{base}/encoder-test.gguf"},
        {"category": "vae", "id": "vae-test.safetensors", "size_gb": 0.001, "url": f"{base}/vae-test.safetensors"},
        {"category": "vision", "id": "vision-test.gguf", "size_gb": 0.001, "url": f"{base}/vision-test.gguf"},
    ])
    jid = downloads.start_family_install("qwen_image_2.1")
    job = wait_job(jid)

    assert job["status"] == "done", job["message"]
    assert [f["status"] for f in job["files"]] == ["done"] * 4
    for cat, name in (("diffusion", "model-test.gguf"), ("text_encoder", "encoder-test.gguf"),
                      ("vae", "vae-test.safetensors"), ("vision", "vision-test.gguf")):
        assert (model_dir("qwen_image_2.1", cat) / name).exists()
    # les fichiers téléchargés deviennent la sélection active
    sel = config.load()["selections"]["qwen_image_2.1"]
    assert sel["diffusion"] == "model-test.gguf" and sel["vision"] == "vision-test.gguf"


def test_fichier_deja_present_est_ignore(sandbox, monkeypatch):
    monkeypatch.setattr(downloads, "install_plan", lambda *a, **k: [
        {"category": "diffusion", "id": "deja.gguf", "size_gb": 0.001, "url": "http://127.0.0.1:1/inutile"},
    ])
    (model_dir("flux2_klein_4b", "diffusion") / "deja.gguf").write_bytes(b"ok")
    job = wait_job(downloads.start_family_install("flux2_klein_4b"))
    assert job["status"] == "done"
    assert job["files"][0]["message"] == "déjà présent"


def test_erreur_de_telechargement_est_rapportee(sandbox, monkeypatch):
    monkeypatch.setattr(downloads, "install_plan", lambda *a, **k: [
        {"category": "diffusion", "id": "absent.gguf", "size_gb": 0.001, "url": "http://127.0.0.1:9/absent.gguf"},
    ])
    job = wait_job(downloads.start_family_install("flux2_klein_4b"))
    assert job["status"] == "error"
    assert "absent.gguf" in job["message"]


def test_telechargement_simple_refuse_si_deja_present(sandbox):
    (model_dir("flux2_klein_4b", "vae") / "flux2-vae.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError):
        downloads.start_model_download("flux2_klein_4b", "vae", "flux2-vae.safetensors")
    # un identifiant absent du catalogue est aussi refusé
    with pytest.raises(ValueError):
        downloads.start_model_download("flux2_klein_4b", "vae", "inconnu.safetensors")


# ------------------------------------------------------------------------ API
@pytest.fixture()
def client(sandbox, monkeypatch):
    from fastapi.testclient import TestClient

    from app import server

    monkeypatch.setattr(server, "MODELS_DIR", sandbox)
    return TestClient(server.app)


def test_status_expose_packs_et_progression(client):
    data = client.get("/api/status").json()
    assert len(data["families"]) == 4
    qwen = next(f for f in data["families"] if f["id"] == "qwen_image_2.1")
    assert qwen["pack"]["vision"].startswith("mmproj-")
    assert qwen["pack_total_gb"] > 5
    assert data["families_status"]["flux2_klein_4b"]["edit_ready"] is True
    assert data["families_status"]["qwen_image_2.1"]["edit_ready"] is False
    assert "paths" in data and "jobs" in data


def test_page_index_recharge_les_assets(client):
    html = client.get("/").text
    assert "data-tab=\"models\"" in html          # onglet de téléchargement des modèles
    assert "app.js?v=" in html and "style.css?v=" in html  # anti-cache


def test_endpoint_install_met_en_file_les_fichiers_du_pack(client, monkeypatch, file_server):
    base, _ = file_server
    monkeypatch.setattr(downloads, "install_plan", lambda *a, **k: [
        {"category": "diffusion", "id": "model-test.gguf", "size_gb": 0.001, "url": f"{base}/model-test.gguf"},
    ])
    r = client.post("/api/install", json={"family": "qwen_image_2.1", "include_vision": True})
    assert r.status_code == 200, r.text
    job_id = r.json()["job"]
    job = wait_job(job_id)
    assert job["status"] == "done"
    assert client.post(f"/api/jobs/{job_id}/cancel").json() == {"cancelled": False}
    assert client.post("/api/jobs/clear").json()["ok"] is True


def test_endpoint_install_refuse_une_famille_inconnue(client):
    assert client.post("/api/install", json={"family": "pas_un_modele"}).status_code == 400


def test_download_url_personnalisee_et_validation(client, file_server):
    base, _ = file_server
    r = client.post("/api/download", json={"family": "flux2_klein_4b", "category": "lora",
                                           "url": f"{base}/vae-test.safetensors"})
    assert r.status_code == 200
    assert wait_job(r.json()["job"])["status"] == "done"
    assert (model_dir("flux2_klein_4b", "lora") / "vae-test.safetensors").exists()
    assert client.post("/api/download", json={"family": "x", "category": "diffusion", "file_id": "a.gguf"}).status_code == 400


def test_suppression_de_modele_nettoie_le_fichier(client):
    p = model_dir("flux2_klein_4b", "vae") / "a-supprimer.safetensors"
    p.write_bytes(b"x")
    assert client.delete("/api/models/flux2_klein_4b/vae/a-supprimer.safetensors").status_code == 200
    assert not p.exists()


def test_selection_config(client):
    p = model_dir("flux2_klein_9b", "diffusion") / "flux-2-klein-9b-Q4_K_M.gguf"
    p.write_bytes(b"x")
    r = client.post("/api/config", json={"selections": {"flux2_klein_9b": {"diffusion": p.name}}})
    assert r.status_code == 200
    assert client.get("/api/status").json()["config"]["selections"]["flux2_klein_9b"]["diffusion"] == p.name


# ------------------------------------------------------------- moteur sd-cli
def _assets():
    """Noms d'archives tels que publiés par stable-diffusion.cpp (release master-889)."""
    return [{"name": n} for n in [
        "cudart-sd-bin-win-cu12-x64.zip",
        "sd-master-c678dfe-bin-Darwin-macOS-26.6.2-arm64.zip",
        "sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip",
        "sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64.zip",
        "sd-master-c678dfe-bin-win-cuda12-x64.zip",
        "sd-master-c678dfe-bin-win-rocm-7.14.0-x64.zip",
        "sd-master-c678dfe-bin-win-vulkan-x64.zip",
        "sd-master-c678dfe-bin-win-cpu-x64.zip",
    ]]


@pytest.mark.parametrize("system,flavor,attendu", [
    ("Windows", "cuda", ["sd-master-c678dfe-bin-win-cuda12-x64.zip", "cudart-sd-bin-win-cu12-x64.zip"]),
    ("Windows", "vulkan", ["sd-master-c678dfe-bin-win-vulkan-x64.zip"]),
    ("Windows", "cpu", ["sd-master-c678dfe-bin-win-cpu-x64.zip"]),
    ("Linux", "vulkan", ["sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"]),
    ("Linux", "cpu", ["sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64.zip"]),
    ("Linux", "rocm", ["sd-master-c678dfe-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"]),
    ("Darwin", "metal", ["sd-master-c678dfe-bin-Darwin-macOS-26.6.2-arm64.zip"]),
])
def test_selection_du_binaire_du_moteur(monkeypatch, system, flavor, attendu):
    from app import backend

    monkeypatch.setattr(backend.platform, "system", lambda: system)
    assert [a["name"] for a in backend._pick_assets(_assets(), flavor)] == attendu


# --------------------------------------------------------- ligne de commande
def _prepare(family: str, files: dict) -> None:
    for cat, name in files.items():
        (model_dir(family, cat) / name).write_bytes(b"x")


def test_commande_qwen_edition_utilise_mmproj(sandbox, fake_engine):
    _prepare("qwen_image_2.1", {"diffusion": "d.gguf", "text_encoder": "te.gguf", "vae": "vae.safetensors", "vision": "mm.gguf"})
    cfg = config.load()
    cfg["family"] = "qwen_image_2.1"
    cfg["selections"]["qwen_image_2.1"] = {"diffusion": "d.gguf", "text_encoder": "te.gguf", "vae": "vae.safetensors", "vision": "mm.gguf"}
    params = {"prompt": "change le ciel", "width": 1024, "height": 1024, "steps": 30, "cfg_scale": 6.0,
              "sampler": "euler", "seed": 1, "ref_images": ["/tmp/ref.png"]}
    cmd = generator.build_command(cfg, params, __import__("pathlib").Path("/tmp/out.png"))
    assert "--llm_vision" in cmd and cmd[cmd.index("--llm_vision") + 1].endswith("mm.gguf")
    assert "-r" in cmd and "/tmp/ref.png" in cmd
    assert cmd[cmd.index("--diffusion-model") + 1].endswith("d.gguf")


def test_commande_klein_edition_sans_mmproj(sandbox, fake_engine):
    _prepare("flux2_klein_4b", {"diffusion": "d.gguf", "text_encoder": "te.gguf", "vae": "vae.safetensors"})
    cfg = config.load()
    cfg["family"] = "flux2_klein_4b"
    cfg["selections"]["flux2_klein_4b"] = {"diffusion": "d.gguf", "text_encoder": "te.gguf", "vae": "vae.safetensors", "vision": ""}
    params = {"prompt": "edit", "width": 1024, "height": 1024, "steps": 4, "cfg_scale": 1.0,
              "sampler": "euler", "seed": 2, "ref_images": ["/tmp/ref.png"]}
    cmd = generator.build_command(cfg, params, __import__("pathlib").Path("/tmp/out.png"))
    assert "--llm_vision" not in cmd
    assert "-r" in cmd
    assert cmd[cmd.index("--steps") + 1] == "4"


def test_commande_refuse_si_fichier_manquant(sandbox, fake_engine):
    cfg = config.load()
    cfg["selections"]["flux2_klein_9b"] = {"diffusion": "absent.gguf", "text_encoder": "", "vae": ""}
    cfg["family"] = "flux2_klein_9b"
    with pytest.raises(RuntimeError) as e:
        generator.build_command(cfg, {"prompt": "x", "width": 512, "height": 512}, __import__("pathlib").Path("/tmp/o.png"))
    assert "manquant" in str(e.value)
