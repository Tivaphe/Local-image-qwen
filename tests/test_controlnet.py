"""Tests ControlNet : détection automatique des personnages, pose, contours, ligne de commande.

Les fonctions pures de ``app.pose`` (décodage YOLOv8-pose, NMS, rendu du squelette,
Canny) sont testées sans modèle réel ; la détection de bout en bout utilise un faux
réseau ONNX de 1,3 ko (``tests/data/fake_yolov8n-pose.onnx``) qui encode une pose connue.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app import catalog, config, generator
from app.paths import model_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
FAKE_ONNX = DATA / "fake_yolov8n-pose.onnx"

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from app import pose  # noqa: E402  (après importorskip : dépend de numpy/OpenCV)


# ------------------------------------------------------------------ catalogue
def test_famille_sd15_controlnet_complete():
    fam = catalog.FAMILIES["sd15_control"]
    assert fam["supports_controlnet"] is True
    assert fam["bundled"] is True                      # un seul fichier : UNet + CLIP + VAE
    assert fam["uses_init_image"] is True              # img2img pour la modification d'image
    assert set(fam["control_modes"]) == {"pose", "canny"}
    assert fam["required_selections"] == ["diffusion"]

    plan = catalog.install_plan("sd15_control")
    assert [e["category"] for e in plan] == ["diffusion", "controlnet", "pose_detector"]
    assert all(e["url"].startswith("https://huggingface.co/") for e in plan)
    # modèle de diffusion + ControlNet OpenPose + détecteur de personnages
    assert plan[0]["id"].endswith(".safetensors") and plan[0]["size_gb"] > 4
    assert plan[1]["id"] == "control_v11p_sd15_openpose.safetensors"
    assert plan[2]["id"] == "yolov8n-pose.onnx"
    assert catalog.pack_total_gb("sd15_control") == pytest.approx(
        sum(e["size_gb"] for e in plan), abs=0.05)


def test_controlnet_disponible_pour_pose_et_contours():
    modes = {c["id"]: c["mode"] for c in catalog.controlnet_files()}
    assert modes["control_v11p_sd15_openpose.safetensors"] == "pose"
    assert modes["control_v11p_sd15_canny.safetensors"] == "canny"
    assert [p["id"] for p in catalog.pose_models()] == ["yolov8n-pose.onnx"]
    assert catalog.family_for_control("pose") == "sd15_control"
    assert catalog.family_for_control("canny") == "sd15_control"


def test_les_modeles_dit_expliquent_le_refus():
    for fid in ("qwen_image_2.1", "flux2_klein_9b", "flux2_klein_4b"):
        fam = catalog.FAMILIES[fid]
        assert fam["supports_controlnet"] is False
        assert "stable-diffusion.cpp" in fam["control_reason"]
        assert "SD 1.5" in fam["control_reason"]


def test_categorie_dun_fichier_de_controle():
    assert catalog.category_of_file("sd15_control", "yolov8n-pose.onnx") == "pose_detector"
    assert catalog.category_of_file("sd15_control", "control_v11p_sd15_canny.safetensors") == "controlnet"


# ------------------------------------------------- décodage des sorties YOLO
def _fausse_sortie(cx=320.0, cy=320.0, w=128.0, h=256.0, conf=0.92, kpt_conf=0.85):
    """Tableau [1, 4+1+3*17, 8400] identique à la sortie d'un YOLOv8-pose."""
    out = np.zeros((1, 4 + 1 + 3 * 17, 8400), dtype="float32")
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = cx, cy, w, h
    out[0, 4, 0] = conf
    for i in range(17):
        out[0, 5 + 3 * i, 0] = 200.0 + 12 * i
        out[0, 6 + 3 * i, 0] = 150.0 + 9 * i
        out[0, 7 + 3 * i, 0] = kpt_conf
    return out


def test_decodage_pose_et_retour_aux_coordonnees_origine():
    # image 640x480 : letterbox vers 640x640 (échelle 1.0, 80 px de marge en haut/bas)
    persons = pose.decode_yolo_pose(_fausse_sortie(), pad=(0, 80), scale=1.0, size=(640, 480))
    assert len(persons) == 1
    p = persons[0]
    assert p["conf"] == pytest.approx(0.92, abs=1e-3)
    assert p["box"] == [256.0, 112.0, 384.0, 368.0]          # 320-64, 320-80-128, ...
    assert len(p["keypoints"]) == 17
    assert p["keypoints"][0]["x"] == 200.0 and p["keypoints"][0]["y"] == 70.0
    assert p["keypoints"][0]["name"] == "nez"
    assert all(k["visible"] for k in p["keypoints"])
    # la coordonnée est ramenée dans l'image d'origine
    assert all(0 <= k["x"] <= 640 and 0 <= k["y"] <= 480 for k in p["keypoints"])


def test_decodage_ignore_les_personnages_peu_surs():
    assert pose.decode_yolo_pose(_fausse_sortie(conf=0.10)) == []
    # un point articulaire sous le seuil est marqué invisible (mais conservé)
    persons = pose.decode_yolo_pose(_fausse_sortie(kpt_conf=0.05))
    assert persons and persons[0]["keypoints"][0]["visible"] is False


def test_decodage_refuse_un_modele_incompatible():
    with pytest.raises(pose.PoseError) as e:
        pose.decode_yolo_pose(np.zeros((1, 20, 30), dtype="float32"))
    assert "YOLOv8-pose" in str(e.value)


def test_nms_supprime_les_doublons():
    boxes = [(0, 0, 100, 100), (5, 5, 105, 105), (500, 500, 600, 600)]
    keep = pose.nms(boxes, [0.9, 0.8, 0.7])
    assert keep == [0, 2]


# ------------------------------------------------------------------ rendu
def test_rendu_squelette_silhouette_et_contours(tmp_path):
    persons = pose.decode_yolo_pose(_fausse_sortie(), pad=(0, 80), scale=1.0, size=(640, 480))
    skel = pose.render_skeleton(persons, 640, 480)
    assert skel.shape == (480, 640, 3)
    assert int(skel.sum()) > 0                                # quelque chose a été dessiné
    sil = pose.render_skeleton(persons, 640, 480, kind="silhouette")
    assert sil[200, 320].tolist() == [255, 255, 255]          # rectangle plein de la boîte

    # une photo de test, puis les deux images de contrôle qui en dérivent
    photo = tmp_path / "photo.png"
    img = np.full((600, 800, 3), 40, dtype="uint8")
    cv2.rectangle(img, (100, 100), (700, 500), (200, 200, 200), -1)
    cv2.imwrite(str(photo), img)

    edges = pose.canny_control(photo)
    assert edges.shape == (600, 800, 3) and int(edges.sum()) > 0
    out = pose.prepare_control(photo, tmp_path / "ctrl.png", 512, 768, "pose")
    assert cv2.imread(str(out)).shape == (768, 512, 3)        # exactement la taille de génération
    assert pose.image_size(photo) == (800, 600)


def test_image_illisible_message_explicite(tmp_path):
    with pytest.raises(pose.PoseError) as e:
        pose.canny_control(tmp_path / "absent.png")
    assert "illisible" in str(e.value).lower()


# ------------------------------------------------------- détection de bout en bout
def test_faux_modele_pose_detecte_et_dessine(tmp_path, monkeypatch):
    """Chargement ONNX + inférence + décodage + rendu, avec un réseau de synthèse."""
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.full((480, 640, 3), 30, dtype="uint8"))
    detector = pose.PoseDetector(FAKE_ONNX)
    result = detector.detect_file(photo, tmp_path / "skel.png")
    assert result["width"] == 640 and result["height"] == 480
    assert result["count"] == 1
    assert result["backend"].startswith(("OpenCV", "onnxruntime"))
    assert result["persons"][0]["box"] == [256.0, 112.0, 384.0, 368.0]
    assert (tmp_path / "skel.png").exists()


def test_detection_sans_modele_message_explicite():
    with pytest.raises(pose.PoseError) as e:
        pose.PoseDetector(Path("/tmp/modele-absent-yolov8n-pose.onnx")).load()
    assert "téléchargez" in str(e.value).lower()


# --------------------------------------------------------- ligne de commande
def _prepare_control_files(family="sd15_control"):
    """Place un modèle de diffusion, un ControlNet et le détecteur dans le bac à sable."""
    diffusion = catalog.FAMILIES[family]["pack"]["diffusion"]
    (model_dir(family, "diffusion") / diffusion).write_bytes(b"x")
    shutil.copy(FAKE_ONNX, model_dir(family, "pose_detector") / "yolov8n-pose.onnx")
    for c in catalog.controlnet_files():
        (model_dir(family, "controlnet") / c["id"]).write_bytes(b"x")
    config.save({"family": family, "selections": {family: {"diffusion": diffusion}}})
    return config.load()


def test_commande_controlnet_pose_et_img2img(sandbox, fake_engine, tmp_path):
    cfg = _prepare_control_files()
    ctrl = tmp_path / "squelette.png"
    ctrl.write_bytes(b"x")
    init = tmp_path / "photo.png"
    init.write_bytes(b"x")
    params = {
        "prompt": "la même personne, en armure dorée", "width": 512, "height": 768, "steps": 25,
        "cfg_scale": 7.0, "sampler": "euler_a", "seed": 3, "ref_images": [],
        "control_type": "pose", "control_image": str(ctrl), "control_strength": 0.85,
        "control_net": "control_v11p_sd15_openpose.safetensors",
        "init_image": str(init), "strength": 0.45,
    }
    cmd = generator.build_command(cfg, params, tmp_path / "out.png")
    assert cmd[1] == "-m"                                     # modèle « groupé » (fichier unique)
    assert cmd[cmd.index("--control-net") + 1].endswith("control_v11p_sd15_openpose.safetensors")
    assert cmd[cmd.index("--control-image") + 1] == str(ctrl)
    assert cmd[cmd.index("--control-strength") + 1] == "0.85"
    assert cmd[cmd.index("-i") + 1] == str(init)
    assert cmd[cmd.index("--strength") + 1] == "0.45"
    assert "--llm_vision" not in cmd and "-r" not in cmd       # pas d'encodeur de vision ici


def test_commande_contours_canny(sandbox, fake_engine, tmp_path):
    cfg = _prepare_control_files()
    ctrl = tmp_path / "contours.png"
    ctrl.write_bytes(b"x")
    params = {"prompt": "paysage", "width": 512, "height": 512, "control_type": "canny",
              "control_image": str(ctrl), "control_net": "control_v11p_sd15_canny.safetensors"}
    cmd = generator.build_command(cfg, params, tmp_path / "o.png")
    assert cmd[cmd.index("--control-net") + 1].endswith("control_v11p_sd15_canny.safetensors")
    assert cmd[cmd.index("--control-strength") + 1] == "0.90"  # valeur par défaut


def test_commande_refuse_controlnet_sur_modele_dit(sandbox, fake_engine, tmp_path):
    (model_dir("qwen_image_2.1", "diffusion") / "d.gguf").write_bytes(b"x")
    (model_dir("qwen_image_2.1", "text_encoder") / "te.gguf").write_bytes(b"x")
    (model_dir("qwen_image_2.1", "vae") / "vae.safetensors").write_bytes(b"x")
    cfg = config.load()
    cfg["family"] = "qwen_image_2.1"
    cfg["selections"]["qwen_image_2.1"] = {"diffusion": "d.gguf", "text_encoder": "te.gguf", "vae": "vae.safetensors"}
    params = {"prompt": "x", "width": 512, "height": 512, "control_type": "pose",
              "control_image": str(tmp_path / "c.png")}
    with pytest.raises(RuntimeError) as e:
        generator.build_command(cfg, params, tmp_path / "o.png")
    assert "ControlNet" in str(e.value) and "SD 1.5" in str(e.value)


def test_commande_controlnet_sans_image_ni_modele(sandbox, fake_engine, tmp_path):
    cfg = _prepare_control_files()
    params = {"prompt": "x", "width": 512, "height": 512, "control_type": "pose"}
    with pytest.raises(RuntimeError) as e:
        generator.build_command(cfg, params, tmp_path / "o.png")
    assert "Image de contrôle absente" in str(e.value)

    ctrl = tmp_path / "c.png"
    ctrl.write_bytes(b"x")
    (model_dir("sd15_control", "controlnet") / "control_v11p_sd15_openpose.safetensors").unlink()
    params = {"prompt": "x", "width": 512, "height": 512, "control_type": "pose",
              "control_image": str(ctrl), "control_net": "control_v11p_sd15_openpose.safetensors"}
    with pytest.raises(RuntimeError) as e:
        generator.build_command(cfg, params, tmp_path / "o.png")
    assert "Modèle ControlNet manquant" in str(e.value)


def test_commande_controlnet_utilise_la_selection_du_catalogue(sandbox, fake_engine, tmp_path):
    """Sans ``control_net`` explicite, le fichier choisi dans l'onglet Modèles est utilisé."""
    _prepare_control_files()
    config.save({"selections": {"sd15_control": {"controlnet": "control_v11p_sd15_canny.safetensors"}}})
    cfg = config.load()
    ctrl = tmp_path / "c.png"
    ctrl.write_bytes(b"x")
    params = {"prompt": "x", "width": 512, "height": 512, "control_type": "canny", "control_image": str(ctrl)}
    cmd = generator.build_command(cfg, params, tmp_path / "o.png")
    assert cmd[cmd.index("--control-net") + 1].endswith("control_v11p_sd15_canny.safetensors")


def test_type_de_controle_inconnu_refuse(sandbox, fake_engine, tmp_path):
    cfg = _prepare_control_files()
    ctrl = tmp_path / "c.png"
    ctrl.write_bytes(b"x")
    params = {"prompt": "x", "width": 512, "height": 512, "control_type": "depth", "control_image": str(ctrl)}
    with pytest.raises(RuntimeError) as e:
        generator.build_command(cfg, params, tmp_path / "o.png")
    assert "Type de contrôle inconnu" in str(e.value)


# ------------------------------------------------------------- téléchargements
def test_installation_de_la_famille_controlnet(sandbox, file_server, monkeypatch):
    """Le pack SD 1.5 + ControlNet installe aussi le modèle de contrôle et le détecteur."""
    from conftest import wait_job

    from app import downloads

    base, _ = file_server
    monkeypatch.setattr(downloads, "install_plan", lambda *a, **k: [
        {"category": "diffusion", "id": "modele-test.safetensors", "size_gb": 0.001,
         "url": f"{base}/control-test.safetensors"},
        {"category": "controlnet", "id": "control-test.safetensors", "size_gb": 0.001,
         "url": f"{base}/control-test.safetensors"},
        {"category": "pose_detector", "id": "pose-test.onnx", "size_gb": 0.001, "url": f"{base}/pose-test.onnx"},
    ])
    job = wait_job(downloads.start_family_install("sd15_control"))
    assert job["status"] == "done", job["message"]
    assert (model_dir("sd15_control", "controlnet") / "control-test.safetensors").exists()
    assert (model_dir("sd15_control", "pose_detector") / "pose-test.onnx").exists()
    sel = config.load()["selections"]["sd15_control"]
    assert sel["controlnet"] == "control-test.safetensors"
    assert sel["pose_detector"] == "pose-test.onnx"       # utilisé ensuite par la détection


def test_telechargement_dun_seul_fichier_de_controle(sandbox, file_server):
    from conftest import wait_job

    from app import downloads

    base, _ = file_server
    jid = downloads.start_model_download("sd15_control", "pose_detector", "",
                                         custom_url=f"{base}/pose-test.onnx")
    assert wait_job(jid)["status"] == "done"
    assert (model_dir("sd15_control", "pose_detector") / "pose-test.onnx").exists()
    # un fichier absent du catalogue est refusé
    with pytest.raises(ValueError):
        downloads.start_model_download("sd15_control", "controlnet", "inconnu.safetensors")


# ------------------------------------------------------------------ API
def _png(path: Path, width=640, height=480) -> Path:
    cv2.imwrite(str(path), np.full((height, width, 3), 30, dtype="uint8"))
    return path


def test_status_expose_le_bloc_control(client):
    data = client.get("/api/status").json()
    control = data["control"]
    assert set(control["types"]) == {"pose", "canny"}
    assert control["control_family"] == "sd15_control"
    assert control["pose_model_present"] is False
    assert [m["id"] for m in control["pose_models"]] == ["yolov8n-pose.onnx"]
    assert {c["id"] for c in control["controlnets"]} == {
        "control_v11p_sd15_openpose.safetensors", "control_v11p_sd15_canny.safetensors"}
    assert "cv" in control["pose"]
    # état par famille : seule la famille SD 1.5 peut piloter la pose
    assert data["families_status"]["sd15_control"]["control_ready"] is False
    assert "control_ready" in data["families_status"]["qwen_image_2.1"]
    sd15 = next(f for f in data["families"] if f["id"] == "sd15_control")
    assert sd15["supports_controlnet"] is True and sd15["uses_init_image"] is True


def test_detecte_les_contours(client, tmp_path):
    photo = _png(tmp_path / "photo.png")
    r = client.post("/api/control/detect",
                    files={"image": ("photo.png", photo.read_bytes(), "image/png")},
                    data={"kind": "canny", "low": "100", "high": "200"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["kind"] == "canny" and body["control"]["width"] == 640
    assert body["control"]["height"] == 480
    assert body["control"]["url"].startswith("/uploads/controls/")
    assert body["control"]["id"].startswith("controls/")
    assert body["message"] and body["persons"] == []


def test_generation_refuse_controlnet_sur_modele_dit(client, uploads, fake_engine):
    """Le refus est explicite côté API (message prêt à afficher), et cite la famille à utiliser."""
    for cat, name in (("diffusion", "d.gguf"), ("text_encoder", "te.gguf"), ("vae", "v.safetensors")):
        (model_dir("qwen_image_2.1", cat) / name).write_bytes(b"x")
    config.save({"family": "qwen_image_2.1",
                 "selections": {"qwen_image_2.1": {"diffusion": "d.gguf", "text_encoder": "te.gguf",
                                                   "vae": "v.safetensors"}}})
    _png(uploads / "controls" / "contours.png")
    r = client.post("/api/generate", data={
        "prompt": "un paysage", "width": "512", "height": "512", "control_type": "canny",
        "control_id": "controls/contours.png", "control_net": "control_v11p_sd15_canny.safetensors",
    })
    assert r.status_code == 400
    assert "SD 1.5" in r.json()["detail"]


def test_generation_refuse_controlnet_sans_image_de_controle(client, fake_engine):
    config.save({"family": "sd15_control"})
    r = client.post("/api/generate", data={
        "prompt": "un paysage", "control_type": "pose", "width": "512", "height": "512"})
    assert r.status_code == 400
    assert "image de contrôle" in r.json()["detail"].lower()


def test_detecte_la_pose_avec_le_faux_modele(client, sandbox):
    shutil.copy(FAKE_ONNX, model_dir("sd15_control", "pose_detector") / "yolov8n-pose.onnx")
    assert client.get("/api/status").json()["control"]["pose_model_present"] is True
    r = client.post("/api/control/detect",
                    files={"image": ("photo.png", _png(Path("/tmp/liq-photo2.png")).read_bytes(), "image/png")},
                    data={"kind": "pose", "background": "image"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["persons"][0]["box"] == [256.0, 112.0, 384.0, 368.0]
    assert body["control"]["width"] == 640 and body["control"]["height"] == 480
    assert "personnage" in body["message"]


def test_detecte_la_pose_sans_modele_message_actionnable(client):
    r = client.post("/api/control/detect",
                    files={"image": ("photo.png", _png(Path("/tmp/liq-photo3.png")).read_bytes(), "image/png")},
                    data={"kind": "pose"})
    assert r.status_code == 400
    assert "yolov8n-pose.onnx" in r.json()["detail"]


def test_detecte_la_pose_sans_personnage(client, sandbox, monkeypatch):
    """Une image uniforme ne contient personne : message clair plutôt que plantage."""
    shutil.copy(FAKE_ONNX, model_dir("sd15_control", "pose_detector") / "yolov8n-pose.onnx")
    detector = pose.PoseDetector(FAKE_ONNX)
    vidage = {"width": 640, "height": 480, "persons": [], "count": 0, "backend": "test"}
    monkeypatch.setattr(pose.PoseDetector, "detect", lambda self, p: vidage)
    r = client.post("/api/control/detect",
                    files={"image": ("photo.png", _png(Path("/tmp/liq-photo4.png")).read_bytes(), "image/png")},
                    data={"kind": "pose"})
    assert r.status_code == 400
    assert "Aucun personnage détecté" in r.json()["detail"]
    assert detector  # le vrai détecteur reste utilisable


def test_control_pose_redessine_le_squelette_ajuste(client, monkeypatch):
    persons = pose.decode_yolo_pose(_fausse_sortie(), pad=(0, 80), scale=1.0, size=(640, 480))
    persons[0]["keypoints"][9]["x"] += 40          # l'utilisateur déplace un poignet
    persons[0]["keypoints"][9]["moved"] = True
    r = client.post("/api/control/pose", json={"persons": persons, "width": 640, "height": 480, "kind": "pose"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["control"]["url"].startswith("/uploads/controls/")
    assert body["control"]["width"] == 640
    assert body["kind"] == "pose"
    # sans point : erreur explicite
    assert client.post("/api/control/pose", json={"persons": [], "width": 640, "height": 480}).status_code == 400


def test_type_de_controle_inconnu_refuse_par_lapi(client):
    r = client.post("/api/control/detect",
                    files={"image": ("photo.png", _png(Path("/tmp/liq-photo5.png")).read_bytes(), "image/png")},
                    data={"kind": "depth"})
    assert r.status_code == 400 and "inconnu" in r.json()["detail"]


def test_les_images_de_controle_sont_servies_par_le_serveur(sandbox):
    """Les aperçus d'images de contrôle sont bien exposés par le serveur statique."""
    from fastapi.testclient import TestClient

    from app import server

    client = TestClient(server.app)          # montage réel de /uploads
    name = "pytest-controle-temporaire.png"
    target = server.CONTROLS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), np.zeros((8, 8, 3), dtype="uint8"))
    try:
        r = client.get(f"/uploads/controls/{name}")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("image/")
        jpg = server.CONTROLS_DIR / "pytest-controle-temporaire.jpg"
        r2 = client.post("/api/open-folder", data={"which": "models"})
        assert r2.status_code == 200
        assert not jpg.exists()
    finally:
        target.unlink(missing_ok=True)
        assert not target.exists()
