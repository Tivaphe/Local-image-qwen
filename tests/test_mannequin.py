"""Mannequin articulé : géométrie (proportions fixes), rendus et API.

Le moteur de pose vit dans le navigateur (``app/static/mannequin.js``) ; le serveur
redessine la pose reçue avec exactement la même géométrie (``app/mannequin.py``).
Ces tests vérifient les deux côtés : la mécanique des longueurs d'os, le cadrage,
les cinq rendus, et les routes ``/api/mannequin/*`` (+ l'envoi d'un rendu comme
image de référence à la génération).
"""

from __future__ import annotations

import json
import math

import cv2

import pytest

from app import config, generator, mannequin


# ------------------------------------------------------------------ géométrie
def _ik2(root, l1, l2, target, pole):
    """IK analytique 2 os — même calcul que ``solveTwoBoneIK`` du navigateur."""
    def sub(a, b): return [a[i] - b[i] for i in range(3)]
    def add(a, b): return [a[i] + b[i] for i in range(3)]
    def mul(a, k): return [a[i] * k for i in range(3)]
    def norm(a):
        n = math.sqrt(sum(v * v for v in a))
        return [v / n for v in a] if n > 1e-9 else [0, 0, 0]
    def dot(a, b): return sum(a[i] * b[i] for i in range(3))
    direction = norm(sub(target, root))
    d = math.dist(root, target)
    d = min(l1 + l2 - 1e-4, max(abs(l1 - l2) + 1e-4, d))
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    u = norm(sub(pole, mul(direction, dot(pole, direction))))
    if sum(v * v for v in u) < 1e-12:
        u = norm(sub([0, 1, 0], mul(direction, dot([0, 1, 0], direction))))
    mid = add(add(root, mul(direction, a)), mul(u, h))
    return mid, add(root, mul(direction, d))


def _plier_bras(pose: dict, cote: str, cible, pole=None) -> dict:
    """Pose le poignet du côté demandé (coude par IK, main dans l'axe de l'avant-bras)."""
    lengths = mannequin.bone_lengths({})
    epaule, poignet = pose[f"shoulder_{cote}"], pose[f"wrist_{cote}"]
    pôle = pole or ([0.35, -0.25, -1.0] if cote == "l" else [-0.35, -0.25, -1.0])
    coude, poignet = _ik2(epaule, lengths[f"elbow_{cote}"], lengths[f"wrist_{cote}"], list(cible), pôle)
    axe = [poignet[i] - coude[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in axe)) or 1.0
    pose[f"elbow_{cote}"] = coude
    pose[f"wrist_{cote}"] = poignet
    pose[f"hand_{cote}"] = [poignet[i] + axe[i] / n * lengths[f"hand_{cote}"] for i in range(3)]
    return pose


def _lengths_of(pose: dict) -> dict:
    out = {}
    for parent, child, *_ in mannequin.BONES:
        a, b = pose[parent], pose[child]
        out[child] = math.dist(a, b)
    return out


def test_pose_par_defaut_respecte_les_longueurs_dos():
    pose = mannequin.default_pose(mannequin.morphed_dimensions("athletique")[0])
    lengths = mannequin.bone_lengths(mannequin.morphed_dimensions("athletique")[0])
    for bone, measured in _lengths_of(pose).items():
        assert measured == pytest.approx(lengths[bone], abs=1e-9)
        assert measured > 0


def test_toutes_les_articulations_du_squelette_sont_definies():
    pose = mannequin.default_pose()
    assert set(pose) == set(mannequin.JOINTS)
    # points OpenPose disponibles (les 18 points du format ControlNet)
    assert set(mannequin.OPENPOSE_18) <= set(pose)
    assert len(mannequin.OPENPOSE_LIMBS) == 17


@pytest.mark.parametrize("morphology,girth", [("fin", 0.84), ("athletique", 1.16), ("fort", 1.34)])
def test_morphologies_reglent_longueurs_et_epaisseurs(morphology, girth):
    lengths, thickness = mannequin.morphed_dimensions(morphology)
    ref_l, ref_t = mannequin.morphed_dimensions("neutre")
    # la stature pilote tout le squelette, la carrure seulement les épaules
    assert lengths["knee_l"] / ref_l["knee_l"] == pytest.approx(
        mannequin.MORPHOLOGIES[morphology]["stature"] * mannequin.MORPHOLOGIES[morphology]["legs"])
    assert lengths["shoulder_l"] / ref_l["shoulder_l"] == pytest.approx(
        mannequin.MORPHOLOGIES[morphology]["shoulders"] * mannequin.MORPHOLOGIES[morphology]["stature"])
    assert thickness["chest"] == pytest.approx(mannequin.BASE_THICK["chest"] * girth)
    # le tableau reste complet : un segment par os + les points du visage
    assert set(lengths) >= set(mannequin.BASE_LENGTHS)
    assert set(thickness) >= set(mannequin.BASE_THICK)
    radii = mannequin.bone_radii(thickness)
    assert len(radii) == len(mannequin.BONES)


def test_tableau_des_dimensions_fixe_la_pose():
    """Les dimensions envoyées par l'interface sont respectées à la lettre, quelle que soit la pose."""
    lengths = {k: v * 1.1 for k, v in mannequin.BASE_LENGTHS.items()}
    lengths.pop("wrist_l")                     # non renseigné : la valeur de référence s'applique
    build, pose, _cam = mannequin.validate({"pose": {}, "lengths": lengths})
    mesure = _lengths_of(pose)
    for bone, valeur in build["lengths"].items():
        if bone in mesure:
            # au micromètre près : les dimensions du tableau ne bougent jamais
            assert mesure[bone] == pytest.approx(valeur, abs=1e-6)
    assert build["lengths"]["wrist_l"] == pytest.approx(mannequin.BASE_LENGTHS["wrist_l"])
    assert build["lengths"]["knee_l"] == pytest.approx(mannequin.BASE_LENGTHS["knee_l"] * 1.1)

    # les épaisseurs pilotent les capsules du rendu
    mince = mannequin.render({"pose": {}, "lengths": mannequin.morphed_dimensions("neutre")[0],
                              "thickness": {k: 0.05 for k in mannequin.BASE_THICK}}, "silhouette", 256, 384)
    epais = mannequin.render({"pose": {}, "lengths": mannequin.morphed_dimensions("neutre")[0],
                              "thickness": {k: 0.20 for k in mannequin.BASE_THICK}}, "silhouette", 256, 384)
    assert (epais > 128).sum() > (mince > 128).sum() * 1.3


def test_taille_du_personnage_suit_les_longueurs():
    petit = mannequin.default_pose(mannequin.morphed_dimensions("femme")[0])
    grand = mannequin.default_pose({k: v * 1.2 for k, v in mannequin.morphed_dimensions("neutre")[0].items()})
    assert (grand["head_top"][1] - grand["ankle_l"][1]) / (petit["head_top"][1] - petit["ankle_l"][1]) > 1.15


def test_trone_suit_les_epaisseurs_reglees():
    """Le tronc est reconstruit à partir des épaisseurs : plus large quand on épaissit poitrine/bassin."""
    def largeur(thickness):
        pose = mannequin.default_pose()
        rings = mannequin._torso_rings(pose, thickness)
        haut = rings[-6][0]
        return max(p[0] for p in haut) - min(p[0] for p in haut)
    ref = dict(mannequin.BASE_THICK)
    gros = dict(mannequin.BASE_THICK)
    gros["spine"] = gros["chest"] = 0.45
    assert largeur(gros) > largeur(ref) * 1.2
    assert largeur(ref) > 0.15


def test_validation_ecarte_les_valeurs_absurdes():
    build, pose, camera = mannequin.validate({"morphology": "fort", "pose": {}})
    assert build["morphology"] == "fort"
    assert build["lengths"]["knee_l"] == pytest.approx(mannequin.morphed_dimensions("fort")[0]["knee_l"])
    assert camera["yaw"] == pytest.approx(0.42)
    with pytest.raises(mannequin.MannequinError) as e:
        mannequin.validate({"pose": {"hips": ["x", 1, 1], "neck": [0, 1, 0]}})
    assert "incomplète" in str(e.value)
    with pytest.raises(mannequin.MannequinError):
        mannequin.validate({"pose": {}, "lengths": {"knee_l": "beaucoup"}})
    with pytest.raises(mannequin.MannequinError):
        mannequin.validate({"pose": {}, "lengths": {"knee_l": 12}})
    with pytest.raises(mannequin.MannequinError):
        mannequin.validate({"pose": {}, "thickness": {"chest": -1}})
    with pytest.raises(mannequin.MannequinError):
        mannequin.render({"pose": {"hips": [0, 1, 0], "neck": [0, 1.4, 0]}}, "flou", 256, 256)


def test_pose_du_navigateur_conservee_telle_quelle():
    """La géométrie envoyée par le navigateur n'est pas retouchée par le serveur."""
    pose = _plier_bras(mannequin.default_pose(), "l", [-0.5, 1.5, 0.4])
    build, out, _ = mannequin.validate({"pose": pose, "build": {}})
    assert out["wrist_l"] == pytest.approx(pose["wrist_l"])
    assert out["elbow_l"] == pytest.approx(pose["elbow_l"])


def test_pose_etiree_refusee_avec_un_message_clair():
    """Une pose bricolée qui étire un membre est refusée (les proportions sont une garantie)."""
    pose = mannequin.default_pose()
    pose["wrist_l"] = [0.9, 1.9, 0.3]                      # main gardée à la hanche → bras étiré
    with pytest.raises(mannequin.MannequinError) as e:
        mannequin.render({"pose": pose}, "volume", 256, 256)
    assert "longueur fixe" in str(e.value) and "hand_l" in str(e.value)

    # reposer le bras proprement (coude par IK + main dans l'axe) rend la pose valide
    propre = _plier_bras(mannequin.default_pose(), "l", [0.55, 1.75, 0.25])
    assert mannequin.check_bone_lengths(propre, {}) == []
    assert mannequin.render({"pose": propre}, "volume", 256, 256).shape == (256, 256, 3)


@pytest.mark.parametrize("cible", [[0.9, 1.6, 0.4], [0.05, 1.9, -0.3], [-0.2, 0.7, 0.5]])
def test_pose_apres_ik_du_navigateur_acceptee(cible):
    """Les poses produites par l'IK 2 os (comme dans le navigateur) sont toujours valides."""
    pose = _plier_bras(mannequin.default_pose(), "l", cible)
    assert mannequin.check_bone_lengths(pose, {}) == []
    assert mannequin.render({"pose": pose}, "volume", 256, 256).shape == (256, 256, 3)


def test_cadrage_automatique_garde_le_corps_entier():
    """Sans caméra fournie, le rendu cadre automatiquement la silhouette."""
    img = mannequin.render({"pose": {}, "build": {}}, "silhouette", 384, 512)
    mask = img.reshape(-1, 3).max(axis=1) > 100
    ys, xs = mask.reshape(512, 384).nonzero()
    assert len(ys) > 400                                     # le corps est bien dessiné
    assert ys.min() > 4 and ys.max() < 508                   # ni collé au bord haut, ni au bas
    assert xs.min() > 4 and xs.max() < 380
    assert (ys.max() - ys.min()) > 200                        # il occupe la hauteur


# --------------------------------------------------------------------- rendus
@pytest.mark.parametrize("mode", mannequin.MODES)
def test_rendus_disponibles(mode):
    img = mannequin.render({"pose": {}, "build": {}}, mode, 320, 416)
    assert img.shape == (416, 320, 3)
    assert img.dtype.name == "uint8"
    assert int(img.max()) > 60                                # quelque chose est dessiné


def test_rendu_squelette_openpose_utilise_les_couleurs_canoniques():
    img = mannequin.render({"pose": {}, "build": {}}, "openpose", 320, 416)
    pixels = img.reshape(-1, 3)
    colored = pixels[(pixels.max(axis=1) > 180) & (pixels.min(axis=1) < 90)]
    assert len(colored) > 100                                 # membres en couleur
    assert int((pixels.sum(axis=1) == 0).sum()) > 320 * 416 * 0.8   # fond noir


def test_rendu_silhouette_est_binaire():
    img = mannequin.render({"pose": {}, "build": {}}, "silhouette", 320, 416)
    pixels = img.reshape(-1, 3)
    assert set(pixels[:, 0].tolist()) == {0, 255}


def test_rendu_profondeur_est_une_carte_de_gris():
    img = mannequin.render({"pose": {}, "build": {}}, "depth", 320, 416)
    pixels = img.reshape(-1, 3)
    assert int(pixels[:, 0].max()) > 120                      # du plus proche (clair)…
    assert int(pixels[:, 0].min()) == 0                       # …au plus lointain (fond noir)
    assert (pixels[:, 0] == pixels[:, 1]).all()               # niveaux de gris


def test_rendu_volume_est_ombre():
    img = mannequin.render({"pose": {}, "build": {}}, "volume", 320, 416)
    body = img.reshape(-1, 3)[img.reshape(-1, 3).min(axis=1) > 120]
    assert len(body) > 500
    assert body.std() > 6                                     # dégradé, pas un aplat


def test_le_rendu_suit_la_pose_envoyee():
    """Les pixels dessinés tombent bien sur les articulations envoyées (pose non figée)."""
    bas = mannequin.default_pose()
    haut = _plier_bras(json.loads(json.dumps(bas)), "l", [0.32, 1.98, 0.10])   # bras levé

    for pose in (bas, haut):
        payload = {"pose": pose, "build": {}}
        img = mannequin.render(payload, "silhouette", 256, 320)
        lit = img[:, :, 0] > 128
        for joint in ("wrist_l", "ankle_r", "nose"):
            x, y = mannequin.openpose_points(payload, 256, 320)[joint]
            zone = lit[max(0, int(y) - 8):int(y) + 8, max(0, int(x) - 8):int(x) + 8]
            assert zone.any(), f"aucun pixel dessiné au niveau de {joint} ({x}, {y})"

    poignet_bas = mannequin.openpose_points({"pose": bas, "build": {}}, 256, 320)["wrist_l"]
    poignet_haut = mannequin.openpose_points({"pose": haut, "build": {}}, 256, 320)["wrist_l"]
    assert poignet_haut[1] < poignet_bas[1]              # le poignet est plus haut à l'image
    img_bas = mannequin.render({"pose": bas, "build": {}}, "silhouette", 256, 320)
    img_haut = mannequin.render({"pose": haut, "build": {}}, "silhouette", 256, 320)
    assert abs(img_bas.astype("int16") - img_haut.astype("int16")).mean() > 1


def test_points_openpose_2d_dans_le_cadre():
    points = mannequin.openpose_points({"pose": {}, "build": {}}, 384, 512)
    assert set(points) == set(mannequin.OPENPOSE_18)
    for x, y in points.values():
        assert 0 <= x <= 384 and 0 <= y <= 512


def test_save_ecrit_un_png_valide(tmp_path):
    dst = tmp_path / "pose.png"
    mannequin.save({"pose": {}, "build": {}}, dst, "openpose", 256, 256)
    assert dst.exists() and dst.stat().st_size > 1000
    from app import pose as pose_module
    width, height = pose_module.image_size(dst)
    assert (width, height) == (256, 256)


# ----------------------------------------------------------------------- API
def test_api_rendu_du_mannequin(client, uploads):
    body = {"pose": {}, "build": {"build": 1.2}, "mode": "openpose", "width": 384, "height": 512}
    r = client.post("/api/mannequin/render", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] and data["kind"] == "pose"
    assert data["control"]["id"].startswith("controls/")
    fichier = uploads / data["control"]["id"]
    assert fichier.exists() and fichier.stat().st_size > 500
    assert (data["control"]["width"], data["control"]["height"]) == (384, 512)


def test_api_rendu_du_mannequin_accepte_une_pose_ajustee(client):
    pose = _plier_bras(mannequin.default_pose(), "r", [-0.45, 1.9, 0.2])
    r = client.post("/api/mannequin/render", json={"pose": pose, "mode": "volume", "width": 256, "height": 256})
    assert r.status_code == 200
    assert r.json()["mode"] == "volume"


def test_api_rendu_respecte_le_tableau_des_dimensions(client, uploads):
    """Les dimensions envoyées par l'interface pilotent le rendu, et les valeurs absurdes sont refusées."""
    def couverture(payload):
        r = client.post("/api/mannequin/render", json=payload)
        assert r.status_code == 200, r.text
        img = cv2.imread(str(uploads / r.json()["control"]["id"]), cv2.IMREAD_GRAYSCALE)
        assert img is not None, r.json()["control"]["id"]
        return float((img > 128).mean())

    base = {"pose": {}, "mode": "silhouette", "width": 320, "height": 448}
    reference = couverture(base)
    grand = couverture({**base, "lengths": {"knee_l": 0.5, "knee_r": 0.5, "ankle_l": 0.46, "ankle_r": 0.46}})
    large = couverture({**base, "thickness": {k: 0.30 for k in mannequin.BASE_THICK}})
    athletique = couverture({**base, "morphology": "athletique"})
    assert grand != pytest.approx(reference, abs=0.002)       # des jambes plus longues changent la silhouette
    assert large > reference * 1.2                            # des capsules épaisses couvrent plus de pixels
    assert athletique != pytest.approx(reference, abs=1e-9)

    r = client.post("/api/mannequin/render", json={**base, "lengths": {"inconnu": 0.3}})
    assert r.status_code == 200                    # un segment inconnu est simplement ignoré
    for mauvais in ({"lengths": {"knee_l": 12}}, {"lengths": {"knee_l": "beaucoup"}},
                    {"thickness": {"chest": -2}}):
        r = client.post("/api/mannequin/render", json={**base, **mauvais})
        assert r.status_code == 400, mauvais
        assert "limites" in r.json()["detail"] or "invalide" in r.json()["detail"]


def test_api_rendu_refuse_un_mode_inconnu(client):
    r = client.post("/api/mannequin/render", json={"pose": {}, "mode": "hologramme", "width": 256, "height": 256})
    assert r.status_code == 400
    assert "hologramme" in r.json()["detail"]


def test_api_pose_renvoie_longueurs_et_points(client):
    r = client.post("/api/mannequin/pose", json={"pose": {}, "width": 384, "height": 512})
    assert r.status_code == 200
    data = r.json()
    assert len(data["lengths"]) == len(mannequin.BONES) + len(mannequin.FACE)   # os + points du visage
    assert set(data["points"]) == set(mannequin.OPENPOSE_18)
    assert set(data["modes"]) == set(mannequin.MODES)


def test_status_expose_les_modes_du_mannequin(client):
    data = client.get("/api/status").json()
    assert data["control"]["mannequin"]["modes"] == list(mannequin.MODES)
    assert data["control"]["mannequin"]["default_size"] == [768, 1024]


def test_page_principale_charge_le_module_mannequin(client):
    html = client.get("/").text
    assert "/static/mannequin.js" in html and "?v=" in html
    assert 'id="mqCanvas"' in html and 'id="btnMqGenerate"' in html


def test_generation_accepte_un_rendu_comme_reference(tmp_path, sandbox, uploads, fake_engine):
    """Le bouton « Générer » du mannequin envoie le rendu comme image de référence."""
    rendu = uploads / "controls" / "mannequin-test.png"
    mannequin.save({"pose": {}, "build": {}}, rendu, "volume", 256, 256)

    famille = "flux2_klein_9b"                      # modèle d'édition : réf. par image
    for categorie in ("diffusion", "text_encoder", "vae", "vision"):
        (sandbox / famille / categorie / f"fichier-{categorie}.safetensors").write_bytes(b"0" * 32)
    config.save({"family": famille,
                 "selections": {famille: {cat: f"fichier-{cat}.safetensors"
                                          for cat in ("diffusion", "text_encoder", "vae", "vision")}}})

    params = {
        "prompt": "Reproduis exactement la pose du personnage de l'image de référence. personne debout",
        "ref_images": [str(rendu)], "mode": "generate", "width": 512, "height": 768,
    }
    cmd = generator.build_command(config.load(), params, tmp_path / "out.png")
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == str(rendu)
    assert "--llm_vision" in cmd
