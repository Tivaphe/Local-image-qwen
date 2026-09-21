"""Mannequin anatomique : maillage MakeHuman (CC0), morphologies, poses et rendus.

Ces tests couvrent le moteur ``app/mannequin_mesh.py`` (celui qui remplace le
bonhomme dessiné à la main) et les routes ``/api/mannequin/model``,
``/api/mannequin/mesh/pose`` et ``/api/mannequin/mesh/render``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app import mannequin_mesh as MM


# --------------------------------------------------------------------- assets
def test_assets_du_mannequin_presents():
    """Les assets dérivés (CC0) sont livrés avec l'application."""
    for fichier in ("maillage.npz", "morphes.npz", "squelette.json", "LISEZ-MOI.md"):
        assert (MM.ASSETS / fichier).exists(), f"asset manquant : {fichier}"
    donnees = MM._charge()
    assert len(donnees["sommets"]) > 15000
    assert len(donnees["faces"]) > 20000
    assert len(donnees["os"]) == 21
    assert np.isfinite(donnees["sommets"]).all()


def test_maillage_oriente_et_normales_unitaires():
    """Les facettes sont orientées de façon cohérente et les normales sont unitaires."""
    donnees = MM._charge()
    a, b, c = (donnees["sommets"][donnees["faces"][:, i]] for i in range(3))
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())
    assert volume > 0, "le maillage doit être orienté vers l'extérieur"
    utilises = np.unique(donnees["faces"])
    normes = np.linalg.norm(donnees["normales"][utilises], axis=1)
    assert np.allclose(normes, 1.0, atol=1e-4)
    assert (np.linalg.norm(donnees["normales"], axis=1) > 0).sum() == len(utilises)


# --------------------------------------------------------------- morphologies
def test_stature_realiste():
    """Le mannequin mesure une taille humaine, dans toutes les morphologies."""
    for nom in MM.MORPHOLOGIES:
        d = MM.dimensions(MM.MORPHOLOGIES[nom])
        assert 1.45 < d["stature"] < 2.05, f"{nom} : stature {d['stature']:.2f} m"


def test_morphologies_femme_homme_differentes():
    """Femme et homme donnent des silhouettes différentes (hanches, carure, taille)."""
    femme = MM.applique_morphs(MM.MORPHOLOGIES["femme"])
    homme = MM.applique_morphs(MM.MORPHOLOGIES["homme"])
    assert femme.shape == homme.shape == MM._charge()["sommets"].shape

    def largeur(sommets, y, tolerance=0.04):
        tranche = sommets[np.abs(sommets[:, 1] - y) < tolerance]
        return float(np.ptp(tranche[:, 0]))

    y_hanches = 0.10
    assert largeur(femme, y_hanches) != pytest.approx(largeur(homme, y_hanches), rel=0.02)
    assert MM.dimensions(MM.MORPHOLOGIES["homme"])["carure"] > MM.dimensions(MM.MORPHOLOGIES["femme"])["carure"]


def test_morphologie_inconnue_refusee():
    with pytest.raises(MM.MannequinError) as e:
        MM.applique_morphs({"morphologie_imaginaire": 1.0})
    assert "inconnue" in str(e.value)


def test_morphs_fins_modifient_les_mensurations():
    """Les curseurs de mensuration (hanches, taille, cuisses…) agissent vraiment."""
    base = MM.applique_morphs({})
    hanches = MM.applique_morphs({"hanches": 1.0})

    def tour(sommets, y, tolerance=0.05):
        tranche = sommets[np.abs(sommets[:, 1] - y) < tolerance]
        return float(np.ptp(tranche[:, 0]) + np.ptp(tranche[:, 2]))

    y_bassin = 0.10
    assert tour(hanches, y_bassin) > tour(base, y_bassin) * 1.01


# ---------------------------------------------------------------------- poses
def test_articulations_completes():
    """Les 21 articulations sont présentes et rattachées (sauf le bassin, racine)."""
    positions = MM.articulations({})
    assert set(positions) == set(MM.ORDRE)
    for nom in MM.ORDRE:
        assert len(positions[nom]) == 3
        assert all(math.isfinite(v) for v in positions[nom])
    assert set(MM.PARENT) == set(MM.ORDRE) - {"hips"}


def test_os_rigides_quelles_que_soient_les_poses():
    """Un squelette « à angles » ne peut pas étirer les os : vérifions-le."""
    reference = MM.articulations({})

    def longueurs(angles, origine=None):
        pos = MM.articulations(angles, origine=origine)
        return {nom: math.dist(pos[nom], pos[MM.PARENT[nom]]) for nom in MM.ORDRE if nom in MM.PARENT}

    attendues = longueurs({})
    for nom, angles in MM.POSES.items():
        mesure = longueurs(angles, MM.POSES_ORIGINE.get(nom))
        for os, valeur in attendues.items():
            assert mesure[os] == pytest.approx(valeur, abs=1e-5), f"os {os} étiré dans « {nom} »"
    assert math.dist(reference["hips"], reference["head"]) == pytest.approx(
        math.dist(MM.articulations({}, origine=(0, -0.4, 0))["hips"],
                  MM.articulations({}, origine=(0, -0.4, 0))["head"]))


def test_butees_respectees():
    """Au-delà des butées, l'angle est ramené au maximum anatomique."""
    p = MM.articulations({"elbow_l": (500, 0, 0)})
    q = MM.articulations({"elbow_l": (MM.LIMITES["elbow_l"][1], 0, 0)})
    assert p["wrist_l"] == pytest.approx(q["wrist_l"], abs=1e-6)
    assert MM.articulations({"knee_l": (-90, 0, 0)})["ankle_l"] == pytest.approx(
        MM.articulations({"knee_l": (0, 0, 0)})["ankle_l"], abs=1e-6)


def test_articulation_inconnue_refusee():
    with pytest.raises(MM.MannequinError) as e:
        MM.articulations({"museau": (10, 0, 0)})
    assert "inconnue" in str(e.value)


def test_poser_leve_la_jambe_et_plie_le_genou():
    """Contrôle de bon sens : lever la cuisse monte le genou, plier le genou recule le pied."""
    repos = MM.articulations({})
    leve = MM.articulations({"hip_l": (70, 0, 0)})
    assert leve["knee_l"][1] > repos["knee_l"][1] + 0.2

    plie = MM.articulations({"knee_l": (90, 0, 0)})
    assert plie["ankle_l"][2] < repos["ankle_l"][2] - 0.1        # le talon part vers l'arrière


def test_bras_le_long_du_corps_par_defaut():
    """La pose « debout » ramène les bras le long du corps (le maillage est modélisé en T)."""
    repos = MM.articulations({})
    debout = MM.articulations(MM.POSES["debout"])
    assert abs(debout["hand_l"][0]) < abs(repos["hand_l"][0])
    assert math.dist(debout["hand_l"], debout["hips"]) < math.dist(repos["hand_l"], repos["hips"])


def test_les_mains_ne_traversent_pas_le_corps():
    """Anti-collision : dans toutes les poses livrées, les mains restent hors du tronc."""
    for nom, angles in MM.POSES.items():
        pos = MM.articulations(angles, origine=MM.POSES_ORIGINE.get(nom))
        axe = pos["hips"]
        for cote in ("l", "r"):
            main, coude = pos["hand_" + cote], pos["elbow_" + cote]
            rayon = math.dist([main[0], axe[1], main[2]], [axe[0], axe[1], axe[2]])
            assert rayon > 0.09, f"main {cote} dans le tronc pour « {nom} » (rayon {rayon:.2f} m)"
            assert abs(coude[0]) < 0.75, f"coude {cote} aberrant dans « {nom} »"


def test_peau_ponderee_conserve_les_sommets():
    """Le skinning renvoie un maillage de même taille, sans valeur aberrante."""
    sommets = MM.applique_morphs(MM.MORPHOLOGIES["femme"])
    deformes = MM._skinnage(sommets, MM.matrices(MM.POSES["marche"]))
    assert deformes.shape == sommets.shape
    assert np.isfinite(deformes).all()
    assert np.ptp(deformes[:, 1]) == pytest.approx(np.ptp(sommets[:, 1]), rel=0.25)


def test_decalage_du_bassin_descend_le_corps():
    """« S'asseoir » descend tout le corps, sans changer les longueurs d'os."""
    haut = MM.articulations(MM.POSES["assis"])
    bas = MM.articulations(MM.POSES["assis"], origine=MM.POSES_ORIGINE["assis"])
    assert bas["hips"][1] < haut["hips"][1] - 0.3
    assert math.dist(bas["hips"], bas["knee_l"]) == pytest.approx(math.dist(haut["hips"], haut["knee_l"]))


# --------------------------------------------------------------------- rendus
@pytest.mark.parametrize("mode", ["volume", "openpose", "depth", "silhouette"])
def test_rendus_disponibles(mode):
    img = MM.rend({"morphology": "neutre", "pose": MM.POSES["debout"]}, mode, 320, 448)
    assert img.shape == (448, 320, 3)
    assert img.dtype == np.uint8
    if mode == "silhouette":
        assert img.max() == 255 and img.min() == 0
        part = float((img[:, :, 0] > 0).mean())
        assert 0.02 < part < 0.75, f"silhouette occupant {part:.1%} de l'image"
    elif mode == "openpose":
        couleurs = {tuple(v) for v in img.reshape(-1, 3).tolist()}
        assert len(couleurs) > 4, "le squelette OpenPose doit être multicolore"
    else:
        assert img.std() > 3.0


def test_rendu_suit_la_pose():
    """Deux poses différentes donnent deux images différentes."""
    a = MM.rend({"morphology": "neutre", "pose": MM.POSES["debout"]}, "volume", 256, 352)
    b = MM.rend({"morphology": "neutre", "pose": MM.POSES["bras_leves"]}, "volume", 256, 352)
    assert float(np.abs(a.astype("int16") - b.astype("int16")).mean()) > 1.0


def test_rendu_suit_la_morphologie():
    """Changer de morphologie change la silhouette."""
    a = MM.rend({"morphology": "femme"}, "silhouette", 256, 352)
    b = MM.rend({"morphology": "homme"}, "silhouette", 256, 352)
    assert float(np.abs(a.astype("int16") - b.astype("int16")).mean()) > 0.5


def test_mode_de_rendu_inconnu_refuse():
    with pytest.raises(MM.MannequinError) as e:
        MM.rend({}, "aquarelle", 128, 128)
    assert "inconnu" in str(e.value)


# ----------------------------------------------------------------------- API
def test_api_modele_du_mannequin(client):
    d = client.get("/api/mannequin/model").json()
    assert d["ok"] and len(d["articulations"]) == 21
    assert "femme" in d["morphologies"] and "homme" in d["morphologies"]
    assert "debout" in d["poses"] and "assis" in d["poses"]
    limites = {a["nom"]: a["limites"] for a in d["articulations"]}
    assert limites["elbow_l"][1] == MM.LIMITES["elbow_l"][1]
    assert limites["knee_l"][0] == MM.LIMITES["knee_l"][0]


def test_api_pose_du_mannequin_renvoie_les_articulations(client):
    r = client.post("/api/mannequin/mesh/pose", json={"morphology": "femme", "pose": {"hip_l": [45, 0, 0]}})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and len(d["articulations"]) == 21
    assert d["etiquettes"]["hips"] == "bassin"
    haut = client.post("/api/mannequin/mesh/pose", json={"pose": {"hip_l": [45, 0, 0]}}).json()
    assert d["articulations"]["knee_l"][1] > haut["articulations"]["knee_l"][1] - 1e-6


def test_api_rendu_du_mannequin_anatomique(client):
    r = client.post("/api/mannequin/mesh/render",
                    json={"morphology": "homme", "pose": MM.POSES["marche"], "mode": "volume",
                          "width": 384, "height": 512})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["mode"] == "volume"
    assert d["control"]["width"] == 384 and d["control"]["height"] == 512
    import cv2
    from app import server
    image = cv2.imread(str(server.CONTROLS_DIR / d["control"]["id"].split("/")[-1]))
    assert image is not None and image.shape[:2] == (512, 384)


def test_api_rendu_refuse_les_demandes_invalides(client):
    assert client.post("/api/mannequin/mesh/render", json={"morphology": "zorg"}).status_code == 400
    assert client.post("/api/mannequin/mesh/render", json={"pose": {"museau": [10, 0, 0]}}).status_code == 400
    assert client.post("/api/mannequin/mesh/render", json={"mode": "flou"}).status_code == 400
    assert client.post("/api/mannequin/mesh/render",
                       json={"pose": {"elbow_l": ["beaucoup"]}}).status_code == 400


def test_api_status_annonce_le_mannequin_anatomique(client):
    d = client.get("/api/status").json()
    bloc = d["control"]["mannequin_mesh"]
    assert bloc["disponible"] is True
    assert "femme" in bloc["morphologies"] and "marche" in bloc["poses"]


# ------------------------------------------- copie du projet incomplète / dégradée
def test_run_detecte_une_copie_incomplete(tmp_path):
    """Le démarrage explique ce qui manque au lieu d'afficher une trace de pile."""
    import run

    assert run.verifie_copie(tmp_path) == 2                     # rien n'est présent → bloquant
    assert set(run.fichiers_manquants(tmp_path)) == set(run.FICHIERS_REQUIS)

    for chemin in run.FICHIERS_REQUIS:                          # copie complète → démarre
        fichier = tmp_path / chemin
        fichier.parent.mkdir(parents=True, exist_ok=True)
        fichier.write_text("")
    assert run.verifie_copie(tmp_path) == 0

    # il ne manque que le mannequin anatomique : signalé, mais l'application démarre
    assert run.fichiers_manquants(tmp_path, [c for c, _ in run.FICHIERS_CONSEILLES]) != []
    assert run.verifie_copie(tmp_path) == 0


def test_api_sans_le_mannequin_anatomique(client, monkeypatch):
    """Copie incomplète : l'API répond clairement au lieu de casser le serveur."""
    from app import server

    monkeypatch.setattr(server, "mannequin_mesh", None)
    d = client.get("/api/mannequin/model").json()
    assert d["ok"] is True and d["disponible"] is False
    assert "mannequin_mesh" in d["aide"]

    r = client.post("/api/mannequin/mesh/render", json={"morphology": "femme"})
    assert r.status_code == 503 and "mannequin_mesh.py" in r.json()["detail"]
    r = client.post("/api/mannequin/mesh/pose", json={})
    assert r.status_code == 503

    etat = client.get("/api/status").json()["control"]["mannequin_mesh"]
    assert etat["disponible"] is False and etat["aide"]


def test_api_sans_les_assets_du_mannequin(client, monkeypatch, tmp_path):
    """Assets absents : message d'action (comment les reconstruire) plutôt qu'une erreur serveur."""
    d = client.get("/api/status").json()["control"]["mannequin_mesh"]
    assert d["disponible"] is True

    monkeypatch.setattr(MM, "ASSETS", tmp_path / "vide")
    MM._CACHE.clear()                       # sinon les assets déjà chargés restent en mémoire
    etat = client.get("/api/status").json()["control"]["mannequin_mesh"]
    assert etat["disponible"] is False and "assets" in etat["raison"]

    modele = client.get("/api/mannequin/model").json()
    assert modele["disponible"] is False and "build_mannequin_assets" in modele["aide"]

    r = client.post("/api/mannequin/mesh/render", json={"morphology": "femme"})
    assert r.status_code == 400 and "assets du mannequin absents" in r.json()["detail"]
    monkeypatch.undo()
    MM._CACHE.clear()                       # les tests suivants rechargent les vrais assets
