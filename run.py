"""Point d'entrée : python run.py  (ouvre http://127.0.0.1:7860)"""
import argparse
import pathlib
import sys
import threading
import webbrowser

import uvicorn

RACINE = pathlib.Path(__file__).resolve().parent

# Fichiers sans lesquels l'application ne peut pas démarrer.
FICHIERS_REQUIS = (
    "app/__init__.py",
    "app/server.py",
    "app/generator.py",
    "app/mannequin.py",
    "app/static/index.html",
    "app/static/app.js",
    "app/static/mannequin.js",
    "app/static/style.css",
)
# Fichiers ajoutés par les versions récentes : leur absence dégrade une fonctionnalité,
# elle est signalée au démarrage mais ne bloque pas l'application.
FICHIERS_CONSEILLES = (
    ("app/mannequin_mesh.py", "mannequin anatomique (corps réaliste)"),
    ("app/assets/mannequin/maillage.npz", "maillage du mannequin anatomique"),
    ("app/assets/mannequin/morphes.npz", "morphologies (femme, homme, mensurations)"),
    ("app/assets/mannequin/squelette.json", "squelette et poids de peau"),
)


def fichiers_manquants(racine=RACINE, fichiers=FICHIERS_REQUIS):
    """Fichiers absents de la copie du projet (contrôle de démarrage, et tests)."""
    racine = pathlib.Path(racine)
    return [chemin for chemin in fichiers if not (racine / chemin).exists()]


def _comment_reparer():
    print("\n  Comment réparer :")
    print("    - récupérez la dernière version du projet (bouton « Code » → « Download ZIP »")
    print("      sur la branche concernée) puis décompressez-la dans un dossier neuf ;")
    print("    - ou, si vous utilisez git dans ce dossier :  git pull")
    print("    - pensez à l'antivirus : certains suppriment des fichiers .py à la décompression.")
    print("    - le dossier a peut-être été décompressé deux fois de suite : décompressez dans")
    print("      un dossier vide plutôt que par-dessus une ancienne copie.\n")


def verifie_copie(racine=RACINE) -> int:
    """Contrôle la copie du projet ; renvoie 0 si l'application peut démarrer."""
    manquants = fichiers_manquants(racine, FICHIERS_REQUIS)
    if manquants:
        print("\n  Cette copie du projet est incomplète — l'application ne peut pas démarrer.\n")
        print("  Fichiers manquants :")
        for chemin in manquants:
            print(f"    - {chemin}")
        _comment_reparer()
        return 2

    absents = [f"{chemin} ({usage})" for chemin, usage in FICHIERS_CONSEILLES
               if not (pathlib.Path(racine) / chemin).exists()]
    if absents:
        print("\n  Attention : cette copie n'est pas à jour — l'application démarre sans :")
        for ligne in absents:
            print(f"    - {ligne}")
        _comment_reparer()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Local Image Qwen")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour accéder depuis un autre appareil du réseau")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="vérifie la copie du projet puis quitte")
    a = ap.parse_args()

    etat = verifie_copie()
    if a.check or etat != 0:
        return etat

    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    print(f"\n  Local Image Qwen  ->  http://{a.host}:{a.port}\n  (Ctrl+C pour arrêter)\n")
    try:
        uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="warning")
    except ImportError as e:                      # dépendance manquante : message lisible
        print(f"\n  Impossible de démarrer : {e}")
        print("  Installez les dépendances :  pip install -r requirements.txt\n")
        return 1
    except KeyboardInterrupt:                     # arrêt normal (Ctrl+C)
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
