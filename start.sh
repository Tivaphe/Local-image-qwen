#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[1/2] Création de l'environnement Python..."
  python3 -m venv .venv
fi
echo "[2/2] Installation / vérification des dépendances..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
exec .venv/bin/python run.py "$@"
