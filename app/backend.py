"""
Gestion du moteur stable-diffusion.cpp (sd-cli) :
  - détection d'un binaire déjà présent (bin/ ou PATH)
  - téléchargement automatique de la dernière release GitHub
    adaptée au système (Windows/Linux/macOS, CUDA/Vulkan/ROCm/CPU)
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .paths import BIN_DIR

GITHUB_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
EXE = "sd-cli.exe" if os.name == "nt" else "sd-cli"


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "local-image-qwen"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def has_amd_rocm() -> bool:
    if os.name == "nt":
        return False
    return Path("/opt/rocm").exists() or shutil.which("rocminfo") is not None


def detect_flavor() -> str:
    """Retourne : cuda | rocm | vulkan | cpu | metal."""
    sysname = platform.system()
    if sysname == "Darwin":
        return "metal"
    if has_nvidia():
        return "cuda"
    if has_amd_rocm():
        return "rocm"
    return "vulkan"


def available_flavors() -> list[str]:
    sysname = platform.system()
    if sysname == "Darwin":
        return ["metal"]
    if sysname == "Windows":
        return ["cuda", "vulkan", "rocm", "cpu"]
    return ["vulkan", "rocm", "cpu"]  # Linux : pas de build CUDA précompilé


def _pick_assets(assets: list[dict], flavor: str) -> list[dict]:
    """Choisit le/les zip(s) à télécharger pour ce système et ce flavor."""
    sysname = platform.system()
    names = {a["name"]: a for a in assets}
    chosen: list[dict] = []

    def find(*needles: str, exclude: tuple[str, ...] = ()) -> dict | None:
        for n, a in names.items():
            if all(x in n for x in needles) and not any(x in n for x in exclude):
                return a
        return None

    if sysname == "Windows":
        if flavor == "cuda":
            a = find("bin-win-cuda12")
            rt = find("cudart", "win-cu12")
            if a:
                chosen.append(a)
            if rt:
                chosen.append(rt)  # DLL runtime CUDA (pas besoin d'installer le toolkit)
        elif flavor == "rocm":
            a = find("bin-win-rocm")
            if a:
                chosen.append(a)
        elif flavor == "vulkan":
            a = find("bin-win-vulkan")
            if a:
                chosen.append(a)
        else:
            a = find("bin-win-cpu")
            if a:
                chosen.append(a)
    elif sysname == "Darwin":
        a = find("Darwin", "arm64")
        if a:
            chosen.append(a)
    else:  # Linux
        if flavor == "rocm":
            a = find("Linux", "rocm")
        elif flavor == "vulkan":
            a = find("Linux", "vulkan")
        else:
            a = find("Linux", "x86_64", exclude=("rocm", "vulkan"))
        if a:
            chosen.append(a)
    return chosen


def find_binary() -> Path | None:
    """Cherche sd-cli dans bin/ (récursif) puis dans le PATH."""
    if BIN_DIR.exists():
        for p in BIN_DIR.rglob(EXE):
            return p
        # anciens noms
        for alt in ("sd.exe", "sd"):
            for p in BIN_DIR.rglob(alt):
                if p.is_file():
                    return p
    w = shutil.which("sd-cli") or shutil.which("sd")
    return Path(w) if w else None


def binary_version(path: Path) -> str:
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr).strip().splitlines()[0][:200]
    except Exception as e:  # pragma: no cover
        return f"inconnu ({e})"


def install(flavor: str | None = None, progress=None) -> Path:
    """Télécharge et extrait la dernière release dans bin/. Retourne le chemin de sd-cli."""
    flavor = flavor or detect_flavor()
    rel = _http_json(GITHUB_API)
    assets = _pick_assets(rel.get("assets", []), flavor)
    if not assets:
        raise RuntimeError(f"Aucun binaire précompilé trouvé pour {platform.system()} / {flavor}. "
                           "Compilez stable-diffusion.cpp et placez sd-cli dans le dossier bin/.")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    # nettoyer l'ancienne installation
    for child in BIN_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

    for a in assets:
        url, total = a["browser_download_url"], a.get("size", 0)
        if progress:
            progress(f"Téléchargement {a['name']}", 0, total)
        buf = io.BytesIO()
        req = urllib.request.Request(url, headers={"User-Agent": "local-image-qwen"})
        with urllib.request.urlopen(req, timeout=60) as r:
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                done += len(chunk)
                if progress:
                    progress(f"Téléchargement {a['name']}", done, total)
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall(BIN_DIR)

    exe = find_binary()
    if exe is None:
        raise RuntimeError("Archive extraite mais sd-cli introuvable dans bin/.")
    if os.name != "nt":
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (BIN_DIR / "VERSION.txt").write_text(f"{rel.get('tag_name','')}\n{flavor}\n", encoding="utf-8")
    return exe


def installed_info() -> dict:
    exe = find_binary()
    info = {"installed": exe is not None, "path": str(exe) if exe else "", "tag": "", "flavor": ""}
    vf = BIN_DIR / "VERSION.txt"
    if vf.exists():
        lines = vf.read_text(encoding="utf-8").splitlines()
        info["tag"] = lines[0] if lines else ""
        info["flavor"] = lines[1] if len(lines) > 1 else ""
    info["detected_flavor"] = detect_flavor()
    info["available_flavors"] = available_flavors()
    info["system"] = f"{platform.system()} {platform.machine()} / Python {sys.version.split()[0]}"
    return info
