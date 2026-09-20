"""Lance sd-cli pour générer / éditer une image, avec suivi de progression."""
from __future__ import annotations

import json
import os
import random
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from . import backend
from .paths import MODEL_SUBDIRS, OUTPUTS_DIR

SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2m", "dpm++2mv2", "dpm++2s_a", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd"]

_state = {
    "running": False,
    "log": [],
    "progress": 0.0,
    "step": 0,
    "steps": 0,
    "phase": "",
    "result": None,
    "error": None,
    "command": "",
    "started": 0.0,
    "elapsed": 0.0,
}
_proc: subprocess.Popen | None = None
_lock = threading.Lock()

STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*-")            # barre "|====> 12/30 - 1.23s/it"
STEP_RE2 = re.compile(r"(?:step|sampling)[^\d]{0,10}(\d+)\s*/\s*(\d+)", re.I)


def state() -> dict:
    with _lock:
        s = dict(_state)
        s["log"] = s["log"][-200:]
        if s["running"]:
            s["elapsed"] = time.time() - s["started"]
    return s


def _log(line: str):
    with _lock:
        _state["log"].append(line)
        if len(_state["log"]) > 2000:
            del _state["log"][:1000]
        low = line.lower()
        m = STEP_RE.search(line) or STEP_RE2.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            _state["step"], _state["steps"] = cur, tot
            _state["progress"] = cur / tot if tot else 0
            _state["phase"] = "Génération"
        elif "loading" in low and "model" in low:
            _state["phase"] = "Chargement des modèles"
        elif "vae" in low and "decod" in low:
            _state["phase"] = "Décodage VAE"
        elif "computing" in low and ("cond" in low or "text" in low or "llm" in low):
            _state["phase"] = "Encodage du prompt"
        elif "save" in low and "result" in low:
            _state["phase"] = "Enregistrement"


def _model_path(category: str, name: str) -> Path | None:
    if not name:
        return None
    p = MODEL_SUBDIRS[category] / name
    return p if p.exists() else None


def build_command(cfg: dict, params: dict, out_path: Path) -> list[str]:
    exe = backend.find_binary()
    if not exe:
        raise RuntimeError("Moteur non installé : allez dans l'onglet Configuration.")

    diffusion = _model_path("diffusion", cfg.get("diffusion_model", ""))
    te = _model_path("text_encoder", cfg.get("text_encoder", ""))
    vae = _model_path("vae", cfg.get("vae", ""))
    vision = _model_path("vision", cfg.get("vision_encoder", ""))
    missing = [n for n, p in (("modèle de diffusion", diffusion), ("encodeur de texte", te), ("VAE", vae)) if p is None]
    if missing:
        raise RuntimeError("Fichier(s) manquant(s) ou non sélectionné(s) : " + ", ".join(missing))

    ref_images: list[str] = params.get("ref_images") or []
    if ref_images and vision is None:
        raise RuntimeError("L'édition d'image nécessite l'encodeur de vision (mmproj). Téléchargez-le dans Configuration.")

    seed = int(params.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    params["seed"] = seed

    cmd = [
        str(exe),
        "--diffusion-model", str(diffusion),
        "--llm", str(te),
        "--vae", str(vae),
        "-p", params["prompt"],
        "--cfg-scale", str(float(params.get("cfg_scale", 6.0))),
        "--sampling-method", str(params.get("sampler", "euler")),
        "--steps", str(int(params.get("steps", 30))),
        "-W", str(int(params.get("width", 1024))),
        "-H", str(int(params.get("height", 1024))),
        "--seed", str(seed),
        "-o", str(out_path),
        "-v",
    ]
    neg = (params.get("negative_prompt") or "").strip()
    if neg:
        cmd += ["-n", neg]
    if vision is not None and ref_images:
        cmd += ["--llm_vision", str(vision)]
        for r in ref_images:
            cmd += ["-r", r]
    if cfg.get("offload_to_cpu", True):
        cmd.append("--offload-to-cpu")
    if cfg.get("flash_attention", True):
        cmd.append("--diffusion-fa")
    if cfg.get("vae_tiling", False):
        cmd.append("--vae-tiling")
    threads = int(cfg.get("threads", -1) or -1)
    if threads > 0:
        cmd += ["-t", str(threads)]
    lora_dir = MODEL_SUBDIRS["lora"]
    if any(p.suffix.lower() in (".safetensors", ".gguf") for p in lora_dir.glob("*")):
        cmd += ["--lora-model-dir", str(lora_dir)]
    extra = (cfg.get("extra_args") or "").strip()
    if extra:
        cmd += shlex.split(extra, posix=(os.name != "nt"))
    return cmd


def start(cfg: dict, params: dict) -> None:
    global _proc
    with _lock:
        if _state["running"]:
            raise RuntimeError("Une génération est déjà en cours.")
        _state.update(running=True, log=[], progress=0.0, step=0, steps=int(params.get("steps", 30)),
                      phase="Démarrage", result=None, error=None, started=time.time(), elapsed=0.0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"qwen_{ts}.png"
    try:
        cmd = build_command(cfg, params, out_path)
    except Exception as e:
        with _lock:
            _state.update(running=False, error=str(e), phase="Erreur")
        raise

    with _lock:
        _state["command"] = subprocess.list2cmdline(cmd) if os.name == "nt" else " ".join(shlex.quote(c) for c in cmd)

    def run():
        global _proc
        try:
            creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
            _proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, creationflags=creation,
                cwd=str(backend.find_binary().parent),
            )
            assert _proc.stdout
            buf = ""
            while True:
                ch = _proc.stdout.read(1)
                if not ch:
                    break
                if ch in "\r\n":
                    if buf.strip():
                        _log(buf.rstrip())
                    buf = ""
                else:
                    buf += ch
            if buf.strip():
                _log(buf)
            rc = _proc.wait()
            elapsed = time.time() - _state["started"]
            if rc == 0 and out_path.exists():
                meta = {
                    "file": out_path.name, "prompt": params["prompt"], "negative_prompt": params.get("negative_prompt", ""),
                    "seed": params["seed"], "steps": params.get("steps"), "cfg_scale": params.get("cfg_scale"),
                    "sampler": params.get("sampler"), "width": params.get("width"), "height": params.get("height"),
                    "ref_images": [Path(r).name for r in params.get("ref_images") or []],
                    "diffusion_model": cfg.get("diffusion_model"), "elapsed_s": round(elapsed, 1),
                    "date": datetime.now().isoformat(timespec="seconds"),
                }
                out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                with _lock:
                    _state.update(running=False, result=out_path.name, progress=1.0, phase="Terminé", elapsed=elapsed)
            else:
                with _lock:
                    tail = "\n".join(_state["log"][-15:])
                    _state.update(running=False, error=f"sd-cli a quitté avec le code {rc}.\n{tail}", phase="Erreur", elapsed=elapsed)
        except Exception as e:
            with _lock:
                _state.update(running=False, error=str(e), phase="Erreur")
        finally:
            _proc = None

    threading.Thread(target=run, daemon=True).start()


def cancel() -> bool:
    global _proc
    p = _proc
    if p and p.poll() is None:
        p.kill()
        with _lock:
            _state.update(running=False, error="Annulé par l'utilisateur", phase="Annulé")
        return True
    return False


def gallery() -> list[dict]:
    items = []
    for png in sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = {}
        mj = png.with_suffix(".json")
        if mj.exists():
            try:
                meta = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:
                pass
        items.append({"file": png.name, "meta": meta, "mtime": png.stat().st_mtime})
    return items
