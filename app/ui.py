"""Rendu HTML côté serveur de la page d'accueil.

Les modèles et leurs boutons « Télécharger » sont écrits directement dans le HTML :
la page est utilisable même si le JavaScript est bloqué, si l'API ne répond pas,
ou si l'utilisateur ouvre le fichier index.html directement. Le JavaScript ne fait
qu'hydrater ce contenu (progression, sélection, rafraîchissement).
"""
from __future__ import annotations

import json
from html import escape

from .catalog import CATEGORIES, DEFAULT_TIER, TIERS, TIER_LABELS

CAT_LABELS = {
    "diffusion": "Modèle de diffusion (GGUF)",
    "text_encoder": "Encodeur de texte (GGUF)",
    "vae": "VAE",
    "vision": "Encodeur de vision (mmproj) — édition d'image",
    "lora": "LoRA (optionnel)",
}
CAT_SHORT = {"diffusion": "Diffusion", "text_encoder": "Encodeur texte", "vae": "VAE", "vision": "Vision (édition)"}


def boot_data(payload: dict) -> str:
    """Catalogue + état courant, embarqué dans la page pour un affichage immédiat."""
    return '<script id="bootData" type="application/json">' + json.dumps(payload, ensure_ascii=False) + "</script>"


# ------------------------------------------------------------- cartes « un clic »
def render_bundle_cards(payload: dict) -> str:
    cards = []
    for fam in payload["families"]:
        fid = fam["id"]
        ready = payload["families_status"][fid]["ready"]
        bundles = payload["bundles"][fid]
        cur = bundles[DEFAULT_TIER]
        active = payload["config"]["family"] == fid

        options = "".join(
            '<option value="{t}"{sel}>{label} — {gb} Go</option>'.format(
                t=t, sel=" selected" if t == DEFAULT_TIER else "",
                label=escape(TIER_LABELS[t]), gb=bundles[t]["total_gb"])
            for t in TIERS
        )
        rows = []
        for f in cur["files"]:
            state = '<span class="have">✔ présent</span>' if f["present"] else f"<span>{f['size_gb']} Go</span>"
            rows.append(f"<div><span>{CAT_SHORT[f['category']]} — {escape(f['id'])}</span>{state}</div>")

        if not cur["missing"]:
            label = "⬇ Tout est présent — relancer / autre qualité"
        elif ready:
            label = f"⬇ Ajouter cette qualité ({cur['missing_gb']} Go)"
        else:
            label = f"⬇ Télécharger {fam['name']} ({cur['missing_gb']} Go)"

        st_txt = "✔ installé" if ready else f"{len(cur['missing'])}/{len(cur['files'])} à télécharger"
        st_cls = "ok" if ready else "ko"
        vision = ""
        if fam["edit_requires_vision"]:
            vision = ('<label class="check"><input type="checkbox" class="visSel" checked> '
                      "Inclure l'édition d'image (encodeur de vision)</label>")
        use = ""
        if ready and not active:
            use = '<button class="ghost" data-use="' + fid + '" style="margin-top:6px">✔ Utiliser ce modèle</button>'

        cards.append(
            '<div class="bundle{rc}" data-fam="{fid}">'
            '<div class="bhead"><h4>{name}</h4><span class="st {st_cls}">{st_txt}</span></div>'
            '<div class="bdesc">{desc}</div>'
            '<label>Qualité <select class="tierSel">{options}</select></label>'
            "{vision}"
            '<div class="bfiles">{files}</div>'
            '<button class="primary" data-dl="{fid}">{label}</button>'
            "{use}"
            "</div>".format(
                rc=" ready" if ready else "", fid=fid, name=escape(fam["name"]), st_cls=st_cls,
                st_txt=st_txt, desc=escape(fam["description"]), options=options, vision=vision,
                files="".join(rows), label=escape(label), use=use)
        )
    return "\n".join(cards)


# --------------------------------------------------------- détail fichier à fichier
def render_model_sections(payload: dict) -> str:
    running = {j.get("label") for j in payload["jobs"] if j["status"] == "running"}
    out = []
    for fam in payload["families"]:
        fid = fam["id"]
        ready = payload["families_status"][fid]["ready"]
        sel = payload["config"]["selections"][fid]
        binfo = payload["bundles"][fid][DEFAULT_TIER]
        secs = []

        for cat in CATEGORIES:
            local = payload["models"][fid][cat]
            catalog = fam.get(cat) or []
            if cat == "vision" and not fam["edit_requires_vision"] and not local:
                continue
            rows = []
            for m in local:
                radio = ""
                if cat != "lora":
                    chk = " checked" if sel.get(cat) == m["name"] else ""
                    radio = f'<input type="radio" name="sel-{fid}-{cat}" data-sel="{fid}/{cat}/{escape(m["name"])}"{chk}>'
                rows.append(
                    f'<div class="mrow">{radio}<span class="name">{escape(m["name"])}</span>'
                    f'<span class="small">{m["size_gb"]} Go</span><span class="tag ok">présent</span>'
                    f'<button class="ghost" data-del="{fid}/{cat}/{escape(m["name"])}" title="Supprimer">🗑</button></div>')
            names = {m["name"] for m in local}
            for item in catalog:
                if item["id"] in names:
                    continue
                busy = f"{fid}/{cat}/{item['id']}" in running
                rec = '<span class="tag rec">recommandé</span>' if item["recommended"] else ""
                cls = "primary" if item["recommended"] else ""
                txt = "⏳ en cours" if busy else "⬇ Télécharger"
                dis = " disabled" if busy else ""
                rows.append(
                    f'<div class="mrow"><span class="name">{escape(item["label"])}</span>{rec}'
                    f'<button class="{cls}" data-file="{fid}/{cat}/{escape(item["id"])}"{dis}>{txt}</button></div>')
            ph = "URL directe d'un fichier .gguf/.safetensors à télécharger dans ce dossier"
            rows.append(f'<div class="mrow"><input type="text" placeholder="{ph}" style="margin:0">'
                        f'<button data-url="{fid}/{cat}">⬇</button></div>')
            secs.append(f'<div class="msec"><h4>{CAT_LABELS[cat]}</h4>{"".join(rows)}</div>')

        head = ('<div class="mrow"><span class="name"><b>Modèle complet</b> — '
                f'{len(binfo["files"])} fichiers ({binfo["total_gb"]} Go)</span>'
                f'<button class="primary" data-dl="{fid}">⬇ Télécharger le modèle</button></div>')
        st_txt = "✔ prêt" if ready else "fichiers manquants"
        st_cls = "ok" if ready else "ko"
        out.append(
            f'<details class="fam" data-fam="{fid}" open>'
            f'<summary><span>{escape(fam["name"])} <span class="small">— {escape(fam["description"])}</span></span>'
            f'<span class="st {st_cls}">{st_txt}</span></summary>'
            f'<div class="body">{head}{"".join(secs)}</div></details>')
    return "\n".join(out)


def render_index(template: str, payload: dict) -> str:
    return (template
            .replace("<!--BOOT_DATA-->", boot_data(payload))
            .replace("<!--BUNDLE_CARDS-->", render_bundle_cards(payload))
            .replace("<!--MODEL_SECTIONS-->", render_model_sections(payload)))
