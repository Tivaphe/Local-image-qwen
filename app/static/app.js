const $ = (s) => document.querySelector(s);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
};
const fmtGb = (b) => (b / 1e9).toFixed(2) + " Go";
const fmtTime = (s) => (s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min ${(s % 60).toFixed(0)} s`);
// Version de l'interface (horodatage de app.js ajouté par le serveur) : permet de
// vérifier qu'on n'exécute pas un vieux fichier resté dans le cache du navigateur.
const APP_VERSION = (() => {
  try {
    const src = (document.currentScript && document.currentScript.src) || "";
    const m = src.match(/v=(\d+)/);
    return m ? new Date(Number(m[1]) * 1000).toLocaleString() : "dev";
  } catch (e) { return "dev"; }
})();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let STATUS = null;
let MODELS_SIG = "";
const CAT_LABELS = {
  diffusion: "Modèle de diffusion (GGUF)",
  text_encoder: "Encodeur de texte (GGUF)",
  vae: "VAE",
  vision: "Encodeur de vision (mmproj) — édition d'image",
  controlnet: "Modèles ControlNet (pose, contours)",
  pose_detector: "Détecteur de personnages (pose automatique)",
  lora: "LoRA (optionnel)",
};
const CAT_ORDER = ["diffusion", "text_encoder", "vae", "vision", "controlnet", "pose_detector", "lora"];
const SELECTABLE = ["diffusion", "text_encoder", "vae", "vision"];
const postJSON = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const famById = (id) => (STATUS && STATUS.families ? STATUS.families.find((f) => f.id === id) : null);

// Choix de quantification mémorisés (survivent au rechargement de la page)
const CHOICES = (() => {
  try { return JSON.parse(localStorage.getItem("liq.choices") || "{}") || {}; } catch (e) { return {}; }
})();
function choice(famId) {
  if (!CHOICES[famId]) CHOICES[famId] = {};
  return CHOICES[famId];
}
function saveChoices() { try { localStorage.setItem("liq.choices", JSON.stringify(CHOICES)); } catch (e) {} }

function renderError(msg) {
  const box = $("#renderError");
  if (!box) return;
  if (!msg) { box.classList.add("hidden"); return; }
  box.innerHTML = `⚠️ <b>Erreur d'affichage de l'interface.</b> Rechargez la page (Ctrl+F5).<br><code>${esc(msg)}</code>`;
  box.classList.remove("hidden");
}

// ------------------------------------------------------------- onglets
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "gallery") loadGallery();
  if (name === "models") refreshStatus();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-goto]");
  if (a) { e.preventDefault(); showTab(a.dataset.goto); }
});

// ------------------------------------------------------------- status
async function refreshStatus() {
  try {
    STATUS = await api("/api/status");
  } catch (e) {
    renderError("Impossible de lire /api/status : " + e.message);
    return;
  }
  try {
    renderAll();
    renderError("");
  } catch (e) {
    renderError(e && e.stack ? e.stack : String(e));
    console.error(e);
  }
}

function renderAll() {
  const c = STATUS.config;
  $("#notReady").classList.toggle("hidden", STATUS.ready);
  $("#modelsBadge").classList.toggle("hidden", STATUS.ready);
  $("#btnGenerate").disabled = !STATUS.ready || STATUS.generation.running;
  $("#editStatus").textContent = STATUS.edit_ready ? "disponible ✔" : "encodeur de vision manquant";
  $("#editFix").classList.toggle("hidden", STATUS.edit_ready);

  // sélecteur de famille
  const fs = $("#family");
  if (!fs.options.length) STATUS.families.forEach((f) => fs.add(new Option(f.name, f.id)));
  [...fs.options].forEach((o) => {
    const st = STATUS.families_status[o.value];
    const f = famById(o.value);
    if (f) o.text = f.name + (st && st.ready ? "" : "  (non installé)");
  });
  fs.value = c.family;
  const fam = famById(c.family) || STATUS.families[0];
  $("#familyDesc").textContent = fam ? fam.description : "";
  $("#editInfo").textContent = fam ? fam.edit_info || "" : "";

  // sampler
  const sel = $("#sampler");
  if (!sel.options.length) STATUS.samplers.forEach((s) => sel.add(new Option(s, s)));
  if (!window.__formInit) {
    sel.value = c.sampler || "euler";
    $("#width").value = c.width; $("#height").value = c.height;
    $("#steps").value = c.steps; $("#stepsVal").textContent = c.steps;
    $("#cfg").value = c.cfg_scale; $("#cfgVal").textContent = Number(c.cfg_scale).toFixed(1);
    $("#negative").value = c.negative_prompt || "";
    syncPreset();
    window.__formInit = true;
  }
  // perf
  $("#offload").checked = !!c.offload_to_cpu;
  $("#fa").checked = !!c.flash_attention;
  $("#vaetiling").checked = !!c.vae_tiling;
  $("#threads").value = c.threads;
  $("#extra").value = c.extra_args || "";

  renderEngine();
  renderModels();
  renderJobs($("#jobs"), STATUS.jobs);
  renderJobs($("#modelJobs"), STATUS.jobs);
  renderActiveFiles();
  renderPaths();
  renderGeneration(STATUS.generation);
  renderControl();
  bindPoseEditor();

// Les tests d'interface (tests/ui_render.mjs, jsdom) ont besoin de lire l'état du
// contrôle : les déclarations `const` ne sont pas visibles depuis un autre script.
// (fusion et non remplacement : le module du mannequin expose aussi son état ici)
window.__liq = Object.assign(window.__liq || {}, { control: CONTROL, applyPoseEdits, drawPoseEditor });
}

function renderEngine() {
  const e = STATUS.engine;
  $("#engineInfo").innerHTML = e.installed
    ? `✅ Installé : <code>${esc(e.path)}</code> ${e.tag ? `(${esc(e.tag)}, ${esc(e.flavor)})` : ""}<br>${esc(e.system)}`
    : `❌ Non installé — ${esc(e.system)} — GPU détecté : <b>${esc(e.detected_flavor)}</b>`;
  const f = $("#flavor");
  if (!f.options.length) {
    e.available_flavors.forEach((x) => f.add(new Option(x + (x === e.detected_flavor ? " (détecté)" : ""), x)));
    f.value = e.available_flavors.includes(e.detected_flavor) ? e.detected_flavor : e.available_flavors[0];
  }
}

// ------------------------------------------------------------- modèles
/** Fichier retenu pour une catégorie : choix de l'utilisateur, fichier déjà
 *  sélectionné/installé, puis fichier recommandé du pack. */
function currentPick(fam, cat, ch) {
  const catalog = fam[cat] || [];
  const local = (STATUS.models[fam.id][cat] || []).map((m) => m.name);
  const candidates = [ch[cat], ((STATUS.config.selections || {})[fam.id] || {})[cat], (fam.pack || {})[cat]];
  for (const c of candidates) {
    if (c && (catalog.some((x) => x.id === c) || local.includes(c))) return c;
  }
  return ((catalog.find((x) => x.recommended) || catalog[0] || {}).id) || local[0] || "";
}

/** Options du menu déroulant : catalogue + fichiers locaux hors catalogue (fine-tunes…). */
function optionsFor(fam, cat) {
  const catalog = (fam[cat] || []).map((x) => ({ id: x.id, label: x.label, recommended: x.recommended }));
  const known = new Set(catalog.map((x) => x.id));
  for (const m of STATUS.models[fam.id][cat] || []) {
    if (!known.has(m.name)) catalog.push({ id: m.name, label: `${m.name} (${m.size_gb} Go, local)`, recommended: false });
  }
  return catalog;
}

/** Fichiers du pack (génération + édition) et état de présence de chacun. */
function packPlan(fam) {
  const ch = choice(fam.id);
  let cats = (fam.required_selections || ["diffusion", "text_encoder", "vae"]).slice();
  if (fam.edit_requires_vision && !ch.visionOff) cats = cats.concat(["vision"]);
  // familles ControlNet : le pack comprend le modèle de contrôle et le détecteur de personnages
  if (fam.supports_controlnet) cats = cats.concat(["controlnet", "pose_detector"]);
  const out = [];
  for (const cat of cats) {
    const options = optionsFor(fam, cat);
    const id = currentPick(fam, cat, ch);
    const item = options.find((x) => x.id === id);
    if (!item) continue;
    const present = (STATUS.models[fam.id][cat] || []).some((m) => m.name === item.id);
    const inCatalog = (fam[cat] || []).some((x) => x.id === item.id);
    out.push({ category: cat, id: item.id, label: item.label, size_gb: inCatalog ? (fam[cat].find((x) => x.id === item.id).size_gb) : 0, present, missing: !present });
  }
  return out;
}

function isFamilyBusy(famId) {
  return STATUS.jobs.some((j) => j.status === "running" && j.family === famId);
}

function installButtonLabel(fam, busy) {
  if (busy) return "⏳ Téléchargement en cours…";
  const missing = packPlan(fam).filter((p) => p.missing);
  if (!missing.length) return "⟳ Tout est installé — compléter si besoin";
  const gb = missing.reduce((s, p) => s + p.size_gb, 0);
  return `⬇ Tout télécharger (~${gb.toFixed(1)} Go)`;
}

function renderModels() {
  const root = $("#modelSections");
  if (!root) return;
  $("#diskFree").textContent = `— ${STATUS.disk_free_gb} Go libres sur le disque`;

  const sig = JSON.stringify([
    STATUS.models, STATUS.config.selections, STATUS.families_status, STATUS.config.family,
    STATUS.jobs.map((j) => [j.id, j.status, j.done, j.total, (j.files || []).map((f) => [f.label, f.status, f.done])]),
    CHOICES, STATUS.disk_free_gb,
  ]);
  if (sig === MODELS_SIG && root.children.length) return;
  MODELS_SIG = sig;
  root.innerHTML = "";

  const anyInstalled = Object.values(STATUS.families_status).some((s) => s.file_count > 0);
  const notice = $("#modelsNotice");
  if (!STATUS.engine.installed) {
    notice.innerHTML = `⬇️ Première utilisation : installez d'abord le <b>moteur</b> (bloc 1 ci‑dessous), puis cliquez sur « Tout télécharger » pour un modèle.`;
    notice.classList.remove("hidden");
  } else if (!anyInstalled) {
    notice.innerHTML = `👉 Aucun modèle installé. Choisissez une quantification puis cliquez sur <b>Tout télécharger</b> : les fichiers de <b>génération</b> et d'<b>édition d'image</b> sont récupérés automatiquement.`;
    notice.classList.remove("hidden");
  } else {
    notice.classList.add("hidden");
  }

  for (const fam of STATUS.families) {
    const st = STATUS.families_status[fam.id] || {};
    const ch = choice(fam.id);
    const busy = isFamilyBusy(fam.id);
    const isOpen = ch.open === undefined ? (fam.id === STATUS.config.family || !anyInstalled) : !!ch.open;

    const card = document.createElement("div");
    card.className = "fam";
    card.dataset.fam = fam.id;
    const isControl = !!fam.supports_controlnet;
    const pill = st.ready && st.edit_ready ? "ok" : st.ready ? "part" : st.file_count ? "part" : "ko";
    let pillTxt = st.ready && st.edit_ready ? "✔ prêt (génération + édition)"
      : st.ready && !st.edit_ready ? "✔ génération · édition : mmproj manquant"
        : st.file_count ? "⚠ incomplet" : "❌ non installé";
    if (isControl) {
      pillTxt = st.ready
        ? (st.control_ready ? "✔ prêt (génération + pose ControlNet)" : "✔ génération · pose : modèle ControlNet manquant")
        : st.file_count ? "⚠ incomplet" : "❌ non installé";
    }

    const diffusionCat = optionsFor(fam, "diffusion");
    const teCat = optionsFor(fam, "text_encoder");
    const visionCat = fam.vision || [];
    const plan = packPlan(fam);
    const options = (list, cat) => list.map((x) =>
      `<option value="${esc(x.id)}" ${x.id === currentPick(fam, cat, ch) ? "selected" : ""}>${esc(x.label)}</option>`).join("");

    card.innerHTML = `
      <div class="fam-head">
        <div>
          <div class="fam-title">${esc(fam.name)} <span class="st ${pill}">${pillTxt}</span></div>
          <div class="small">${esc(fam.description)}</div>
          <div class="small">Occupation disque : ${st.installed_gb || 0} Go · pack complet ≈ ${fam.pack_total_gb} Go${st.file_count ? ` · ${st.file_count} fichier(s)` : ""}</div>
        </div>
        <button class="ghost fam-toggle">${isOpen ? "▾ réduire" : "▸ détails"}</button>
      </div>
      <div class="fam-body" ${isOpen ? "" : "hidden"}>
        <ul class="plan">
          ${plan.map((p) => `<li>${p.present ? "✅" : "⬇️"} <b>${esc(CAT_LABELS[p.category] || p.category)}</b> — ${esc(p.label)}
            ${p.present ? '<span class="tag ok">présent</span>' : '<span class="tag warn">à télécharger</span>'}</li>`).join("")
          || '<li class="small">Aucun fichier au catalogue pour ce modèle.</li>'}
        </ul>
        <div class="grid2">
          <label>Quantification du modèle de diffusion
            <select class="q-diffusion">${options(diffusionCat, "diffusion")}</select>
          </label>
          <label>Encodeur de texte
            <select class="q-te">${options(teCat, "text_encoder")}</select>
          </label>
        </div>
        ${visionCat.length ? `<label class="check"><input type="checkbox" class="q-vision" ${ch.visionOff ? "" : "checked"}>
            Télécharger aussi l'encodeur de vision (mmproj) — indispensable à l'édition d'image</label>` : ""}
        <p class="hint">${esc(fam.edit_info || "")}</p>
        <div class="row wrap">
          <button class="primary btn-install" ${busy ? "disabled" : ""}>${esc(installButtonLabel(fam, busy))}</button>
          <span class="small">${esc((fam.pack && fam.pack.hint) || "")}</span>
        </div>
        <details class="files">
          <summary>Fichiers installés, autres variantes et URL directe</summary>
          <div class="body"></div>
        </details>
      </div>`;

    card.querySelector(".fam-toggle").onclick = () => {
      const willOpen = card.querySelector(".fam-body").hidden;
      card.querySelector(".fam-body").hidden = !willOpen;
      card.querySelector(".fam-toggle").textContent = willOpen ? "▾ réduire" : "▸ détails";
      ch.open = willOpen; saveChoices();
    };
    const selDiff = card.querySelector(".q-diffusion");
    if (selDiff) selDiff.onchange = () => { ch.diffusion = selDiff.value; saveChoices(); refreshStatus(); };
    const selTe = card.querySelector(".q-te");
    if (selTe) selTe.onchange = () => { ch.text_encoder = selTe.value; saveChoices(); refreshStatus(); };
    const cbVision = card.querySelector(".q-vision");
    if (cbVision) cbVision.onchange = () => { ch.visionOff = !cbVision.checked; saveChoices(); refreshStatus(); };

    card.querySelector(".btn-install").onclick = async () => {
      const btn = card.querySelector(".btn-install");
      btn.disabled = true; btn.textContent = "⏳ démarrage…";
      const diffSel = card.querySelector(".q-diffusion");
      const teSelCard = card.querySelector(".q-te");
      const visionBox = card.querySelector(".q-vision");
      try {
        await postJSON("/api/install", {
          family: fam.id,
          diffusion: (diffSel && diffSel.value) || ch.diffusion || "",
          text_encoder: (teSelCard && teSelCard.value) || ch.text_encoder || "",
          include_vision: visionBox ? visionBox.checked : !ch.visionOff,
        });
      } catch (e) {
        alert("Téléchargement impossible : " + e.message);
      }
      refreshStatus();
    };

    // liste des fichiers : locaux (sélection / suppression), catalogue, URL directe
    const body = card.querySelector(".files .body");
    for (const cat of CAT_ORDER) {
      if (cat === "vision" && !fam.edit_requires_vision && !(STATUS.models[fam.id].vision || []).length) continue;
      const local = STATUS.models[fam.id][cat] || [];
      const catalog = fam[cat] || [];
      if (!local.length && !catalog.length) continue;
      const localNames = new Set(local.map((m) => m.name));
      const sec = document.createElement("div");
      sec.className = "msec";
      sec.innerHTML = `<h4>${esc(CAT_LABELS[cat] || cat)}</h4>`;
      for (const m of local) {
        const row = document.createElement("div");
        row.className = "mrow";
        row.innerHTML = `${SELECTABLE.includes(cat) ? `<input type="radio" name="sel-${fam.id}-${cat}" ${STATUS.config.selections[fam.id][cat] === m.name ? "checked" : ""}>` : ""}
          <span class="name">${esc(m.name)}</span><span class="small">${m.size_gb} Go</span>
          <span class="tag ok">présent</span><button class="ghost" title="Supprimer">🗑</button>`;
        if (SELECTABLE.includes(cat)) {
          row.querySelector("input").onchange = () => postJSON("/api/config", { selections: { [fam.id]: { [cat]: m.name } } })
            .then(refreshStatus)
            .catch((e) => renderError(e.message));
        }
        row.querySelector("button").onclick = async () => {
          if (!confirm(`Supprimer ${m.name} ?`)) return;
          try { await api(`/api/models/${fam.id}/${cat}/${encodeURIComponent(m.name)}`, { method: "DELETE" }); }
          catch (e) { renderError(e.message); }
          refreshStatus();
        };
        sec.appendChild(row);
      }
      for (const item of catalog) {
        if (localNames.has(item.id)) continue;
        const row = document.createElement("div");
        row.className = "mrow";
        row.innerHTML = `<span class="name">${esc(item.label)}</span>${item.recommended ? '<span class="tag rec">recommandé</span>' : ""}
          <button class="${item.recommended ? "primary" : ""}">⬇ Télécharger</button>`;
        row.querySelector("button").onclick = async (ev) => {
          ev.target.disabled = true; ev.target.textContent = "⏳…";
          try { await postJSON("/api/download", { family: fam.id, category: cat, file_id: item.id }); }
          catch (e) { alert(e.message); }
          refreshStatus();
        };
        sec.appendChild(row);
      }
      const custom = document.createElement("div");
      custom.className = "mrow";
      custom.innerHTML = `<input type="text" placeholder="URL directe d'un fichier .gguf/.safetensors à télécharger dans ce dossier"><button>⬇</button>`;
      custom.querySelector("button").onclick = async () => {
        const url = custom.querySelector("input").value.trim();
        if (!url) return;
        try { await postJSON("/api/download", { family: fam.id, category: cat, file_id: "", url }); }
        catch (e) { alert(e.message); }
        refreshStatus();
      };
      sec.appendChild(custom);
      body.appendChild(sec);
    }

    root.appendChild(card);
  }
}

// ------------------------------------------------------------- tâches
function shortStatus(s) {
  return s === "done" ? "✅" : s === "error" ? "❌" : s === "cancelled" ? "⛔" : s === "running" ? "⏳" : "•";
}

function renderJobs(root, jobs) {
  if (!root) return;
  if (!jobs.length) { root.innerHTML = '<div class="small">Aucun téléchargement.</div>'; return; }
  root.innerHTML = jobs.map((j) => {
    const pct = j.total ? Math.min(100, (100 * j.done) / j.total) : (j.status === "done" ? 100 : 0);
    const files = (j.files || []).map((f) => {
      const fp = f.total ? Math.min(100, (100 * f.done) / f.total) : (f.status === "done" ? 100 : 0);
      return `<div class="job-file">${shortStatus(f.status)} <span class="name">${esc(f.label)}</span>
        <span class="small">${f.status === "running" && f.total ? `${fmtGb(f.done)} / ${fmtGb(f.total)} (${fp.toFixed(0)} %)` : esc(f.message || (f.size_gb ? f.size_gb + " Go" : ""))}</span>
        ${f.status === "running" && f.total ? `<div class="bar"><div style="width:${fp}%"></div></div>` : ""}</div>`;
    }).join("");
    return `<div class="job">
      <div class="row"><b>${shortStatus(j.status)} ${esc(j.label)}</b>
        <span class="small">${j.total ? `${fmtGb(j.done)} / ${fmtGb(j.total)} (${pct.toFixed(0)} %)` : esc(j.message || "")}
        ${j.status === "running" ? `<button class="ghost btn-cancel-job" data-job="${j.id}">■ annuler</button>` : ""}</span></div>
      ${j.status === "running" && !(j.files || []).length ? `<div class="bar"><div style="width:${pct}%"></div></div>` : ""}
      ${j.status === "error" ? `<div class="small err">${esc(j.message)}</div>` : ""}
      ${files}
    </div>`;
  }).join("");
  root.querySelectorAll(".btn-cancel-job").forEach((b) => {
    b.onclick = () => api(`/api/jobs/${b.dataset.job}/cancel`, { method: "POST" }).then(refreshStatus);
  });
}

function renderActiveFiles() {
  const root = $("#activeFiles");
  if (!root || !STATUS) return;
  const fam = STATUS.config.family;
  const sel = STATUS.config.selections[fam] || {};
  root.innerHTML = Object.entries(sel).map(([cat, name]) =>
    `${name ? "✅" : "❌"} <b>${esc(CAT_LABELS[cat] || cat)}</b> : ${name ? `<code>${esc(name)}</code>` : "<i>aucun fichier sélectionné</i>"}`
  ).join("<br>");
}

function renderPaths() {
  const root = $("#paths");
  if (!root) return;
  root.innerHTML = `Modèles : <code>${esc(STATUS.paths.models)}</code><br>Images : <code>${esc(STATUS.paths.outputs)}</code>
    <br>Interface chargée le ${esc(APP_VERSION)} — si une nouveauté manque, rechargez avec <b>Ctrl+F5</b>.`;
}

// changement de modèle : applique les réglages par défaut de la famille
$("#family").onchange = async (e) => {
  const fam = famById(e.target.value);
  if (!fam) return;
  await postJSON("/api/config", { family: fam.id });
  const d = fam.defaults;
  $("#steps").value = d.steps; $("#stepsVal").textContent = d.steps;
  $("#cfg").value = d.cfg_scale; $("#cfgVal").textContent = Number(d.cfg_scale).toFixed(1);
  $("#sampler").value = d.sampler;
  $("#familyDesc").textContent = fam.description;
  $("#editInfo").textContent = fam.edit_info || "";
  await refreshStatus();
};

// ------------------------------------------------------------- mode générer / éditer
let MODE = "generate";
// Références ajoutées par l'application (mannequin, pose générée) : elles ne passent pas
// par le champ fichier, on les garde donc à part et on les ajoute à l'envoi.
let EXTRA_REFS = [];
const refFiles = () => [...$("#refs").files, ...EXTRA_REFS];
function renderRefPreviews() {
  const box = $("#refPreview");
  box.innerHTML = "";
  [...$("#refs").files].forEach((f) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    box.appendChild(img);
  });
  EXTRA_REFS.forEach((f, i) => {
    const wrap = document.createElement("div");
    wrap.className = "ref";
    wrap.style.position = "relative";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    const del = document.createElement("button");
    del.className = "ghost";
    del.textContent = "✖";
    del.title = "Retirer cette référence";
    del.onclick = () => { EXTRA_REFS.splice(i, 1); renderRefPreviews(); };
    wrap.appendChild(img); wrap.appendChild(del);
    box.appendChild(wrap);
  });
}
function addExtraRef(file) {
  if (!file) return false;
  if (EXTRA_REFS.some((f) => f.name === file.name && f.size === file.size)) return false;
  EXTRA_REFS.push(file);
  renderRefPreviews();
  return true;
}
function setMode(m) {
  MODE = m === "edit" ? "edit" : "generate";
  const editing = MODE === "edit";
  $("#modeGen").classList.toggle("active", !editing);
  $("#modeEdit").classList.toggle("active", editing);
  $("#editCard").classList.toggle("highlight", editing);
  $("#btnGenerate").textContent = editing ? "✏️ Appliquer la modification" : "✨ Générer";
  $("#prompt").placeholder = editing
    ? "Décrivez la modification : ex « remplace le fond par une plage au coucher du soleil, garde le sujet identique »"
    : "Ex : un chat roux tenant une pancarte « Bonjour », lumière douce, photo réaliste";
  const col = $("#editCard").parentElement;
  if (editing) col.prepend($("#editCard"));
  else col.insertBefore($("#editCard"), $("#formatRow"));
  // en modification d'image avec un modèle ControlNet : on repart de la photo par défaut,
  // pour garder la personne (couleurs, identité) tout en suivant la pose détectée
  if (editing && controlCapable() && !CONTROL.id) $("#useInit").checked = true;
}
$("#modeGen").onclick = () => setMode("generate");
$("#modeEdit").onclick = () => setMode("edit");

// ------------------------------------------------------------- formulaire
$("#steps").oninput = (e) => ($("#stepsVal").textContent = e.target.value);
$("#cfg").oninput = (e) => ($("#cfgVal").textContent = Number(e.target.value).toFixed(1));
$("#preset").onchange = (e) => {
  if (e.target.value === "custom") return;
  const [w, h] = e.target.value.split("x");
  $("#width").value = w; $("#height").value = h;
};
function syncPreset() {
  const v = `${$("#width").value}x${$("#height").value}`;
  $("#preset").value = [...$("#preset").options].some((o) => o.value === v) ? v : "custom";
}
$("#width").onchange = $("#height").onchange = syncPreset;
$("#refs").onchange = () => renderRefPreviews();

function showGenerateError(msg) {
  $("#errorBox").textContent = msg;
  $("#errorBox").classList.remove("hidden");
}

$("#btnGenerate").onclick = async () => {
  const refs = refFiles();
  if (MODE === "edit" && !refs.length) {
    showGenerateError("Mode édition : ajoutez au moins une image de référence dans la carte « Édition d'image ».");
    return;
  }
  if (CONTROL.type && CONTROL.id && !controlCapable()) {
    showGenerateError("Le modèle sélectionné n'accepte pas ControlNet (architectures DiT). "
      + "Choisissez le modèle « SD 1.5 + ControlNet (pose) » : la pose détectée sera appliquée.");
    return;
  }
  if (CONTROL.type && CONTROL.id && CONTROL.type === "pose" && !controlnetPresent("pose")) {
    showGenerateError("Modèle ControlNet OpenPose manquant : téléchargez-le dans l'onglet Modèles "
      + "(famille « SD 1.5 + ControlNet »).");
    return;
  }
  if (MODE === "edit" && !STATUS.edit_ready) {
    showGenerateError("Mode édition : ce modèle a besoin de l'encodeur de vision (mmproj). "
      + "Cliquez sur « ⬇ Télécharger l'encodeur de vision (édition) » ou allez dans l'onglet Modèles.");
    return;
  }
  const fd = new FormData();
  fd.append("prompt", $("#prompt").value);
  fd.append("negative_prompt", $("#negative").value);
  fd.append("width", $("#width").value);
  fd.append("height", $("#height").value);
  fd.append("steps", $("#steps").value);
  fd.append("cfg_scale", $("#cfg").value);
  fd.append("sampler", $("#sampler").value);
  fd.append("seed", $("#seed").value);
  fd.append("mode", MODE);
  if (CONTROL.type && CONTROL.id) {
    fd.append("control_type", CONTROL.type);
    fd.append("control_id", CONTROL.id);
    fd.append("control_strength", $("#controlStrength").value);
    const cn = controlnetFor(CONTROL.type);
    if (cn) fd.append("control_net", cn.id);
  }
  if (controlCapable()) {
    fd.append("use_init", $("#useInit").checked ? "true" : "false");
    fd.append("strength", $("#strength").value);
  }
  refs.forEach((f) => fd.append("ref_images", f));
  $("#errorBox").classList.add("hidden");
  try {
    await api("/api/generate", { method: "POST", body: fd });
    $("#btnGenerate").disabled = true;
    pollGeneration();
  } catch (e) {
    $("#errorBox").textContent = e.message;
    $("#errorBox").classList.remove("hidden");
  }
};
$("#btnCancel").onclick = () => api("/api/cancel", { method: "POST" });

let pollTimer = null;
function pollGeneration() {
  clearTimeout(pollTimer);
  api("/api/generation").then((g) => {
    renderGeneration(g);
    if (MANNEQUIN.pending && g.result && g.result !== MANNEQUIN.result) mqGenerationReady(g.result);
    else if (MANNEQUIN.pending && g.error) { MANNEQUIN.pending = false; mqSetGenStatus("⚠️ " + g.error, "warn"); }
    if (g.running) pollTimer = setTimeout(pollGeneration, 800);
    else { $("#btnGenerate").disabled = !(STATUS && STATUS.ready); loadGallery(); }
  });
}

function renderGeneration(g) {
  const box = $("#progressBox");
  $("#btnCancel").classList.toggle("hidden", !g.running);
  if (g.running || g.error || g.result) box.classList.remove("hidden");
  $("#phase").textContent = g.phase || "";
  $("#elapsed").textContent = g.elapsed ? fmtTime(g.elapsed) : "";
  $("#bar").style.width = (100 * (g.progress || 0)) + "%";
  $("#stepTxt").textContent = g.steps ? `étape ${g.step}/${g.steps}` : "";
  if (g.running && g.step > 1 && g.progress > 0) {
    const eta = g.elapsed / g.progress - g.elapsed;
    $("#eta").textContent = `~${fmtTime(Math.max(0, eta))} restantes`;
  } else $("#eta").textContent = "";
  const log = $("#log");
  log.textContent = (g.log || []).join("\n");
  log.scrollTop = log.scrollHeight;
  if (g.error) { $("#errorBox").textContent = g.error; $("#errorBox").classList.remove("hidden"); }
  if (g.result && $("#resultBox").dataset.file !== g.result) {
    $("#resultBox").dataset.file = g.result;
    $("#resultBox").innerHTML = `<img src="/outputs/${g.result}?t=${Date.now()}" alt="">`;
    $("#resultBox img").onclick = () => openLightbox(`/outputs/${g.result}`, g.result);
    fetch(`/outputs/${g.result.replace(".png", ".json")}`).then((r) => r.json()).then((m) => {
      const f = famById(m.family);
      $("#resultMeta").innerHTML = `${m.family ? (f ? f.name : m.family) + " · " : ""}Seed <b>${m.seed}</b> · ${m.width}×${m.height} · ${m.steps} étapes · CFG ${m.cfg_scale} · ${m.elapsed_s} s
        <button class="ghost" id="reuseSeed">↺ réutiliser la seed</button>`;
      $("#reuseSeed").onclick = () => ($("#seed").value = m.seed);
      $("#resultControl").innerHTML = m.control_type
        ? `🧍 contrôle : ${m.control_type === "pose" ? "pose des personnages" : "contours (Canny)"} · force ${m.control_strength}`
          + (m.init_image ? ` · img2img depuis ${esc(m.init_image)} (force ${m.strength})` : "")
        : "";
    }).catch(() => {});
  }
}

// ------------------------------------------------------------- galerie
async function loadGallery() {
  const items = await api("/api/gallery");
  const root = $("#gallery");
  root.innerHTML = items.length ? "" : '<div class="small">Aucune image pour l\'instant.</div>';
  for (const it of items) {
    const d = document.createElement("div");
    d.className = "item";
    const m = it.meta || {};
    d.innerHTML = `<img src="/outputs/${it.file}" loading="lazy" alt="">
      <div class="cap" title="${esc(m.prompt || "")}">${esc(m.prompt || it.file)}</div>
      <div class="tools"><button title="Réutiliser le prompt et les réglages">↺ Réutiliser</button><a href="/outputs/${it.file}" download><button>⬇</button></a><button title="Supprimer">🗑</button></div>`;
    d.querySelector("img").onclick = () => openLightbox(`/outputs/${it.file}`, m.prompt || it.file);
    const [reuse, , del] = d.querySelectorAll("button");
    reuse.onclick = async () => {
      if (m.family && m.family !== STATUS.config.family && famById(m.family)) { await postJSON("/api/config", { family: m.family }); await refreshStatus(); }
      if (m.prompt) $("#prompt").value = m.prompt;
      if (m.negative_prompt != null) $("#negative").value = m.negative_prompt;
      if (m.seed != null) $("#seed").value = m.seed;
      if (m.steps) { $("#steps").value = m.steps; $("#stepsVal").textContent = m.steps; }
      if (m.cfg_scale) { $("#cfg").value = m.cfg_scale; $("#cfgVal").textContent = Number(m.cfg_scale).toFixed(1); }
      if (m.width) $("#width").value = m.width;
      if (m.height) $("#height").value = m.height;
      if (m.sampler) $("#sampler").value = m.sampler;
      syncPreset();
      showTab("generate");
    };
    del.onclick = async () => { if (confirm("Supprimer cette image ?")) { await api(`/api/gallery/${it.file}`, { method: "DELETE" }); loadGallery(); } };
    root.appendChild(d);
  }
}
$("#btnOpenOutputs").onclick = () => { const fd = new FormData(); fd.append("which", "outputs"); api("/api/open-folder", { method: "POST", body: fd }).then((r) => console.log(r.path)); };

function openLightbox(src, cap) {
  $("#lightboxImg").src = src; $("#lightboxCap").textContent = cap || "";
  $("#lightbox").classList.remove("hidden");
}
$("#lightbox").onclick = () => $("#lightbox").classList.add("hidden");

// ------------------------------------------------------------- setup
$("#btnEngine").onclick = async () => {
  const btn = $("#btnEngine");
  try { await postJSON("/api/engine/install", { flavor: $("#flavor").value }); }
  catch (e) { renderError(e.message); }
  btn.disabled = true; setTimeout(() => (btn.disabled = false), 5000);
  refreshStatus();
};
$("#btnSavePerf").onclick = async () => {
  await postJSON("/api/config", {
    offload_to_cpu: $("#offload").checked, flash_attention: $("#fa").checked, vae_tiling: $("#vaetiling").checked,
    threads: parseInt($("#threads").value || "-1", 10), extra_args: $("#extra").value,
  });
  $("#btnSavePerf").textContent = "Enregistré ✔"; setTimeout(() => ($("#btnSavePerf").textContent = "Enregistrer"), 1500);
};
$("#btnClearJobs").onclick = () => api("/api/jobs/clear", { method: "POST" }).then(refreshStatus);
$("#btnCancelJobs").onclick = () => api("/api/jobs/cancel-all", { method: "POST" }).then(refreshStatus);
$("#btnOpenModels").onclick = () => { const fd = new FormData(); fd.append("which", "models"); api("/api/open-folder", { method: "POST", body: fd }).then((r) => console.log(r.path)); };
$("#btnFixEdit").onclick = async () => {
  const fam = famById(STATUS.config.family);
  if (!fam) return;
  const vision = (fam.vision || []).find((x) => x.recommended) || (fam.vision || [])[0];
  if (!vision) { showTab("models"); return; }
  try { await postJSON("/api/download", { family: fam.id, category: "vision", file_id: vision.id }); }
  catch (e) { alert(e.message); }
  refreshStatus();
};


// ======================================================================
//  Personnages, pose et composition (ControlNet)
// ======================================================================
const SKELETON_LIMBS = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];
const JOINT_NAMES = ["nez", "œil g.", "œil d.", "oreille g.", "oreille d.", "épaule g.", "épaule d.",
  "coude g.", "coude d.", "poignet g.", "poignet d.", "hanche g.", "hanche d.",
  "genou g.", "genou d.", "cheville g.", "cheville d."];

const CONTROL = {
  type: "", id: "", url: "", width: 0, height: 0,
  persons: [], detected: null, sourceUrl: "", sourceId: "", sourceKind: "ref",
  backend: "", status: "", busy: false, editor: false,
};

function controlFamily() {
  if (!STATUS || !STATUS.control) return null;
  return famById(STATUS.control.control_family);
}

function controlCapable() {
  const fam = famById(STATUS?.config?.family);
  return !!(fam && fam.supports_controlnet);
}

function controlnetFor(type) {
  const list = (STATUS.control.controlnets || []);
  const want = type === "canny" ? "canny" : "openpose";
  return list.find((c) => (c.mode || "").includes(want)) || list[0];
}

function controlnetPresent(type) {
  const cat = controlFamily();
  if (!cat) return false;
  const file = controlnetFor(type);
  return !!file && (STATUS.models[cat.id].controlnet || []).some((m) => m.name === file.id);
}

function poseModelPresent() {
  const cat = controlFamily();
  if (!cat) return false;
  return (STATUS.models[cat.id].pose_detector || []).length > 0;
}

function setControlStatus(msg, kind = "") {
  const el = $("#controlStatus");
  el.textContent = msg || "";
  el.className = "small " + kind;
}

// --------------------------------------------------------------- affichage
function renderControl() {
  const card = $("#poseCard");
  if (!card || !STATUS || !STATUS.control) return;
  const capable = controlCapable();
  const fam = famById(STATUS.config.family);
  const warn = $("#controlUnsupported");
  const body = $("#controlBody");
  const cat = controlFamily();

  if (!capable) {
    warn.innerHTML = `ℹ️ <b>${esc(fam ? fam.name : "Ce modèle")} n'accepte pas ControlNet.</b>
      ${esc((fam && fam.control_reason) || "")}
      <button class="primary" id="btnSwitchControl" style="margin-top:8px">→ Utiliser « ${esc(cat ? cat.name : "SD 1.5 + ControlNet")} »</button>`;
    warn.classList.remove("hidden");
    const sw = $("#btnSwitchControl");
    if (sw) sw.onclick = () => switchFamily((cat && cat.id) || "sd15_control");
  } else {
    warn.classList.add("hidden");
  }
  body.classList.toggle("hidden", false);
  card.classList.toggle("dim", !capable);

  // type de contrôle
  const sel = $("#controlType");
  if (sel.value !== CONTROL.type) sel.value = CONTROL.type;
  const needsPoseModel = CONTROL.type === "pose";
  const missing = [];
  if (CONTROL.type && !controlnetPresent(CONTROL.type)) missing.push(controlnetFor(CONTROL.type));
  if (needsPoseModel && !poseModelPresent()) missing.push((STATUS.control.pose_models || [])[0]);
  const box = $("#controlMissing");
  if (missing.length && capable) {
    box.innerHTML = "⬇️ Fichier(s) nécessaire(s) au contrôle de la pose encore absent(s) : <b>"
      + missing.map((m) => esc(m ? m.label : "?")).join("</b>, <b>")
      + "</b> <button class=\"primary\" id=\"btnControlDownload\">Télécharger maintenant</button>";
    box.classList.remove("hidden");
    const b = $("#btnControlDownload");
    if (b) b.onclick = async () => {
      b.disabled = true; b.textContent = "⏳ téléchargement…";
      try {
        for (const m of missing) {
          if (!m) continue;
          const category = (m.mode === undefined) ? "pose_detector" : "controlnet";
          await postJSON("/api/download", { family: cat.id, category, file_id: m.id });
        }
      } catch (e) { alert(e.message); }
      refreshStatus();
    };
  } else box.classList.add("hidden");

  // état
  const pose = STATUS.control.pose || {};
  const bits = [];
  if (!capable) bits.push("modèle actif incompatible");
  if (pose.cv === false) bits.push("NumPy/OpenCV manquants — installez requirements.txt");
  if (CONTROL.type === "pose" && !poseModelPresent()) bits.push("détecteur de personnages à télécharger (13 Mo)");
  if (CONTROL.type && !controlnetPresent(CONTROL.type)) bits.push("modèle ControlNet à télécharger (0,7 Go)");
  if (CONTROL.status) bits.push(CONTROL.status);
  else if (CONTROL.backend) bits.push("détection : " + CONTROL.backend);
  setControlStatus(bits.join(" · "), bits.length ? "warn" : "ok");

  // aperçus
  const show = !!CONTROL.url;
  $("#controlPreviewWrap").classList.toggle("hidden", !show);
  if (show) {
    $("#controlPreview").src = CONTROL.url;
    $("#controlSourcePreview").src = CONTROL.sourceUrl || CONTROL.url;
  }
  $("#poseEditor").classList.toggle("hidden", !(CONTROL.editor && CONTROL.persons.length));
  $("#controlInitWrap").classList.toggle("hidden", !capable);
  $("#controlHint").textContent = CONTROL.status
    || (CONTROL.type === "pose"
      ? "Détectez les personnages d'une photo (ou importez un squelette), ajustez la pose, puis générez : la posture est conservée."
      : "Choisissez un type de contrôle puis cliquez sur « Détecter les personnages » (pose) ou lancez le calcul des contours (Canny).");
  if (CONTROL.persons.length && CONTROL.editor) drawPoseEditor();
}

function switchFamily(id) {
  const fs = $("#family");
  if (!fs || !famById(id)) return;
  fs.value = id;
  fs.onchange({ target: fs });
}

// --------------------------------------------------------------- détection
function quietError(msg) {
  const e = new Error(msg);
  e.silent = true;          // pas de fenêtre d'alerte : le message s'affiche dans la carte
  return e;
}

async function sourceImageFile() {
  const kind = $("#controlSource").value;
  if (kind === "ref") {
    const f = [...$("#refs").files][0];
    if (!f) throw quietError("Ajoutez d'abord une image dans « Édition d'image — images de référence », "
      + "ou choisissez « Autre fichier… ».");
    return f;
  }
  if (kind === "file") {
    const f = $("#controlFile").files[0];
    if (!f) { $("#controlFile").click(); throw quietError("Choisissez une image avec « Autre fichier… »."); }
    return f;
  }
  // dernière image générée
  const items = await api("/api/gallery");
  if (!items.length) throw new Error("Aucune image générée pour l'instant.");
  const r = await fetch("/outputs/" + items[0].file);
  const blob = await r.blob();
  return new File([blob], items[0].file, { type: blob.type || "image/png" });
}

async function detectControl(kind) {
  const btn = $("#btnDetect");
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = kind === "canny" ? "⏳ calcul des contours…" : "⏳ détection des personnages…";
  setControlStatus("Traitement de l'image…");
  try {
    const file = await sourceImageFile();
    const fd = new FormData();
    fd.append("image", file);
    fd.append("kind", kind);
    fd.append("background", "black");
    const r = await api("/api/control/detect", { method: "POST", body: fd });
    CONTROL.type = r.kind;
    CONTROL.id = r.control.id;
    CONTROL.url = r.control.url;
    CONTROL.width = r.control.width;
    CONTROL.height = r.control.height;
    CONTROL.persons = r.persons || [];
    CONTROL.detected = r.persons ? JSON.parse(JSON.stringify(r.persons)) : null;
    CONTROL.sourceUrl = r.source.url;
    CONTROL.sourceId = r.source.id;
    CONTROL.backend = r.backend || "";
    CONTROL.status = r.message || "";
    CONTROL.editor = kind === "pose";
    $("#controlType").value = kind;
    if (kind === "pose" && CONTROL.persons.length) {
      $("#controlSourcePreview").onload = () => drawPoseEditor();
    }
    renderControl();
    $("#controlSourcePreview").src = CONTROL.sourceUrl;
  } catch (e) {
    CONTROL.status = "";
    renderControl();                       // réaffiche l'état réel (fichiers manquants, etc.)
    setControlStatus("⚠️ " + e.message, "warn");
    if (!e.silent) alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// ------------------------------------------------------- éditeur de squelette
function poseImage() {
  return window.__poseImg && window.__poseImg.src === CONTROL.sourceUrl ? window.__poseImg : null;
}

function loadPoseImage(url) {
  if (window.__poseImg && window.__poseImg.src === url) return Promise.resolve(window.__poseImg);
  return new Promise((resolve) => {
    let done = false;
    const finish = (img) => {
      if (done) return;
      done = true;
      window.__poseImg = img;
      resolve(img);
    };
    const img = new Image();
    img.onload = () => finish(img);
    img.onerror = () => finish(null);
    setTimeout(() => finish(null), 2000);      // image lente : on dessine quand même le squelette
    img.src = url;
  });
}

async function drawPoseEditor() {
  const canvas = $("#poseCanvas");
  if (!canvas || !CONTROL.persons.length) return;
  const img = await loadPoseImage(CONTROL.sourceUrl);
  const natural = img ? { w: img.naturalWidth, h: img.naturalHeight } : { w: CONTROL.width, h: CONTROL.height };
  const maxW = Math.min(520, canvas.parentElement.clientWidth - 20 || 520);
  const scale = Math.min(maxW / natural.w, 520 / natural.h);
  canvas.width = Math.round(natural.w * scale);
  canvas.height = Math.round(natural.h * scale);
  canvas.dataset.scale = scale;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;                      // navigateur sans canvas 2D : aperçu uniquement
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (img) {
    ctx.globalAlpha = 0.55;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1;
  }
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#7c5cff";
  for (const p of CONTROL.persons) {
    for (const [a, b] of SKELETON_LIMBS) {
      const ka = p.keypoints[a], kb = p.keypoints[b];
      if (!ka || !kb || !ka.visible || !kb.visible) continue;
      ctx.beginPath();
      ctx.moveTo(ka.x * scale, ka.y * scale);
      ctx.lineTo(kb.x * scale, kb.y * scale);
      ctx.stroke();
    }
    p.keypoints.forEach((k, i) => {
      if (!k.visible && !k.moved) return;
      ctx.beginPath();
      ctx.arc(k.x * scale, k.y * scale, k.moved ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = k.moved ? "#ffd479" : (i < 5 ? "#ff64ff" : "#48d1ff");
      ctx.fill();
      ctx.strokeStyle = "#0a0c10";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#7c5cff";
    });
  }
  $("#poseEditorInfo").textContent =
    `${CONTROL.persons.length} personnage(s) · ${CONTROL.persons.reduce((n, p) => n + p.keypoints.filter((k) => k.moved).length, 0)} point(s) déplacé(s) · image ${natural.w}×${natural.h} px`;
}

function canvasPoint(ev, canvas) {
  const r = canvas.getBoundingClientRect();
  const scale = Number(canvas.dataset.scale || 1);
  return { x: (ev.clientX - r.left) / scale, y: (ev.clientY - r.top) / scale };
}

let poseDrag = null;
let poseRenderTimer = null;

function bindPoseEditor() {
  const canvas = $("#poseCanvas");
  if (!canvas || canvas.dataset.bound) return;
  canvas.dataset.bound = "1";
  canvas.addEventListener("pointerdown", (ev) => {
    const pt = canvasPoint(ev, canvas);
    let best = null, bestD = 18;
    CONTROL.persons.forEach((p, pi) => p.keypoints.forEach((k, ki) => {
      if (!k.visible && !k.moved) return;
      const d = Math.hypot(k.x - pt.x, k.y - pt.y);
      if (d < bestD) { bestD = d; best = { pi, ki }; }
    }));
    if (!best) return;
    poseDrag = best;
    if (canvas.setPointerCapture && ev.pointerId !== undefined) {
      try { canvas.setPointerCapture(ev.pointerId); } catch (e) { /* non supporté */ }
    }
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!poseDrag) return;
    const pt = canvasPoint(ev, canvas);
    const k = CONTROL.persons[poseDrag.pi].keypoints[poseDrag.ki];
    k.x = Math.round(Math.max(0, Math.min(CONTROL.width, pt.x)));
    k.y = Math.round(Math.max(0, Math.min(CONTROL.height, pt.y)));
    k.moved = true;
    k.visible = true;
    drawPoseEditor();
  });
  canvas.addEventListener("pointerup", async () => {
    if (!poseDrag) return;
    poseDrag = null;
    clearTimeout(poseRenderTimer);
    poseRenderTimer = setTimeout(applyPoseEdits, 150);
  });
  canvas.addEventListener("pointercancel", () => { poseDrag = null; });
}

async function applyPoseEdits() {
  if (!CONTROL.persons.length) return;
  setControlStatus("Mise à jour du squelette…");
  try {
    const r = await postJSON("/api/control/pose", {
      persons: CONTROL.persons, width: CONTROL.width, height: CONTROL.height, kind: "pose",
    });
    CONTROL.id = r.control.id;
    CONTROL.url = r.control.url + "?t=" + Date.now();
    CONTROL.status = "Pose ajustée à la main — " + CONTROL.persons.length + " personnage(s).";
    renderControl();
  } catch (e) {
    setControlStatus("⚠️ " + e.message, "warn");
  }
}

$("#btnDetect").onclick = () => detectControl($("#controlType").value === "canny" ? "canny" : "pose");
$("#controlType").onchange = (e) => {
  const v = e.target.value;
  if (!v) { clearControl(); return; }
  CONTROL.type = v;
  CONTROL.editor = v === "pose";
  detectControl(v);
};
$("#controlStrength").oninput = (e) => ($("#controlStrengthVal").textContent = Number(e.target.value).toFixed(2));
$("#strength").oninput = (e) => ($("#strengthVal").textContent = Number(e.target.value).toFixed(2));
$("#controlSource").onchange = (e) => {
  if (e.target.value === "file") $("#controlFile").click();
};
$("#controlFile").onchange = () => { if (CONTROL.type) detectControl(CONTROL.type); };
$("#btnReference").onclick = () => $("#referencePose").click();
$("#referencePose").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  // squelette de référence : on l'utilise tel quel comme image de contrôle
  const fd = new FormData();
  fd.append("image", f);
  fd.append("kind", "canny");
  try {
    const up = await api("/api/control/detect", { method: "POST", body: fd });
    CONTROL.type = "pose";
    CONTROL.id = up.control.id;
    CONTROL.url = up.control.url;
    CONTROL.width = up.control.width;
    CONTROL.height = up.control.height;
    CONTROL.persons = [];
    CONTROL.detected = null;
    CONTROL.editor = false;
    CONTROL.sourceUrl = up.source.url;
    CONTROL.status = "Squelette de référence importé : il sera utilisé comme contrôle de pose (utilisez un "
      + "ControlNet Canny si c'est une simple silhouette).";
    $("#controlType").value = "pose";
    renderControl();
  } catch (err) { alert(err.message); }
};
$("#btnPoseReset").onclick = () => {
  if (!CONTROL.detected) return;
  CONTROL.persons = JSON.parse(JSON.stringify(CONTROL.detected));
  drawPoseEditor();
  applyPoseEdits();
};
$("#btnPoseHide").onclick = () => {
  CONTROL.editor = false;
  $("#poseEditor").classList.add("hidden");
};
$("#btnControlClear").onclick = () => clearControl();

function clearControl() {
  CONTROL.type = ""; CONTROL.id = ""; CONTROL.url = ""; CONTROL.persons = [];
  CONTROL.detected = null; CONTROL.sourceUrl = ""; CONTROL.status = ""; CONTROL.editor = false;
  CONTROL.width = CONTROL.height = 0;
  if ($("#controlType")) $("#controlType").value = "";
  renderControl();
  setControlStatus("Contrôle désactivé : la génération se fait librement à partir du prompt.");
}

bindPoseEditor();


// ======================================================================
//  Mannequin articulé — poser, puis exporter une image de référence
// ======================================================================
const MQ_JOINTS = {
  hips: "bassin", spine: "bas du dos", chest: "poitrine", neck: "cou", head: "tête", head_top: "sommet du crâne",
  nose: "nez", eye_l: "œil gauche", eye_r: "œil droit", ear_l: "oreille gauche", ear_r: "oreille droite",
  shoulder_l: "épaule gauche", elbow_l: "coude gauche", wrist_l: "poignet gauche", hand_l: "main gauche",
  shoulder_r: "épaule droite", elbow_r: "coude droit", wrist_r: "poignet droit", hand_r: "main droite",
  hip_l: "hanche gauche", knee_l: "genou gauche", ankle_l: "cheville gauche", toe_l: "pied gauche (pointe)",
  heel_l: "talon gauche",
  hip_r: "hanche droite", knee_r: "genou droit", ankle_r: "cheville droite", toe_r: "pied droit (pointe)",
  heel_r: "talon droit",
};
const MQ_BODY_LABELS = { neutre: "Neutre", fin: "Fine", athletique: "Athlétique", fort: "Forte", femme: "Féminine" };
const MQ_PROMPTS = {
  photo: "personne debout, tenue simple, fond gris uni, éclairage de studio, photo réaliste, corps entier visible",
  sport: "athlète en tenue de sport, mouvement dynamique, fond neutre, photo réaliste, corps entier visible",
  ville: "personne en tenue de ville, rue ensoleillée, photo réaliste, corps entier visible",
  dessin: "illustration, personnage stylisé, fond clair, dessin net, corps entier visible",
};
const MANNEQUIN = {
  rig: null, style: "volume", output: "openpose", selected: "wrist_l",
  depthDone: 0, drag: null, history: [], render: null, renderKey: "",
  pending: false, result: "", jsdomFallback: false,
};

const MQ_KIT = (typeof MannequinKit !== "undefined" && MannequinKit)
  ? MannequinKit : { BUILDS: {}, SEGMENTS: [], morphedDimensions: () => ({ lengths: {}, thickness: {} }) };
function mqCanvas() { return $("#mqCanvas"); }
function mqScale(canvas) {
  const w = canvas.getBoundingClientRect().width || canvas.width;
  return canvas.width / (w || canvas.width);
}

function mqInit() {
  const canvas = mqCanvas();
  if (!canvas) return;
  if (typeof Mannequin !== "function") {
    MANNEQUIN.jsdomFallback = true;
    return;
  }
  MANNEQUIN.rig = new Mannequin({ style: "volume" });
  MANNEQUIN.rig.fitCamera(canvas.width, canvas.height);
  const sel = $("#mqJoint");
  Object.entries(MQ_JOINTS).forEach(([id, label]) => {
    const o = document.createElement("option");
    o.value = id; o.textContent = label;
    sel.appendChild(o);
  });
  sel.value = MANNEQUIN.selected;
  mqBuildTable();
  mqBind();
  mqPaint();
  mqFillEngines();
}

function mqInfo() {
  const rig = MANNEQUIN.rig;
  if (!rig) return;
  const stature = rig.pose.head_top.y - Math.min(rig.pose.ankle_l.y, rig.pose.ankle_r.y);
  $("#mqInfo").textContent = `Taille ${stature.toFixed(2).replace(".", ",")} m · pose « ${
    rig.preset === "personnalise" ? "personnalisée" : rig.preset.replace("_", " ")} » · point : ${MQ_JOINTS[MANNEQUIN.selected] || MANNEQUIN.selected}`;
}

function mqPaint() {
  const canvas = mqCanvas();
  const rig = MANNEQUIN.rig;
  if (!canvas || !rig) return;
  let ctx = null;
  try { ctx = canvas.getContext("2d"); } catch (e) { ctx = null; }
  if (!ctx) { MANNEQUIN.jsdomFallback = true; return; }
  ctx.save();
  ctx.setTransform ? ctx.setTransform(1, 0, 0, 1, 0, 0) : null;
  ctx.fillStyle = MANNEQUIN.style === "volume" ? "#0f1218" : "#0a0d13";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  rig.selected = MANNEQUIN.selected;
  try {
    rig.render(ctx, canvas.width, canvas.height, MANNEQUIN.style);
  } catch (e) {
    // contexte 2D incomplet (navigateur ancien, test sans canvas) : on garde l'application utilisable
    MANNEQUIN.lastDrawError = e.message;
  }
  ctx.restore();
  mqInfo();
  window.__liq = Object.assign(window.__liq || {}, { mannequin: MANNEQUIN, mqPaint });
}

function mqSnapshot() {
  if (!MANNEQUIN.rig) return;
  MANNEQUIN.history.push(JSON.stringify(MANNEQUIN.rig.toJSON()));
  if (MANNEQUIN.history.length > 40) MANNEQUIN.history.shift();
}

function mqUndo() {
  const snap = MANNEQUIN.history.pop();
  if (!snap || !MANNEQUIN.rig) { mqSetStatus("Rien à annuler."); return; }
  MANNEQUIN.rig = Mannequin.fromJSON(JSON.parse(snap));
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height);
  mqPaint();
  mqSetStatus("Dernier déplacement annulé.");
}

function mqSetStatus(msg, kind = "") {
  const el = $("#mqStatus");
  el.textContent = msg || "";
  el.className = "small " + kind;
}
function mqSetGenStatus(msg, kind = "") {
  const el = $("#mqGenStatus");
  el.textContent = msg || "";
  el.className = "small " + kind;
}

function mqPoint(ev, canvas) {
  const r = canvas.getBoundingClientRect();
  const k = mqScale(canvas);
  return { x: (ev.clientX - r.left) * k, y: (ev.clientY - r.top) * k };
}

function mqBind() {
  const canvas = mqCanvas();
  canvas.tabIndex = 0;
  canvas.addEventListener("pointerdown", (ev) => {
    const rig = MANNEQUIN.rig;
    if (!rig) return;
    const pt = mqPoint(ev, canvas);
    const joint = rig.jointAt(pt.x, pt.y, canvas.width, canvas.height, 30);
    mqSnapshot();
    if (joint) {
      MANNEQUIN.selected = joint;
      $("#mqJoint").value = joint;
      MANNEQUIN.depthDone = 0;
      $("#mqDepth").value = 0;
      $("#mqDepthVal").textContent = "0";
      MANNEQUIN.drag = { kind: "joint", joint };
      mqPaint();
    } else {
      MANNEQUIN.drag = { kind: "orbit", x: ev.clientX, y: ev.clientY };
      canvas.classList.add("dragging");
    }
    if (canvas.setPointerCapture && ev.pointerId !== undefined) {
      try { canvas.setPointerCapture(ev.pointerId); } catch (e) { /* non supporté */ }
    }
  });
  canvas.addEventListener("pointermove", (ev) => {
    const rig = MANNEQUIN.rig;
    if (!rig || !MANNEQUIN.drag) return;
    if (MANNEQUIN.drag.kind === "joint") {
      const pt = mqPoint(ev, canvas);
      const joint = MANNEQUIN.drag.joint;
      const opts = rig.cameraOptions(canvas.width, canvas.height);
      const ref = MQ_KIT.project ? MQ_KIT.project(rig.pose[joint], opts).depth : opts.distance;
      rig.moveJoint(joint, rig.screenToWorld(pt.x, pt.y, canvas.width, canvas.height, ref));
      mqPaint();
    } else {
      rig.orbit(ev.clientX - MANNEQUIN.drag.x, ev.clientY - MANNEQUIN.drag.y);
      MANNEQUIN.drag.x = ev.clientX; MANNEQUIN.drag.y = ev.clientY;
      rig.fitCamera(canvas.width, canvas.height, 1.06);
      mqPaint();
    }
  });
  const stop = () => { MANNEQUIN.drag = null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("wheel", (ev) => {
    if (!MANNEQUIN.rig) return;
    ev.preventDefault();
    MANNEQUIN.rig.zoom(ev.deltaY > 0 ? 1.08 : 0.93);
    mqPaint();
  }, { passive: false });
  canvas.addEventListener("keydown", (ev) => {
    const steps = { ArrowLeft: [-0.03, 0], ArrowRight: [0.03, 0], ArrowUp: [0, 0.03], ArrowDown: [0, -0.03] };
    const d = steps[ev.key];
    if (!d || !MANNEQUIN.rig) return;
    ev.preventDefault();
    if (MANNEQUIN.history.length === 0 || ev.repeat === false) mqSnapshot();
    const canvasEl = mqCanvas();
    const rig = MANNEQUIN.rig;
    const cur = rig.pose[MANNEQUIN.selected];
    const right = { x: Math.cos(rig.camera.yaw), y: 0, z: -Math.sin(rig.camera.yaw) };
    const up = { x: 0, y: 1, z: 0 };
    const target = {
      x: cur.x + d[0] * right.x + d[1] * up.x,
      y: cur.y + d[0] * right.y + d[1] * up.y,
      z: cur.z + d[0] * right.z + d[1] * up.z,
    };
    rig.moveJoint(MANNEQUIN.selected, target);
    mqPaint();
  });
}

$("#mqJoint").onchange = (e) => {
  MANNEQUIN.selected = e.target.value;
  MANNEQUIN.depthDone = 0;
  $("#mqDepth").value = 0;
  $("#mqDepthVal").textContent = "0";
  mqPaint();
};
$("#mqDepth").oninput = (e) => {
  const rig = MANNEQUIN.rig;
  if (!rig) return;
  $("#mqDepthVal").textContent = e.target.value;
  const delta = (Number(e.target.value) - MANNEQUIN.depthDone) / 100 * 0.5;
  MANNEQUIN.depthDone = Number(e.target.value);
  if (!delta) return;
  // avance ou recule le long de l'axe de la caméra
  const yaw = rig.camera.yaw, pitch = rig.camera.pitch;
  const dir = { x: Math.sin(yaw), y: -Math.cos(yaw) * Math.sin(pitch), z: Math.cos(yaw) * Math.cos(pitch) };
  const cur = rig.pose[MANNEQUIN.selected];
  if (!cur) return;
  rig.moveJoint(MANNEQUIN.selected, { x: cur.x + dir.x * delta, y: cur.y + dir.y * delta, z: cur.z + dir.z * delta });
  mqPaint();
};
$("#mqDepth").onpointerdown = () => mqSnapshot();

$("#mqStyle").onchange = (e) => {
  MANNEQUIN.style = e.target.value === "wireframe" ? "wireframe" : "volume";
  if (MANNEQUIN.rig) MANNEQUIN.rig.setStyle(MANNEQUIN.style);
  mqPaint();
};
$("#mqPreset").onchange = (e) => {
  if (!MANNEQUIN.rig) return;
  mqSnapshot();
  MANNEQUIN.rig.applyPreset(e.target.value);
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height);
  mqPaint();
  mqSetStatus(`Pose « ${e.target.selectedOptions[0].textContent} » appliquée — ajustez ensuite les articulations.`);
};
$("#btnMqUndo").onclick = () => mqUndo();
$("#btnMqMirror").onclick = () => {
  if (!MANNEQUIN.rig) return;
  mqSnapshot();
  MANNEQUIN.rig.mirror();
  mqPaint();
  mqSetStatus("Pose inversée (miroir gauche/droite).");
};
$("#btnMqFit").onclick = () => {
  if (!MANNEQUIN.rig) return;
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height);
  mqPaint();
};
$("#btnMqFront").onclick = () => mqView(0, 0.02);
$("#btnMqSide").onclick = () => mqView(Math.PI / 2, 0.02);
function mqView(yaw, pitch) {
  if (!MANNEQUIN.rig) return;
  MANNEQUIN.rig.camera.yaw = yaw;
  MANNEQUIN.rig.camera.pitch = pitch;
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height);
  mqPaint();
}

// --- tableau des dimensions : une ligne par segment (longueur + épaisseur) ---
const MQ_REF = (() => (typeof MannequinKit !== "undefined" ? MannequinKit.morphedDimensions({}) : { lengths: {}, thickness: {} }))();
let mqTableBuilt = false;

function mqBuildTable() {
  const body = $("#mqTableBody");
  if (!body || mqTableBuilt) return;
  mqTableBuilt = true;
  let group = "";
  for (const seg of MQ_KIT.SEGMENTS) {
    if (seg.group !== group) {
      group = seg.group;
      const tr = document.createElement("tr");
      tr.className = "group";
      tr.innerHTML = `<td colspan="4">${esc(group)}</td>`;
      body.appendChild(tr);
    }
    const tr = document.createElement("tr");
    tr.dataset.bone = seg.bone;
    tr.innerHTML = `<td>${esc(seg.label)}</td>
      <td><input type="number" step="0.5" min="1" max="120" data-kind="length" data-bone="${seg.bone}"><span class="unit">cm</span></td>
      <td><input type="number" step="0.5" min="1" max="60" data-kind="thickness" data-bone="${seg.bone}"><span class="unit">cm</span></td>
      <td><button class="ghost mq-link" title="Appliquer cette longueur à l'autre côté">⇄</button></td>`;
    body.appendChild(tr);
  }
  // saisie : on applique la valeur au pantin, la pose ne bouge pas
  body.addEventListener("pointerdown", (ev) => {
    if (ev.target && ev.target.dataset && ev.target.dataset.bone) mqSnapshot();
  });
  body.addEventListener("input", (ev) => {
    const input = ev.target;
    if (!input.dataset || !input.dataset.bone) return;
    mqApplyField(input.dataset.bone, input.dataset.kind, Number(input.value) / 100);
  });
  body.addEventListener("click", (ev) => {
    const btn = ev.target.closest ? ev.target.closest(".mq-link") : null;
    if (!btn) return;
    const bone = btn.closest("tr").dataset.bone;
    mqLinkSides(bone, true);
  });
  mqFillTable();
}

/** Écrit une valeur (mètres) dans le tableau, en signalant les segments modifiés. */
function mqSetField(bone, kind, meters, opts) {
  const input = $(`#mqTableBody input[data-bone="${bone}"][data-kind="${kind}"]`);
  if (!input) return;
  input.value = (meters * 100).toFixed(1);
  const ref = kind === "length" ? MQ_REF.lengths[bone] : MQ_REF.thickness[bone];
  const row = input.closest("tr");
  const modifie = ref !== undefined && Math.abs(meters - ref) > 0.0005;
  if (row) row.classList.toggle("changed", !!modifie || !!(row.dataset.dirty));
  if (modifie && row) row.dataset.dirty = "1";
  if (!opts || opts.info !== false) mqSizeInfo();
}

/** Remplit tout le tableau depuis le pantin. */
function mqFillTable() {
  const rig = MANNEQUIN.rig;
  if (!rig) return;
  const body = $("#mqTableBody");
  if (body) body.querySelectorAll("tr").forEach((tr) => { delete tr.dataset.dirty; tr.classList.remove("changed"); });
  for (const seg of MQ_KIT.SEGMENTS) {
    mqSetField(seg.bone, "length", rig.lengths[seg.bone], { info: false });
    mqSetField(seg.bone, "thickness", rig.thickness[seg.bone], { info: false });
  }
  mqSizeInfo();
}

/** Applique une saisie (longueur ou épaisseur) au pantin, pose conservée. */
let mqSizeTimer = null;
function mqApplyField(bone, kind, meters) {
  const rig = MANNEQUIN.rig;
  if (!rig || !isFinite(meters) || meters <= 0.005) return;
  const patch = {};
  patch[bone] = meters;
  if (kind === "length") rig.setLengths(patch, { keepPose: true });
  else rig.setThickness(patch, { keepPose: true });
  if (kind === "length" && $("#mqSymmetry").checked) {
    const other = bone.endsWith("_l") ? bone.slice(0, -2) + "_r" : bone.endsWith("_r") ? bone.slice(0, -2) + "_l" : "";
    if (other) {
      const p2 = {};
      p2[other] = meters;
      rig.setLengths(p2, { keepPose: true });
      mqSetField(other, kind, meters, { info: false });
      if (kind === "thickness") rig.setThickness(p2, { keepPose: true });
      if (kind === "thickness") mqSetField(other, "thickness", meters, { info: false });
    }
  }
  rig.fitCamera(mqCanvas().width, mqCanvas().height, 1.1);
  clearTimeout(mqSizeTimer);
  mqSizeTimer = setTimeout(mqPaint, 16);            // rendu au fil de la frappe, sans à-coups
  mqSizeInfo();
}

/** Recopie une longueur (ou une épaisseur) sur le côté symétrique. */
function mqLinkSides(bone, both) {
  const rig = MANNEQUIN.rig;
  if (!rig) return;
  const other = bone.endsWith("_l") ? bone.slice(0, -2) + "_r" : bone.endsWith("_r") ? bone.slice(0, -2) + "_l" : "";
  if (!other) return;
  for (const kind of (both ? ["length", "thickness"] : ["length"])) {
    const value = kind === "length" ? rig.lengths[bone] : rig.thickness[bone];
    if (kind === "length") { const p = {}; p[other] = value; rig.setLengths(p, { keepPose: true }); }
    else { const p = {}; p[other] = value; rig.setThickness(p, { keepPose: true }); }
    mqSetField(other, kind, value, { info: false });
  }
  rig.fitCamera(mqCanvas().width, mqCanvas().height, 1.1);
  mqPaint();
  mqSizeInfo();
  setMqStatusOnly(`Dimensions de « ${MQ_JOINTS[other] || other} » alignées sur l'autre côté.`);
}

function setMqStatusOnly(msg) { mqSetStatus(msg); }

/** Hauteur réelle du personnage, pour le résumé du tableau. */
function mqSizeInfo() {
  const rig = MANNEQUIN.rig;
  const el = $("#mqSizeInfo");
  if (!rig || !el) return;
  const debout = MQ_KIT.defaultPose ? MQ_KIT.defaultPose(rig.lengths) : rig.pose;
  const haut = debout.head_top.y - Math.min(debout.ankle_l.y, debout.ankle_r.y);
  const larg = Math.max(debout.shoulder_l.x, debout.shoulder_r.x) - Math.min(debout.shoulder_l.x, debout.shoulder_r.x);
  el.textContent = `Debout : ${haut.toFixed(2).replace(".", ",")} m · épaules ${(larg * 100).toFixed(0)} cm`;
}

$("#mqMorphology").onchange = (e) => {
  if (!MANNEQUIN.rig) return;
  mqSnapshot();
  MANNEQUIN.rig.applyMorphology(e.target.value, { keepPose: true });
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height, 1.1);
  mqFillTable();
  mqPaint();
  mqSetStatus(`Morphologie « ${MQ_BODY_LABELS[e.target.value] || e.target.value} » appliquée — ajustez ensuite le tableau.`);
};
$("#btnMqReset").onclick = () => {
  if (!MANNEQUIN.rig) return;
  mqSnapshot();
  MANNEQUIN.rig.applyMorphology($("#mqMorphology").value || "neutre", { keepPose: true });
  MANNEQUIN.rig.fitCamera(mqCanvas().width, mqCanvas().height, 1.1);
  mqFillTable();
  mqPaint();
  mqSetStatus("Dimensions remises aux valeurs de la morphologie choisie.");
};

// ------------------------------------------------------------- export API
function mqMode() { return $("#mqOutput").value || "volume"; }
function mqSizeValue() {
  const [w, h] = ($("#mqSize").value || "768x1024").split("x").map(Number);
  return { width: w, height: h };
}
function mqPayload(mode, size) {
  const goal = size || mqSizeValue();
  const data = MANNEQUIN.rig.toJSON();
  return {
    pose: data.pose, lengths: data.lengths, thickness: data.thickness,
    mode: mode || mqMode(), width: goal.width, height: goal.height,
    camera: { yaw: data.camera.yaw, pitch: data.camera.pitch, target: [0, MANNEQUIN.rig.camera.target.y, 0] },
  };
}
function mqNeedsRender(mode) {
  const key = JSON.stringify([MANNEQUIN.rig.toJSON(), mode, mqSizeValue()]);
  return !MANNEQUIN.render || MANNEQUIN.renderKey !== key;
}
async function mqRender(mode) {
  const goal = mode || mqMode();
  const payload = mqPayload(goal);
  const r = await postJSON("/api/mannequin/render", payload);
  MANNEQUIN.render = r.control;
  MANNEQUIN.renderKey = JSON.stringify([MANNEQUIN.rig.toJSON(), goal, mqSizeValue()]);
  MANNEQUIN.renderMode = goal;
  const img = $("#mqPreview");
  img.src = r.control.url + "?t=" + Date.now();
  img.classList.remove("hidden");
  img.onclick = () => openLightbox(r.control.url, "Mannequin — " + goal);
  return r.control;
}
$("#btnMqPreview").onclick = async () => {
  const btn = $("#btnMqPreview");
  btn.disabled = true;
  mqSetStatus("Rendu de la pose…");
  try {
    const c = await mqRender();
    mqSetStatus(`Rendu ${c.width}×${c.height} prêt — cliquez sur l'image pour l'agrandir.`, "ok");
  } catch (e) {
    mqSetStatus("⚠️ " + e.message, "warn");
  }
  btn.disabled = false;
};

$("#btnMqControl").onclick = async () => {
  const btn = $("#btnMqControl");
  btn.disabled = true;
  mqSetStatus("Rendu de la pose pour ControlNet…");
  try {
    const mode = mqMode();
    const c = await mqRender(mode);
    mqApplyControl(c, mode);
  } catch (e) {
    mqSetStatus("⚠️ " + e.message, "warn");
  }
  btn.disabled = false;
};

function mqApplyControl(control, mode) {
  const kind = mode === "openpose" ? "pose" : "canny";
  CONTROL.type = kind;
  CONTROL.id = control.id;
  CONTROL.url = control.url + "?t=" + Date.now();
  CONTROL.width = control.width;
  CONTROL.height = control.height;
  CONTROL.persons = [];
  CONTROL.detected = null;
  CONTROL.editor = false;
  CONTROL.sourceUrl = $("#mqPreview").src;
  CONTROL.status = mode === "openpose"
    ? "Pose du mannequin (squelette OpenPose) utilisée comme contrôle."
    : `Rendu « ${mode} » du mannequin utilisé comme contrôle (contours Canny).`;
  if ($("#controlType")) $("#controlType").value = kind;
  renderControl();
  if (!controlCapable()) {
    mqSetStatus("⚠️ Le modèle actif n'utilise pas ControlNet. L'image est prête : utilisez « SD 1.5 + ControlNet » pour "
      + "la pose exacte, ou envoyez-la comme référence avec un modèle d'édition.", "warn");
  } else if (!controlnetPresent(kind)) {
    mqSetStatus("⚠️ Modèle ControlNet manquant : téléchargez-le dans l'onglet Modèles (famille « SD 1.5 + ControlNet »).", "warn");
  } else {
    mqSetStatus("✔ Pose envoyée dans la carte « Personnages, pose et composition » : forcez le contrôle puis cliquez sur Générer.", "ok");
  }
  const card = $("#poseCard");
  if (card && card.scrollIntoView) card.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---------------------------------------------------- génération directe
function mqFillEngines() {
  const sel = $("#mqEngine");
  if (!sel || !STATUS || !STATUS.families) return;
  const cat = controlFamily();
  const items = [];
  if (cat) {
    items.push({ id: cat.id, label: `${cat.name} — pose exacte (ControlNet)`, ready: !!(STATUS.families_status[cat.id] || {}).ready });
  }
  STATUS.families.filter((f) => !f.supports_controlnet).forEach((f) => {
    items.push({ id: f.id, label: `${f.name} — image de référence`, ready: !!(STATUS.families_status[f.id] || {}).ready });
  });
  const current = sel.value;
  sel.innerHTML = "";
  items.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id;
    o.textContent = it.label + (it.ready ? "" : " (fichiers à télécharger)");
    sel.appendChild(o);
  });
  if (items.some((i) => i.id === current)) sel.value = current;
  else if (STATUS.config && items.some((i) => i.id === STATUS.config.family)) sel.value = STATUS.config.family;
  mqEngineHint();
}
$("#mqEngine").onchange = () => mqEngineHint();
function mqEngineHint() {
  const id = $("#mqEngine").value;
  const fam = famById(id);
  const ready = STATUS && STATUS.families_status ? (STATUS.families_status[id] || {}).ready : false;
  const el = $("#mqEngineHint");
  if (!fam) { el.textContent = ""; return; }
  if (fam.supports_controlnet) {
    el.textContent = (ready ? "✔ Fichiers installés." : "⬇ Fichiers manquants : passez par l'onglet Modèles (famille "
      + fam.name + "). ")
      + " La pose du mannequin est envoyée comme contrôle : le résultat suit exactement le squelette. "
      + "Sortie recommandée ci-dessus : « Squelette OpenPose » (ou « mannequin filaire » pour les contours).";
  } else {
    el.textContent = (ready ? "✔ Fichiers installés." : "⬇ Fichiers manquants : passez par l'onglet Modèles (famille "
      + fam.name + "). ")
      + " Ce modèle ne peut pas utiliser ControlNet : le mannequin est envoyé comme image de référence "
      + "avec l'instruction de pose. Sortie recommandée : « Mannequin ombré ».";
  }
}

$("#mqPromptPreset").onchange = (e) => {
  const txt = MQ_PROMPTS[e.target.value];
  if (txt) $("#mqPrompt").value = txt;
};

function mqSd15Size(w, h) {
  const max = Math.max(w, h);
  if (max <= 768) return { width: w, height: h };
  const k = 768 / max;
  return { width: Math.max(512, Math.round(w * k / 64) * 64), height: Math.max(512, Math.round(h * k / 64) * 64) };
}

$("#btnMqGenerate").onclick = async () => {
  const btn = $("#btnMqGenerate");
  const engine = $("#mqEngine").value;
  const fam = famById(engine);
  const prompt = ($("#mqPrompt").value || "").trim();
  if (!fam) { mqSetGenStatus("⚠️ Aucun moteur disponible.", "warn"); return; }
  if (!prompt) { mqSetGenStatus("⚠️ Décrivez l'image souhaitée (par exemple « athlète en tenue de sport, studio »).", "warn"); return; }
  if (MANNEQUIN.pending) { mqSetGenStatus("⏳ Une génération est déjà en cours.", "warn"); return; }
  btn.disabled = true;
  try {
    const mode = mqMode();
    mqSetGenStatus("Préparation du rendu du mannequin…");
    if (fam.supports_controlnet) {
      if (!controlnetPresent(mode === "openpose" ? "pose" : "canny")) {
        throw new Error("modèle ControlNet manquant : téléchargez-le dans l'onglet Modèles (famille « " + fam.name + " »).");
      }
      const c = await mqRender(mode);
      mqApplyControl(c, mode);
      const size = mqSd15Size(c.width, c.height);
      const fd = new FormData();
      fd.append("prompt", prompt);
      fd.append("negative_prompt", "flou, déformé, mauvaise anatomie, membres en trop, texte, filigrane");
      fd.append("width", size.width);
      fd.append("height", size.height);
      fd.append("steps", $("#steps").value);
      fd.append("cfg_scale", $("#cfg").value);
      fd.append("sampler", $("#sampler").value);
      fd.append("seed", $("#seed").value);
      fd.append("mode", "generate");
      fd.append("control_type", mode === "openpose" ? "pose" : "canny");
      fd.append("control_id", c.id);
      fd.append("control_strength", $("#controlStrength").value);
      const cn = controlnetFor(mode === "openpose" ? "pose" : "canny");
      if (cn) fd.append("control_net", cn.id);
      await api("/api/generate", { method: "POST", body: fd });
      mqSetGenStatus(`Génération lancée avec ${fam.name} (${size.width}×${size.height}) : la pose est imposée par le mannequin.`);
    } else {
      const c = await mqRender(mode);
      if (STATUS.config.family !== engine) {
        await postJSON("/api/config", { family: engine });
        await refreshStatus();
        mqFillEngines();
      }
      const size = mqSizeValue();
      const instruction = $("#mqPoseInstruction").checked
        ? "Reproduis exactement la pose du personnage de l'image de référence. " : "";
      const fd = new FormData();
      fd.append("prompt", instruction + prompt);
      fd.append("negative_prompt", "flou, déformé, mauvaise anatomie, membres en trop, texte, filigrane");
      fd.append("width", size.width);
      fd.append("height", size.height);
      fd.append("steps", $("#steps").value);
      fd.append("cfg_scale", $("#cfg").value);
      fd.append("sampler", $("#sampler").value);
      fd.append("seed", $("#seed").value);
      fd.append("mode", "generate");
      fd.append("ref_id", c.id);
      await api("/api/generate", { method: "POST", body: fd });
      mqSetGenStatus(`Génération lancée avec ${fam.name} : le mannequin sert d'image de référence (pose).`);
    }
    MANNEQUIN.pending = true;
    MANNEQUIN.result = "";
    $("#btnMqRef").disabled = true;
    pollGeneration();
  } catch (e) {
    mqSetGenStatus("⚠️ " + e.message, "warn");
  }
  btn.disabled = false;
};

function mqGenerationReady(file) {
  MANNEQUIN.pending = false;
  MANNEQUIN.result = file;
  const box = $("#mqResult");
  box.innerHTML = "";
  const img = document.createElement("img");
  img.src = "/outputs/" + file;
  img.title = "Agrandir";
  img.onclick = () => openLightbox("/outputs/" + file, file);
  box.appendChild(img);
  $("#btnMqRef").disabled = false;
  mqSetGenStatus("✔ Image générée : " + file + ". Ajoutez-la aux références pour vous en servir comme pose de départ.", "ok");
}

$("#btnMqRef").onclick = async () => {
  if (!MANNEQUIN.result) return;
  const btn = $("#btnMqRef");
  btn.disabled = true;
  try {
    const r = await fetch("/outputs/" + MANNEQUIN.result + "?t=" + Date.now());
    const blob = await r.blob();
    const file = new File([blob], MANNEQUIN.result, { type: blob.type || "image/png" });
    const added = addExtraRef(file);
    mqSetGenStatus(added
      ? `📎 « ${MANNEQUIN.result} » ajoutée aux références (champ « Édition d'image »). Passez en ✏️ Modification pour la réutiliser, ou relancez une génération.`
      : "Cette image est déjà dans les références.", added ? "ok" : "warn");
    if (added) {
      const refs = $("#refs");
      if (refs && refs.scrollIntoView) refs.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  } catch (e) {
    mqSetGenStatus("⚠️ " + e.message, "warn");
  }
  btn.disabled = false;
};

mqInit();

// ------------------------------------------------------------- boucle
refreshStatus().then(() => {
  mqFillEngines();
  if (STATUS && STATUS.generation.running) pollGeneration();
});
setInterval(() => {
  if (!STATUS) return;
  const busy = STATUS.jobs.some((j) => j.status === "running");
  const onModels = $("#tab-models") && $("#tab-models").classList.contains("active");
  const onSetup = $("#tab-setup") && $("#tab-setup").classList.contains("active");
  if (busy || onModels || onSetup) refreshStatus();
}, 2000);
