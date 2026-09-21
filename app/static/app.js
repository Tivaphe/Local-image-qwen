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
  lora: "LoRA (optionnel)",
};
const CAT_ORDER = ["diffusion", "text_encoder", "vae", "vision", "lora"];
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
  let cats = ["diffusion", "text_encoder", "vae"];
  if (fam.edit_requires_vision && !ch.visionOff) cats = cats.concat(["vision"]);
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
    const pill = st.ready && st.edit_ready ? "ok" : st.ready ? "part" : st.file_count ? "part" : "ko";
    const pillTxt = st.ready && st.edit_ready ? "✔ prêt (génération + édition)"
      : st.ready && !st.edit_ready ? "✔ génération · édition : mmproj manquant"
        : st.file_count ? "⚠ incomplet" : "❌ non installé";

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
$("#refs").onchange = () => {
  const box = $("#refPreview");
  box.innerHTML = "";
  [...$("#refs").files].forEach((f) => { const img = document.createElement("img"); img.src = URL.createObjectURL(f); box.appendChild(img); });
};

function showGenerateError(msg) {
  $("#errorBox").textContent = msg;
  $("#errorBox").classList.remove("hidden");
}

$("#btnGenerate").onclick = async () => {
  const refs = [...$("#refs").files];
  if (MODE === "edit" && !refs.length) {
    showGenerateError("Mode édition : ajoutez au moins une image de référence dans la carte « Édition d'image ».");
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

// ------------------------------------------------------------- boucle
refreshStatus().then(() => { if (STATUS && STATUS.generation.running) pollGeneration(); });
setInterval(() => {
  if (!STATUS) return;
  const busy = STATUS.jobs.some((j) => j.status === "running");
  const onModels = $("#tab-models") && $("#tab-models").classList.contains("active");
  const onSetup = $("#tab-setup") && $("#tab-setup").classList.contains("active");
  if (busy || onModels || onSetup) refreshStatus();
}, 2000);
