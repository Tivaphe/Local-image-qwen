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
window.__liq = { control: CONTROL, applyPoseEdits, drawPoseEditor };
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

// ------------------------------------------------------------- boucle
refreshStatus().then(() => { if (STATUS && STATUS.generation.running) pollGeneration(); });
setInterval(() => {
  if (!STATUS) return;
  const busy = STATUS.jobs.some((j) => j.status === "running");
  const onModels = $("#tab-models") && $("#tab-models").classList.contains("active");
  const onSetup = $("#tab-setup") && $("#tab-setup").classList.contains("active");
  if (busy || onModels || onSetup) refreshStatus();
}, 2000);
