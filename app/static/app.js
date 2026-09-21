const $ = (s) => document.querySelector(s);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
};
const fmtGb = (b) => (b / 1e9).toFixed(2) + " Go";
const fmtTime = (s) => (s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min ${(s % 60).toFixed(0)} s`);

let STATUS = null;
const CAT_LABELS = {
  diffusion: "Modèle de diffusion (GGUF)",
  text_encoder: "Encodeur de texte (GGUF)",
  vae: "VAE",
  vision: "Encodeur de vision (mmproj) — édition d'image",
  lora: "LoRA (optionnel)",
};
const SELECTABLE = ["diffusion", "text_encoder", "vae", "vision"];
const CAT_SHORT = { diffusion: "Diffusion", text_encoder: "Encodeur texte", vae: "VAE", vision: "Vision (édition)" };
const postJSON = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const famById = (id) => STATUS.families.find((f) => f.id === id);
const bundleRunning = (famId) => STATUS.jobs.find((j) => j.kind === "bundle" && j.family === famId && j.status === "running");
// choix de l'utilisateur conservés entre deux rafraîchissements (le DOM est reconstruit toutes les 2 s)
const TIER_CHOICE = {};
const VISION_CHOICE = {};

// ------------------------------------------------------------- onglets
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "gallery") loadGallery();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-goto]");
  if (a) { e.preventDefault(); showTab(a.dataset.goto); }
});

// ------------------------------------------------------------- status
async function refreshStatus() {
  STATUS = await api("/api/status");
  const c = STATUS.config;
  $("#notReady").classList.toggle("hidden", STATUS.ready);
  $("#setupBadge").classList.toggle("hidden", STATUS.ready);
  $("#btnGenerate").disabled = !STATUS.ready || STATUS.generation.running;
  $("#editStatus").textContent = STATUS.edit_ready ? "disponible ✔" : "encodeur de vision manquant";

  // sélecteur de famille
  const fs = $("#family");
  if (!fs.options.length) STATUS.families.forEach((f) => fs.add(new Option(f.name, f.id)));
  [...fs.options].forEach((o) => { const st = STATUS.families_status[o.value]; o.text = famById(o.value).name + (st.ready ? "" : "  (non installé)"); });
  fs.value = c.family;
  $("#familyDesc").textContent = famById(c.family).description;

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
  renderBundles();
  renderQuickDl();
  renderModels();
  renderJobs();
  renderGeneration(STATUS.generation);
}

function renderEngine() {
  const e = STATUS.engine;
  $("#engineInfo").innerHTML = e.installed
    ? `✅ Installé : <code>${e.path}</code> ${e.tag ? `(${e.tag}, ${e.flavor})` : ""}<br>${e.system}`
    : `❌ Non installé — ${e.system} — GPU détecté : <b>${e.detected_flavor}</b>`;
  const f = $("#flavor");
  if (!f.options.length) {
    e.available_flavors.forEach((x) => f.add(new Option(x + (x === e.detected_flavor ? " (détecté)" : ""), x)));
    f.value = e.available_flavors.includes(e.detected_flavor) ? e.detected_flavor : e.available_flavors[0];
  }
}

// ------------------------------------------- téléchargement d'un modèle entier
async function downloadBundle(famId) {
  const tier = TIER_CHOICE[famId] || STATUS.default_tier;
  const body = { family: famId, tier };
  if (famById(famId).edit_requires_vision) body.include_vision = VISION_CHOICE[famId] !== false;
  try {
    await postJSON("/api/download/bundle", body);
  } catch (e) {
    alert(e.message);
  }
  refreshStatus();
}

function renderBundles() {
  const root = $("#bundleCards");
  if (!root) return;
  root.innerHTML = "";

  for (const fam of STATUS.families) {
    const st = STATUS.families_status[fam.id];
    const tiers = STATUS.bundles[fam.id];
    const tier = TIER_CHOICE[fam.id] || STATUS.default_tier;
    const b = tiers[tier] || tiers[STATUS.default_tier];
    const job = bundleRunning(fam.id);
    const active = STATUS.config.family === fam.id;

    const card = document.createElement("div");
    card.className = "bundle" + (st.ready ? " ready" : "");
    card.innerHTML = `
      <div class="bhead"><h4>${fam.name}</h4>
        <span class="st ${st.ready ? "ok" : "ko"}">${st.ready ? "✔ installé" : `${b.missing.length}/${b.files.length} à télécharger`}</span></div>
      <div class="bdesc">${fam.description}</div>
      <label>Qualité
        <select class="tierSel">${STATUS.tiers.map((t) =>
          `<option value="${t.id}" ${t.id === tier ? "selected" : ""}>${t.label} — ${tiers[t.id].total_gb} Go</option>`).join("")}</select>
      </label>
      ${fam.edit_requires_vision ? `<label class="check"><input type="checkbox" class="visSel" ${VISION_CHOICE[fam.id] !== false ? "checked" : ""}>
        Inclure l'édition d'image (encodeur de vision)</label>` : ""}
      <div class="bfiles">${b.files.map((f) => `<div><span>${CAT_SHORT[f.category]} — ${f.id}</span>
        <span class="${f.present ? "have" : ""}">${f.present ? "✔ présent" : `${f.size_gb} Go`}</span></div>`).join("")}</div>`;

    card.querySelector(".tierSel").onchange = (e) => { TIER_CHOICE[fam.id] = e.target.value; renderBundles(); };
    const vis = card.querySelector(".visSel");
    if (vis) vis.onchange = (e) => { VISION_CHOICE[fam.id] = e.target.checked; renderBundles(); };

    if (job) {
      const pct = job.total ? (100 * job.done) / job.total : 0;
      card.insertAdjacentHTML("beforeend", `
        <div class="bbar"><div class="row small"><span>${job.message || "…"}</span><span>${fmtGb(job.done)} / ${fmtGb(job.total)}</span></div>
        <div class="bar"><div style="width:${pct}%"></div></div></div>`);
      const busy = document.createElement("button");
      busy.className = "primary";
      busy.disabled = true;
      busy.textContent = `⏳ Téléchargement… ${pct.toFixed(0)} %`;
      card.appendChild(busy);
    } else {
      const btn = document.createElement("button");
      btn.className = "primary";
      btn.textContent = !b.missing.length ? "⬇ Tout est présent — relancer / autre qualité"
        : st.ready ? `⬇ Ajouter cette qualité (${b.missing_gb} Go)`
        : `⬇ Télécharger ${fam.name} (${b.missing_gb} Go)`;
      btn.onclick = () => downloadBundle(fam.id);
      card.appendChild(btn);
    }
    if (st.ready && !active) {
      const use = document.createElement("button");
      use.className = "ghost";
      use.style.marginTop = "6px";
      use.textContent = "✔ Utiliser ce modèle";
      use.onclick = async () => {
        await postJSON("/api/config", { family: fam.id });
        const d = fam.defaults;
        $("#steps").value = d.steps; $("#stepsVal").textContent = d.steps;
        $("#cfg").value = d.cfg_scale; $("#cfgVal").textContent = Number(d.cfg_scale).toFixed(1);
        $("#sampler").value = d.sampler;
        await refreshStatus();
        showTab("generate");
      };
      card.appendChild(use);
    }
    root.appendChild(card);
  }
}

function renderQuickDl() {
  const box = $("#quickDl");
  if (!box) return;
  const famId = STATUS.config.family;
  const st = STATUS.families_status[famId];
  const b = STATUS.bundles[famId][TIER_CHOICE[famId] || STATUS.default_tier];
  const hidden = !!st.ready || !!bundleRunning(famId);
  box.classList.toggle("hidden", hidden);
  if (hidden) return;
  $("#quickDlName").textContent = famById(famId).name;
  $("#quickDlSize").textContent = `${b.missing_gb} Go`;
}

function renderModels() {
  $("#diskFree").textContent = `— ${STATUS.disk_free_gb} Go libres sur le disque`;
  const root = $("#modelSections");
  // toutes les familles sont dépliées au premier affichage : les boutons « Télécharger » sont visibles d'emblée
  const first = !root.children.length;
  const openFams = new Set([...root.querySelectorAll("details.fam[open]")].map((d) => d.dataset.fam));
  if (first) STATUS.families.forEach((f) => openFams.add(f.id));
  root.innerHTML = "";
  const runningJobs = new Set(STATUS.jobs.filter((j) => j.status === "running").map((j) => j.label));

  for (const fam of STATUS.families) {
    const st = STATUS.families_status[fam.id];
    const det = document.createElement("details");
    det.className = "fam"; det.dataset.fam = fam.id; det.open = openFams.has(fam.id);
    det.innerHTML = `<summary><span>${fam.name} <span class="small">— ${fam.description}</span></span>
      <span class="st ${st.ready ? "ok" : "ko"}">${st.ready ? "✔ prêt" : "fichiers manquants"}${fam.edit_requires_vision ? (st.edit_ready ? " · édition ✔" : " · édition : mmproj manquant") : ""}</span></summary>
      <div class="body"></div>`;
    const body = det.querySelector(".body");
    const sel = STATUS.config.selections[fam.id];

    // raccourci : tout le modèle d'un coup
    const binfo = STATUS.bundles[fam.id][TIER_CHOICE[fam.id] || STATUS.default_tier];
    const bjob = bundleRunning(fam.id);
    const quick = document.createElement("div");
    quick.className = "mrow";
    quick.innerHTML = `<span class="name"><b>Modèle complet</b> — ${binfo.files.length} fichiers (${binfo.total_gb} Go)</span>
      <button class="primary" ${bjob ? "disabled" : ""}>${bjob ? "⏳ en cours" : "⬇ Télécharger le modèle"}</button>`;
    quick.querySelector("button").onclick = () => downloadBundle(fam.id);
    body.appendChild(quick);

    for (const cat of ["diffusion", "text_encoder", "vae", "vision", "lora"]) {
      const local = STATUS.models[fam.id][cat] || [];
      const catalog = fam[cat] || [];
      if (cat === "vision" && !fam.edit_requires_vision && !local.length) continue;
      const localNames = new Set(local.map((m) => m.name));
      const sec = document.createElement("div");
      sec.className = "msec";
      sec.innerHTML = `<h4>${CAT_LABELS[cat]}</h4>`;
      const selectable = SELECTABLE.includes(cat);

      for (const m of local) {
        const row = document.createElement("div");
        row.className = "mrow";
        row.innerHTML = `${selectable ? `<input type="radio" name="sel-${fam.id}-${cat}" ${sel[cat] === m.name ? "checked" : ""}>` : ""}
          <span class="name">${m.name}</span><span class="small">${m.size_gb} Go</span>
          <span class="tag ok">présent</span><button class="ghost" title="Supprimer">🗑</button>`;
        if (selectable) row.querySelector("input").onchange = () => postJSON("/api/config", { selections: { [fam.id]: { [cat]: m.name } } }).then(refreshStatus);
        row.querySelector("button").onclick = async () => {
          if (confirm(`Supprimer ${m.name} ?`)) { await api(`/api/models/${fam.id}/${cat}/${encodeURIComponent(m.name)}`, { method: "DELETE" }); refreshStatus(); }
        };
        sec.appendChild(row);
      }
      for (const item of catalog) {
        if (localNames.has(item.id)) continue;
        const row = document.createElement("div");
        row.className = "mrow";
        const busy = runningJobs.has(`${fam.id}/${cat}/${item.id}`);
        row.innerHTML = `<span class="name">${item.label}</span>${item.recommended ? '<span class="tag rec">recommandé</span>' : ""}
          <button class="${item.recommended ? "primary" : ""}" ${busy ? "disabled" : ""}>${busy ? "⏳ en cours" : "⬇ Télécharger"}</button>`;
        row.querySelector("button").onclick = async () => {
          try { await postJSON("/api/download", { family: fam.id, category: cat, file_id: item.id }); } catch (e) { alert(e.message); }
          refreshStatus();
        };
        sec.appendChild(row);
      }
      const custom = document.createElement("div");
      custom.className = "mrow";
      custom.innerHTML = `<input type="text" placeholder="URL directe d'un fichier .gguf/.safetensors à télécharger dans ce dossier" style="margin:0"><button>⬇</button>`;
      custom.querySelector("button").onclick = async () => {
        const url = custom.querySelector("input").value.trim();
        if (!url) return;
        try { await postJSON("/api/download", { family: fam.id, category: cat, file_id: "", url }); } catch (e) { alert(e.message); }
        refreshStatus();
      };
      sec.appendChild(custom);
      body.appendChild(sec);
    }
    root.appendChild(det);
  }
}

// changement de modèle : applique les réglages par défaut de la famille
$("#family").onchange = async (e) => {
  const fam = famById(e.target.value);
  await postJSON("/api/config", { family: fam.id });
  const d = fam.defaults;
  $("#steps").value = d.steps; $("#stepsVal").textContent = d.steps;
  $("#cfg").value = d.cfg_scale; $("#cfgVal").textContent = Number(d.cfg_scale).toFixed(1);
  $("#sampler").value = d.sampler;
  $("#familyDesc").textContent = fam.description;
  refreshStatus();
};

function renderJobs() {
  const root = $("#jobs");
  if (!STATUS.jobs.length) { root.innerHTML = '<div class="small">Aucun téléchargement.</div>'; return; }
  root.innerHTML = STATUS.jobs.map((j) => {
    const pct = j.total ? (100 * j.done / j.total) : 0;
    const icon = j.status === "done" ? "✅" : j.status === "error" ? "❌" : "⏳";
    const files = (j.files || []).length
      ? `<div class="files">${j.files.map((f) => `<div class="f"><span>${f.id}</span>
          <span>${f.status === "done" ? "✔" : f.status === "error" ? "✖" : f.status === "running" ? `${(100 * f.done / (f.total || 1)).toFixed(0)} %` : "en attente"}</span></div>`).join("")}</div>`
      : "";
    return `<div class="job">${icon} <b>${j.label}</b> — ${j.total ? `${fmtGb(j.done)} / ${fmtGb(j.total)} (${pct.toFixed(0)} %)` : j.message || ""}
      ${j.status === "running" ? `<div class="bar"><div style="width:${pct}%"></div></div>` : `<div class="small">${j.message}</div>`}${files}</div>`;
  }).join("");
}

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

$("#btnGenerate").onclick = async () => {
  const fd = new FormData();
  fd.append("prompt", $("#prompt").value);
  fd.append("negative_prompt", $("#negative").value);
  fd.append("width", $("#width").value);
  fd.append("height", $("#height").value);
  fd.append("steps", $("#steps").value);
  fd.append("cfg_scale", $("#cfg").value);
  fd.append("sampler", $("#sampler").value);
  fd.append("seed", $("#seed").value);
  [...$("#refs").files].forEach((f) => fd.append("ref_images", f));
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
    else { $("#btnGenerate").disabled = !STATUS?.ready; loadGallery(); }
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
      $("#resultMeta").innerHTML = `${m.family ? (famById(m.family)?.name || m.family) + " · " : ""}Seed <b>${m.seed}</b> · ${m.width}×${m.height} · ${m.steps} étapes · CFG ${m.cfg_scale} · ${m.elapsed_s} s
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
      <div class="cap" title="${(m.prompt || "").replace(/"/g, "&quot;")}">${m.prompt || it.file}</div>
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
  await postJSON("/api/engine/install", { flavor: $("#flavor").value });
  refreshStatus();
};
$("#btnQuickDl").onclick = async () => {
  await downloadBundle(STATUS.config.family);
  showTab("setup");
};
$("#btnSavePerf").onclick = async () => {
  await postJSON("/api/config", {
    offload_to_cpu: $("#offload").checked, flash_attention: $("#fa").checked, vae_tiling: $("#vaetiling").checked,
    threads: parseInt($("#threads").value || "-1", 10), extra_args: $("#extra").value,
  });
  $("#btnSavePerf").textContent = "Enregistré ✔"; setTimeout(() => ($("#btnSavePerf").textContent = "Enregistrer"), 1500);
};
$("#btnClearJobs").onclick = () => api("/api/jobs/clear", { method: "POST" }).then(refreshStatus);

// ------------------------------------------------------------- boucle
refreshStatus().then(() => { if (STATUS.generation.running) pollGeneration(); });
setInterval(() => {
  // rafraîchit status si téléchargements en cours ou onglet config affiché
  if (!STATUS) return;
  const busy = STATUS.jobs.some((j) => j.status === "running");
  if (busy || $("#tab-setup").classList.contains("active")) refreshStatus();
}, 2000);
