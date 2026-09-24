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
const postJSON = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const famById = (id) => STATUS.families.find((f) => f.id === id);

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

function renderModels() {
  $("#diskFree").textContent = `— ${STATUS.disk_free_gb} Go libres sur le disque`;
  const root = $("#modelSections");
  const openFams = new Set([...root.querySelectorAll("details.fam[open]")].map((d) => d.dataset.fam));
  if (!root.children.length) openFams.add(STATUS.config.family);
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
    return `<div class="job">${icon} <b>${j.label}</b> — ${j.total ? `${fmtGb(j.done)} / ${fmtGb(j.total)} (${pct.toFixed(0)} %)` : j.message || ""}
      ${j.status === "running" ? `<div class="bar"><div style="width:${pct}%"></div></div>` : `<div class="small">${j.message}</div>`}</div>`;
  }).join("");
}

// ------------------------------------------------------------- formulaire & format
const KNOWN_RATIOS = [
  { label: "1:1", name: "Carré 1:1", ratio: 1.0, w: 1024, h: 1024 },
  { label: "4:3", name: "Paysage 4:3", ratio: 4 / 3, w: 1152, h: 864 },
  { label: "3:4", name: "Portrait 3:4", ratio: 3 / 4, w: 864, h: 1152 },
  { label: "3:2", name: "Paysage 3:2", ratio: 3 / 2, w: 1248, h: 832 },
  { label: "2:3", name: "Portrait 2:3", ratio: 2 / 3, w: 832, h: 1248 },
  { label: "16:9", name: "Paysage 16:9", ratio: 16 / 9, w: 1344, h: 768 },
  { label: "9:16", name: "Portrait 9:16", ratio: 9 / 16, w: 768, h: 1344 },
  { label: "21:9", name: "Cinéma 21:9", ratio: 21 / 9, w: 1536, h: 640 },
  { label: "9:21", name: "Bannière 9:21", ratio: 9 / 21, w: 640, h: 1536 },
];

function getRatioLabel(w, h) {
  if (!w || !h) return "1:1";
  const r = w / h;
  for (const k of KNOWN_RATIOS) {
    if (Math.abs(r - k.ratio) / k.ratio < 0.035) return k.label;
  }
  return r >= 1 ? `${r.toFixed(2)}:1` : `1:${(1 / r).toFixed(2)}`;
}

function calculateOptimalSourceDimensions(origW, origH, targetArea = 1048576) {
  const targetRatio = origW / origH;
  for (const k of KNOWN_RATIOS) {
    if (Math.abs(targetRatio - k.ratio) / k.ratio < 0.025) {
      return { width: k.w, height: k.h, ratioLabel: k.label };
    }
  }
  let bestW = 1024, bestH = 1024, minDiff = Infinity;
  for (let w = 256; w <= 2048; w += 32) {
    let h = Math.round((w / targetRatio) / 32) * 32;
    h = Math.max(256, Math.min(2048, h));
    const area = w * h;
    const r = w / h;
    const diff = Math.abs(Math.log(r / targetRatio)) * 3.0 + Math.abs(Math.log(area / targetArea));
    if (diff < minDiff) {
      minDiff = diff;
      bestW = w;
      bestH = h;
    }
  }
  return { width: bestW, height: bestH, ratioLabel: getRatioLabel(origW, origH) };
}

function calculateExactSourceDimensions(origW, origH) {
  let scale = 1.0;
  if (origW > 2048 || origH > 2048) {
    scale = Math.min(2048 / origW, 2048 / origH);
  } else if (origW < 256 && origH < 256) {
    scale = Math.max(256 / origW, 256 / origH);
  }
  const w = Math.max(256, Math.min(2048, Math.round((origW * scale) / 32) * 32));
  const h = Math.max(256, Math.min(2048, Math.round((origH * scale) / 32) * 32));
  return { width: w, height: h };
}

let isRatioLocked = false;
let lockedRatio = 1.0;
let refItems = [];
let activeSourceId = null;

function getActiveSource() {
  return refItems.find((r) => r.id === activeSourceId) || refItems[0] || null;
}

function updateRatioInfo(w, h) {
  const badge = $("#ratioInfo");
  if (!badge) return;
  const label = getRatioLabel(w, h);
  const mp = ((w * h) / 1e6).toFixed(2);
  badge.textContent = `${label} · ${mp} MP`;
}

function applyDimensions(w, h, updatePresetSelect = true) {
  w = Math.max(256, Math.min(2048, Math.round(w / 32) * 32));
  h = Math.max(256, Math.min(2048, Math.round(h / 32) * 32));
  $("#width").value = w;
  $("#height").value = h;
  lockedRatio = w / h;
  if (updatePresetSelect) {
    syncPreset();
  } else {
    updateRatioInfo(w, h);
  }
  renderRefList();
}

function syncPreset() {
  const w = parseInt($("#width").value, 10);
  const h = parseInt($("#height").value, 10);
  lockedRatio = (w && h) ? (w / h) : 1.0;
  const active = getActiveSource();

  if (active) {
    if (w === active.optimal.width && h === active.optimal.height) {
      $("#preset").value = "source-ratio";
      updateRatioInfo(w, h);
      renderRefList();
      return;
    }
    if (w === active.exact.width && h === active.exact.height) {
      $("#preset").value = "source-exact";
      updateRatioInfo(w, h);
      renderRefList();
      return;
    }
  }

  const v = `${w}x${h}`;
  const exists = [...$("#preset").options].some((o) => o.value === v);
  $("#preset").value = exists ? v : "custom";
  updateRatioInfo(w, h);
  renderRefList();
}

$("#steps").oninput = (e) => ($("#stepsVal").textContent = e.target.value);
$("#cfg").oninput = (e) => ($("#cfgVal").textContent = Number(e.target.value).toFixed(1));

$("#preset").onchange = (e) => {
  const val = e.target.value;
  if (val === "custom") return;
  const active = getActiveSource();
  if (val === "source-ratio" && active) {
    applyDimensions(active.optimal.width, active.optimal.height, false);
    return;
  }
  if (val === "source-exact" && active) {
    applyDimensions(active.exact.width, active.exact.height, false);
    return;
  }
  const parts = val.split("x").map(Number);
  if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
    applyDimensions(parts[0], parts[1], false);
  }
};

$("#btnSwapDims").onclick = () => {
  const w = parseInt($("#width").value, 10);
  const h = parseInt($("#height").value, 10);
  $("#width").value = h;
  $("#height").value = w;
  lockedRatio = h / w;
  syncPreset();
};

$("#btnLockRatio").onclick = () => {
  isRatioLocked = !isRatioLocked;
  $("#btnLockRatio").classList.toggle("active", isRatioLocked);
  $("#btnLockRatio").textContent = isRatioLocked ? "🔒" : "🔓";
  $("#btnLockRatio").title = isRatioLocked ? "Ratio verrouillé (cliquer pour déverrouiller)" : "Verrouiller le ratio";
  if (isRatioLocked) {
    const w = parseInt($("#width").value, 10) || 1024;
    const h = parseInt($("#height").value, 10) || 1024;
    lockedRatio = w / h;
  }
};

$("#width").oninput = $("#width").onchange = (e) => {
  const w = parseInt(e.target.value, 10);
  if (isRatioLocked && w && lockedRatio) {
    const newH = Math.max(256, Math.min(2048, Math.round((w / lockedRatio) / 32) * 32));
    $("#height").value = newH;
  }
  syncPreset();
};

$("#height").oninput = $("#height").onchange = (e) => {
  const h = parseInt(e.target.value, 10);
  if (isRatioLocked && h && lockedRatio) {
    const newW = Math.max(256, Math.min(2048, Math.round((h * lockedRatio) / 32) * 32));
    $("#width").value = newW;
  }
  syncPreset();
};

// ------------------------------------------------------------- gestion des références & drop
function updateSourcePresetGroup() {
  const group = $("#sourcePresetGroup");
  const active = getActiveSource();
  if (!active) {
    group.classList.add("hidden");
    return;
  }
  group.classList.remove("hidden");
  $("#optSourceRatio").textContent = `📐 Conserver le ratio source (~1 MP : ${active.optimal.width}×${active.optimal.height})`;
  $("#optSourceExact").textContent = `📏 Dimensions source d'origine (${active.exact.width}×${active.exact.height})`;
}

function renderRefList() {
  const list = $("#refPreview");
  const controls = $("#refControls");
  list.innerHTML = "";
  if (!refItems.length) {
    controls.classList.add("hidden");
    updateSourcePresetGroup();
    return;
  }
  controls.classList.remove("hidden");
  updateSourcePresetGroup();

  const currentW = parseInt($("#width").value, 10);
  const currentH = parseInt($("#height").value, 10);
  const active = getActiveSource();

  refItems.forEach((item) => {
    const card = document.createElement("div");
    const isActiveSource = active && active.id === item.id;
    card.className = `ref-card ${isActiveSource ? "active-format" : ""}`;

    const isOptActive = isActiveSource && currentW === item.optimal.width && currentH === item.optimal.height;
    const isExactActive = isActiveSource && currentW === item.exact.width && currentH === item.exact.height;

    card.innerHTML = `
      <img src="${item.url}" class="ref-thumb" alt="${item.file.name}">
      <div class="ref-info">
        <div class="ref-name" title="${item.file.name}">${item.file.name}</div>
        <div class="ref-meta">
          <span>${item.origW} × ${item.origH} px</span>
          <span class="badge-ratio-mini">${item.ratioLabel}</span>
          ${isActiveSource ? '<span class="tag ok" style="font-size:10px;padding:1px 5px">Format actif</span>' : ""}
        </div>
        <div class="ref-actions">
          <button type="button" class="btn-ref-opt btn-apply-opt ${isOptActive ? "active" : ""}" title="Adapter au ratio source (~1 MP, aligné 32px)">📐 Ratio adapté (${item.optimal.width}×${item.optimal.height})</button>
          <button type="button" class="btn-ref-opt btn-apply-exact ${isExactActive ? "active" : ""}" title="Conserver les dimensions d'origine (arrondies à 32px)">📏 Taille originale (${item.exact.width}×${item.exact.height})</button>
        </div>
      </div>
      <button type="button" class="btn-ref-remove" title="Supprimer cette référence">✕</button>
    `;

    card.querySelector(".btn-apply-opt").onclick = () => {
      activeSourceId = item.id;
      applyDimensions(item.optimal.width, item.optimal.height, false);
      $("#preset").value = "source-ratio";
      updateSourcePresetGroup();
      renderRefList();
    };

    card.querySelector(".btn-apply-exact").onclick = () => {
      activeSourceId = item.id;
      applyDimensions(item.exact.width, item.exact.height, false);
      $("#preset").value = "source-exact";
      updateSourcePresetGroup();
      renderRefList();
    };

    card.querySelector(".btn-ref-remove").onclick = () => {
      removeRefItem(item.id);
    };

    list.appendChild(card);
  });
}

function removeRefItem(id) {
  const idx = refItems.findIndex((it) => it.id === id);
  if (idx !== -1) {
    URL.revokeObjectURL(refItems[idx].url);
    refItems.splice(idx, 1);
  }
  if (activeSourceId === id) {
    activeSourceId = refItems[0]?.id || null;
  }
  if (!refItems.length) {
    const curPreset = $("#preset").value;
    if (curPreset === "source-ratio" || curPreset === "source-exact") {
      $("#preset").value = "1024x1024";
      applyDimensions(1024, 1024);
    }
  }
  renderRefList();
}

function handleNewFiles(files) {
  const fileArray = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name));
  if (!fileArray.length) return;

  let loadedCount = 0;
  const isFirstUpload = refItems.length === 0;

  fileArray.forEach((file) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const origW = img.naturalWidth;
      const origH = img.naturalHeight;
      const optimal = calculateOptimalSourceDimensions(origW, origH);
      const exact = calculateExactSourceDimensions(origW, origH);
      const ratioLabel = getRatioLabel(origW, origH);
      const item = {
        id: Math.random().toString(36).slice(2) + Date.now().toString(36),
        file,
        url,
        origW,
        origH,
        ratioLabel,
        optimal,
        exact,
      };
      refItems.push(item);
      if (!activeSourceId) {
        activeSourceId = item.id;
      }
      loadedCount++;
      if (loadedCount === fileArray.length) {
        if (isFirstUpload && $("#autoAdaptFormat")?.checked) {
          const first = getActiveSource();
          if (first) {
            applyDimensions(first.optimal.width, first.optimal.height, false);
            $("#preset").value = "source-ratio";
          }
        }
        renderRefList();
      }
    };
    img.onerror = () => {
      loadedCount++;
      if (loadedCount === fileArray.length) renderRefList();
    };
    img.src = url;
  });
}

// Zone de glisser-déposer
const dropZone = $("#dropZone");
if (dropZone) {
  dropZone.onclick = (e) => {
    if (e.target.id !== "refs") $("#refs").click();
  };
  dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  };
  dropZone.ondragleave = () => {
    dropZone.classList.remove("dragover");
  };
  dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) {
      handleNewFiles(e.dataTransfer.files);
    }
  };
}

$("#refs").onchange = (e) => {
  if (e.target.files?.length) {
    handleNewFiles(e.target.files);
    e.target.value = "";
  }
};

$("#btnClearRefs").onclick = () => {
  refItems.forEach((it) => URL.revokeObjectURL(it.url));
  refItems = [];
  activeSourceId = null;
  const curPreset = $("#preset").value;
  if (curPreset === "source-ratio" || curPreset === "source-exact") {
    $("#preset").value = "1024x1024";
    applyDimensions(1024, 1024);
  }
  renderRefList();
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
  refItems.forEach((it) => fd.append("ref_images", it.file));
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
      <div class="tools">
        <button class="btn-reuse" title="Réutiliser le prompt et les réglages">↺ Réutiliser</button>
        <button class="btn-edit" title="Utiliser comme image de référence pour la modifier">🎨 Éditer</button>
        <a href="/outputs/${it.file}" download><button title="Télécharger">⬇</button></a>
        <button class="btn-del" title="Supprimer">🗑</button>
      </div>`;
    d.querySelector("img").onclick = () => openLightbox(`/outputs/${it.file}`, m.prompt || it.file);
    const reuse = d.querySelector(".btn-reuse");
    const editBtn = d.querySelector(".btn-edit");
    const del = d.querySelector(".btn-del");

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

    editBtn.onclick = async () => {
      try {
        const res = await fetch(`/outputs/${it.file}`);
        const blob = await res.blob();
        const file = new File([blob], it.file, { type: blob.type || "image/png" });
        if (m.family && m.family !== STATUS.config.family && famById(m.family)) {
          await postJSON("/api/config", { family: m.family });
          await refreshStatus();
        }
        if (m.prompt) $("#prompt").value = m.prompt;
        if (m.negative_prompt != null) $("#negative").value = m.negative_prompt;
        showTab("generate");
        handleNewFiles([file]);
      } catch (err) {
        console.error("Erreur lors du chargement de l'image de galerie:", err);
      }
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
