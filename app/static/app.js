const $ = (s) => document.querySelector(s);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
};

const fmtGb = (b) => {
  if (b == null || isNaN(b) || b <= 0) return "0 Go";
  if (b < 1e8) return `${(b / 1e6).toFixed(0)} Mo`;
  return `${(b / 1e9).toFixed(2)} Go`;
};

const fmtSpeed = (bps) => {
  if (!bps || bps <= 0) return "";
  if (bps < 1e6) return `${(bps / 1e3).toFixed(0)} Ko/s`;
  return `${(bps / 1e6).toFixed(1)} Mo/s`;
};

const fmtTime = (s) => {
  if (s == null || isNaN(s) || s < 0) return "";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${m} min ${rem} s`;
};

let STATUS = null;
const CAT_LABELS = {
  diffusion: "Modèle de diffusion (GGUF)",
  text_encoder: "Encodeur de texte (GGUF)",
  vae: "VAE (.safetensors / .gguf)",
  vision: "Encodeur de vision (mmproj) — édition d'image",
  lora: "LoRA (optionnel)",
};
const SELECTABLE = ["diffusion", "text_encoder", "vae", "vision"];
const postJSON = (url, body) =>
  api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const famById = (id) => STATUS.families.find((f) => f.id === id);

// ------------------------------------------------------------- Navigation & Onglets
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "gallery") loadGallery();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-goto]");
  if (a) {
    e.preventDefault();
    showTab(a.dataset.goto);
  }
  const openFam = e.target.closest("[data-open-fam]");
  if (openFam) {
    e.preventDefault();
    showTab("setup");
    setTimeout(() => {
      const det = document.querySelector(`details.fam[data-fam="${openFam.dataset.openFam}"]`);
      if (det) {
        det.open = true;
        det.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 100);
  }
});

// ------------------------------------------------------------- Rafraîchissement de l'état
async function refreshStatus() {
  try {
    STATUS = await api("/api/status");
  } catch (err) {
    console.error("Erreur api status:", err);
    return;
  }
  const c = STATUS.config;
  const runningJobs = STATUS.jobs.filter((j) => j.status === "running");
  const hasRunningJobs = runningJobs.length > 0;

  // Badge configuration
  const setupBadge = $("#setupBadge");
  if (hasRunningJobs) {
    setupBadge.textContent = `⏳ ${runningJobs.length}`;
    setupBadge.classList.remove("hidden");
  } else if (!STATUS.ready) {
    setupBadge.textContent = "!";
    setupBadge.classList.remove("hidden");
  } else {
    setupBadge.classList.add("hidden");
  }

  // Alerte moteur manquant sur l'onglet Générer
  const engineInstalled = STATUS.engine && STATUS.engine.installed;
  const engineNotice = $("#engineNotice");
  if (engineNotice) engineNotice.classList.toggle("hidden", !!engineInstalled);

  // Bouton générer
  $("#btnGenerate").disabled = !STATUS.ready || STATUS.generation.running;
  $("#editStatus").textContent = STATUS.edit_ready ? "disponible ✔" : "encodeur de vision (mmproj) manquant";

  // Sélecteur de modèle / famille
  const fs = $("#family");
  if (!fs.options.length) {
    STATUS.families.forEach((f) => fs.add(new Option(f.name, f.id)));
  }
  [...fs.options].forEach((o) => {
    const st = STATUS.families_status[o.value];
    const tag = st.is_downloading ? " ⏳ (téléchargement…)" : st.ready ? " ✔" : " (non installé)";
    o.text = famById(o.value).name + tag;
  });
  fs.value = c.family;
  $("#familyDesc").textContent = famById(c.family).description;

  // Rendu de la carte d'état sur l'onglet Générer
  renderModelStatusCard(c.family);

  // Sampler
  const sel = $("#sampler");
  if (!sel.options.length) STATUS.samplers.forEach((s) => sel.add(new Option(s, s)));
  if (!window.__formInit) {
    sel.value = c.sampler || "euler";
    $("#width").value = c.width;
    $("#height").value = c.height;
    $("#steps").value = c.steps;
    $("#stepsVal").textContent = c.steps;
    $("#cfg").value = c.cfg_scale;
    $("#cfgVal").textContent = Number(c.cfg_scale).toFixed(1);
    $("#negative").value = c.negative_prompt || "";
    syncPreset();
    window.__formInit = true;
  }

  // Performances
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

// ------------------------------------------------------------- Carte statut modèle (Onglet Générer)
function renderModelStatusCard(familyId) {
  const card = $("#modelStatusCard");
  if (!card) return;
  const fam = famById(familyId);
  const st = STATUS.families_status[familyId];
  if (!fam || !st) {
    card.innerHTML = "";
    return;
  }

  if (st.is_downloading) {
    card.className = "model-status-box dl";
    const doneBytes = st.running_jobs.reduce((a, b) => a + (b.done || 0), 0);
    const totalBytes = st.running_jobs.reduce((a, b) => a + (b.total || 0), 0);
    const pct = totalBytes > 0 ? (100 * doneBytes) / totalBytes : 0;
    const speed = st.running_jobs.reduce((a, b) => a + (b.speed || 0), 0);
    const maxEta = Math.max(0, ...st.running_jobs.map((j) => j.eta || 0));

    card.innerHTML = `
      <div class="title">
        <span>⏳ Téléchargement de <b>${fam.name}</b> en cours… (${pct.toFixed(0)} %)</span>
        <span class="small">${fmtSpeed(speed)} ${maxEta > 0 ? `· restant : ~${fmtTime(maxEta)}` : ""}</span>
      </div>
      <div class="bar sm"><div style="width:${pct}%"></div></div>
      <div class="small" style="margin-top:4px">${fmtGb(doneBytes)} téléchargés sur ${fmtGb(totalBytes || st.recommended_total_gb * 1e9)}</div>
      <div class="actions">
        <button class="ghost small-btn" id="btnCancelFamDl">■ Annuler le téléchargement</button>
        <button class="ghost small-btn" data-goto="setup">Voir les détails dans Configuration</button>
      </div>`;

    const cancelBtn = card.querySelector("#btnCancelFamDl");
    if (cancelBtn) {
      cancelBtn.onclick = async () => {
        for (const j of st.running_jobs) {
          await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }).catch(() => {});
        }
        refreshStatus();
      };
    }
  } else if (!st.ready) {
    card.className = "model-status-box warn";
    card.innerHTML = `
      <div class="title">
        <span>⚠️ Ce modèle n'est pas encore installé sur votre machine.</span>
        <span class="badge" style="background:#ffb648">Non installé</span>
      </div>
      <p class="hint" style="margin:4px 0 8px">Fichiers manquants : <b>${st.missing.join(", ")}</b>. Vous pouvez télécharger le pack recommandé complet en un clic.</p>
      <div class="actions">
        <button class="primary" id="btnQuickDlPack">📥 Télécharger le modèle complet (~${st.recommended_total_gb} Go)</button>
        <button class="ghost small-btn" data-open-fam="${fam.id}">⚙️ Choisir manuellement les variantes</button>
      </div>`;

    const dlBtn = card.querySelector("#btnQuickDlPack");
    if (dlBtn) {
      dlBtn.onclick = async () => {
        dlBtn.disabled = true;
        dlBtn.textContent = "⏳ Lancement du téléchargement…";
        try {
          await postJSON("/api/download-family", { family: fam.id, include_vision: true });
        } catch (e) {
          alert("Erreur: " + e.message);
        }
        refreshStatus();
      };
    }
  } else {
    card.className = "model-status-box ok";
    const editNotice = (!st.edit_ready && fam.edit_requires_vision)
      ? `<div class="small" style="color:var(--warn);margin-top:4px">⚠️ Édition d'image désactivée : l'encodeur de vision (mmproj) est manquant. <a href="#" data-open-fam="${fam.id}">Télécharger mmproj</a></div>`
      : "";

    card.innerHTML = `
      <div class="row">
        <span>✔ <b>${fam.name}</b> est prêt à générer des images.</span>
        <button class="ghost small-btn" data-open-fam="${fam.id}">⚙️ Gérer les fichiers & quantifications</button>
      </div>
      ${editNotice}`;
  }
}

// ------------------------------------------------------------- Moteur (stable-diffusion.cpp)
function renderEngine() {
  const e = STATUS.engine;
  $("#engineInfo").innerHTML = e.installed
    ? `✅ <b>Moteur prêt</b> : <code>${e.path}</code> ${e.tag ? `(${e.tag}, variante <b>${e.flavor}</b>)` : ""}<br><span class="small">${e.system}</span>`
    : `❌ <b>Moteur non installé</b> — ${e.system} — GPU détecté : <b>${e.detected_flavor}</b>`;

  const f = $("#flavor");
  if (!f.options.length) {
    e.available_flavors.forEach((x) =>
      f.add(new Option(x + (x === e.detected_flavor ? " (recommandé)" : ""), x))
    );
    f.value = e.available_flavors.includes(e.detected_flavor) ? e.detected_flavor : e.available_flavors[0];
  }
}

// ------------------------------------------------------------- Modèles (Onglet Configuration)
function renderModels() {
  $("#diskFree").textContent = `💾 ${STATUS.disk_free_gb} Go libres sur le disque`;

  // Sélecteur personnalisé
  const customFam = $("#customFam");
  if (customFam && !customFam.options.length) {
    STATUS.families.forEach((f) => customFam.add(new Option(f.name, f.id)));
  }

  const root = $("#modelSections");
  const openFams = new Set([...root.querySelectorAll("details.fam[open]")].map((d) => d.dataset.fam));
  if (!root.children.length) openFams.add(STATUS.config.family);
  root.innerHTML = "";

  const runningJobLabels = new Set(
    STATUS.jobs.filter((j) => j.status === "running").map((j) => j.label)
  );

  for (const fam of STATUS.families) {
    const st = STATUS.families_status[fam.id];
    const det = document.createElement("details");
    det.className = "fam";
    det.dataset.fam = fam.id;
    det.open = openFams.has(fam.id);

    const statusBadgeClass = st.is_downloading ? "dl" : st.ready ? "ok" : "ko";
    const statusText = st.is_downloading
      ? `⏳ Téléchargement (${(st.download_progress * 100).toFixed(0)} %)`
      : st.ready
      ? "✔ Prêt"
      : "❌ Fichiers manquants";

    det.innerHTML = `
      <summary>
        <span>${fam.name} <span class="small">— ${fam.description}</span></span>
        <div class="fam-header-actions">
          <span class="st ${statusBadgeClass}">${statusText}</span>
        </div>
      </summary>
      <div class="body">
        <div class="fam-banner">
          <div>
            <b>Pack complet recommandé (${st.recommended_total_gb} Go)</b>
            <div class="small">Installe en 1 clic le modèle de diffusion optimal, l'encodeur de texte, le VAE${fam.edit_requires_vision ? " et le mmproj" : ""}.</div>
          </div>
          <button class="${st.has_recommended ? "ghost" : "primary"} btnDlPack" ${st.is_downloading ? "disabled" : ""}>
            ${st.is_downloading ? "⏳ Téléchargement en cours…" : st.has_recommended ? "✔ Pack complet déjà installé" : `📥 Télécharger le pack complet (~${st.recommended_total_gb} Go)`}
          </button>
        </div>
        <div class="categories-container"></div>
      </div>`;

    const body = det.querySelector(".categories-container");
    const dlPackBtn = det.querySelector(".btnDlPack");
    if (dlPackBtn && !st.is_downloading && !st.has_recommended) {
      dlPackBtn.onclick = async () => {
        dlPackBtn.disabled = true;
        dlPackBtn.textContent = "⏳ Démarrage…";
        try {
          await postJSON("/api/download-family", { family: fam.id, include_vision: true });
        } catch (e) {
          alert(e.message);
        }
        refreshStatus();
      };
    }

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

      // Fichiers présents
      for (const m of local) {
        const row = document.createElement("div");
        row.className = "mrow";
        row.innerHTML = `
          ${selectable ? `<input type="radio" name="sel-${fam.id}-${cat}" ${sel[cat] === m.name ? "checked" : ""} title="Activer pour la génération">` : ""}
          <span class="name"><b>${m.name}</b> <span class="small">(${m.size_gb} Go)</span></span>
          <span class="tag ok">installé</span>
          <button class="ghost small-btn" title="Supprimer ce fichier">🗑</button>`;

        if (selectable) {
          row.querySelector("input").onchange = () =>
            postJSON("/api/config", { selections: { [fam.id]: { [cat]: m.name } } }).then(refreshStatus);
        }
        row.querySelector("button").onclick = async () => {
          if (confirm(`Supprimer le fichier ${m.name} ?`)) {
            await api(`/api/models/${fam.id}/${cat}/${encodeURIComponent(m.name)}`, { method: "DELETE" });
            refreshStatus();
          }
        };
        sec.appendChild(row);
      }

      // Fichiers du catalogue à télécharger
      for (const item of catalog) {
        if (localNames.has(item.id)) continue;
        const row = document.createElement("div");
        row.className = "mrow";
        const jobLabel = `${fam.id}/${cat}/${item.id}`;
        const isBusy = runningJobLabels.has(jobLabel);

        row.innerHTML = `
          <span class="name">${item.label}</span>
          ${item.recommended ? '<span class="tag rec">recommandé</span>' : ""}
          <button class="${item.recommended ? "primary" : ""} small-btn" ${isBusy ? "disabled" : ""}>
            ${isBusy ? "⏳ en cours…" : "⬇ Télécharger"}
          </button>`;

        row.querySelector("button").onclick = async () => {
          try {
            await postJSON("/api/download", { family: fam.id, category: cat, file_id: item.id });
          } catch (e) {
            alert(e.message);
          }
          refreshStatus();
        };
        sec.appendChild(row);
      }

      body.appendChild(sec);
    }
    root.appendChild(det);
  }
}

// Changement de modèle : applique les réglages par défaut de la famille
$("#family").onchange = async (e) => {
  const fam = famById(e.target.value);
  await postJSON("/api/config", { family: fam.id });
  const d = fam.defaults;
  $("#steps").value = d.steps;
  $("#stepsVal").textContent = d.steps;
  $("#cfg").value = d.cfg_scale;
  $("#cfgVal").textContent = Number(d.cfg_scale).toFixed(1);
  $("#sampler").value = d.sampler;
  $("#familyDesc").textContent = fam.description;
  refreshStatus();
};

// ------------------------------------------------------------- Liste des Téléchargements
function renderJobs() {
  const root = $("#jobs");
  if (!STATUS.jobs.length) {
    root.innerHTML = '<div class="small" style="padding:8px 0">Aucun téléchargement dans l\'historique.</div>';
    return;
  }

  root.innerHTML = STATUS.jobs
    .map((j) => {
      const isRunning = j.status === "running";
      const pct = j.total ? (100 * j.done) / j.total : 0;
      const icon = j.status === "done" ? "✅" : j.status === "error" ? "❌" : j.status === "cancelled" ? "⏹" : "⏳";
      const speedTxt = isRunning && j.speed > 0 ? ` · <b>${fmtSpeed(j.speed)}</b>` : "";
      const etaTxt = isRunning && j.eta > 0 ? ` · restant : ~${fmtTime(j.eta)}` : "";

      return `
        <div class="job ${j.status}">
          <div class="row">
            <span>${icon} <b>${j.label}</b></span>
            ${isRunning ? `<button class="ghost small-btn" data-cancel-job="${j.id}" title="Annuler">✕ Annuler</button>` : `<span class="small">${j.status === "done" ? "Terminé ✔" : j.message || ""}</span>`}
          </div>
          ${isRunning ? `
            <div class="bar sm"><div style="width:${pct}%"></div></div>
            <div class="meta">
              <span>${j.total ? `${fmtGb(j.done)} / ${fmtGb(j.total)} (${pct.toFixed(0)} %)` : `${fmtGb(j.done)}`}</span>
              <span>${speedTxt}${etaTxt}</span>
            </div>` : ""}
        </div>`;
    })
    .join("");

  root.querySelectorAll("[data-cancel-job]").forEach((btn) => {
    btn.onclick = async () => {
      btn.disabled = true;
      btn.textContent = "…";
      await api(`/api/jobs/${btn.dataset.cancelJob}/cancel`, { method: "POST" }).catch(() => {});
      refreshStatus();
    };
  });
}

// ------------------------------------------------------------- Téléchargement Personnalisé (URL / Hugging Face)
$("#btnCustomDownload").onclick = async () => {
  const family = $("#customFam").value;
  const category = $("#customCat").value;
  const url = $("#customUrl").value.trim();
  const file_id = $("#customFilename").value.trim();

  if (!url) {
    alert("Veuillez saisir une URL de fichier à télécharger.");
    return;
  }

  const btn = $("#btnCustomDownload");
  btn.disabled = true;
  btn.textContent = "⏳ Démarrage…";

  try {
    await postJSON("/api/download", { family, category, file_id, url });
    $("#customUrl").value = "";
    $("#customFilename").value = "";
    refreshStatus();
    showTab("setup");
  } catch (e) {
    alert("Erreur lors du téléchargement : " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "📥 Lancer le téléchargement";
  }
};

// ------------------------------------------------------------- Boutons Moteur & Config
$("#btnEngine").onclick = async () => {
  const btn = $("#btnEngine");
  btn.disabled = true;
  btn.textContent = "⏳ Installation en cours…";
  try {
    await postJSON("/api/engine/install", { flavor: $("#flavor").value });
  } catch (e) {
    alert("Erreur : " + e.message);
  }
  refreshStatus();
  btn.disabled = false;
  btn.textContent = "⬇ Installer / mettre à jour le moteur";
};

const btnQuickEngine = $("#btnQuickEngine");
if (btnQuickEngine) {
  btnQuickEngine.onclick = () => $("#btnEngine").click();
}

$("#btnSavePerf").onclick = async () => {
  await postJSON("/api/config", {
    offload_to_cpu: $("#offload").checked,
    flash_attention: $("#fa").checked,
    vae_tiling: $("#vaetiling").checked,
    threads: parseInt($("#threads").value || "-1", 10),
    extra_args: $("#extra").value,
  });
  $("#btnSavePerf").textContent = "Enregistré ✔";
  setTimeout(() => ($("#btnSavePerf").textContent = "Enregistrer les réglages"), 1500);
};

$("#btnClearJobs").onclick = () => api("/api/jobs/clear", { method: "POST" }).then(refreshStatus);

// ------------------------------------------------------------- Formulaire de Génération
$("#steps").oninput = (e) => ($("#stepsVal").textContent = e.target.value);
$("#cfg").oninput = (e) => ($("#cfgVal").textContent = Number(e.target.value).toFixed(1));
$("#preset").onchange = (e) => {
  if (e.target.value === "custom") return;
  const [w, h] = e.target.value.split("x");
  $("#width").value = w;
  $("#height").value = h;
};
function syncPreset() {
  const v = `${$("#width").value}x${$("#height").value}`;
  $("#preset").value = [...$("#preset").options].some((o) => o.value === v) ? v : "custom";
}
$("#width").onchange = $("#height").onchange = syncPreset;
$("#refs").onchange = () => {
  const box = $("#refPreview");
  box.innerHTML = "";
  [...$("#refs").files].forEach((f) => {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    box.appendChild(img);
  });
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
    else {
      $("#btnGenerate").disabled = !STATUS?.ready;
      loadGallery();
    }
  });
}

function renderGeneration(g) {
  const box = $("#progressBox");
  $("#btnCancel").classList.toggle("hidden", !g.running);
  if (g.running || g.error || g.result) box.classList.remove("hidden");
  $("#phase").textContent = g.phase || "";
  $("#elapsed").textContent = g.elapsed ? fmtTime(g.elapsed) : "";
  $("#bar").style.width = 100 * (g.progress || 0) + "%";
  $("#stepTxt").textContent = g.steps ? `étape ${g.step}/${g.steps}` : "";
  if (g.running && g.step > 1 && g.progress > 0) {
    const eta = g.elapsed / g.progress - g.elapsed;
    $("#eta").textContent = `~${fmtTime(Math.max(0, eta))} restantes`;
  } else $("#eta").textContent = "";
  const log = $("#log");
  log.textContent = (g.log || []).join("\n");
  log.scrollTop = log.scrollHeight;
  if (g.error) {
    $("#errorBox").textContent = g.error;
    $("#errorBox").classList.remove("hidden");
  }
  if (g.result && $("#resultBox").dataset.file !== g.result) {
    $("#resultBox").dataset.file = g.result;
    $("#resultBox").innerHTML = `<img src="/outputs/${g.result}?t=${Date.now()}" alt="">`;
    $("#resultBox img").onclick = () => openLightbox(`/outputs/${g.result}`, g.result);
    fetch(`/outputs/${g.result.replace(".png", ".json")}`)
      .then((r) => r.json())
      .then((m) => {
        $("#resultMeta").innerHTML = `${m.family ? (famById(m.family)?.name || m.family) + " · " : ""}Seed <b>${m.seed}</b> · ${m.width}×${m.height} · ${m.steps} étapes · CFG ${m.cfg_scale} · ${m.elapsed_s} s
          <button class="ghost small-btn" id="reuseSeed">↺ réutiliser la seed</button>`;
        $("#reuseSeed").onclick = () => ($("#seed").value = m.seed);
      })
      .catch(() => {});
  }
}

// ------------------------------------------------------------- Galerie
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
      if (m.family && m.family !== STATUS.config.family && famById(m.family)) {
        await postJSON("/api/config", { family: m.family });
        await refreshStatus();
      }
      if (m.prompt) $("#prompt").value = m.prompt;
      if (m.negative_prompt != null) $("#negative").value = m.negative_prompt;
      if (m.seed != null) $("#seed").value = m.seed;
      if (m.steps) {
        $("#steps").value = m.steps;
        $("#stepsVal").textContent = m.steps;
      }
      if (m.cfg_scale) {
        $("#cfg").value = m.cfg_scale;
        $("#cfgVal").textContent = Number(m.cfg_scale).toFixed(1);
      }
      if (m.width) $("#width").value = m.width;
      if (m.height) $("#height").value = m.height;
      if (m.sampler) $("#sampler").value = m.sampler;
      syncPreset();
      showTab("generate");
    };
    del.onclick = async () => {
      if (confirm("Supprimer cette image ?")) {
        await api(`/api/gallery/${it.file}`, { method: "DELETE" });
        loadGallery();
      }
    };
    root.appendChild(d);
  }
}
$("#btnOpenOutputs").onclick = () => {
  const fd = new FormData();
  fd.append("which", "outputs");
  api("/api/open-folder", { method: "POST", body: fd }).then((r) => console.log(r.path));
};

function openLightbox(src, cap) {
  $("#lightboxImg").src = src;
  $("#lightboxCap").textContent = cap || "";
  $("#lightbox").classList.remove("hidden");
}
$("#lightbox").onclick = () => $("#lightbox").classList.add("hidden");

// ------------------------------------------------------------- Boucle de rafraîchissement
refreshStatus().then(() => {
  if (STATUS?.generation?.running) pollGeneration();
});

setInterval(() => {
  if (!STATUS) return;
  const busy = STATUS.jobs && STATUS.jobs.some((j) => j.status === "running");
  const isSetupActive = $("#tab-setup") && $("#tab-setup").classList.contains("active");
  // Rafraîchit si des téléchargements sont en cours ou sur l'onglet configuration
  if (busy || isSetupActive) refreshStatus();
}, 1000);
