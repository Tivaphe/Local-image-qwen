/**
 * Tests de l'interface sans navigateur (jsdom).
 *
 *   npm install jsdom                # une seule fois (à la racine du projet)
 *   node tests/ui_render.mjs [status.json]
 *
 * Sans argument, le script interroge http://127.0.0.1:7860/api/status (application
 * lancée) ; sinon il utilise le JSON fourni (ex. : `python -c "..."` qui enregistre
 * le vrai /api/status dans un fichier).
 *
 * Vérifie que l'onglet Modèles affiche bien, pour Qwen‑Image‑2.1, FLUX.2 klein 9B
 * et FLUX.2 klein 4B : l'état d'installation, les fichiers du pack génération +
 * édition, le bouton « Tout télécharger » et son comportement (POST /api/install),
 * puis le suivi des téléchargements (progression fichier par fichier + annulation).
 */
import { JSDOM, VirtualConsole } from "jsdom";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "app/static/index.html"), "utf8");
const script = readFileSync(join(root, "app/static/app.js"), "utf8");

const statusPath = process.argv[2];
let currentStatus = statusPath
  ? JSON.parse(readFileSync(statusPath, "utf8"))
  : await (await fetch("http://127.0.0.1:7860/api/status")).json();

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => errors.push("jsdomError: " + e.message));
virtualConsole.on("error", (m) => errors.push("console.error: " + m));
virtualConsole.on("warn", () => {});

const dom = new JSDOM(html, { url: "http://127.0.0.1:7860/", runScripts: "outside-only", virtualConsole, pretendToBeVisual: true });
const { window } = dom;
const calls = [];

window.fetch = async (url, opts = {}) => {
  const u = String(url);
  calls.push({ url: u, opts });
  const body = u.includes("/api/status") ? currentStatus
    : u.includes("/api/generation") ? currentStatus.generation
      : u.includes("/api/gallery") ? []
        : { ok: true, job: "job-test" };
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
};

window.eval(script);
const tick = (ms = 250) => new Promise((r) => setTimeout(r, ms));
await tick(400);

const doc = window.document;
const checks = [];
const check = (name, ok, detail = "") => checks.push({ name, ok, detail });
const text = (sel) => (doc.querySelector(sel)?.textContent || "").trim();
const report = () => {
  for (const c of checks) console.log(`${c.ok ? "✅" : "❌"} ${c.name}${c.detail ? " — " + c.detail : ""}`);
  if (errors.length) { console.log("--- erreurs JS ---"); errors.forEach((e) => console.log("❌ " + e)); }
  const failed = checks.filter((c) => !c.ok).length + errors.length;
  console.log(failed ? `\n${failed} échec(s)` : "\nTous les rendus sont OK");
  process.exit(failed ? 1 : 0);
};
process.on("uncaughtException", (e) => { errors.push("exception: " + e.message); report(); });

check("onglet Générer présent", !!doc.querySelector("#tab-generate"));
check("onglet Modèles présent", !!doc.querySelector('[data-tab="models"]'));
check("pas de bannière d'erreur de rendu", !doc.querySelector("#renderError:not(.hidden)"), text("#renderError"));
check("rappel « premier lancement » affiché", !doc.querySelector("#modelsNotice").classList.contains("hidden"));

const attendus = {
  "qwen_image_2.1": "Qwen‑Image‑2.1",
  "flux2_klein_9b": "FLUX.2 klein 9B",
  "flux2_klein_4b": "FLUX.2 klein 4B",
};
for (const [id, name] of Object.entries(attendus)) {
  const card = doc.querySelector(`.fam[data-fam="${id}"]`);
  check(`carte « ${name} »`, !!card);
  if (!card) continue;
  const buttons = [...card.querySelectorAll("button")].map((b) => b.textContent.trim());
  check(`bouton d'installation en un clic pour ${name}`,
    buttons.some((b) => /Tout télécharger/i.test(b)), buttons.find((b) => /Tout télécharger/i.test(b)));
  const rows = card.querySelectorAll(".mrow").length;
  check(`fichiers listés pour ${name}`, rows > 0, `${rows} ligne(s)`);
  check(`édition d'image annoncée pour ${name}`, /édition/i.test(card.textContent));
  check(`quantification au choix pour ${name}`, !!card.querySelector(".q-diffusion"));
  const options = [...card.querySelectorAll(".q-diffusion option")].map((o) => o.value);
  check(`${name} : variantes de quantification proposées`, options.length >= 3, options.slice(0, 3).join(", "));
}

// --- Qwen : le pack doit inclure l'encodeur de vision (édition) -----------------
const qwen = doc.querySelector('.fam[data-fam="qwen_image_2.1"]');
check("Qwen : mmproj proposé dans le pack", !!qwen.querySelector(".q-vision"));
check("Qwen : mmproj coché par défaut", !!qwen.querySelector(".q-vision").checked);
check("Qwen : plan d'installation détaillé", /Encodeur de vision/.test(qwen.querySelector(".plan").textContent));

// --- clic sur « Tout télécharger » ---------------------------------------------
qwen.querySelector(".btn-install").click();
await tick(120);
const installCall = calls.find((c) => c.url.includes("/api/install"));
check("clic → POST /api/install", !!installCall);
if (installCall) {
  const body = JSON.parse(installCall.opts.body);
  check("installation : bonne famille", body.family === "qwen_image_2.1", body.family);
  check("installation : édition incluse", body.include_vision === true, String(body.include_vision));
  check("installation : quantification du pack", /Q4_K/.test(body.diffusion), body.diffusion);
  check("installation : encodeur de texte du pack", /Qwen3VL-8B/.test(body.text_encoder), body.text_encoder);
}

// --- suivi d'une tâche en cours -------------------------------------------------
currentStatus = {
  ...currentStatus,
  jobs: [{
    id: "job-1", kind: "bundle", family: "qwen_image_2.1", label: "Qwen‑Image‑2.1 — installation complète",
    status: "running", done: 1.2e9, total: 4.2e9, message: "",
    files: [
      { label: "qwen_image_2.1-Q4_K.gguf", category: "diffusion", status: "running", done: 1.2e9, total: 4.2e9, size_gb: 4.2, message: "" },
      { label: "Qwen3VL-8B-Instruct-Q4_K_M.gguf", category: "text_encoder", status: "pending", done: 0, total: 5.03e9, size_gb: 5.03, message: "" },
    ],
  }],
};
window.refreshStatus();
await tick(200);
const jobsBox = doc.querySelector("#modelJobs");
check("tâche en cours affichée dans l'onglet Modèles", /installation complète/.test(jobsBox.textContent));
check("progression fichier par fichier", /qwen_image_2\.1-Q4_K\.gguf/.test(jobsBox.textContent) && /Go \//.test(jobsBox.textContent));
const cancelBtn = jobsBox.querySelector(".btn-cancel-job");
check("bouton d'annulation de la tâche", !!cancelBtn);
if (cancelBtn) {
  cancelBtn.click();
  await tick(120);
  check("annulation → POST /api/jobs/job-1/cancel", calls.some((c) => c.url.includes("/api/jobs/job-1/cancel")));
  check("bouton d'installation désactivé pendant le téléchargement",
    doc.querySelector('.fam[data-fam="qwen_image_2.1"] .btn-install').disabled);
}

// --- cas « partiellement installé » : Qwen sans mmproj ---------------------------
currentStatus = {
  ...currentStatus,
  jobs: [],
  models: {
    ...currentStatus.models,
    "qwen_image_2.1": { ...currentStatus.models["qwen_image_2.1"], diffusion: [{ name: "qwen_image_2.1-Q6_K.gguf", size_gb: 6 }] },
  },
  families_status: {
    ...currentStatus.families_status,
    "qwen_image_2.1": { ready: false, edit_ready: false, installed_gb: 6, file_count: 1 },
  },
  // /api/status sélectionne automatiquement le fichier présent (comme le fait le serveur)
  config: {
    ...currentStatus.config,
    selections: {
      ...currentStatus.config.selections,
      "qwen_image_2.1": { ...currentStatus.config.selections["qwen_image_2.1"], diffusion: "qwen_image_2.1-Q6_K.gguf" },
    },
  },
};
window.refreshStatus();
await tick(200);
const qwen2 = doc.querySelector('.fam[data-fam="qwen_image_2.1"]');
check("fichiers présents listés avec sélection", qwen2.querySelectorAll(".mrow input[type=radio]").length > 0);
check("le pack ne propose que les fichiers manquants", !/Tout télécharger \(~11/.test(qwen2.querySelector(".btn-install").textContent),
  qwen2.querySelector(".btn-install").textContent);

// --- mode édition ---------------------------------------------------------------
currentStatus = { ...currentStatus, ready: true, engine: { ...currentStatus.engine, installed: true } };
window.refreshStatus();
await tick(150);
check("sélecteur de mode Générer / Éditer", !!doc.querySelector("#modeGen") && !!doc.querySelector("#modeEdit"));
doc.querySelector("#modeEdit").click();
await tick(60);
check("mode édition : bouton renommé", /modification/i.test(doc.querySelector("#btnGenerate").textContent),
  doc.querySelector("#btnGenerate").textContent.trim());
check("mode édition : carte des images de référence mise en avant",
  doc.querySelector("#editCard").classList.contains("highlight"));
check("mode édition : carte placée en tête du formulaire",
  doc.querySelector("#editCard").parentElement.firstElementChild === doc.querySelector("#editCard"));
doc.querySelector("#btnGenerate").click();
await tick(60);
check("mode édition sans image de référence : message d'erreur clair",
  !doc.querySelector("#errorBox").classList.contains("hidden") && /image de référence/i.test(doc.querySelector("#errorBox").textContent));
check("mode édition : aucun appel /api/generate",
  !calls.some((c) => c.url.includes("/api/generate")));
doc.querySelector("#modeGen").click();
await tick(60);
check("retour en mode génération : carte d'édition replacée après le prompt",
  doc.querySelector("#editCard").parentElement.querySelector("#formatRow") !== null);

check("bouton moteur présent", !!doc.querySelector("#btnEngine"));
check("bouton Générer présent", !!doc.querySelector("#btnGenerate"));
check("réglages de performance présents", !!doc.querySelector("#btnSavePerf"));
report();
