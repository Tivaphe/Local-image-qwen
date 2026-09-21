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

// Réponse simulée du serveur pour la détection de pose (une personne détectée).
const POSE_DETECTION = {
  ok: true, kind: "pose", count: 1, backend: "OpenCV 5.0.0 (dnn)",
  persons: [{
    box: [256, 112, 384, 368], conf: 0.92,
    keypoints: Array.from({ length: 17 }, (_, i) => ({
      x: 200 + 12 * i, y: 150 + 9 * i, c: 0.85, visible: true, name: "pt" + i,
    })),
  }],
  control: { id: "controls/pose-test.png", url: "/uploads/controls/pose-test.png", width: 640, height: 480 },
  source: { id: "photo.png", url: "/uploads/photo.png" },
  image: { width: 640, height: 480 },
  message: "1 personnage(s) détecté(s) — vous pouvez ajuster le squelette dans l'éditeur.",
};

window.fetch = async (url, opts = {}) => {
  const u = String(url);
  calls.push({ url: u, opts });
  const body = u.includes("/api/status") ? currentStatus
    : u.includes("/api/generation") ? currentStatus.generation
      : u.includes("/api/gallery") ? []
        : u.includes("/api/control/detect") ? POSE_DETECTION
          : u.includes("/api/control/pose") ? {
            ok: true, kind: "pose",
            control: { id: "controls/pose-edite.png", url: "/uploads/controls/pose-edite.png", width: 640, height: 480 },
          }
            : { ok: true, job: "job-test" };
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) };
};

// jsdom ne charge pas les images : on simule un chargement immédiat (640x480).
window.Image = class {
  set src(value) {
    this._src = value;
    this.naturalWidth = 640;
    this.naturalHeight = 480;
    setTimeout(() => this.onload && this.onload(), 0);
  }
  get src() { return this._src; }
};

// jsdom n'implémente pas le dessin : on fournit un contexte 2D factice qui compte les tracés.
const drawn = { lines: 0, arcs: 0, images: 0 };
window.HTMLCanvasElement.prototype.getContext = function () {
  return {
    clearRect() {}, fillRect() {}, beginPath() {}, stroke() {}, fill() {},
    moveTo() {}, lineTo() { drawn.lines++; }, arc() { drawn.arcs++; },
    drawImage() { drawn.images++; }, set strokeStyle(v) {}, set fillStyle(v) {},
    set lineWidth(v) {}, set globalAlpha(v) {},
  };
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

// ---------------------------------------------------------------- ControlNet
const controlStatus = (family, extra = {}) => ({
  ...currentStatus,
  config: { ...currentStatus.config, family },
  ...extra,
});

check("carte « Personnages, pose et composition » présente",
  !!doc.querySelector("#poseCard") && !!doc.querySelector("#controlType"));
check("carte de contrôle : sélecteur de type de contrôle",
  [...doc.querySelectorAll("#controlType option")].some((o) => o.value === "pose")
  && [...doc.querySelectorAll("#controlType option")].some((o) => o.value === "canny"));
check("carte de contrôle : curseur de force + valeur affichée",
  !!doc.querySelector("#controlStrength") && /0\.90/.test(doc.querySelector("#controlStrengthVal").textContent));
check("carte de contrôle : bouton de détection automatique", !!doc.querySelector("#btnDetect"));

// 1) modèle DiT (Qwen) : ControlNet expliqué et refusé, avec bascule proposée
currentStatus = controlStatus("qwen_image_2.1");
window.refreshStatus();
await tick(150);
const warn = doc.querySelector("#controlUnsupported");
check("Qwen : ControlNet annoncé comme indisponible", !warn.classList.contains("hidden"));
check("Qwen : explication « UNet / SD 1.5 » affichée", /UNet/.test(warn.textContent) && /SD 1\.5/.test(warn.textContent),
  warn.textContent.trim().slice(0, 90));
const switchBtn = doc.querySelector("#btnSwitchControl");
check("Qwen : bouton pour basculer vers la famille ControlNet", !!switchBtn);
if (switchBtn) {
  switchBtn.click();
  await tick(150);
  const conf = calls.filter((c) => c.url.includes("/api/config")).pop();
  check("bascule → POST /api/config vers sd15_control",
    !!conf && JSON.parse(conf.opts.body).family === "sd15_control");
}

// 2) famille SD 1.5 sans fichiers de contrôle : téléchargement proposé
const sd15 = {
  ...currentStatus.families.find((f) => f.id === "sd15_control"),
};
currentStatus = {
  ...currentStatus,
  config: { ...currentStatus.config, family: "sd15_control" },
  models: { ...currentStatus.models, sd15_control: { diffusion: [{ name: "v1-5-pruned-emaonly.safetensors", size_gb: 4.27 }], controlnet: [], pose_detector: [], text_encoder: [], vae: [], vision: [], lora: [] } },
  families_status: { ...currentStatus.families_status, sd15_control: { ready: false, edit_ready: true, control_ready: false, installed_gb: 4.27, file_count: 1 } },
  control: { ...currentStatus.control, pose_model_present: false, control_family: "sd15_control" },
};
window.refreshStatus();
await tick(150);
check("SD 1.5 : plus d'avertissement d'incompatibilité", doc.querySelector("#controlUnsupported").classList.contains("hidden"));

// choix de la pose dans le sélecteur : les fichiers manquants sont proposés au téléchargement
const missing = doc.querySelector("#controlMissing");
doc.querySelector("#controlType").value = "pose";
doc.querySelector("#controlType").onchange({ target: doc.querySelector("#controlType") });
await tick(300);
check("SD 1.5 : fichiers de contrôle manquants signalés", !missing.classList.contains("hidden")
  && /ControlNet|détecteur/i.test(missing.textContent), missing.textContent.trim().slice(0, 110).replace(/\s+/g, " "));
check("SD 1.5 : bouton de téléchargement des fichiers de contrôle", !!doc.querySelector("#btnControlDownload"));
calls.length = 0;
doc.querySelector("#btnControlDownload").click();
await tick(300);
const dl = calls.filter((c) => c.url.includes("/api/download")).map((c) => JSON.parse(c.opts.body));
check("SD 1.5 : téléchargement des fichiers de contrôle en un clic", dl.length === 2,
  dl.map((d) => `${d.category}/${d.file_id}`).join(", "));
check("SD 1.5 : modèle OpenPose et détecteur de pose demandés",
  dl.some((d) => d.category === "controlnet" && /openpose/.test(d.file_id))
  && dl.some((d) => d.category === "pose_detector" && /yolov8n-pose/.test(d.file_id)),
  dl.map((d) => d.file_id).join(", "));
check("SD 1.5 : bonne famille pour le téléchargement", dl.every((d) => d.family === "sd15_control"));

// 3) détection automatique : la pose détectée remplit l'éditeur
currentStatus = {
  ...currentStatus,
  control: { ...currentStatus.control, pose_model_present: true },
  models: { ...currentStatus.models, sd15_control: { ...currentStatus.models.sd15_control, controlnet: [{ name: "control_v11p_sd15_openpose.safetensors", size_gb: 0.72 }], pose_detector: [{ name: "yolov8n-pose.onnx", size_gb: 0.01 }] } },
  families_status: { ...currentStatus.families_status, sd15_control: { ...currentStatus.families_status.sd15_control, control_ready: true } },
};
window.refreshStatus();
await tick(150);
check("SD 1.5 : fichiers de contrôle présents → plus d'avertissement", doc.querySelector("#controlMissing").classList.contains("hidden"));
// source : un fichier choisi par l'utilisateur (l'autre source est la galerie)
const photo = new window.File([new Uint8Array([137, 80, 78, 71])], "photo.png", { type: "image/png" });
Object.defineProperty(doc.querySelector("#controlFile"), "files", { value: [photo], configurable: true });
doc.querySelector("#controlSource").value = "file";
doc.querySelector("#btnDetect").click();
await tick(400);
const detectCall = calls.find((c) => c.url.includes("/api/control/detect"));
check("détection → POST /api/control/detect", !!detectCall);
check("détection : type « pose » envoyé", detectCall && detectCall.opts.body.get("kind") === "pose",
  detectCall && String(detectCall.opts.body.get("kind")));
check("aperçu de l'image de contrôle affiché", /pose-test\.png/.test(doc.querySelector("#controlPreview").src),
  doc.querySelector("#controlPreview").src);
check("éditeur de squelette ouvert sur les personnages détectés",
  !doc.querySelector("#poseEditor").classList.contains("hidden"));
check("squelette dessiné (segments + articulations)", drawn.lines > 10 && drawn.arcs >= 17,
  `${drawn.lines} segments, ${drawn.arcs} articulations`);
check("état de la détection affiché", /personnage/i.test(doc.querySelector("#controlStatus").textContent),
  doc.querySelector("#controlStatus").textContent.trim());

// 4) ajustement à la souris : le point déplacé est renvoyé au serveur
const canvas = doc.querySelector("#poseCanvas");
canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480, right: 640, bottom: 480 });
const before = window.__liq.control.persons[0].keypoints[9].x;
const send = (type, x, y) => canvas.dispatchEvent(new window.MouseEvent(type, { clientX: x, clientY: y, bubbles: true }));
const scale = Number(canvas.dataset.scale || 1);
send("pointerdown", before * scale, (150 + 9 * 9) * scale);   // poignet gauche
send("pointermove", (before + 60) * scale, (150 + 9 * 9) * scale);
send("pointerup", (before + 60) * scale, (150 + 9 * 9) * scale);
await tick(400);
const after = window.__liq.control.persons[0].keypoints[9].x;
check("glisser-déposer d'une articulation pris en compte", after > before, `${before} → ${after}`);
check("pose ajustée renvoyée au serveur (POST /api/control/pose)",
  calls.some((c) => c.url.includes("/api/control/pose")));
check("point déplacé signalé dans l'éditeur", /déplacé/.test(doc.querySelector("#poseEditorInfo").textContent),
  doc.querySelector("#poseEditorInfo").textContent.trim());

// 5) génération : les paramètres de contrôle accompagnent la requête
currentStatus = { ...currentStatus, ready: true, engine: { ...currentStatus.engine, installed: true } };
window.refreshStatus();
await tick(150);
doc.querySelector("#useInit").checked = true;
doc.querySelector("#controlStrength").value = "0.75";
doc.querySelector("#controlStrength").oninput({ target: doc.querySelector("#controlStrength") });
calls.length = 0;
doc.querySelector("#btnGenerate").click();
await tick(250);
const gen = calls.find((c) => c.url.includes("/api/generate"));
check("génération → POST /api/generate", !!gen);
if (gen) {
  const fd = gen.opts.body;
  check("génération : type de contrôle transmis", fd.get("control_type") === "pose", String(fd.get("control_type")));
  check("génération : image de contrôle transmise", String(fd.get("control_id")).includes("pose-edite"),
    String(fd.get("control_id")));
  check("génération : force du contrôle personnalisée", fd.get("control_strength") === "0.75", String(fd.get("control_strength")));
  check("génération : modèle ControlNet transmis", /control_v11p_sd15_openpose/.test(String(fd.get("control_net"))),
    String(fd.get("control_net")));
  check("génération : img2img activé (modification d'image)", fd.get("use_init") === "true", String(fd.get("use_init")));
}
check("force du contrôle affichée à côté du curseur", /0\.75/.test(doc.querySelector("#controlStrengthVal").textContent));

// 6) retrait du contrôle
doc.querySelector("#btnControlClear").click();
await tick(120);
check("bouton « Retirer le contrôle » réinitialise le formulaire",
  doc.querySelector("#controlType").value === "" && doc.querySelector("#controlPreviewWrap").classList.contains("hidden"));

currentStatus = controlStatus("qwen_image_2.1");
window.refreshStatus();
await tick(120);
check("bouton moteur présent", !!doc.querySelector("#btnEngine"));
check("bouton Générer présent", !!doc.querySelector("#btnGenerate"));
check("réglages de performance présents", !!doc.querySelector("#btnSavePerf"));
report();
