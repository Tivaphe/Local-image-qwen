/**
 * Mannequin articulé — moteur de pose (aucune dépendance, aucun asset externe).
 *
 * Un pantin 3D filaire/volumétrique manipulable à la souris, qui produit une image
 * de pose utilisable comme référence (ControlNet OpenPose, image de référence pour
 * les modèles d'édition, carte de profondeur ou silhouette).
 *
 * Principes :
 *   - chaque os a une longueur FIXE (proportions conservées, on ne peut rien étirer) ;
 *   - déplacer un poignet/une cheville plie le coude/le genou (IK analytique 2 os) ;
 *   - déplacer une articulation montante (tête, colonne, poitrine) fait tourner la
 *     chaîne : les os suivent, les longueurs restent justes ;
 *   - les paramètres morphologiques (taille, carrure, jambes, silhouette) recalculent
 *     les longueurs d'os à la volée.
 *
 * API : window.Mannequin (classe) + helpers purs exposés pour les tests.
 */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------ modèle
  // Repère : x = droite (gauche/droite du personnage), y = haut, z = vers l'avant.
  const JOINTS = [
    "hips", "spine", "chest", "neck", "head", "head_top",
    "nose", "eye_l", "eye_r", "ear_l", "ear_r",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "hip_l", "knee_l", "ankle_l", "toe_l", "heel_l",
    "hip_r", "knee_r", "ankle_r", "toe_r", "heel_r",
  ];

  // os : [parent, enfant, rayon au départ, rayon à l'arrivée, type]
  const BONES = [
    ["hips", "spine", 0.112, 0.150, "torso"],
    ["spine", "chest", 0.150, 0.165, "torso"],
    ["chest", "neck", 0.120, 0.048, "neck"],
    ["neck", "head", 0.072, 0.100, "head"],
    ["head", "head_top", 0.100, 0.055, "head"],
    ["neck", "shoulder_l", 0.062, 0.066, "shoulder"],
    ["shoulder_l", "elbow_l", 0.062, 0.048, "arm"],
    ["elbow_l", "wrist_l", 0.048, 0.036, "forearm"],
    ["wrist_l", "hand_l", 0.036, 0.044, "hand"],
    ["neck", "shoulder_r", 0.062, 0.066, "shoulder"],
    ["shoulder_r", "elbow_r", 0.062, 0.048, "arm"],
    ["elbow_r", "wrist_r", 0.048, 0.036, "forearm"],
    ["wrist_r", "hand_r", 0.036, 0.044, "hand"],
    ["hips", "hip_l", 0.082, 0.098, "pelvis"],
    ["hip_l", "knee_l", 0.088, 0.062, "thigh"],
    ["knee_l", "ankle_l", 0.062, 0.040, "shin"],
    ["ankle_l", "toe_l", 0.042, 0.038, "foot"],
    ["ankle_l", "heel_l", 0.040, 0.034, "foot"],
    ["hips", "hip_r", 0.082, 0.098, "pelvis"],
    ["hip_r", "knee_r", 0.088, 0.062, "thigh"],
    ["knee_r", "ankle_r", 0.062, 0.040, "shin"],
    ["ankle_r", "toe_r", 0.042, 0.038, "foot"],
    ["ankle_r", "heel_r", 0.040, 0.034, "foot"],
  ];

  // points décoratifs rattachés à la tête (ils la suivent et tournent avec elle)
  const FACE = { nose: "head", eye_l: "head", eye_r: "head", ear_l: "head", ear_r: "head" };

  const PARENT = {};
  for (const [a, b] of BONES) if (PARENT[b] === undefined) PARENT[b] = a;
  for (const name in FACE) PARENT[name] = FACE[name];
  PARENT.hips = null;

  // Chaînes cinématiques (2 os) résolues par IK quand on tire l'extrémité
  // seules les extrémités de chaîne (poignet, cheville) déclenchent l'IK : le coude et le
  // genou se plient alors que les os gardent leur longueur
  const IK_CHAINS = {
    wrist_l: { root: "shoulder_l", mid: "elbow_l", pole: [0.35, -0.25, -1.0] },
    wrist_r: { root: "shoulder_r", mid: "elbow_r", pole: [-0.35, -0.25, -1.0] },
    ankle_l: { root: "hip_l", mid: "knee_l", pole: [0.20, 0.10, 1.0] },
    ankle_r: { root: "hip_r", mid: "knee_r", pole: [-0.20, 0.10, 1.0] },
  };
  // le pied est rigide : bouger la pointe tourne aussi le talon (et inversement)
  const FOOT = { toe_l: "heel_l", heel_l: "toe_l", toe_r: "heel_r", heel_r: "toe_r" };

  const LIMB_COLOR = {
    torso: "#5b6bd6", neck: "#5b6bd6", head: "#6b7ade", shoulder: "#5b6bd6",
    pelvis: "#5b6bd6", arm: "#4b5cc4", forearm: "#4b5cc4", hand: "#8b93e8",
    thigh: "#4b5cc4", shin: "#4b5cc4", foot: "#8b93e8",
  };

  // Ordre canonique ControlNet/OpenPose (18 points)
  const OPENPOSE_18 = [
    "nose", "neck", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l", "elbow_l", "wrist_l",
    "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l",
    "eye_r", "eye_l", "ear_r", "ear_l",
  ];
  const OPENPOSE_LIMBS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
  ];
  const OPENPOSE_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],
  ];

  // ------------------------------------------------------------- utilitaires
  const v = (x, y, z) => ({ x: x || 0, y: y || 0, z: z || 0 });
  const sub = (a, b) => v(a.x - b.x, a.y - b.y, a.z - b.z);
  const add = (a, b) => v(a.x + b.x, a.y + b.y, a.z + b.z);
  const mul = (a, s) => v(a.x * s, a.y * s, a.z * s);
  const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
  const len = (a) => Math.sqrt(dot(a, a));
  const norm = (a) => (len(a) < 1e-9 ? v(0, 0, 0) : mul(a, 1 / len(a)));
  const dist = (a, b) => len(sub(a, b));
  const clone = (p) => v(p.x, p.y, p.z);
  const lerp = (a, b, t) => v(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);

  /** Rotation d'un vecteur autour d'un axe (formule de Rodrigues). */
  function rotateAround(vector, axis, angle) {
    const c = Math.cos(angle), sn = Math.sin(angle);
    const k = norm(axis);
    const kxv = v(k.y * vector.z - k.z * vector.y, k.z * vector.x - k.x * vector.z, k.x * vector.y - k.y * vector.x);
    const kv = dot(k, vector);
    return v(
      vector.x * c + kxv.x * sn + k.x * kv * (1 - c),
      vector.y * c + kxv.y * sn + k.y * kv * (1 - c),
      vector.z * c + kxv.z * sn + k.z * kv * (1 - c),
    );
  }

  /**
   * IK analytique 2 os : place l'articulation intermédiaire (coude/genou) pour que
   * la chaîne root→mid→end respecte les longueurs l1/l2 et atteigne `target`.
   * Le vecteur `pole` choisit le plan de pliage (coude vers l'arrière, genou vers l'avant).
   */
  function solveTwoBoneIK(root, mid, end, target, l1, l2, pole) {
    const toTarget = sub(target, root);
    let d = len(toTarget);
    const dMax = l1 + l2 - 1e-4;
    const dMin = Math.abs(l1 - l2) + 1e-4;
    const dir = norm(toTarget);
    if (d > dMax || d < dMin) {                 // hors de portée : on garde la direction
      d = Math.min(dMax, Math.max(dMin, d));
      target = add(root, mul(dir, d));
    }
    const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    // plan de pliage : perpendiculaire à la chaîne, orienté vers le pôle
    let u = norm(sub(pole, mul(dir, dot(pole, dir))));
    if (len(u) < 1e-6) {                        // pôle parallèle : on prend un axe quelconque
      u = norm(sub(v(0, 1, 0), mul(dir, dot(v(0, 1, 0), dir))));
      if (len(u) < 1e-6) u = v(1, 0, 0);
    }
    return {
      mid: add(add(root, mul(dir, a)), mul(u, h)),
      end: add(root, mul(dir, d)),
    };
  }

  /**
   * Ramène chaque os à sa longueur de repos en ne déplaçant QUE l'enfant
   * (le parent garde sa place) — appelé du haut vers le bas de la hiérarchie.
   */
  function enforceBoneLengths(pose, lengths, order) {
    for (const name of order) {
      const parent = PARENT[name];
      if (!parent) continue;
      const rest = lengths[name];
      if (!rest) continue;
      const p = pose[parent], c = pose[name];
      const direction = norm(sub(c, p));
      if (len(direction) < 1e-9) {               // superposés : on pousse vers le bas
        pose[name] = add(p, v(0, -rest, 0));
      } else {
        pose[name] = add(p, mul(direction, rest));
      }
    }
  }

  /** Ordre hiérarchique (parents avant enfants). */
  function hierarchyOrder() {
    const order = [], seen = {};
    const visit = (name) => {
      if (seen[name]) return;
      const parent = PARENT[name];
      if (parent) visit(parent);
      seen[name] = true;
      order.push(name);
    };
    JOINTS.forEach(visit);
    return order;
  }
  const ORDER = hierarchyOrder();

  // -------------------------------------------------------- dimensions
  // Toutes les dimensions sont FIXES : le tableau de réglages de l'interface écrit ici.
  // Longueurs de chaque segment (mètres) et épaisseurs (diamètre au milieu, mètres).
  const BASE_LENGTHS = {
    spine: 0.16, chest: 0.20, neck: 0.10, head: 0.13, head_top: 0.10,
    shoulder_l: 0.19, elbow_l: 0.28, wrist_l: 0.25, hand_l: 0.10,
    shoulder_r: 0.19, elbow_r: 0.28, wrist_r: 0.25, hand_r: 0.10,
    hip_l: 0.11, knee_l: 0.44, ankle_l: 0.42, toe_l: 0.17, heel_l: 0.08,
    hip_r: 0.11, knee_r: 0.44, ankle_r: 0.42, toe_r: 0.17, heel_r: 0.08,
  };
  const BASE_THICK = {
    spine: 0.262, chest: 0.315, neck: 0.168, head: 0.176, head_top: 0.155,
    shoulder_l: 0.108, elbow_l: 0.110, wrist_l: 0.088, hand_l: 0.075,
    shoulder_r: 0.108, elbow_r: 0.110, wrist_r: 0.088, hand_r: 0.075,
    hip_l: 0.180, knee_l: 0.150, ankle_l: 0.102, toe_l: 0.080, heel_l: 0.074,
    hip_r: 0.180, knee_r: 0.150, ankle_r: 0.102, toe_r: 0.080, heel_r: 0.074,
  };

  // Morphologies types : de simples jeux de facteurs qui remplissent le tableau
  // (ensuite, seules les valeurs du tableau comptent).
  const MORPHOLOGIES = {
    neutre: { stature: 1, shoulders: 1, legs: 1, arms: 1, girth: 1 },
    fin: { stature: 1, shoulders: 0.88, legs: 1.04, arms: 0.97, girth: 0.84 },
    athletique: { stature: 1.01, shoulders: 1.12, legs: 1.0, arms: 1.0, girth: 1.16 },
    fort: { stature: 1, shoulders: 1.18, legs: 0.97, arms: 0.99, girth: 1.34 },
    femme: { stature: 0.97, shoulders: 0.88, legs: 1.02, arms: 0.96, girth: 0.92 },
  };
  const BUILDS = MORPHOLOGIES;                      // ancien nom, conservé

  /** Dimensions complètes pour une morphologie (longueurs + épaisseurs), en mètres. */
  function morphedDimensions(morph) {
    const preset = typeof morph === "string" ? (MORPHOLOGIES[morph] || {}) : (morph || {});
    const o = Object.assign({}, MORPHOLOGIES.neutre, preset);
    const lengths = {}, thickness = {};
    for (const bone in BASE_LENGTHS) {
      const side = bone.slice(-2);
      let f = o.stature;                                          // tout le squelette suit la stature
      if (side === "_l" || side === "_r") {
        if (bone.startsWith("shoulder")) f *= o.shoulders;
        if (bone.startsWith("elbow") || bone.startsWith("wrist")) f *= o.arms;
        if (bone.startsWith("knee") || bone.startsWith("ankle")) f *= o.legs;
      }
      lengths[bone] = BASE_LENGTHS[bone] * f;
      thickness[bone] = BASE_THICK[bone] * o.girth;
    }
    // points du visage : distances à la tête (ils tournent avec elle)
    const h = lengths.head;
    lengths.nose = 0.985 * h; lengths.eye_l = lengths.eye_r = 0.930 * h;
    lengths.ear_l = lengths.ear_r = 0.625 * h;
    thickness.nose = thickness.eye_l = thickness.eye_r = 0.06;
    thickness.ear_l = thickness.ear_r = 0.05;
    return { lengths, thickness };
  }

  /** Longueurs d'une morphologie (compatibilité : sert à remplir le tableau). */
  function boneLengths(opts) {
    return morphedDimensions(opts).lengths;
  }

  // Épaisseurs de référence (compatibilité) : diamètre de chaque segment
  function boneThickness(opts) {
    return morphedDimensions(opts).thickness;
  }

  // Rapport de conicité de chaque segment (rayon au départ / rayon à l'arrivée),
  // dérivé de la table de capsules ci-dessus : une seule source de vérité.
  const BONE_TAPER = {};
  for (const [, child, r1, r2] of BONES) {
    const mean = (r1 + r2) / 2;
    BONE_TAPER[child] = [r1 / mean, r2 / mean];
  }

  /** Rayons des capsules à partir des épaisseurs fournies (ou des valeurs de référence). */
  function boneRadii(thickness) {
    const t = thickness || BASE_THICK;
    return BONES.map(([, child]) => {
      const dia = t[child] !== undefined ? t[child] : BASE_THICK[child];
      const [k1, k2] = BONE_TAPER[child] || [1, 1];
      return [dia / 2 * k1, dia / 2 * k2];
    });
  }

  // Tronc : suite de coupes elliptiques le long de la colonne (hanches → bassin → taille →
  // poitrine → épaules). Le rendu est une silhouette lisse et pleine, qui suit la pose,
  // au lieu d'un empilement de capsules qui s'écrasaient au bassin et au buste.
  const TORSO_PROFILE = [
    { at: 0.00, width: 0.170, depth: 0.165 },
    { at: 0.16, width: 0.190, depth: 0.198 },   // bassin (renforcé par le bombé ci-dessous)
    { at: 0.45, width: 0.158, depth: 0.172 },   // taille
    { at: 0.70, width: 0.245, depth: 0.200 },   // bas des côtes
    { at: 0.88, width: 0.215, depth: 0.195 },   // haut de la cage thoracique
    { at: 1.00, width: 0.145, depth: 0.145 },   // raccord au cou
  ];
  // bombés anatomiques (gaussiennes le long du tronc) : hanches et épaules
  const TORSO_BUMPS = [
    { centre: 0.16, sigma: 0.15, scale: 1.05 },   // bassin
    { centre: 0.90, sigma: 0.10, scale: 0.93 },   // ceinture scapulaire
  ];
  const TORSO_SLICES = 13;
  const TORSO_RING = 14;
  // os remplacés par le tronc en rendu 3D (ischions : le bassin les englobe)
  const SKIP_IN_VOLUME = { hip_l: true, hip_r: true };
  const CAP_JOINTS = ["elbow_l", "elbow_r", "knee_l", "knee_r", "wrist_l", "wrist_r", "ankle_l", "ankle_r"];

  // Libellés du tableau de réglages (l'interface les affiche tels quels)
  const SEGMENTS = [
    { bone: "spine", label: "Tronc — bas du dos (taille)", group: "Tronc" },
    { bone: "chest", label: "Tronc — poitrine", group: "Tronc" },
    { bone: "neck", label: "Cou", group: "Tronc" },
    { bone: "head", label: "Tête (cou → crâne)", group: "Tronc" },
    { bone: "head_top", label: "Crâne (sommet)", group: "Tronc" },
    { bone: "shoulder_l", label: "Clavicule gauche", group: "Bras gauche" },
    { bone: "elbow_l", label: "Bras gauche (épaule → coude)", group: "Bras gauche" },
    { bone: "wrist_l", label: "Avant-bras gauche", group: "Bras gauche" },
    { bone: "hand_l", label: "Main gauche", group: "Bras gauche" },
    { bone: "shoulder_r", label: "Clavicule droite", group: "Bras droit" },
    { bone: "elbow_r", label: "Bras droit (épaule → coude)", group: "Bras droit" },
    { bone: "wrist_r", label: "Avant-bras droit", group: "Bras droit" },
    { bone: "hand_r", label: "Main droite", group: "Bras droit" },
    { bone: "hip_l", label: "Hanche gauche (bassin)", group: "Jambe gauche" },
    { bone: "knee_l", label: "Cuisse gauche", group: "Jambe gauche" },
    { bone: "ankle_l", label: "Jambe gauche (genou → cheville)", group: "Jambe gauche" },
    { bone: "toe_l", label: "Pied gauche", group: "Jambe gauche" },
    { bone: "heel_l", label: "Talon gauche", group: "Jambe gauche" },
    { bone: "hip_r", label: "Hanche droite (bassin)", group: "Jambe droite" },
    { bone: "knee_r", label: "Cuisse droite", group: "Jambe droite" },
    { bone: "ankle_r", label: "Jambe droite (genou → cheville)", group: "Jambe droite" },
    { bone: "toe_r", label: "Pied droit", group: "Jambe droite" },
    { bone: "heel_r", label: "Talon droit", group: "Jambe droite" },
  ];

  // -------------------------------------------------------------- poses types
  // Directions de repos (normalisées) : les positions sont construites à partir des
  // LONGUEURS d'os, donc les proportions sont exactes dès la pose de départ.
  const REST_DIRS = {
    spine: [0, 1, 0], chest: [0, 1, 0], neck: [0, 1, 0],
    head: [0, 0.99, 0.12], head_top: [0, 0.98, 0.18],
    shoulder_l: [0.94, -0.34, 0], elbow_l: [0.06, -0.995, 0.08],
    wrist_l: [0.03, -0.998, 0.05], hand_l: [0, -1, 0.05],
    shoulder_r: [-0.94, -0.34, 0], elbow_r: [-0.06, -0.995, 0.08],
    wrist_r: [-0.03, -0.998, 0.05], hand_r: [0, -1, 0.05],
    hip_l: [0.66, -0.75, 0], knee_l: [0.09, -0.99, 0.09],
    ankle_l: [0, -1, -0.02], toe_l: [0, -0.18, 0.98], heel_l: [0, -0.32, -0.95],
    hip_r: [-0.66, -0.75, 0], knee_r: [-0.09, -0.99, 0.09],
    ankle_r: [0, -1, -0.02], toe_r: [0, -0.18, 0.98], heel_r: [0, -0.32, -0.95],
    nose: [0, -0.10, 0.98], eye_l: [0.28, 0.24, 0.86], eye_r: [-0.28, 0.24, 0.86],
    ear_l: [0.62, 0.06, 0.05], ear_r: [-0.62, 0.06, 0.05],
  };
  // Ordre de construction (parent avant enfant)
  const BUILD_ORDER = ["hips", "spine", "chest", "neck", "head", "head_top",
    "nose", "eye_l", "eye_r", "ear_l", "ear_r",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l", "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "hip_l", "knee_l", "ankle_l", "toe_l", "heel_l", "hip_r", "knee_r", "ankle_r", "toe_r", "heel_r"];

  /** Pose debout exacte, construite à partir des longueurs d'os. */
  function defaultPose(lengthsOrMorph) {
    // un tableau de dimensions contient « spine » ; une morphologie contient « stature »
    const estTableau = !!lengthsOrMorph && (lengthsOrMorph.spine !== undefined || lengthsOrMorph.hips !== undefined);
    const lengths = estTableau ? Object.assign(boneLengths(), lengthsOrMorph) : boneLengths(lengthsOrMorph);
    const p = {};
    p.hips = v(0, 0, 0);
    for (const name of BUILD_ORDER) {
      if (name === "hips") continue;
      const parent = PARENT[name];
      const dir = REST_DIRS[name];
      const rest = lengths[name] || 0;
      p[name] = add(p[parent], mul(norm(v(dir[0], dir[1], dir[2])), rest));
    }
    // les pieds reposent sur le sol (y = 0)
    const sol = Math.min(p.ankle_l.y, p.ankle_r.y, p.toe_l.y, p.toe_r.y, p.heel_l.y, p.heel_r.y);
    for (const j in p) p[j] = v(p[j].x, p[j].y - sol + 0.015, p[j].z);
    return p;
  }

  /** Pose debout à partir d'une morphologie type (raccourci). */
  function defaultPoseFromMorph(opts) {
    return defaultPose(boneLengths(opts));
  }

  // Positions articulaires exprimées en angles/offsets relatifs au gabarit debout.
  const PRESETS = {
    debout: {},
    main_levee: {
      elbow_l: [-0.14, 0.06, 0.05], wrist_l: [-0.02, 0.42, 0.06], hand_l: [0.01, 0.10, 0.03],
      head: [-0.01, 0, -0.01],
    },
    marche: {
      knee_l: [0.02, -0.02, -0.22], ankle_l: [0.02, -0.03, -0.36], toe_l: [0.02, -0.01, 0.06],
      knee_r: [-0.02, -0.02, 0.18], ankle_r: [-0.03, -0.02, 0.24], toe_r: [-0.03, 0.0, 0.03],
      elbow_l: [0.03, 0.05, 0.16], wrist_l: [0.05, 0.12, 0.30],
      elbow_r: [-0.03, 0.05, -0.14], wrist_r: [-0.05, 0.12, -0.26],
      hips: [0, -0.02, 0], spine: [0, -0.01, 0.01],
    },
    assis: {
      hips: [0, -0.46, -0.02], spine: [0, -0.34, -0.02], chest: [0, -0.18, 0.0],
      neck: [0, -0.08, 0], head: [0, -0.06, 0], head_top: [0, -0.05, 0], nose: [0, -0.06, 0.02],
      eye_l: [0, -0.06, 0], eye_r: [0, -0.06, 0], ear_l: [0, -0.06, 0], ear_r: [0, -0.06, 0],
      shoulder_l: [0, -0.14, 0], elbow_l: [0.02, -0.22, 0.10], wrist_l: [0.04, -0.26, 0.28], hand_l: [0.04, -0.26, 0.10],
      shoulder_r: [0, -0.14, 0], elbow_r: [-0.02, -0.22, 0.10], wrist_r: [-0.04, -0.26, 0.28], hand_r: [-0.04, -0.26, 0.10],
      hip_l: [0, -0.02, 0.02], knee_l: [0.02, -0.06, 0.44], ankle_l: [0.02, -0.42, 0.36], toe_l: [0.02, -0.03, 0.14], heel_l: [0.02, -0.03, -0.04],
      hip_r: [0, -0.02, 0.02], knee_r: [-0.02, -0.06, 0.44], ankle_r: [-0.02, -0.42, 0.36], toe_r: [-0.02, -0.03, 0.14], heel_r: [-0.02, -0.03, -0.04],
    },
    accroupi: {
      hips: [0, -0.52, -0.10], spine: [0, -0.38, -0.06], chest: [0, -0.22, -0.02],
      neck: [0, -0.10, 0.02], head: [0, -0.07, 0.02], head_top: [0, -0.06, 0.02],
      nose: [0, -0.07, 0.03], eye_l: [0, -0.07, 0.02], eye_r: [0, -0.07, 0.02],
      ear_l: [0, -0.07, 0.02], ear_r: [0, -0.07, 0.02],
      shoulder_l: [0, -0.16, 0.02], elbow_l: [0.02, -0.26, 0.16], wrist_l: [0.03, -0.34, 0.28], hand_l: [0.03, -0.08, 0.02],
      shoulder_r: [0, -0.16, 0.02], elbow_r: [-0.02, -0.26, 0.16], wrist_r: [-0.03, -0.34, 0.28], hand_r: [-0.03, -0.08, 0.02],
      hip_l: [0.02, -0.04, 0.02], knee_l: [0.04, -0.10, 0.40], ankle_l: [0.03, -0.44, 0.30], toe_l: [0.03, -0.03, 0.14], heel_l: [0.03, -0.03, -0.04],
      hip_r: [-0.02, -0.04, 0.02], knee_r: [-0.04, -0.10, 0.40], ankle_r: [-0.03, -0.44, 0.30], toe_r: [-0.03, -0.03, 0.14], heel_r: [-0.03, -0.03, -0.04],
    },
    danse: {
      shoulder_l: [0, -0.02, 0], elbow_l: [0.10, 0.34, 0.06], wrist_l: [0.06, 0.34, 0.10], hand_l: [0.01, 0.10, 0.02],
      shoulder_r: [0, -0.02, 0], elbow_r: [-0.06, 0.30, -0.10], wrist_r: [0.16, 0.26, -0.06], hand_r: [0.06, 0.10, -0.02],
      spine: [0, 0.01, 0], chest: [0, 0.02, 0], neck: [0, 0.01, 0], head: [0.02, 0.01, 0],
      knee_l: [0.04, -0.02, -0.10], ankle_l: [0.04, -0.02, 0.10], toe_l: [0.04, 0, 0.02],
      knee_r: [-0.02, -0.02, 0.06], ankle_r: [-0.04, -0.04, -0.06], toe_r: [-0.04, 0, 0.02],
      hips: [0, -0.02, 0],
    },
    course: {
      hips: [0, -0.04, 0.04], spine: [0, -0.02, 0.06], chest: [0, -0.01, 0.06],
      neck: [0, -0.02, 0.03], head: [0, 0, 0.02], head_top: [0, 0, 0.02],
      knee_l: [0.02, 0.06, 0.36], ankle_l: [0.02, -0.10, 0.34], toe_l: [0.02, 0, 0.04],
      knee_r: [-0.02, 0.04, -0.30], ankle_r: [-0.02, -0.06, -0.42], toe_r: [-0.02, 0, 0.04],
      elbow_l: [0.04, 0.16, 0.18], wrist_l: [0.06, 0.20, 0.24], hand_l: [0.02, 0.08, 0.04],
      elbow_r: [-0.04, 0.20, -0.10], wrist_r: [-0.06, 0.28, -0.06], hand_r: [-0.02, 0.08, 0.02],
      shoulder_l: [0, 0, 0.02], shoulder_r: [0, 0, -0.02],
    },
  };

  // ------------------------------------------------------------- projection
  function rotation(yaw, pitch) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
    // yaw autour de y, puis pitch autour de x
    return [
      [cy, 0, sy],
      [sy * sx, cx, -cy * sx],
      [-sy * cx, sx, cy * cx],
    ];
  }

  /**
   * Projette un point 3D en coordonnées écran (perspective simple).
   * Renvoie {x, y, depth, scale} ; depth > 0 = devant la caméra.
   */
  function project(point, opts) {
    const { width, height, yaw, pitch, distance, target, focal } = opts;
    const r = rotation(yaw, pitch);
    const p = sub(point, target);
    const cam = v(
      r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z,
      r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z,
      r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z,
    );
    const depth = distance - cam.z;
    const f = focal || Math.min(width, height) * 1.25;
    const k = f / Math.max(0.15, depth);
    return { x: width / 2 + cam.x * k, y: height / 2 - cam.y * k, depth, scale: k, cam: cam };
  }

  // ------------------------------------------------------------------ rendu
  const COLORS = {
    skin: [198, 193, 188], skin_dark: [168, 160, 154], bone: [92, 104, 255],
  };

  function shade(rgb, factor) {
    return `rgb(${Math.max(0, Math.min(255, Math.round(rgb[0] * factor)))},` +
      `${Math.max(0, Math.min(255, Math.round(rgb[1] * factor)))},` +
      `${Math.max(0, Math.min(255, Math.round(rgb[2] * factor)))})`;
  }

  /** Dessine une capsule (segment épais à bouts ronds) entre deux projections. */
  function capsule(ctx, a, b, r1, r2, fill) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const l = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx);
    const rA = Math.max(1, r1 * a.scale), rB = Math.max(1, r2 * b.scale);
    ctx.save();
    ctx.translate(a.x, a.y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, -rA);
    ctx.lineTo(l, -rB);
    ctx.arc(l, 0, rB, -Math.PI / 2, Math.PI / 2);
    ctx.lineTo(0, rA);
    ctx.arc(0, 0, rA, Math.PI / 2, -Math.PI / 2);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.restore();
    return Math.max(rA, rB);
  }

  // ------------------------------------------------------------- mannequin
  class Mannequin {
    constructor(opts) {
      const o = opts || {};
      const dims = morphedDimensions(o.morphology);
      this.morphology = o.morphology || "neutre";
      // dimensions FIXES du pantin : elles ne changent plus quand on pose un membre
      this.lengths = Object.assign(dims.lengths, o.lengths || {});
      this.thickness = Object.assign(dims.thickness, o.thickness || {});
      this.pose = defaultPose(this.lengths);
      this.preset = "debout";
      this.camera = { yaw: 0.42, pitch: 0.12, distance: 3.0, target: v(0, 0.85, 0) };
      this.selected = "wrist_l";
      this.style = (opts && opts.style) || "volume";   // volume | wireframe
      this.showHandles = true;                        // repères visibles seulement dans l'éditeur
    }

    /** Dimensions du pantin (longueurs + épaisseurs), pour l'affichage/le partage. */
    dimensions() {
      return { lengths: Object.assign({}, this.lengths), thickness: Object.assign({}, this.thickness) };
    }

    /** Rayons des capsules, d'après les épaisseurs courantes. */
    radii() {
      return boneRadii(this.thickness);
    }

    /** Change les longueurs de segments (tableau de réglages) ; la pose est conservée. */
    setLengths(partial, opts) {
      for (const bone in partial) {
        const value = Number(partial[bone]);
        if (!isFinite(value) || value <= 0.005) continue;      // garde-fou : pas de segment nul
        this.lengths[bone] = Math.min(3, value);
      }
      // points du visage : ils restent solidaires de la tête
      const h = this.lengths.head;
      this.lengths.nose = 0.985 * h; this.lengths.eye_l = this.lengths.eye_r = 0.930 * h;
      this.lengths.ear_l = this.lengths.ear_r = 0.625 * h;
      if (!(opts && opts.keepPose)) this.applyPreset(this.preset);
      else { this.enforce(); this.fitCamera(560, 720, 1.1); }
      return this;
    }

    /** Change les épaisseurs (diamètre au milieu de chaque segment). */
    setThickness(partial, opts) {
      for (const bone in partial) {
        const value = Number(partial[bone]);
        if (!isFinite(value) || value <= 0.005) continue;
        this.thickness[bone] = Math.min(1, value);
      }
      if (opts && opts.keepPose) this.fitCamera(560, 720, 1.1);
      return this;
    }

    /** Remplit le tableau de réglages depuis une morphologie type. */
    applyMorphology(name, opts) {
      const preset = MORPHOLOGIES[name] || MORPHOLOGIES.neutre;
      const scaled = morphedDimensions(preset);
      this.morphology = MORPHOLOGIES[name] ? name : "neutre";
      this.lengths = scaled.lengths;
      this.thickness = scaled.thickness;
      if (opts && opts.keepPose) { this.enforce(); this.fitCamera(560, 720, 1.1); }
      else this.applyPreset(this.preset);
      return this;
    }

    setStyle(style) {
      this.style = style === "wireframe" ? "wireframe" : "volume";
    }

    /** Applique une pose type (réinitialise puis pose les décalages). */
    applyPreset(name, keepBuild) {
      const preset = PRESETS[name] || PRESETS.debout;
      const base = defaultPose(this.lengths);
      this.pose = base;
      // les décalages de la pose type sont exprimés en mètres : ils ne dépendent pas des dimensions
      const scale = 1;
      for (const joint in preset) {
        const off = preset[joint];
        if (!base[joint]) continue;
        base[joint] = add(base[joint], v(off[0] * scale, off[1] * scale, off[2] * scale));
      }
      this.descendants("hips").forEach((n) => {
        if (n === "hips") return;
        const parent = PARENT[n];
        if (!parent) return;
        const rest = this.lengths[n];
        if (!rest) return;
        const dir = norm(sub(this.pose[n], this.pose[parent]));
        this.pose[n] = add(this.pose[parent], mul(dir, rest));
      });
      // les chaînes de membres repassent par l'IK : le coude et le genou se placent
      // toujours du bon côté, même si la pose type demande une position extrême
      for (const ext of ["wrist_l", "wrist_r", "ankle_l", "ankle_r"]) {
        this.moveJoint(ext, this.pose[ext]);
      }
      this.preset = name;
      return this;
    }

    /** Tous les descendants d'une articulation (profondeur d'abord). */
    descendants(name) {
      const out = [];
      const walk = (n) => {
        out.push(n);
        for (const j of JOINTS) if (PARENT[j] === n && !out.includes(j)) walk(j);
      };
      walk(name);
      return out;
    }

    /**
     * Déplace une articulation : IK si c'est une extrémité, sinon rotation de la chaîne.
     * Les longueurs d'os restent exactes (proportions conservées).
     */
    moveJoint(name, target) {
      if (!this.pose[name]) return this;
      // pointe ou talon : le pied pivote d'un bloc autour de la cheville
      const sibling = FOOT[name];
      if (sibling && this.pose[sibling]) {
        const ankle = this.pose[PARENT[name] || "ankle_l"];
        const oldDir = norm(sub(this.pose[name], ankle));
        const newDir = norm(sub(target, ankle));
        const axis = v(oldDir.y * newDir.z - oldDir.z * newDir.y,
          oldDir.z * newDir.x - oldDir.x * newDir.z,
          oldDir.x * newDir.y - oldDir.y * newDir.x);
        const angle = Math.acos(Math.max(-1, Math.min(1, dot(oldDir, newDir))));
        this.pose[name] = add(ankle, mul(newDir, this.lengths[name] || dist(ankle, target)));
        if (len(axis) > 1e-6) {
          const sibDir = rotateAround(norm(sub(this.pose[sibling], ankle)), axis, angle);
          this.pose[sibling] = add(ankle, mul(sibDir, this.lengths[sibling] || dist(ankle, this.pose[sibling])));
        }
        this.enforce();
        return this;
      }
      const ik = IK_CHAINS[name];
      if (ik) {
        const rootP = this.pose[ik.root];
        const l1 = this.lengths[ik.mid], l2 = this.lengths[name];
        const solved = solveTwoBoneIK(rootP, this.pose[ik.mid], this.pose[name], target, l1, l2, ik.pole);
        this.pose[ik.mid] = solved.mid;
        this.pose[name] = solved.end;
        // les extrémités filles (main/pied) suivent la direction du membre
        const children = JOINTS.filter((j) => PARENT[j] === name);
        for (const c of children) {
          const rest = this.lengths[c];
          if (!rest) continue;
          const dir = norm(sub(this.pose[c], this.pose[name]));
          this.pose[c] = add(this.pose[name], mul(dir, rest));
        }
        this.enforce();
        return this;
      }
      // articulation « montante » : on déplace puis on remet la chaîne à longueur,
      // ce qui fait tourner les os parents sans changer leurs longueurs
      this.pose[name] = clone(target);
      this.enforce();
      return this;
    }

    /** Décalage relatif (utile pour les articulations profondes comme la tête). */
    nudgeJoint(name, delta) {
      return this.moveJoint(name, add(this.pose[name], delta));
    }

    /** Applique les longueurs d'os, du tronc vers les extrémités. */
    enforce() {
      enforceBoneLengths(this.pose, this.lengths, ORDER);
      return this;
    }

    /** Longueurs réellement mesurées (pour les tests : proportions conservées). */
    measure() {
      const out = {};
      for (const [a, b] of BONES) out[b] = dist(this.pose[a], this.pose[b]);
      return out;
    }

    /** Miroir gauche/droite de la pose courante. */
    mirror() {
      const swap = (n) => (n.endsWith("_l") ? n.slice(0, -2) + "_r"
        : n.endsWith("_r") ? n.slice(0, -2) + "_l" : n);
      const out = {};
      for (const j of JOINTS) {
        const p = this.pose[j];
        out[swap(j)] = v(-p.x, p.y, p.z);
      }
      this.pose = out;
      return this;
    }

    cameraOptions(width, height) {
      return {
        width, height, yaw: this.camera.yaw, pitch: this.camera.pitch,
        distance: this.camera.distance, target: this.camera.target,
      };
    }

    /**
     * Cadre automatiquement la pose (le mannequin entier reste visible) :
     * on choisit la distance et le centre vertical à partir de l'encombrement réel.
     */
    fitCamera(width, height, margin) {
      const m = margin || 1.22;
      let top = -Infinity, bottom = Infinity, left = Infinity, right = -Infinity;
      const opts = { width, height, yaw: this.camera.yaw, pitch: this.camera.pitch,
                     distance: this.camera.distance, target: this.camera.target };
      const centerY = (pose) => (pose.head_top.y + Math.min(pose.ankle_l.y, pose.ankle_r.y)) / 2;
      this.camera.target = v(0, centerY(this.pose), 0);
      for (const j of JOINTS) {
        const p = this.pose[j];
        top = Math.max(top, p.y); bottom = Math.min(bottom, p.y);
        left = Math.min(left, p.x); right = Math.max(right, p.x);
      }
      const spanY = Math.max(0.4, (top - bottom)) * m;
      const spanX = Math.max(0.4, (right - left)) * m;
      const focal = Math.min(width, height) * 1.25;
      const needY = spanY * focal / height;
      const needX = spanX * focal / width;
      this.camera.distance = Math.max(1.4, Math.min(14, Math.max(needY, needX)));
      return this;
    }

    orbit(dx, dy) {
      this.camera.yaw += dx * 0.01;
      this.camera.pitch = Math.max(-0.9, Math.min(1.1, this.camera.pitch + dy * 0.008));
      return this;
    }

    zoom(factor) {
      this.camera.distance = Math.max(1.6, Math.min(12, this.camera.distance * factor));
      return this;
    }

    /** Articulation la plus proche d'un point écran (en pixels). */
    jointAt(x, y, width, height, tolerance) {
      const opts = this.cameraOptions(width, height);
      let best = null, bestD = tolerance || 26;
      for (const j of JOINTS) {
        const p = project(this.pose[j], opts);
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < bestD) { bestD = d; best = j; }
      }
      return best;
    }

    /** Projection écran → point 3D sur le plan de profondeur de l'articulation donnée. */
    screenToWorld(x, y, width, height, depth) {
      const opts = this.cameraOptions(width, height);
      const focal = Math.min(width, height) * 1.25;
      const camX = (x - width / 2) * depth / focal;
      const camY = (height / 2 - y) * depth / focal;
      // inverse de la rotation appliquée dans project()
      const r = rotation(this.camera.yaw, this.camera.pitch);
      const cam = v(camX, camY, -(depth - this.camera.distance));
      const world = v(
        r[0][0] * cam.x + r[1][0] * cam.y + r[2][0] * cam.z,
        r[0][1] * cam.x + r[1][1] * cam.y + r[2][1] * cam.z,
        r[0][2] * cam.x + r[1][2] * cam.y + r[2][2] * cam.z,
      );
      return add(world, this.camera.target);
    }

    // ------------------------------------------------------------- dessin
    drawScene(ctx, width, height, options) {
      const opts = this.cameraOptions(width, height);
      const o = options || {};
      const groundY = 0;
      // sol + ombre portée
      if (o.ground !== false) {
        ctx.save();
        const horizon = project(v(0, groundY, 0), Object.assign({}, opts, { target: this.camera.target }));
        ctx.strokeStyle = "rgba(120,130,160,.55)";
        ctx.lineWidth = 1;
        for (let i = -6; i <= 6; i++) {
          const a = project(v(i * 0.35, groundY, -2.2), opts);
          const b = project(v(i * 0.35, groundY, 3.4), opts);
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          const c = project(v(-2.2, groundY, i * 0.35), opts);
          const d = project(v(3.4, groundY, i * 0.35), opts);
          ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
        }
        ctx.restore();
        const shadow = project(v(this.pose.hips.x, groundY + 0.002, this.pose.hips.z), opts);
        ctx.save();
        ctx.fillStyle = "rgba(8,10,16,.55)";
        ctx.beginPath();
        ctx.ellipse(shadow.x, shadow.y, 0.75 * shadow.scale, 0.22 * shadow.scale, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      return { opts, horizon: groundY };
    }

    /**
     * Rend le mannequin. `mode` : "volume" (ombré), "wireframe" (filaire léger),
     * "depth" (carte de profondeur), "silhouette" (blanc sur noir).
     */
    /**
     * Rend le mannequin. ``mode`` : "volume" (ombré), "wireframe" (filaire),
     * "openpose", "depth" (carte de profondeur), "silhouette" (masque blanc/noir).
     */
    render(ctx, width, height, mode) {
      const m = mode || this.style;
      const opts = this.cameraOptions(width, height);
      const draw = mode !== "silhouette" && mode !== "depth";
      if (draw) this.drawScene(ctx, width, height, { ground: m === "volume" });

      const radii = this.radii();
      const projected = {};
      for (const j of JOINTS) projected[j] = project(this.pose[j], opts);

      // éléments : volumes rigides, rotules, puis os (triés du plus loin au plus proche)
      const items = this.collectItems(m, projected, radii, opts);
      const skip = (m === "volume") ? SKIP_IN_VOLUME : {};
      BONES.forEach(([a, b], i) => {
        if (skip[b] || !projected[a] || !projected[b]) return;
        items.push({
          type: "bone", a: projected[a], b: projected[b], r1: radii[i][0], r2: radii[i][1],
          part: BONES[i][4], depth: (projected[a].depth + projected[b].depth) / 2,
        });
      });
      items.sort((p, q) => q.depth - p.depth);

      const depths = items.map((it) => it.depth);
      const dMin = Math.min.apply(null, depths), dMax = Math.max.apply(null, depths);

      for (const it of items) {
        if (it.type === "torso") { this.drawTorso(ctx, it, m); continue; }
        if (it.type === "cap") {
          const r = Math.max(1, it.r * it.p.scale);
          if (m === "silhouette") ctx.fillStyle = "#ffffff";
          else if (m === "depth") {
            const t = dMax === dMin ? 0.5 : (it.depth - dMin) / (dMax - dMin);
            const g = Math.round(255 - 205 * t);
            ctx.fillStyle = `rgb(${g},${g},${g})`;
          } else {
            const grad = ctx.createLinearGradient(it.p.x - r, 0, it.p.x + r, 0);
            grad.addColorStop(0, shade(COLORS.skin, 1.01));
            grad.addColorStop(0.45, shade(COLORS.skin, 0.95));
            grad.addColorStop(1, shade(COLORS.skin_dark, 0.94));
            ctx.fillStyle = grad;
          }
          ctx.beginPath();
          ctx.arc(it.p.x, it.p.y, r, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        if (m === "depth") {
          const t = dMax === dMin ? 0.5 : (it.depth - dMin) / (dMax - dMin);
          capsule(ctx, it.a, it.b, it.r1, it.r2,
            `rgb(${Math.round(255 - 200 * t)},${Math.round(255 - 200 * t)},${Math.round(255 - 200 * t)})`);
          continue;
        }
        if (m === "silhouette") {
          capsule(ctx, it.a, it.b, it.r1, it.r2, "#ffffff");
          continue;
        }
        if (m === "wireframe") {
          ctx.save();
          ctx.strokeStyle = LIMB_COLOR[it.part] || "#8b93e8";
          ctx.lineWidth = Math.max(2, it.r1 * it.a.scale * 1.2);
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(it.a.x, it.a.y);
          ctx.lineTo(it.b.x, it.b.y);
          ctx.stroke();
          ctx.restore();
          continue;
        }
        // volume : dégradé perpendiculaire à l'os (lumière venant du haut-gauche)
        const dx = it.b.x - it.a.x, dy = it.b.y - it.a.y;
        const l = Math.max(1, Math.hypot(dx, dy));
        const nx = -dy / l, ny = dx / l;
        const cx = (it.a.x + it.b.x) / 2, cy = (it.a.y + it.b.y) / 2;
        const half = Math.max(it.r1 * it.a.scale, it.r2 * it.b.scale);
        const depthT = dMax === dMin ? 0.5 : (it.depth - dMin) / (dMax - dMin);
        const light = 0.98 + 0.10 * (1 - depthT);
        const grad = ctx.createLinearGradient(cx + nx * half, cy + ny * half, cx - nx * half, cy - ny * half);
        grad.addColorStop(0, shade(COLORS.skin, light * 1.04));
        grad.addColorStop(0.45, shade(COLORS.skin, light * 0.95));
        grad.addColorStop(1, shade(COLORS.skin_dark, light * 0.88));
        capsule(ctx, it.a, it.b, it.r1, it.r2, grad);
      }
      if ((m === "volume" || m === "wireframe") && this.showHandles !== false) this.drawHandles(ctx, projected, m);
      return projected;
    }

    // -------------------------------------------------------------- tronc
    /** Repère local du tronc à une fraction f du chemin bassin → cou. */
    torsoFrame(f) {
      const chain = ["hips", "spine", "chest", "neck"];
      // longueurs réelles des segments du tronc (elles suivent le tableau de réglages)
      const weights = [this.lengths.spine, this.lengths.chest, this.lengths.neck];
      const total = weights.reduce((a, b) => a + b, 0) || 1;
      let reste = Math.max(0, Math.min(1, f)) * total, i = 0;
      while (i < weights.length - 1 && reste > weights[i]) { reste -= weights[i]; i++; }
      const t = weights[i] > 0 ? Math.max(0, Math.min(1, reste / weights[i])) : 0;
      const A = this.pose[chain[i]], B = this.pose[chain[i + 1]];
      const centre = lerp(A, B, t);
      const axis = norm(sub(B, A));
      // largeur : des hanches (en bas) aux épaules (en haut)
      const bas = norm(sub(this.pose.hip_l, this.pose.hip_r));
      const haut = norm(sub(this.pose.shoulder_l, this.pose.shoulder_r));
      let x = norm(add(mul(bas, 1 - f), mul(haut, f)));
      x = norm(sub(x, mul(axis, dot(x, axis))));
      if (len(x) < 1e-6) x = norm(sub(this.pose.hip_l, this.pose.hip_r));
      const z = norm(v(x.y * axis.z - x.z * axis.y, x.z * axis.x - x.x * axis.z, x.x * axis.y - x.y * axis.x));
      return { centre: centre, axis: axis, x: x, z: z };
    }

    /** Dimensions du tronc à une fraction f (largeur/profondeur), d'après les réglages. */
    torsoSize(f) {
      const profil = TORSO_PROFILE;
      let a = profil[0], b = profil[profil.length - 1];
      for (let i = 0; i + 1 < profil.length; i++) {
        if (f >= profil[i].at && f <= profil[i + 1].at) { a = profil[i]; b = profil[i + 1]; break; }
      }
      const t = b.at === a.at ? 0 : (f - a.at) / (b.at - a.at);
      const interp = (k) => a[k] + (b[k] - a[k]) * t;
      // mise à l'échelle : épaisseurs réglées dans le tableau + largeurs réelles
      const girth = (this.thickness.spine + this.thickness.chest) / (BASE_THICK.spine + BASE_THICK.chest);
      let width = interp("width") * girth;
      const epaules = dist(this.pose.shoulder_l, this.pose.shoulder_r) + 0.05;
      const hanches = dist(this.pose.hip_l, this.pose.hip_r) + 0.09;
      const refs = [hanches, epaules];
      TORSO_BUMPS.forEach((bump, i) => {
        const g = Math.exp(-Math.pow((f - bump.centre) / bump.sigma, 2));
        width = Math.max(width, refs[i] * bump.scale * Math.max(0.12, g));
      });
      const retrecissement = 1 - 0.25 * Math.max(0, Math.min(1, (f - 0.95) / 0.05));
      return { width: width * retrecissement, depth: interp("depth") * girth * retrecissement };
    }

    /** Points 3D de la coupe f (anneau elliptique). */
    torsoRing(f) {
      const frame = this.torsoFrame(f);
      const taille = this.torsoSize(f);
      const pts = [];
      for (let i = 0; i < TORSO_RING; i++) {
        const th = (i / TORSO_RING) * Math.PI * 2;
        pts.push(add(frame.centre, add(mul(frame.x, Math.cos(th) * taille.width / 2),
          mul(frame.z, Math.sin(th) * taille.depth / 2))));
      }
      return { pts: pts, centre: frame.centre, size: taille };
    }

    /** Élément « tronc » à dessiner (silhouette lissée + profondeur moyenne). */
    torsoItem(opts) {
      const rings = [];
      for (let i = 0; i < TORSO_SLICES; i++) {
        const f = i / (TORSO_SLICES - 1);
        const ring = this.torsoRing(f);
        const proj = ring.pts.map((p) => project(p, opts));
        let gauche = proj[0], droite = proj[0];
        for (const p of proj) {
          if (p.x < gauche.x) gauche = p;
          if (p.x > droite.x) droite = p;
        }
        rings.push({ f: f, proj: proj, gauche: gauche, droite: droite, centre: project(ring.centre, opts) });
      }
      const depth = rings.reduce((acc, r) => acc + r.centre.depth, 0) / rings.length;
      return { type: "torso", rings: rings, depth: depth };
    }

    /** Dessine le tronc : silhouette (chaîne gauche + chaîne droite) remplie et ombrée. */
    drawTorso(ctx, item, mode) {
      const points = [];
      for (const r of item.rings) points.push(r.droite);
      for (let i = item.rings.length - 1; i >= 0; i--) points.push(item.rings[i].gauche);
      if (points.length < 3) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of points) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      }
      ctx.beginPath();
      points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      if (mode === "silhouette") {
        ctx.fillStyle = "#ffffff";
      } else if (mode === "depth") {
        const t = Math.max(0, Math.min(1, (item.depth - (this.camera.distance - 0.7)) / 1.4));
        const g = Math.round(255 - 205 * (1 - t));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
      } else {
        // lumière venue de la gauche : dégradé horizontal + léger modelé vertical
        const grad = ctx.createLinearGradient(minX, 0, maxX, 0);
        grad.addColorStop(0, shade(COLORS.skin, 1.02));
        grad.addColorStop(0.42, shade(COLORS.skin, 0.97));
        grad.addColorStop(1, shade(COLORS.skin_dark, 0.92));
        ctx.fillStyle = grad;
      }
      ctx.fill();
      if (mode === "volume") {
        // ceinture un peu marquée (taille) + ombre sous les côtes : lecture anatomique
        const waist = item.rings.reduce((a, r) => (Math.abs(r.f - 0.45) < Math.abs(a.f - 0.45) ? r : a), item.rings[0]);
        const cotes = item.rings.reduce((a, r) => (Math.abs(r.f - 0.72) < Math.abs(a.f - 0.72) ? r : a), item.rings[0]);
        ctx.save();
        ctx.strokeStyle = "rgba(0,0,0,.10)";
        ctx.lineWidth = Math.max(2, waist.centre.scale * 0.02);
        ctx.beginPath();
        ctx.moveTo(waist.gauche.x, waist.gauche.y);
        ctx.lineTo(waist.droite.x, waist.droite.y);
        ctx.stroke();
        ctx.restore();
        void cotes;
      }
    }

    /** Rotule (sphère) à une articulation : évite les découpes entre deux segments. */
    capRadius(joint, radii) {
      let best = 0;
      BONES.forEach(([, child], i) => {
        if (PARENT[child] !== joint) return;
        best = Math.max(best, Math.min(radii[i][0], radii[i][1]));
      });
      return best * 0.99;
    }

    collectItems(mode, projected, radii, opts) {
      const items = [];
      if (mode === "volume" || mode === "silhouette" || mode === "depth") {
        items.push(this.torsoItem(opts));
        for (const joint of CAP_JOINTS) {
          const r = this.capRadius(joint, radii);
          if (r > 0 && projected[joint]) {
            items.push({ type: "cap", p: projected[joint], r: r, depth: projected[joint].depth - 0.002 });
          }
        }
      }
      return items;
    }

    /** Petits repères sur les articulations (pour l'édition). */
    drawHandles(ctx, projected, mode) {
      ctx.save();
      for (const j of JOINTS) {
        const p = projected[j];
        const active = j === this.selected;
        ctx.beginPath();
        ctx.arc(p.x, p.y, active ? 6 : 3.2, 0, Math.PI * 2);
        ctx.fillStyle = active ? "#ffd479" : "rgba(255,255,255,.75)";
        ctx.fill();
        if (active) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#0a0c10";
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    /** Squelette au format OpenPose (18 points, couleurs canoniques ControlNet). */
    renderOpenPose(ctx, width, height) {
      const opts = this.cameraOptions(width, height);
      const pts = OPENPOSE_18.map((n) => project(this.pose[n], opts));
      ctx.save();
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, width, height);
      ctx.lineCap = "round";
      const stick = Math.max(3, Math.min(width, height) / 150);
      OPENPOSE_LIMBS.forEach(([a, b], i) => {
        const c = OPENPOSE_COLORS[i % OPENPOSE_COLORS.length];
        const A = pts[a], B = pts[b];
        if (!A || !B) return;
        ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
        ctx.lineWidth = stick;
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        ctx.stroke();
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
        ctx.beginPath(); ctx.arc(A.x, A.y, stick * 0.85, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(B.x, B.y, stick * 0.85, 0, Math.PI * 2); ctx.fill();
      });
      ctx.restore();
      return pts;
    }

    /** Points 2D (repère image) du squelette OpenPose, pour réutilisation côté serveur. */
    openposePoints(width, height) {
      const opts = this.cameraOptions(width, height);
      const out = {};
      for (const n of OPENPOSE_18) {
        const p = project(this.pose[n], opts);
        out[n] = [Math.round(p.x), Math.round(p.y)];
      }
      return out;
    }

    /** Description sérialisable de la pose (sauvegarde/partage). */
    toJSON() {
      const pose = {};
      for (const j of JOINTS) pose[j] = [Number(this.pose[j].x.toFixed(4)), Number(this.pose[j].y.toFixed(4)), Number(this.pose[j].z.toFixed(4))];
      return { version: 2, morphology: this.morphology, lengths: Object.assign({}, this.lengths),
        thickness: Object.assign({}, this.thickness), pose: pose,
        camera: { yaw: this.camera.yaw, pitch: this.camera.pitch, distance: this.camera.distance }, preset: this.preset };
    }

    static fromJSON(data) {
      const mannequin = new Mannequin({ morphology: data.morphology, lengths: data.lengths, thickness: data.thickness });
      for (const j in data.pose || {}) {
        if (mannequin.pose[j]) mannequin.pose[j] = v(data.pose[j][0], data.pose[j][1], data.pose[j][2]);
      }
      if (data.camera) {
        mannequin.camera.yaw = data.camera.yaw;
        mannequin.camera.pitch = data.camera.pitch;
        mannequin.camera.distance = data.camera.distance;
      }
      mannequin.preset = data.preset || "personnalise";
      mannequin.enforce();
      return mannequin;
    }
  }

  const api = {
    Mannequin, JOINTS, BONES, PARENT, PRESETS, BUILDS, MORPHOLOGIES, SEGMENTS, TORSO_PROFILE,
    BASE_LENGTHS, BASE_THICK, OPENPOSE_18, OPENPOSE_LIMBS, OPENPOSE_COLORS,
    boneLengths, boneThickness, boneRadii, morphedDimensions, defaultPose, solveTwoBoneIK,
    enforceBoneLengths, project, capsule,
    v, sub, add, mul, norm, dist, len,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  global.Mannequin = api.Mannequin;
  global.MannequinKit = api;
})(typeof window !== "undefined" ? window : globalThis);
