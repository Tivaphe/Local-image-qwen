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

  const cross = (a, b) => v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
  const angleEntre = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(norm(a), norm(b)))));
  const rad = (degres) => (degres * Math.PI) / 180;
  const deg = (r) => (r * 180) / Math.PI;

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
    spine: 0.16, chest: 0.20, neck: 0.115, head: 0.13, head_top: 0.10,
    shoulder_l: 0.19, elbow_l: 0.28, wrist_l: 0.25, hand_l: 0.10,
    shoulder_r: 0.19, elbow_r: 0.28, wrist_r: 0.25, hand_r: 0.10,
    hip_l: 0.11, knee_l: 0.44, ankle_l: 0.42, toe_l: 0.17, heel_l: 0.08,
    hip_r: 0.11, knee_r: 0.44, ankle_r: 0.42, toe_r: 0.17, heel_r: 0.08,
  };
  const BASE_THICK = {
    spine: 0.262, chest: 0.315, neck: 0.116, head: 0.172, head_top: 0.150,
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
    homme: { stature: 1.03, shoulders: 1.10, legs: 1.0, arms: 1.03, girth: 1.12 },
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
    { at: -0.14, width: 0.158, depth: 0.168 },   // entrejambe (sous le bassin)
    { at: 0.00, width: 0.180, depth: 0.175 },    // sous le bassin
    { at: 0.16, width: 0.205, depth: 0.205 },    // bassin / hanches (renforcé par le bombé)
    { at: 0.45, width: 0.166, depth: 0.178 },    // taille
    { at: 0.70, width: 0.238, depth: 0.196 },    // bas des côtes
    { at: 0.86, width: 0.302, depth: 0.202 },    // haut du thorax
    { at: 0.94, width: 0.330, depth: 0.192 },    // épaules
    { at: 1.00, width: 0.150, depth: 0.150 },    // base du cou
  ];
  const TORSO_BUMPS = [
    { centre: 0.16, sigma: 0.15, scale: 1.05 },   // bassin
    { centre: 0.90, sigma: 0.10, scale: 0.93 },   // ceinture scapulaire
  ];
  const TORSO_SLICES = 19;
  const TORSO_BAS = -0.12;                       // le tronc descend jusqu'à l'entrejambe
  const TORSO_RING = 14;
  // os remplacés par le tronc en rendu 3D (ischions : le bassin les englobe)
  const SKIP_IN_VOLUME = { hip_l: true, hip_r: true };
  const JOINT_BALLS = [];                 // rotules à dessiner en plus des segments (aucune par défaut)
  const CAP_JOINTS = ["elbow_l", "elbow_r", "knee_l", "knee_r", "wrist_l", "wrist_r", "ankle_l", "ankle_r"];

  // Libellés du tableau de réglages (l'interface les affiche tels quels)
  // Galbe de chaque segment (facteur appliqué au rayon de référence, 0 = milieu
  // du parent, 1 = extrémité). Les profils sont normalisés : l'épaisseur réglée
  // dans le tableau reste le diamètre moyen du segment.
  const LIMB_PROFILE = {
    spine: [[0, 1.0], [1, 1.0]],
    chest: [[0, 1.0], [1, 1.0]],
    neck: [[0, 1.04], [1, 0.88]],
    head: [[0, 0.86], [0.5, 1.0], [1, 0.92]],
    head_top: [[0, 1.0], [1, 0.8]],
    shoulder_l: [[0, 0.98], [1, 0.9]], shoulder_r: [[0, 0.98], [1, 0.9]],
    // bras : deltoïde puis biceps, affiné au coude
    elbow_l: [[0, 0.62], [0.14, 1.34], [0.42, 1.26], [0.72, 1.0], [1, 0.72]],
    elbow_r: [[0, 0.62], [0.14, 1.34], [0.42, 1.26], [0.72, 1.0], [1, 0.72]],
    // avant-bras : galbe du rond pronateur, fin au poignet
    wrist_l: [[0, 0.7], [0.18, 1.18], [0.5, 1.06], [0.85, 0.7], [1, 0.56]],
    wrist_r: [[0, 0.7], [0.18, 1.18], [0.5, 1.06], [0.85, 0.7], [1, 0.56]],
    // main : paume pleine puis doigts
    hand_l: [[0, 0.9], [0.35, 1.06], [0.7, 0.94], [1, 0.6]],
    hand_r: [[0, 0.9], [0.35, 1.06], [0.7, 0.94], [1, 0.6]],
    hip_l: [[0, 1.0], [1, 0.78]], hip_r: [[0, 1.0], [1, 0.78]],
    // cuisse : fessier puis quadriceps, genou fin
    knee_l: [[0, 0.7], [0.14, 1.24], [0.46, 1.16], [0.78, 0.9], [1, 0.66]],
    knee_r: [[0, 0.7], [0.14, 1.24], [0.46, 1.16], [0.78, 0.9], [1, 0.66]],
    // jambe : mollet, cheville très fine
    ankle_l: [[0, 0.72], [0.22, 1.22], [0.5, 1.04], [0.8, 0.68], [1, 0.5]],
    ankle_r: [[0, 0.72], [0.22, 1.22], [0.5, 1.04], [0.8, 0.68], [1, 0.5]],
    toe_l: [[0, 1.0], [0.6, 0.94], [1, 0.78]], toe_r: [[0, 1.0], [0.6, 0.94], [1, 0.78]],
    heel_l: [[0, 1.0], [1, 0.88]], heel_r: [[0, 1.0], [1, 0.88]],
  };
  // normalisation : chaque profil a une moyenne de 1 (l'épaisseur réglée = diamètre moyen)
  Object.keys(LIMB_PROFILE).forEach((k) => {
    const prof = LIMB_PROFILE[k];
    let somme = 0, span = 0;
    for (let i = 0; i + 1 < prof.length; i++) {
      const d = prof[i + 1][0] - prof[i][0];
      somme += ((prof[i][1] + prof[i + 1][1]) / 2) * d;
      span += d;
    }
    const moyenne = span > 0 ? somme / span : 1;
    if (moyenne > 0) prof.forEach((point) => { point[1] /= moyenne; });
  });

  /** Facteur de galbe d'un segment à la fraction u (0 = parent, 1 = extrémité). */
  function profileAt(child, u) {
    const prof = LIMB_PROFILE[child];
    if (!prof) return 1;
    const x = Math.max(0, Math.min(1, u));
    for (let i = 0; i + 1 < prof.length; i++) {
      if (x >= prof[i][0] && x <= prof[i + 1][0]) {
        const d = prof[i + 1][0] - prof[i][0] || 1;
        const t = (x - prof[i][0]) / d;
        return prof[i][1] + (prof[i + 1][1] - prof[i][1]) * t;
      }
    }
    return prof[prof.length - 1][1];
  }

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

  // ------------------------------------------------------- butées articulaires
  // « cone » : écart maximal de l'os par rapport à sa direction de repos (degrés).
  // « pli »  : flexion autorisée par rapport au segment parent (degrés, sens unique :
  //            pas d'hyperextension pour un coude ou un genou).
  // « sens » : direction dans laquelle le bout de l'os doit partir quand l'angle
  // augmente (sert aux curseurs : un bras se lève vers l'extérieur, un genou plie
  // vers l'arrière, la nuque se penche vers l'avant).
  const LIMITS = {
    neck: { cone: 45, pli: [-40, 45], sens: [0, 0, 1] },
    head: { cone: 38, pli: [-35, 40], sens: [0, 0, 1] },
    head_top: { cone: 18 },
    shoulder_l: { cone: 28 }, shoulder_r: { cone: 28 },
    // L'épaule tourne dans tous les plans : un seul cône, l'anti-collision fait le reste.
    elbow_l: { cone: 170, sens: [1, 0, 0] },
    elbow_r: { cone: 170, sens: [-1, 0, 0] },
    wrist_l: { cone: 180, pli: [0, 150], sens: [0, 0, 1] },     // coude
    wrist_r: { cone: 180, pli: [0, 150], sens: [0, 0, 1] },
    hand_l: { cone: 100, pli: [0, 85], sens: [0, 0, 1] },       // poignet
    hand_r: { cone: 100, pli: [0, 85], sens: [0, 0, 1] },
    hip_l: { cone: 30 }, hip_r: { cone: 30 },
    knee_l: { cone: 160, pli: [-25, 115], sens: [0, 0, 1] },    // hanche
    knee_r: { cone: 160, pli: [-25, 115], sens: [0, 0, 1] },
    ankle_l: { cone: 170, pli: [0, 145], sens: [0, 0, -1] },    // genou : vers l'arrière
    ankle_r: { cone: 170, pli: [0, 145], sens: [0, 0, -1] },
    toe_l: { cone: 110, pli: [-75, 40], sens: [0, 1, 0] },      // cheville : orteils vers le haut
    toe_r: { cone: 110, pli: [-75, 40], sens: [0, 1, 0] },
    heel_l: { cone: 110 }, heel_r: { cone: 110 },      // talon : solidaire du pied
  };
  for (const cote of ["l", "r"]) {
    for (const base of ["shoulder", "elbow", "wrist", "hand", "hip", "knee", "ankle", "toe", "heel"]) {
      const lim = LIMITS[base + "_l"];
      if (lim) LIMITS[base + "_" + cote] = lim;
    }
  }
  // Pli de repos (angle os/parent dans la pose debout) : les butées s'ajoutent à cette valeur.
  const REST_BEND = {};
  for (const name of BUILD_ORDER) {
    const parent = PARENT[name];
    const dir = REST_DIRS[name];
    const dirParent = parent && REST_DIRS[parent];
    REST_BEND[name] = (dir && dirParent) ? angleEntre(v(dir[0], dir[1], dir[2]), v(dirParent[0], dirParent[1], dirParent[2])) : 0;
  }
  // Articulations qui ne doivent jamais entrer dans le volume du tronc (coude dans le cou…).
  const MEMBRES_TESTES = ["elbow_l", "wrist_l", "hand_l", "elbow_r", "wrist_r", "hand_r"];

  // Tendons/élastiques : [articulation, os parent, os enfant] — dessinés du côté qui
  // s'ouvre quand l'articulation plie, et d'autant plus tendus que le pli est marqué.
  const TENDONS = [];
  for (const cote of ["l", "r"]) {
    TENDONS.push(["shoulder_" + cote, "shoulder_" + cote, "elbow_" + cote, [-1, 0, 0]]);
    TENDONS.push(["elbow_" + cote, "elbow_" + cote, "wrist_" + cote, [1, 0, 0]]);
    TENDONS.push(["hip_" + cote, "hip_" + cote, "knee_" + cote, [1, 0, 0]]);
    TENDONS.push(["knee_" + cote, "knee_" + cote, "ankle_" + cote, [-1, 0, 0]]);
  }

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
    skin: [214, 176, 152],          // albédo de la peau (peau claire, mat)
    skin_dark: [138, 100, 86],      // bords / ombres
    under: [226, 128, 104],         // liseré chaud (lumière qui traverse la peau)
    tendon: [242, 238, 232],        // élastiques et tendons
    bone: [92, 104, 255],
  };
  const SKIN_AMBIENT = 0.52, SKIN_DIFFUSE = 0.56, SKIN_SPEC = 0.22, SKIN_SHIN = 26;
  const LIGHT = norm(v(-0.38, 0.82, 0.42));       // lumière de haut-gauche, légèrement devant

  function shade(rgb, factor) {
    return `rgb(${Math.max(0, Math.min(255, Math.round(rgb[0] * factor)))},` +
      `${Math.max(0, Math.min(255, Math.round(rgb[1] * factor)))},` +
      `${Math.max(0, Math.min(255, Math.round(rgb[2] * factor)))})`;
  }

  /**
   * Couleur de la peau pour une normale unitaire 3D : Lambert + reflet + liseré chaud
   * sur les bords (la lumière traverse un peu la peau, comme de la chair).
   */
  function skinFor(n, eye, out) {
    const diff = Math.max(0, dot(n, LIGHT));
    const hx = LIGHT.x + eye.x, hy = LIGHT.y + eye.y, hz = LIGHT.z + eye.z;
    const hl = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
    const nh = Math.max(0, (n.x * hx + n.y * hy + n.z * hz) / hl);
    const spec = Math.pow(nh, SKIN_SHIN) * SKIN_SPEC;
    const face = Math.abs(dot(n, eye));                       // 1 = face à la caméra
    const k = SKIN_AMBIENT + SKIN_DIFFUSE * diff;
    const bord = Math.pow(1 - face, 2.2);                     // 1 = bord de silhouette
    const sss = bord * 0.38;
    out[0] = COLORS.skin[0] * k + COLORS.under[0] * sss + 255 * spec;
    out[1] = COLORS.skin[1] * k + COLORS.under[1] * sss + 248 * spec;
    out[2] = COLORS.skin[2] * k + COLORS.under[2] * sss + 242 * spec;
    return out;
  }

  function skinCss(n, eye) {
    const c = skinFor(n, eye, [0, 0, 0]);
    return `rgb(${Math.min(255, Math.round(c[0]))},${Math.min(255, Math.round(c[1]))},${Math.min(255, Math.round(c[2]))})`;
  }

  /** Capsule simple (segments fins du mode filaire, ombres, rotules). */
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

  /** Repère de vue : matrice de rotation + axe caméra (vers l'objectif). */
  function viewOf(opts) {
    const r = rotation(opts.yaw, opts.pitch);
    return { r: r, eye: v(r[2][0], r[2][1], r[2][2]), opts: opts };
  }

  /** Direction écran (x vers la droite, y vers le bas) d'une direction 3D. */
  function screenDir(view, dir) {
    return [
      dir.x * view.r[0][0] + dir.y * view.r[0][1] + dir.z * view.r[0][2],
      -(dir.x * view.r[1][0] + dir.y * view.r[1][1] + dir.z * view.r[1][2]),
    ];
  }

  /** Repère d'ombrage d'un segment : axes du plan perpendiculaire + axe caméra. */
  function limbFrame(a3, b3, view, nx, ny) {
    const w = norm(sub(b3, a3));
    let u = norm(v(w.y * view.eye.z - w.z * view.eye.y, w.z * view.eye.x - w.x * view.eye.z,
      w.x * view.eye.y - w.y * view.eye.x));
    if (len(u) < 1e-6) u = v(1, 0, 0);
    // l'axe « u » doit pointer du même côté que la perpendiculaire écran (nx, ny)
    const sd = screenDir(view, u);
    if (sd[0] * nx + sd[1] * ny < 0) u = mul(u, -1);
    let z = norm(v(u.y * w.z - u.z * w.y, u.z * w.x - u.x * w.z, u.x * w.y - u.y * w.x));
    if (dot(z, view.eye) < 0) z = mul(z, -1);
    return { w: w, u: u, z: z };
  }

  /** Dégradé de peau d'un tube : échantillonne la normale tout autour du cylindre. */
  function tubeGradient(ctx, a3, b3, view, nx, ny, rWorld) {
    const frame = limbFrame(a3, b3, view, nx, ny);
    const cx = (a3.x + b3.x) / 2, cy = (a3.y + b3.y) / 2;
    const s = project({ x: cx, y: cy, z: cx === a3.x && cy === a3.y ? a3.z : (a3.z + b3.z) / 2 }, view.opts || {});
    void s;
    return { frame: frame, nx: nx, ny: ny };
  }

  /**
   * Remplissage de peau d'un segment : dégradé perpendiculaire calculé sur la
   * vraie normale 3D du cylindre (Lambert + reflet + liseré chaud).
   */
  function skinFill(ctx, a, b, a3, b3, view, rScreen) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const l = Math.max(1e-6, Math.hypot(dx, dy));
    const nx = -dy / l, ny = dx / l;
    const frame = limbFrame(a3, b3, view, nx, ny);
    const grad = ctx.createLinearGradient(a.x + nx * rScreen, a.y + ny * rScreen,
      a.x - nx * rScreen, a.y - ny * rScreen);
    const N = 10;
    for (let i = 0; i <= N; i++) {
      const s01 = i / N, th = Math.PI * s01;
      const c = Math.cos(th), sn = Math.sin(th);
      const nrm = v(frame.u.x * c + frame.z.x * sn, frame.u.y * c + frame.z.y * sn, frame.u.z * c + frame.z.z * sn);
      grad.addColorStop(s01, skinCss(nrm, view.eye));
    }
    return grad;
  }

  /**
   * Dessine un segment de chair : silhouette musclée (profil de rayon) + peau ombrée.
   * `radiusAt(u)` donne le rayon (unités monde) à la fraction u du segment.
   */
  function limb(ctx, a, b, radiusAt, a3, b3, view, fill) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const l = Math.max(1e-6, Math.hypot(dx, dy));
    const nx = -dy / l, ny = dx / l;
    const N = 16;
    const gauche = [], droite = [];
    for (let i = 0; i <= N; i++) {
      const u = i / N;
      const cx = a.x + dx * u, cy = a.y + dy * u;
      const scale = a.scale + (b.scale - a.scale) * u;
      const r = Math.max(1.2, radiusAt(u) * scale);
      gauche.push([cx + nx * r, cy + ny * r]);
      droite.push([cx - nx * r, cy - ny * r]);
    }
    const rFin = Math.max(1.2, radiusAt(1) * b.scale);
    const rDeb = Math.max(1.2, radiusAt(0) * a.scale);
    const remplissage = fill ? fill : skinFill(ctx, a, b, a3, b3, view, Math.max(rDeb, rFin));
    ctx.beginPath();
    gauche.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
    ctx.arc(b.x, b.y, rFin, Math.atan2(ny, nx) - Math.PI / 2, Math.atan2(ny, nx) + Math.PI / 2);
    for (let i = droite.length - 1; i >= 0; i--) ctx.lineTo(droite[i][0], droite[i][1]);
    ctx.arc(a.x, a.y, rDeb, Math.atan2(ny, nx) + Math.PI / 2, Math.atan2(ny, nx) + 3 * Math.PI / 2);
    ctx.closePath();
    ctx.fillStyle = remplissage;
    ctx.fill();
    // calottes aux extrémités : bouchent le coin laissé par une articulation pliée
    for (const [c, r] of [[a, rDeb], [b, rFin]]) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    return Math.max(rDeb, rFin);
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
        this.deplacer(ext, this.pose[ext]);
      }
      this.clamp();
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
    /**
     * Règle l'angle d'une articulation en degrés (0 = position de repos, positif =
     * flexion). Sert aux curseurs d'angles : l'os tourne autour de son axe de pliage,
     * puis les butées et l'anti-collision s'appliquent.
     */
    setJointAngle(child, degres) {
      const parent = PARENT[child];
      if (!parent || !this.pose[child] || !this.pose[parent]) return this;
      const lim = LIMITS[child];
      if (!lim) return this;
      const borne = lim.pli ? lim.pli : [-lim.cone, lim.cone];
      const angle = Math.max(borne[0], Math.min(borne[1], degres));
      const grand = PARENT[parent];
      const dirParent = grand && this.pose[grand] ? norm(sub(this.pose[parent], this.pose[grand])) : v(0, 1, 0);
      const dir = REST_DIRS[child];
      if (!dir) return this;
      const repos = norm(v(dir[0], dir[1], dir[2]));
      let axe = norm(cross(repos, dirParent));
      if (len(axe) < 0.2) axe = norm(cross(repos, v(1, 0, 0)));
      if (len(axe) < 0.2) axe = norm(cross(repos, v(0, 1, 0)));
      const candidats = [1, -1].map((sens) => norm(rotateAround(repos, mul(axe, sens), rad(angle))));
      const pref = lim.sens ? v(lim.sens[0], lim.sens[1], lim.sens[2]) : null;
      const note = (dir) => dot(mul(dir, this.lengths[child] || 0.2), pref);
      const direction = pref && note(candidats[0]) < note(candidats[1]) ? candidats[1] : candidats[0];
      const base = this.pose[parent];
      const courant = norm(sub(this.pose[child], base));
      // rotation rigide du membre sous l'articulation : les os suivent, l'articulation plie
      const axeRotation = norm(cross(courant, direction));
      const ecart = angleEntre(courant, direction);
      if (len(axeRotation) > 1e-6 && ecart > 1e-9) {
        for (const nom of this.descendants(child)) {
          this.pose[nom] = add(base, rotateAround(sub(this.pose[nom], base), axeRotation, ecart));
        }
      }
      this.enforce();
      this.clamp();
      return this;
    }

    /** Angle courant d'une articulation (degrés, 0 = repos) — pour les curseurs. */
    jointAngle(child) {
      const parent = PARENT[child];
      if (!parent || !this.pose[child] || !this.pose[parent]) return 0;
      const out = norm(sub(this.pose[child], this.pose[parent]));
      const lim = LIMITS[child] || {};
      const dir = REST_DIRS[child];
      if (!lim.pli && dir) {
        // articulation « à cône » (épaule) : écart signé par rapport à la position de repos
        const repos = norm(v(dir[0], dir[1], dir[2]));
        const pref = lim.sens ? v(lim.sens[0], lim.sens[1], lim.sens[2]) : v(0, 0, 1);
        const sens = dot(out, pref) >= 0 ? 1 : -1;
        return deg(angleEntre(out, repos)) * sens;
      }
      const grand = PARENT[parent];
      const dirParent = grand && this.pose[grand] ? norm(sub(this.pose[parent], this.pose[grand])) : v(0, 1, 0);
      return deg(angleEntre(out, dirParent) - (REST_BEND[child] || 0));
    }

    /** Copie de la pose (pour annuler un geste qui entre dans le corps). */
    snapshot() {
      const out = {};
      for (const key in this.pose) out[key] = clone(this.pose[key]);
      return out;
    }

    /** Restaure une pose copiée par snapshot(). */
    restore(copie) {
      const propre = {};
      for (const key in copie) propre[key] = clone(copie[key]);
      this.pose = propre;
      return this;
    }

    /**
     * Déplacement d'une articulation avec butées anatomiques et anti-collision :
     * si le membre entrerait dans le tronc, la pose se rapproche au plus près par
     * dichotomie — la main s'arrête sur le corps au lieu de le traverser.
     */
    moveJoint(name, target) {
      if (!this.pose[name]) return this;
      const avant = this.snapshot();
      const base = clone(this.pose[name]);
      this.deplacer(name, target);
      if (!this.collides()) return this;
      let bon = 0, mauvais = 1;
      for (let i = 0; i < 7; i++) {
        const t = (bon + mauvais) / 2;
        this.restore(avant);
        this.deplacer(name, lerp(base, target, t));
        if (this.collides()) mauvais = t; else bon = t;
      }
      this.restore(avant);
      this.deplacer(name, lerp(base, target, bon));
      return this;
    }

    /**
     * Vrai si une articulation du bras est entrée dans le volume du tronc
     * (le mannequin butte alors contre lui-même).
     */
    collides() {
      const basse = this.pose.neck;
      const ecart = Math.max(1e-3, basse.y - this.pose.hips.y);
      for (const name of MEMBRES_TESTES) {
        const p = this.pose[name];
        if (!p || p.y > basse.y + 0.02) continue;
        const f = Math.max(0, Math.min(1, (p.y - this.pose.hips.y) / ecart));
        if (f <= 0.03) continue;
        const taille = this.torsoSize(f);
        const centre = this.torsoFrame(f).centre;
        const dx = (p.x - centre.x) / Math.max(1e-3, taille.width / 2);
        const dz = (p.z - centre.z) / Math.max(1e-3, taille.depth / 2);
        if (dx * dx + dz * dz < 0.98) return true;
      }
      return false;
    }

    /** Ramène chaque os dans ses butées (cône autour du repos + charnière du parent). */
    clamp() {
      for (const name of BUILD_ORDER) {
        const parent = PARENT[name];
        if (!parent || !this.pose[name] || !this.pose[parent]) continue;
        const lim = LIMITS[name];
        if (!lim) continue;
        const p = this.pose[parent];
        const long = this.lengths[name] || dist(p, this.pose[name]);
        const grandParent = PARENT[parent];
        const dirParent = grandParent && this.pose[grandParent]
          ? norm(sub(p, this.pose[grandParent])) : v(0, 1, 0);
        let out = norm(sub(this.pose[name], p));
        if (len(out) < 1e-9) continue;
        if (lim.pli) {
          const pli = deg(angleEntre(out, dirParent) - (REST_BEND[name] || 0));   // degrés, comme le tableau
          const cible = Math.max(lim.pli[0], Math.min(lim.pli[1], pli));
          if (Math.abs(cible - pli) > 1e-6) out = rotateAround(out, norm(cross(out, dirParent)), rad(pli - cible));
        }
        const dir = REST_DIRS[name];
        if (lim.cone !== undefined && dir) {
          const repos = norm(v(dir[0], dir[1], dir[2]));
          const ecart = angleEntre(out, repos);
          if (ecart > rad(lim.cone)) out = rotateAround(out, norm(cross(out, repos)), ecart - rad(lim.cone));
        }
        this.pose[name] = add(p, mul(out, long));
      }
      return this;
    }

    /** Déplacement brut (sans anti-collision) : sert à moveJoint et aux poses types. */
    deplacer(name, target) {
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
        this.clamp();
        // le pied reste rigide : le talon suit exactement l'orientation de la pointe,
        // même quand les butées ont retouché celle-ci.
        const dirPointe = norm(sub(this.pose[name], ankle));
        const ecart2 = angleEntre(newDir, dirPointe);
        const axe2 = norm(cross(newDir, dirPointe));
        if (len(axe2) > 1e-6 && ecart2 > 1e-9) {
          const sibDir = rotateAround(norm(sub(this.pose[sibling], ankle)), axe2, ecart2);
          this.pose[sibling] = add(ankle, mul(sibDir, this.lengths[sibling] || dist(ankle, this.pose[sibling])));
        }
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
        this.clamp();
        return this;
      }
      // articulation « montante » : on déplace puis on remet la chaîne à longueur,
      // ce qui fait tourner les os parents sans changer leurs longueurs
      this.pose[name] = clone(target);
      this.enforce();
      this.clamp();
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
      const view = viewOf(opts);
      const drawFond = m !== "silhouette" && m !== "depth";
      if (drawFond) this.drawScene(ctx, width, height, { ground: m === "volume" });

      const projected = {};
      for (const j of JOINTS) projected[j] = project(this.pose[j], opts);

      const anatomique = (m === "volume" || m === "silhouette" || m === "depth");
      const items = [];
      if (anatomique) {
        items.push(this.torsoItem(opts));
        items.push(this.headItem(opts));
        for (const cote of ["l", "r"]) items.push(this.footItem(cote, opts));
      }
      BONES.forEach(([a, b, , , part]) => {
        const child = b;
        if (anatomique && (SKIP_IN_VOLUME[child] || child === "spine" || child === "chest" ||
          child === "shoulder_l" || child === "shoulder_r" ||
          child === "head" || child === "head_top" ||
          child === "nose" || child === "toe_l" || child === "toe_r" || child === "heel_l" || child === "heel_r")) return;
        const pa = projected[a], pb = projected[b];
        if (!pa || !pb) return;
        items.push({ type: "bone", a: pa, b: pb, a3: this.pose[a], b3: this.pose[b], child: child,
          part: part, r: this.meanRadius(child), depth: (pa.depth + pb.depth) / 2 });
      });
      if (anatomique) {
        // aucune rotule visible : le galbe des segments ferme les articulations tout seul
        for (const j of JOINT_BALLS) {
          items.push({ type: "joint", p: projected[j], r: this.jointRadius(j), depth: projected[j].depth - 0.005 });
        }
        if (m === "volume") {
          for (const [joint, parentBone, childBone, axe] of TENDONS) {
            const it = this.tendonItem(joint, parentBone, childBone, axe);
            if (!it || !projected[joint]) continue;
            // dessiné devant les deux segments qu'il relie (sinon le membre le recouvre)
            it.depth = Math.min(projected[parentBone].depth, projected[childBone].depth) - 0.02;
            items.push(it);
          }
        }
      }
      items.sort((p, q) => q.depth - p.depth);

      const depths = items.map((it) => it.depth);
      const dMin = Math.min.apply(null, depths), dMax = Math.max.apply(null, depths);
      const gris = (d) => {
        const t = dMax === dMin ? 0.5 : (d - dMin) / (dMax - dMin);
        const g = Math.round(255 - 205 * t);
        return `rgb(${g},${g},${g})`;
      };

      for (const it of items) {
        if (it.type === "torso") { this.drawTorso(ctx, it, m, view); continue; }
        if (it.type === "head") { this.drawHead(ctx, it, m, view); continue; }
        if (it.type === "foot") { this.drawFoot(ctx, it, m, view); continue; }
        if (it.type === "tendon") { this.drawTendon(ctx, it, view); continue; }
        if (it.type === "joint") {
          const r = Math.max(1, it.r * it.p.scale);
          if (m === "silhouette") { ctx.fillStyle = "#ffffff"; }
          else if (m === "depth") { ctx.fillStyle = gris(it.depth); }
          else { ctx.fillStyle = this.jointFill(ctx, it.p, r, view); }
          ctx.beginPath();
          ctx.arc(it.p.x, it.p.y, r, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        if (m === "wireframe") {
          ctx.save();
          ctx.strokeStyle = LIMB_COLOR[it.part] || "#8b93e8";
          ctx.lineWidth = Math.max(2, it.r * it.a.scale * 2.1);
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(it.a.x, it.a.y);
          ctx.lineTo(it.b.x, it.b.y);
          ctx.stroke();
          ctx.restore();
          continue;
        }
        const fill = m === "silhouette" ? "#ffffff" : (m === "depth" ? gris(it.depth) : null);
        limb(ctx, it.a, it.b, (u) => it.r * profileAt(it.child, u), it.a3, it.b3, view, fill);
      }
      if ((m === "volume" || m === "wireframe") && this.showHandles !== false) this.drawHandles(ctx, projected, m);
      return projected;
    }

    /**
     * Tendon/élastique d'une articulation : attaches de part et d'autre, posées du
     * côté qui s'ouvre quand l'articulation plie. `tension` va de 0 (repos) à 1 (pli
     * maximal autorisé par les butées).
     */
    tendonItem(joint, parentBone, childBone, axePrefere) {
      // `parentBone` se termine à l'articulation : son origine est le parent de cet os.
      const origine = PARENT[parentBone] ? this.pose[PARENT[parentBone]] : null;
      const a = origine || this.pose[parentBone];
      const j = this.pose[joint], b = this.pose[childBone];
      if (!a || !j || !b) return null;
      const da = norm(sub(j, a)), db = norm(sub(b, j));
      if (len(da) < 1e-6 || len(db) < 1e-6) return null;
      const limite = LIMITS[childBone];
      const pliMax = limite && limite.pli ? Math.max(1, limite.pli[1]) : 120;
      const pli = Math.abs(angleEntre(da, db) - (REST_BEND[childBone] || 0));
      const tension = Math.max(0, Math.min(1, (pli * 180 / Math.PI) / pliMax));
      const rayon = Math.max(0.02, this.meanRadius(childBone) * profileAt(childBone, 0.06));
      // Côté extérieur au pli : bissectrice extérieure (da - db). Membre tendu : côté préféré.
      let dehors = norm(sub(da, db));
      if (len(dehors) < 0.2) {
        dehors = norm(cross(v(axePrefere[0], axePrefere[1], axePrefere[2]), da));
      }
      if (len(dehors) < 0.2) dehors = norm(cross(v(0, 1, 0), da));
      if (len(dehors) < 0.2) dehors = v(0, 0, -1);
      const pA = add(lerp(a, j, 0.86), mul(dehors, rayon * 0.34));
      const pB = add(lerp(j, b, 0.18), mul(dehors, rayon * 0.34));
      const ctrl = add(lerp(pA, pB, 0.5), mul(dehors, rayon * (0.06 + tension * 0.55)));
      return { type: "tendon", A: pA, B: pB, ctrl: ctrl, tension: tension, rayon: rayon, joint: joint };
    }

    /** Dessine un tendon : ombre portée douce puis le cordon clair, tendu selon le pli. */
    drawTendon(ctx, item, view) {
      const A = project(item.A, view.opts), B = project(item.B, view.opts), C = project(item.ctrl, view.opts);
      if (!A || !B || !C || !isFinite(A.x) || !isFinite(B.x)) return;
      const largeur = Math.max(1.2, 0.0095 * ((A.scale + B.scale) / 2));
      ctx.save();
      ctx.lineCap = "round";
      const trace = () => {
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.quadraticCurveTo(C.x, C.y, B.x, B.y);
        ctx.stroke();
      };
      ctx.globalAlpha = 0.06 + 0.14 * item.tension;
      ctx.strokeStyle = "#5a3226";
      ctx.lineWidth = largeur * 1.6;
      trace();
      ctx.globalAlpha = 0.14 + 0.44 * item.tension;
      ctx.strokeStyle = "#f2ddcd";
      ctx.lineWidth = largeur * 0.9;
      trace();
      ctx.restore();
    }

    /** Rotule ombrée : dégradé radial décalé vers la lumière. */
    jointFill(ctx, p, r, view) {
      const l = screenDir(view, LIGHT);
      const ll = Math.hypot(l[0], l[1]) || 1;
      const grad = ctx.createRadialGradient(p.x + (l[0] / ll) * r * 0.45, p.y + (l[1] / ll) * r * 0.45, r * 0.05,
        p.x, p.y, r * 1.02);
      grad.addColorStop(0, skinCss(this.melangeNormale(view, 0.85), view.eye));
      grad.addColorStop(0.5, skinCss(this.melangeNormale(view, 0.25), view.eye));
      grad.addColorStop(1, skinCss(this.melangeNormale(view, -0.35), view.eye));
      return grad;
    }

    /** Normale mélangeant la direction caméra et la lumière (nuances des rotules). */
    melangeNormale(view, k) {
      const n = v(view.eye.x * (1 - k) + LIGHT.x * k, view.eye.y * (1 - k) + LIGHT.y * k, view.eye.z * (1 - k) + LIGHT.z * k);
      return len(n) < 1e-6 ? view.eye : norm(n);
    }

    /** Dégradé de peau d'un anneau (ellipse) : normale vraie le long de la largeur. */
    ringSkinFill(ctx, gauche, droite, centre, xAxis, zAxis, view) {
      const sd = screenDir(view, xAxis);
      const demi = Math.max(2, Math.abs(sd[0] * (droite.x - gauche.x) / 2) + Math.abs(sd[1] * (droite.x - gauche.x) / 2));
      const grad = ctx.createLinearGradient(centre.x - sd[0] * demi, centre.y - sd[1] * demi,
        centre.x + sd[0] * demi, centre.y + sd[1] * demi);
      const N = 12;
      for (let i = 0; i <= N; i++) {
        const s01 = i / N, th = Math.PI * s01;
        const c = Math.cos(th), sn = Math.sin(th);
        const nrm = v(xAxis.x * c + zAxis.x * sn, xAxis.y * c + zAxis.y * sn, xAxis.z * c + zAxis.z * sn);
        grad.addColorStop(s01, skinCss(nrm, view.eye));
      }
      return grad;
    }

    /** Dessine un volume d'anneaux (tronc, tête) ombré, avec repères anatomiques. */
    fillRings(ctx, rings, mode, view, options) {
      const o = options || {};
      const points = [];
      for (const r of rings) points.push(r.droite);
      for (let i = rings.length - 1; i >= 0; i--) points.push(rings[i].gauche);
      if (points.length < 3) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of points) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      }
      if (mode === "silhouette") {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.closePath();
        ctx.fill();
      } else if (mode === "depth") {
        const g = Math.round(255 - 205 * Math.max(0, Math.min(1, o.depthT === undefined ? 0.5 : o.depthT)));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.beginPath();
        points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.closePath();
        ctx.fill();
      } else {
        // ombrage par bandes : chaque anneau a sa propre normale (le volume tourne)
        const ref = rings[Math.floor(rings.length / 2)];
        for (let i = 0; i + 1 < rings.length; i++) {
          const a = rings[i], b = rings[i + 1];
          const milieu = { centre: { x: (a.centre.x + b.centre.x) / 2, y: (a.centre.y + b.centre.y) / 2 },
            gauche: Math.abs(a.droite.x - a.gauche.x) > Math.abs(b.droite.x - b.gauche.x) ? a.gauche : b.gauche,
            droite: Math.abs(a.droite.x - a.gauche.x) > Math.abs(b.droite.x - b.gauche.x) ? a.droite : b.droite };
          ctx.beginPath();
          ctx.moveTo(a.droite.x, a.droite.y);
          ctx.lineTo(b.droite.x, b.droite.y);
          ctx.lineTo(b.gauche.x, b.gauche.y);
          ctx.lineTo(a.gauche.x, a.gauche.y);
          ctx.closePath();
          let axeX = a.x || v(1, 0, 0);
          let axeZ = a.z || v(0, 0, 1);
          if (ref && ref.x && dot(axeX, ref.x) < 0) axeX = mul(axeX, -1);
          if (ref && ref.z && dot(axeZ, ref.z) < 0) axeZ = mul(axeZ, -1);
          ctx.fillStyle = this.ringSkinFill(ctx, milieu.gauche, milieu.droite, milieu.centre, axeX, axeZ, view);
          ctx.fill();
        }
      }
      if (mode === "volume" && o.cues && o.surface && this.showAnatomy !== false) {
        this.drawAnatomy(ctx, o.surface, view, o.cues);
      }
    }

    /**
     * Repères anatomiques : taches douces placées sur la vraie surface (f = hauteur,
     * phi = angle autour du volume, 0 = flanc droit, π/2 = avant).
     */
    drawAnatomy(ctx, surface, view, cues) {
      for (const cue of cues) {
        const p3 = surface(cue.f, cue.phi);
        if (!p3) continue;
        const p = project(p3, view.opts);
        if (!p || !isFinite(p.x)) continue;
        const axe = screenDir(view, cue.axe || v(1, 0, 0));
        const angle = Math.atan2(axe[1], axe[0]);
        const rx = Math.max(1.5, (cue.rx || 0.03) * p.scale);
        const ry = Math.max(1.2, (cue.ry || 0.02) * p.scale);
        ctx.save();
        ctx.globalAlpha = cue.alpha === undefined ? 0.12 : cue.alpha;
        if (cue.flou === false) {
          ctx.fillStyle = cue.clair ? "#fff6ef" : "#241611";
        } else {
          const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, Math.max(rx, ry));
          grad.addColorStop(0, cue.clair ? "rgba(255,246,239,.95)" : "rgba(32,18,12,.85)");
          grad.addColorStop(1, cue.clair ? "rgba(255,246,239,0)" : "rgba(32,18,12,0)");
          ctx.fillStyle = grad;
        }
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, rx, ry, angle, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    /** Tête : volume d'anneaux + visage (yeux, nez, bouche, oreilles). */
    drawHead(ctx, item, mode, view) {
      const rings = item.rings.map((r) => {
        const proj = r.pts.map((q) => project(q, view.opts));
        let gauche = proj[0], droite = proj[0];
        for (const q of proj) {
          if (q.x < gauche.x) gauche = q;
          if (q.x > droite.x) droite = q;
        }
        return { u: r.u, gauche: gauche, droite: droite, centre: project(r.centre, view.opts),
          centre3: r.centre, x: item.frame.x, z: item.frame.z };
      });
      this.fillRings(ctx, rings, mode, view, {
        depthT: 0.5,
        cues: [{ f: 0.62, x: 0, y: 0, largeur: 0.3, ratio: 0.2, alpha: 0.10 },
          { f: 0.30, x: 0, y: 0, largeur: 0.26, ratio: 0.16, alpha: 0.10 }],
      });
      if (mode !== "volume" || this.showFace === false) return;
      // visage : yeux, nez, bouche, oreilles placés sur la surface de la tête
      const fr = item.frame;
      const surface = (u, phi) => {
        const r = this.headProfile(u);
        const centre = add(add(fr.bas, mul(fr.axis, fr.longueur * (u * 0.96 - 0.10))),
          mul(fr.z, fr.profondeur * this.headFace(u)));
        return add(centre, add(mul(fr.x, Math.cos(phi) * fr.largeur / 2 * r * 1.02),
          mul(fr.z, Math.sin(phi) * fr.profondeur / 2 * r)));
      };
      const tache = (p3, rx, ry, couleur, alpha) => {
        const p = project(p3, view.opts);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = couleur;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, Math.max(1, rx * p.scale), Math.max(1, ry * p.scale), 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      };
      // yeux (un peu enfoncés dans l'orbite)
      const yeux = 0.60;
      for (const cote of [1, -1]) {
        tache(surface(0.67, Math.PI / 2 + cote * 0.42), fr.largeur * 0.105, fr.largeur * 0.022, "#2b1a14", 0.22);
        tache(surface(yeux, Math.PI / 2 + cote * 0.42), fr.largeur * 0.10, fr.largeur * 0.052, "#4a3229", 0.5);
        tache(surface(yeux + 0.05, Math.PI / 2 + cote * 0.42), fr.largeur * 0.13, fr.largeur * 0.04, "#2b1a14", 0.08);
      }
      // nez : arête claire, flanc ombré, narines
      tache(surface(0.58, Math.PI / 2), fr.largeur * 0.028, fr.largeur * 0.075, "#fff2e8", 0.30);
      tache(surface(0.55, Math.PI / 2 + 0.24), fr.largeur * 0.045, fr.largeur * 0.07, "#4a2c22", 0.16);
      for (const cote of [1, -1]) {
        tache(surface(0.475, Math.PI / 2 + cote * 0.10), fr.largeur * 0.016, fr.largeur * 0.012, "#3c2a24", 0.34);
      }
      // bouche et menton
      tache(surface(0.34, Math.PI / 2), fr.largeur * 0.13, fr.largeur * 0.035, "#6d3f34", 0.42);
      tache(surface(0.20, Math.PI / 2), fr.largeur * 0.065, fr.largeur * 0.032, "#ffe7d8", 0.12);
      // oreilles
      for (const cote of [1, -1]) {
        tache(surface(0.55, cote * 0.02), fr.largeur * 0.045, fr.largeur * 0.085, "#d3a98f", 0.85);
      }
    }

    /** Pied : talon et avant-pied, aplatis (semelle au sol). */
    drawFoot(ctx, item, mode, view) {
      const g = mode === "depth"
        ? (() => { const t = Math.max(0, Math.min(1, (item.depth - (this.camera.distance - 0.7)) / 1.4));
          const v0 = Math.round(255 - 205 * (1 - t)); return `rgb(${v0},${v0},${v0})`; })()
        : null;
      const blanc = mode === "silhouette";
      const talon = (u) => item.rTal * (1 - 0.12 * u);
      const avant = (u) => item.rAvant * (1 - 0.30 * u) + item.rPointe * 0.30 * u;
      limb(ctx, item.a, item.b, talon, item.a3, item.b3, view, blanc ? "#ffffff" : g);
      limb(ctx, item.b, item.c, avant, item.b3, item.c3, view, blanc ? "#ffffff" : g);
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
      let centre = lerp(A, B, t);
      const axis = norm(sub(B, A));
      // largeur : des hanches (en bas) aux épaules (en haut)
      const bas = norm(sub(this.pose.hip_l, this.pose.hip_r));
      const haut = norm(sub(this.pose.shoulder_l, this.pose.shoulder_r));
      let x = norm(add(mul(bas, 1 - f), mul(haut, f)));
      x = norm(sub(x, mul(axis, dot(x, axis))));
      if (len(x) < 1e-6) x = norm(sub(this.pose.hip_l, this.pose.hip_r));
      const z = norm(v(x.y * axis.z - x.z * axis.y, x.z * axis.x - x.x * axis.z, x.x * axis.y - x.y * axis.x));
      if (f < 0) {
        // sous le bassin : le tronc descend vers l'entrejambe (sinon trou entre les cuisses)
        centre = sub(centre, mul(axis, -f * (weights[0] + weights[1])));
      }
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
      if (f < 0) {
        // sous le bassin : on referme doucement vers l'entrejambe
        const k = Math.max(0, Math.min(1, (0 - f) / 0.14));
        const ferme = 1 - 0.42 * k;
        return { width: width * ferme, depth: interp("depth") * girth * ferme };
      }
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
      return { pts: pts, centre: frame.centre, size: taille, frame: frame };
    }

    /** Élément « tronc » à dessiner (silhouette lissée + profondeur moyenne). */
    torsoItem(opts) {
      const rings = [];
      for (let i = 0; i < TORSO_SLICES; i++) {
        const f = TORSO_BAS + (i / (TORSO_SLICES - 1)) * (1 - TORSO_BAS);
        const ring = this.torsoRing(f);
        const proj = ring.pts.map((p) => project(p, opts));
        let gauche = proj[0], droite = proj[0];
        for (const p of proj) {
          if (p.x < gauche.x) gauche = p;
          if (p.x > droite.x) droite = p;
        }
        rings.push({ f: f, proj: proj, gauche: gauche, droite: droite, centre: project(ring.centre, opts),
          centre3: ring.centre, x: ring.frame.x, z: ring.frame.z });
      }
      const self = this;
      const surface = (f, phi) => {
        const frame = self.torsoFrame(f);
        const taille = self.torsoSize(f);
        return add(frame.centre, add(mul(frame.x, Math.cos(phi) * taille.width / 2),
          mul(frame.z, Math.sin(phi) * taille.depth / 2)));
      };
      const depth = rings.reduce((acc, r) => acc + r.centre.depth, 0) / rings.length;
      return { type: "torso", rings: rings, depth: depth, surface: surface };
    }

    /** Dessine le tronc : volume lissé, peau ombrée et repères anatomiques. */
    drawTorso(ctx, item, mode, view) {
      this.fillRings(ctx, item.rings, mode, view, {
        depthT: Math.max(0, Math.min(1, (item.depth - (this.camera.distance - 0.7)) / 1.4)),
        surface: item.surface,
        cues: [
          { f: 0.90, phi: 0.55, rx: 0.055, ry: 0.020, alpha: 0.045, clair: true },  // dessus d'épaule
          { f: 0.90, phi: 2.59, rx: 0.055, ry: 0.020, alpha: 0.045, clair: true },
          { f: 0.755, phi: 1.10, rx: 0.042, ry: 0.030, alpha: 0.05 },               // pectoral
          { f: 0.755, phi: 2.04, rx: 0.042, ry: 0.030, alpha: 0.05 },
          { f: 0.50, phi: 1.57, rx: 0.013, ry: 0.011, alpha: 0.26, flou: false },   // nombril
          { f: 0.26, phi: 4.71, rx: 0.055, ry: 0.024, alpha: 0.07 },                // pli fessier
        ],
      });
    }

    /** Rayon moyen (mètres) d'un segment : la moitié de l'épaisseur réglée dans le tableau. */
    meanRadius(child) {
      const t = this.thickness[child];
      if (t) return t / 2;
      return BASE_THICK[child] ? BASE_THICK[child] / 2 : 0.04;
    }

    /** Rayon de la rotule d'une articulation : celui des tubes qui s'y raccordent. */
    jointRadius(joint) {
      let r = 0.02;
      BONES.forEach(([a, b]) => {
        const child = b;
        if (child === joint) r = Math.max(r, this.meanRadius(child) * profileAt(child, 0));
        if (a === joint && PARENT[child] === a) r = Math.max(r, this.meanRadius(child) * profileAt(child, 1));
      });
      // une articulation de la racine (épaule, hanche) prend aussi l'épaisseur du membre enfant
      BONES.forEach(([a, b]) => {
        if (a === joint) r = Math.max(r, this.meanRadius(b) * profileAt(b, 0) * 0.92);
      });
      if (joint === "hip_l" || joint === "hip_r") r = Math.min(r, 0.052);
      if (joint === "shoulder_l" || joint === "shoulder_r") r = Math.min(r, 0.046);
      if (joint === "elbow_l" || joint === "elbow_r" || joint === "knee_l" || joint === "knee_r") r *= 0.94;
      return r;
    }

    /** Repère de la tête : axe crâne→sommets, largeur et profondeur réglées. */
    headFrame() {
      const bas = this.pose.neck, haut = this.pose.head_top;
      const axis = norm(sub(haut, bas));
      const epaules = norm(sub(this.pose.shoulder_l, this.pose.shoulder_r));
      let x = norm(sub(epaules, mul(axis, dot(epaules, axis))));
      if (len(x) < 1e-6) x = v(1, 0, 0);
      let z = norm(v(x.y * axis.z - x.z * axis.y, x.z * axis.x - x.x * axis.z, x.x * axis.y - x.y * axis.x));
      const front = this.pose.nose ? norm(sub(this.pose.nose, bas)) : z;
      if (dot(z, front) < 0) z = mul(z, -1);
      return { bas: bas, haut: haut, axis: axis, x: x, z: z,
        longueur: dist(bas, haut), largeur: this.thickness.head || 0.176, profondeur: (this.thickness.head || 0.176) * 1.18 };
    }

    /**
     * Galbe de la tête : 0 = menton, 1 = sommet. Crâne arrondi, mâchoire plus
     * étroite et menton légèrement en avant.
     */
    headProfile(u) {
      const prof = [[0, 0.26], [0.06, 0.46], [0.14, 0.62], [0.26, 0.80], [0.40, 0.92],
        [0.56, 0.99], [0.72, 1.0], [0.85, 0.94], [0.94, 0.78], [1, 0.40]];
      for (let i = 0; i + 1 < prof.length; i++) {
        if (u >= prof[i][0] && u <= prof[i + 1][0]) {
          const d = prof[i + 1][0] - prof[i][0] || 1;
          const t = (u - prof[i][0]) / d;
          return prof[i][1] + (prof[i + 1][1] - prof[i][1]) * t;
        }
      }
      return 0.40;
    }

    /** Avancée du menton / du visage (fraction de la profondeur) selon la hauteur. */
    headFace(u) {
      return 0.16 * Math.exp(-Math.pow((u - 0.22) / 0.20, 2))
        + 0.10 * Math.exp(-Math.pow((u - 0.48) / 0.30, 2));
    }

    /** Élément « tête » : anneaux 3D du menton au sommet. */
    headItem(opts) {
      const fr = this.headFrame();
      const rings = [];
      const n = 11;
      for (let i = 0; i < n; i++) {
        const u = i / (n - 1);
        const centre = add(add(fr.bas, mul(fr.axis, fr.longueur * (u * 0.96 - 0.10))),
          mul(fr.z, fr.profondeur * this.headFace(u)));
        const r = this.headProfile(u);
        const pts = [];
        for (let k = 0; k < 12; k++) {
          const th = (k / 12) * Math.PI * 2;
          pts.push(add(centre, add(mul(fr.x, Math.cos(th) * fr.largeur / 2 * r * 1.02),
            mul(fr.z, Math.sin(th) * fr.profondeur / 2 * r))));
        }
        rings.push({ u: u, pts: pts, centre: centre });
      }
      return { type: "head", rings: rings, frame: fr,
        depth: project(fr.bas, opts).depth - 0.01 };
    }

    /** Élément « pied » : talon + orteils (deux tubes galbés). */
    footItem(cote, opts) {
      const ankle = this.pose["ankle_" + cote], toe = this.pose["toe_" + cote], heel = this.pose["heel_" + cote];
      const sol = 0.012;
      const bas = v(ankle.x, sol, ankle.z);
      const basT = v(toe.x, sol + 0.004, toe.z);
      const basH = v(heel.x, sol + 0.002, heel.z);
      const cheville = this.thickness["ankle_" + cote] || 0.102;
      const doigts = this.thickness["toe_" + cote] || 0.08;
      const cote3 = cheville * 0.44;
      const lame = doigts * 0.46;
      const proj = (q) => project(q, opts);
      return { type: "foot", a3: basH, b3: bas, c3: basT, rTal: cote3 * 1.02, rAvant: cote3 * 0.98,
        rPointe: lame * 0.95, a: proj(basH), b: proj(bas), c: proj(basT),
        depth: Math.max(proj(bas).depth, proj(basT).depth) + 0.01 };
    }

    /** Ancien tracé (non utilisé) : conservé pour référence des versions précédentes. */
    footItemLegacy(cote, opts) {
      const ankle = this.pose["ankle_" + cote], toe = this.pose["toe_" + cote], heel = this.pose["heel_" + cote];
      const axe = norm(sub(toe, ankle));
      const cote3 = this.thickness["ankle_" + cote] || 0.102;
      const larg = (this.thickness["toe_" + cote] || 0.08) * 1.25;
      const frame = { x: norm(v(axe.y * 0 - axe.z * 1, 0, axe.x * 1 - axe.y * 0)) };
      void frame;
      const sol = 0.012;
      const bas = v(ankle.x, sol, ankle.z);
      const basT = v(toe.x, sol, toe.z);
      const basH = v(heel.x, sol, heel.z);
      const hautT = add(toe, mul(norm(sub(ankle, toe)), cote3 * 0.35));
      const hautH = add(heel, mul(norm(sub(ankle, heel)), cote3 * 0.42));
      const points = [basH, bas, basT, hautT, add(ankle, v(0, cote3 * 0.15, 0)), hautH];
      const proj = points.map((q) => project(q, opts));
      const projAnkle = project(ankle, opts);
      return { type: "foot", pts: proj, largeur: larg, ankle: projAnkle, a3: ankle, b3: toe,
        depth: (projAnkle.depth + project(toe, opts).depth) / 2 };
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
