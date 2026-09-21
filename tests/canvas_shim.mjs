/**
 * Canvas 2D minimal + encodeur PNG, pour vérifier les rendus du mannequin sans
 * navigateur (jsdom n'implémente pas le contexte 2D).
 *
 * Couverture volontairement limitée à ce que le mannequin utilise :
 * chemins (moveTo/lineTo/arc/ellipse/closePath), remplissage (couleur unie ou
 * dégradé linéaire), contours, transformations (translate/rotate/scale), clipping
 * rectangulaire (fillRect), alpha global.
 *
 * On rasterise en scannant chaque polygone ligne par ligne (remplissage pair-impair),
 * puis on écrit un PNG RVB avec zlib.
 */
import { deflateSync } from "node:zlib";

const TAU = Math.PI * 2;

function parseColor(c) {
  if (typeof c !== "string") return { r: 128, g: 128, b: 128, a: 1 };
  const m = c.trim();
  if (m === "transparent" || m === "none") return { r: 0, g: 0, b: 0, a: 0 };
  let mm = /^#([0-9a-f]{6})$/i.exec(m);
  if (mm) {
    const n = parseInt(mm[1], 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: 1 };
  }
  mm = /^#([0-9a-f]{3})$/i.exec(m);
  if (mm) {
    const n = parseInt(mm[1], 16);
    return { r: ((n >> 8) & 15) * 17, g: ((n >> 4) & 15) * 17, b: (n & 15) * 17, a: 1 };
  }
  mm = /^rgba?\(([^)]+)\)$/i.exec(m);
  if (mm) {
    const parts = mm[1].split(",").map((x) => parseFloat(x));
    return { r: parts[0] || 0, g: parts[1] || 0, b: parts[2] || 0, a: parts.length > 3 ? parts[3] : 1 };
  }
  return { r: 128, g: 128, b: 128, a: 1 };
}

class RadialGradient {
  constructor(x0, y0, r0, x1, y1, r1) {
    this.x0 = x0; this.y0 = y0; this.r0 = Math.max(0, r0);
    this.x1 = x1; this.y1 = y1; this.r1 = Math.max(0.001, r1); this.stops = [];
  }
  addColorStop(pos, color) { this.stops.push([pos, parseColor(color)]); return this; }
  /** Couleur interpolée selon la distance au foyer du dégradé radial. */
  at(x, y) {
    const d = Math.hypot(x - this.x1, y - this.y1);
    const t = Math.max(0, Math.min(1, (d - this.r0) / (this.r1 - this.r0)));
    const stops = this.stops.slice().sort((a, b) => a[0] - b[0]);
    if (!stops.length) return { r: 128, g: 128, b: 128, a: 1 };
    if (t <= stops[0][0]) return stops[0][1];
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
        const k = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
        return {
          r: c0.r + (c1.r - c0.r) * k, g: c0.g + (c1.g - c0.g) * k,
          b: c0.b + (c1.b - c0.b) * k, a: c0.a + (c1.a - c0.a) * k,
        };
      }
    }
    return stops[stops.length - 1][1];
  }
}

class LinearGradient {
  constructor(x0, y0, x1, y1) {
    this.x0 = x0; this.y0 = y0; this.x1 = x1; this.y1 = y1; this.stops = [];
  }
  addColorStop(pos, color) { this.stops.push([pos, parseColor(color)]); return this; }
  /** Couleur interpolée pour un point (projeté sur l'axe du dégradé). */
  at(x, y) {
    const dx = this.x1 - this.x0, dy = this.y1 - this.y0;
    const l2 = dx * dx + dy * dy;
    let t = l2 < 1e-9 ? 0 : ((x - this.x0) * dx + (y - this.y0) * dy) / l2;
    t = Math.max(0, Math.min(1, t));
    const stops = this.stops.slice().sort((a, b) => a[0] - b[0]);
    if (!stops.length) return { r: 128, g: 128, b: 128, a: 1 };
    if (t <= stops[0][0]) return stops[0][1];
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
        const k = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
        return {
          r: c0.r + (c1.r - c0.r) * k, g: c0.g + (c1.g - c0.g) * k,
          b: c0.b + (c1.b - c0.b) * k, a: c0.a + (c1.a - c0.a) * k,
        };
      }
    }
    return stops[stops.length - 1][1];
  }
}

export class Canvas2DShim {
  constructor(width, height) {
    this.width = width; this.height = height;
    this.ops = 0;
    this.data = new Uint8ClampedArray(width * height * 3);
    this.data.fill(0);
    this.fillStyle = "#000000";
    this.strokeStyle = "#000000";
    this.lineWidth = 1;
    this.lineCap = "butt";
    this.globalAlpha = 1;
    this._stack = [];
    this._path = [];
    this._current = null;
    this._m = [1, 0, 0, 1, 0, 0];              // matrice courante
    this._tm = [1, 0, 0, 1, 0, 0];            // matrice de travail (pour pathTo)
  }

  // ------------------------------------------------------- transformations
  save() { this._stack.push({ m: this._m.slice(), fillStyle: this.fillStyle, strokeStyle: this.strokeStyle, lineWidth: this.lineWidth, globalAlpha: this.globalAlpha, lineCap: this.lineCap }); }
  restore() {
    const s = this._stack.pop();
    if (!s) return;
    this._m = s.m; this.fillStyle = s.fillStyle; this.strokeStyle = s.strokeStyle;
    this.lineWidth = s.lineWidth; this.globalAlpha = s.globalAlpha; this.lineCap = s.lineCap;
  }
  translate(x, y) { this._m = mul(this._m, [1, 0, 0, 1, x, y]); }
  rotate(a) { const c = Math.cos(a), s = Math.sin(a); this._m = mul(this._m, [c, s, -s, c, 0, 0]); }
  scale(x, y) { this._m = mul(this._m, [x, 0, 0, y === undefined ? x : y, 0, 0]); }
  apply(x, y) {
    const m = this._m;
    return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
  }

  // ------------------------------------------------------------- chemins
  beginPath() { this._path = []; this._current = null; }
  moveTo(x, y) { this._current = [x, y]; this._path.push({ points: [this._current], closed: false }); }
  lineTo(x, y) { this._current = [x, y]; if (this._path.length) this._path[this._path.length - 1].points.push(this._current); }
  closePath() { if (this._path.length) this._path[this._path.length - 1].closed = true; }
  _ensureSubpath(x, y) {
    const last = this._path[this._path.length - 1];
    if (!last || last.closed) { this._current = [x, y]; this._path.push({ points: [this._current], closed: false }); }
  }
  /** Courbe quadratique (utilisée pour les lèvres et les élastiques). */
  quadraticCurveTo(cx, cy, x, y) {
    const debut = this._current ? [this._current[0], this._current[1]] : [cx, cy];
    this._ensureSubpath(debut[0], debut[1]);
    const steps = 10;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps, u = 1 - t;
      const px = u * u * debut[0] + 2 * u * t * cx + t * t * x;
      const py = u * u * debut[1] + 2 * u * t * cy + t * t * y;
      this._current = [px, py];
      this._path[this._path.length - 1].points.push(this._current);
    }
  }

  arc(cx, cy, r, a0, a1, ccw) {
    // comme un vrai canvas : le sens compte (par défaut, les angles CROISSENT)
    let sweep = a1 - a0;
    if (!ccw && sweep < 0) sweep += TAU;
    if (ccw && sweep > 0) sweep -= TAU;
    this._ensureSubpath(cx + r * Math.cos(a0), cy + r * Math.sin(a0));
    const steps = Math.max(6, Math.ceil(Math.abs(sweep) * r / 2));
    for (let i = 0; i <= steps; i++) {
      const a = a0 + sweep * (i / steps);
      this.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
    }
  }
  ellipse(cx, cy, rx, ry, rot, a0, a1) {
    let sweep = a1 - a0;
    if (sweep <= 0) sweep += TAU;                    // ellipse() trace toujours dans le sens positif
    this._ensureSubpath(cx + rx * Math.cos(a0), cy + ry * Math.sin(a0));
    const steps = Math.max(16, Math.ceil(Math.abs(sweep) * 12));
    for (let i = 0; i <= steps; i++) {
      const a = a0 + sweep * (i / steps);
      this.lineTo(cx + rx * Math.cos(a), cy + ry * Math.sin(a));
    }
  }
  rect(x, y, w, h) { this.moveTo(x, y); this.lineTo(x + w, y); this.lineTo(x + w, y + h); this.lineTo(x, y + h); this.closePath(); }

  fillRect(x, y, w, h) {
    this.beginPath(); this.rect(x, y, w, h);
    const p = this._path[this._path.length - 1];
    this._fillPath(p, this.fillStyle);
  }
  strokeRect(x, y, w, h) {
    this.beginPath(); this.rect(x, y, w, h);
    const p = this._path[this._path.length - 1];
    this._strokePath(p, this.strokeStyle);
  }

  createLinearGradient(x0, y0, x1, y1) { return new LinearGradient(x0, y0, x1, y1); }

  createRadialGradient(x0, y0, r0, x1, y1, r1) { return new RadialGradient(x0, y0, r0, x1, y1, r1); }

  clearRect(x, y, w, h) {
    for (let yy = Math.max(0, y | 0); yy < Math.min(this.height, (y + h) | 0); yy++) {
      for (let xx = Math.max(0, x | 0); xx < Math.min(this.width, (x + w) | 0); xx++) this._set(xx, yy, 0, 0, 0, 1);
    }
  }

  fill() { for (const p of this._path) this._fillPath(p, this.fillStyle); }
  stroke() { for (const p of this._path) this._strokePath(p, this.strokeStyle); }

  // ------------------------------------------------------------ raster
  _set(x, y, r, g, b, a) {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
    const i = (y * this.width + x) * 3;
    const k = Math.max(0, Math.min(1, a * this.globalAlpha));
    this.data[i] = this.data[i] * (1 - k) + r * k;
    this.data[i + 1] = this.data[i + 1] * (1 - k) + g * k;
    this.data[i + 2] = this.data[i + 2] * (1 - k) + b * k;
  }

  _sample(style, x, y) {
    const c = (style instanceof LinearGradient || style instanceof RadialGradient)
      ? style.at(x, y) : parseColor(style);
    return c;
  }

  _fillPath(path, style) {
    const pts = path.points.map(([x, y]) => this.apply(x, y));
    if (pts.length < 3) return;
    // les dégradés sont définis dans l'espace utilisateur AU MOMENT du remplissage :
    // on échantillonne donc les stops via l'inverse de la matrice courante
    const inv = style instanceof LinearGradient ? invert(this._m) : null;
    this.ops++;
    let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
    for (const [x, y] of pts) { minY = Math.min(minY, y); maxY = Math.max(maxY, y); minX = Math.min(minX, x); maxX = Math.max(maxX, x); }
    const y0 = Math.max(0, Math.floor(minY)), y1 = Math.min(this.height - 1, Math.ceil(maxY));
    const edges = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      if (a[1] !== b[1]) edges.push([a, b]);
    }
    for (let y = y0; y <= y1; y++) {
      const sy = y + 0.5;
      const xs = [];
      for (const [a, b] of edges) {
        const [ax, ay] = a, [bx, by] = b;
        if ((sy >= ay && sy < by) || (sy >= by && sy < ay)) {
          xs.push(ax + ((sy - ay) / (by - ay)) * (bx - ax));
        }
      }
      if (xs.length < 2) continue;
      xs.sort((p, q) => p - q);
      for (let i = 0; i + 1 < xs.length; i += 2) {
        const xa = Math.max(0, Math.ceil(xs[i] - 0.5)), xb = Math.min(this.width - 1, Math.floor(xs[i + 1] - 0.5));
        for (let x = xa; x <= xb; x++) {
          const sx = inv ? inv[0] * x + inv[2] * y + inv[4] : x;
          const sy = inv ? inv[1] * x + inv[3] * y + inv[5] : y;
          const c = this._sample(style, sx, sy);
          this._set(x, y, c.r, c.g, c.b, c.a);
        }
      }
    }
  }

  _strokePath(path, style) {
    const pts = path.points.map(([x, y]) => this.apply(x, y));
    const w = Math.max(1, this.lineWidth * Math.abs(this._m[0]) || this.lineWidth);
    const c = parseColor(typeof style === "string" ? style : "#000");
    const half = w / 2;
    for (let i = 0; i + 1 < pts.length; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
      const dx = x1 - x0, dy = y1 - y0;
      const l = Math.max(1e-6, Math.hypot(dx, dy));
      const nx = -dy / l * half, ny = dx / l * half;
      const quad = [[x0 + nx, y0 + ny], [x1 + nx, y1 + ny], [x1 - nx, y1 - ny], [x0 - nx, y0 - ny]];
      this._fillPath({ points: quad.map(([x, y]) => unapply(this._m, x, y)), closed: true }, style);
      if (this.lineCap === "round") {
        for (const [cx, cy] of [[x0, y0], [x1, y1]]) {
          const disque = [];                    // calotte pleine (sinon effet « éventail »)
          for (let a = 0; a < TAU; a += TAU / 16) {
            const px = cx + Math.cos(a) * half, py = cy + Math.sin(a) * half;
            disque.push(unapply(this._m, px, py));
          }
          this._fillPath({ points: disque, closed: true }, style);
        }
      }
    }
  }

  // --------------------------------------------------------------- export
  toPNG() { return encodePNG(this.width, this.height, this.data); }
  /** Nombre de pixels non noirs (indicateur de couverture dans les tests). */
  coverage() {
    let n = 0;
    for (let i = 0; i < this.data.length; i += 3) if (this.data[i] + this.data[i + 1] + this.data[i + 2] > 24) n++;
    return n;
  }
  pixel(x, y) {
    const i = (y * this.width + x) * 3;
    return [this.data[i], this.data[i + 1], this.data[i + 2]];
  }
}

function mul(a, b) {
  return [
    a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
    a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5],
  ];
}
function invert(m) {
  const det = m[0] * m[3] - m[1] * m[2];
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 1, 0, 0];
  return [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det,
    (m[2] * m[5] - m[3] * m[4]) / det, (m[1] * m[4] - m[0] * m[5]) / det];
}
function unapply(m, x, y) {
  const i = invert(m);
  return [i[0] * x + i[2] * y + i[4], i[1] * x + i[3] * y + i[5]];
}

// ------------------------------------------------------------------- PNG
function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = c ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

export function encodePNG(width, height, rgb) {
  const raw = Buffer.alloc(height * (width * 3 + 1));
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    Buffer.from(rgb.buffer, rgb.byteOffset + y * width * 3, width * 3).copy(raw, y * (width * 3 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 6 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

export function createCanvas(width, height) {
  const canvas = { width, height, _ctx: null };
  canvas.getContext = () => (canvas._ctx = canvas._ctx || new Canvas2DShim(width, height));
  return canvas;
}
