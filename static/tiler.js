// YOLOtiler front-end. Plain JS, no build step.
//
// Shows how each image of a directory is cut into fixed-size, overlapping tiles and what
// becomes of its YOLOboxer labels in every tile. Everything here is in image pixels
// (as the browser shows the image, i.e. after EXIF rotation); label boxes arrive in YOLO
// form {cls, cx, cy, w, h}, normalised 0-1, and are converted where needed.
//
// The export runs in yolotiler.py, which computes the same grid: tileStarts and clipBox
// mirror tile_starts and clip_box there; keep them in step.

// same palette and keys as YOLOboxer, so classes keep their colours
const KEYS = "1234567890abcdefghijklmnopqrstuvwxyz";
const COLORS = ["#f5c818", "#e0431c", "#2f9e57", "#c99a06", "#8a3fc7", "#0f9fa8",
                "#d6336c", "#6b8e23", "#b8551f", "#4054b2", "#7a5c3e", "#c2185b"];
const LAST_DIR_KEY = "yolotiler.lastDir";
const SETTINGS_KEY = "yolotiler.settings";
const DEFAULTS = { w: 640, h: 640, ox: 64, oy: 64, minVis: 25 };
const MAX_TILES = 10000;  // per image; more is almost certainly a typo in the overlap
const CLICK_PX = 4;       // a press that moves less than this on screen is a click, not a pan
const MAX_ZOOM = 32;      // relative to the fitted size

const state = {
  dir: null,
  images: [],      // [{name, done, empty, w, h}]
  classes: [],     // ["cat", "dog", ...]
  idx: 0,          // index into images
  boxes: [],       // label boxes of the current image
  settings: { ...DEFAULTS },
  grid: null,      // tile grid of the current image, see tilesFor
  tile: 0,         // selected tile index into grid.tiles
  hover: -1,       // tile under the pointer
  mode: "image",   // "image": whole image with grid, "tile": the selected tile close up
};

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let img = new Image();
let view = { x: 0, y: 0, s: 0 };   // image pixel (px, py) is drawn at canvas (x + px*s, y + py*s)
let fit = { x: 0, y: 0, s: 0 };    // the same, with the focus (image or tile) fitted (zoom 1)
let zoom = 1;                      // view scale relative to fit
let press = null;                  // {x, y, moved} while a mouse button is down
let mouse = null;                  // last pointer position, image px

const colorFor = (c) => COLORS[c % COLORS.length];
const className = (c) => state.classes[c] ?? `#${c}`;
const current = () => state.images[state.idx];
const selTile = () => state.grid?.tiles[state.tile];

// --- tiling ----------------------------------------------------------------

// tile offsets along one axis; the last tile is moved back to end at the image edge,
// so it overlaps its neighbour by more than `overlap` instead of running off the image
function tileStarts(length, size, overlap) {
  if (size >= length) return [0];
  const stride = Math.max(1, size - overlap);
  const starts = [];
  for (let s = 0; s < length - size; s += stride) starts.push(s);
  starts.push(length - size);
  return starts;
}

// same count as tileStarts(...).length, without building the list
function countStarts(length, size, overlap) {
  if (size >= length) return 1;
  return Math.ceil((length - size) / Math.max(1, size - overlap)) + 1;
}

function tileCount(W, H, s) {
  if (!W || !H) return 0;
  return countStarts(W, Math.min(s.w, W), s.ox) * countStarts(H, Math.min(s.h, H), s.oy);
}

// every tile of a W x H image, row by row; tiles never exceed the image
function tilesFor(W, H, s) {
  const tw = Math.min(s.w, W), th = Math.min(s.h, H);
  const xs = tileStarts(W, tw, s.ox), ys = tileStarts(H, th, s.oy);
  const tiles = [];
  ys.forEach((y, r) => xs.forEach((x, c) => tiles.push({ r, c, x, y, w: tw, h: th })));
  return { tiles, xs, ys, tw, th };
}

// a label box cut to tile t: null if it misses the tile, else the visible part in image px,
// the visible fraction of the box and whether that is enough to keep it
function clipBox(b, t, W, H, minVis) {
  const x0 = (b.cx - b.w / 2) * W, x1 = (b.cx + b.w / 2) * W;
  const y0 = (b.cy - b.h / 2) * H, y1 = (b.cy + b.h / 2) * H;
  const ix0 = Math.max(x0, t.x), ix1 = Math.min(x1, t.x + t.w);
  const iy0 = Math.max(y0, t.y), iy1 = Math.min(y1, t.y + t.h);
  if (ix1 <= ix0 || iy1 <= iy0) return null;
  const area = (x1 - x0) * (y1 - y0);
  const vis = area > 0 ? ((ix1 - ix0) * (iy1 - iy0)) / area : 0;
  return { cls: b.cls, x0: ix0, y0: iy0, x1: ix1, y1: iy1, vis, keep: vis * 100 >= minVis - 1e-9 };
}

function clipsIn(t) {
  const W = img.naturalWidth, H = img.naturalHeight;
  return state.boxes.map((b) => clipBox(b, t, W, H, state.settings.minVis)).filter(Boolean);
}

// the tile under image point p whose centre is closest (tiles overlap), or -1
function tileAt(p) {
  if (!state.grid || !p) return -1;
  let best = -1, bestD = Infinity;
  state.grid.tiles.forEach((t, i) => {
    if (p.x < t.x || p.x > t.x + t.w || p.y < t.y || p.y > t.y + t.h) return;
    const d = Math.hypot(p.x - t.x - t.w / 2, p.y - t.y - t.h / 2);
    if (d < bestD) { best = i; bestD = d; }
  });
  return best;
}

// rebuild the grid of the current image, keeping the selection on the same spot
function regrid() {
  if (!img.naturalWidth) { state.grid = null; return; }
  const old = selTile();
  state.grid = tilesFor(img.naturalWidth, img.naturalHeight, state.settings);
  const i = old ? tileAt({ x: old.x + old.w / 2, y: old.y + old.h / 2 }) : -1;
  state.tile = i >= 0 ? i : Math.min(state.tile, state.grid.tiles.length - 1);
  state.hover = -1;
}

// --- api -------------------------------------------------------------------

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : r.statusText);
  return data;
}

function remember(dir) {
  try { localStorage.setItem(LAST_DIR_KEY, dir); } catch {}
}
function recall() {
  try { return localStorage.getItem(LAST_DIR_KEY); } catch { return null; }
}

// --- loading ---------------------------------------------------------------

function applyState(s) {
  state.dir = s.dir;
  state.images = s.images;
  state.classes = s.classes;
  $("dir-input").value = s.dir ?? "";
  $("empty").hidden = !!s.dir && s.images.length > 0;
  if (s.dir && !s.images.length) {
    $("empty").innerHTML = "<p>No images in this directory.</p>";
  }
  renderFiles();
}

async function openDir(path, quiet = false) {
  try {
    applyState(await api("/api/open", { path }));
  } catch (err) {
    if (!quiet) alert(err.message);
    return;
  }
  remember(state.dir);
  if (!state.images.length) { clearStage(); return; }
  const firstDone = state.images.findIndex((x) => x.done && !x.empty);
  await loadImage(firstDone < 0 ? 0 : firstDone);
}

async function loadImage(i) {
  state.idx = i;
  const name = current().name;
  state.boxes = (await api("/api/label/" + encodeURIComponent(name))).boxes;
  img = new Image();
  img.onload = () => { regrid(); layout(); render(); };
  img.src = "/image/" + encodeURIComponent(name) + "?t=" + Date.now();
  $("image-name").textContent = name;
  $("counter").textContent = `${i + 1} / ${state.images.length}`;
  renderFiles();
}

function clearStage() {
  img = new Image();
  state.boxes = [];
  state.grid = null;
  $("image-name").textContent = "";
  $("counter").textContent = "—";
  render();
}

async function go(i) {
  if (!state.images.length || i < 0 || i >= state.images.length || i === state.idx) return;
  await loadImage(i);
}

function selectTile(i, refit = state.mode === "tile") {
  if (!state.grid || i < 0 || i >= state.grid.tiles.length) return;
  state.tile = i;
  if (refit) layout();
  render();
}

// step to the neighbouring tile in the grid
function moveTile(dc, dr) {
  const t = selTile();
  if (!t) return;
  const cols = state.grid.xs.length, rows = state.grid.ys.length;
  const c = Math.min(cols - 1, Math.max(0, t.c + dc)), r = Math.min(rows - 1, Math.max(0, t.r + dr));
  selectTile(r * cols + c);
}

function setMode(mode) {
  if (mode === "tile" && !selTile()) return;
  state.mode = mode;
  state.hover = -1;
  layout();
  render();
}

// --- settings --------------------------------------------------------------

const FIELDS = { w: "set-w", h: "set-h", ox: "set-ox", oy: "set-oy", minVis: "set-vis" };

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY));
    if (s && Object.keys(DEFAULTS).every((k) => Number.isFinite(s[k]))) return { ...DEFAULTS, ...s };
  } catch {}
  return { ...DEFAULTS };
}

function showSettings() {
  for (const [k, id] of Object.entries(FIELDS)) {
    $(id).value = state.settings[k];
    $(id).classList.remove("bad");
  }
  $("settings-error").hidden = true;
}

// read the inputs; apply them if they make a valid grid, otherwise flag what is wrong
function readSettings() {
  const v = {}, bad = new Set();
  for (const [k, id] of Object.entries(FIELDS)) v[k] = Number($(id).value);
  const int = (n, min) => Number.isInteger(n) && n >= min;
  if (!int(v.w, 1)) bad.add("w");
  if (!int(v.h, 1)) bad.add("h");
  if (!int(v.ox, 0) || v.ox >= v.w) bad.add("ox");
  if (!int(v.oy, 0) || v.oy >= v.h) bad.add("oy");
  if ($("set-vis").value === "" || !(v.minVis >= 0 && v.minVis <= 100)) bad.add("minVis");

  let error = "";
  if (bad.has("ox") || bad.has("oy")) error = "overlap must be smaller than the tile size";
  if (!bad.size) {
    const most = Math.max(0, ...state.images.map((im) => tileCount(im.w, im.h, v)),
                          tileCount(img.naturalWidth, img.naturalHeight, v));
    if (most > MAX_TILES) {
      error = `${most.toLocaleString()} tiles for one image; lower the overlap`;
      bad.add("ox").add("oy");
    }
  }
  for (const [k, id] of Object.entries(FIELDS)) $(id).classList.toggle("bad", bad.has(k));
  $("settings-error").textContent = error;
  $("settings-error").hidden = !error;
  if (bad.size) return;

  state.settings = v;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(v)); } catch {}
  regrid();
  if (state.mode === "tile") layout();
  renderFiles();
  render();
}

$("settings").addEventListener("input", readSettings);
$("settings").addEventListener("submit", (e) => e.preventDefault());

// a preset sets a square tile size and keeps the overlap at the same share of the tile
document.querySelector(".presets").addEventListener("click", (e) => {
  const size = +e.target.dataset.size;
  if (!size) return;
  const s = state.settings;
  $("set-w").value = $("set-h").value = size;
  $("set-ox").value = Math.round((s.ox / s.w) * size);
  $("set-oy").value = Math.round((s.oy / s.h) * size);
  readSettings();
});

// --- canvas ----------------------------------------------------------------

// the image region the view fits: the whole image, or the selected tile in tile mode
function focusRect() {
  const t = selTile();
  if (state.mode === "tile" && t) return t;
  return { x: 0, y: 0, w: img.naturalWidth, h: img.naturalHeight };
}
const focusPad = () => (state.mode === "tile" ? 48 : 16);

// size the canvas to its container and fit the focus inside it
function layout() {
  const wrap = $("canvas-wrap");
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!img.naturalWidth) {
    fit = { x: 0, y: 0, s: 0 };
  } else {
    const r = focusRect(), pad = focusPad();
    const s = Math.min((W - 2 * pad) / r.w, (H - 2 * pad) / r.h);
    fit = { x: W / 2 - (r.x + r.w / 2) * s, y: H / 2 - (r.y + r.h / 2) * s, s };
  }
  resetZoom();
}

// --- zoom & pan ------------------------------------------------------------

function resetZoom() {
  zoom = 1;
  view = { ...fit };
  showZoom();
}

// zoom by `factor`, keeping canvas point (px, py) fixed under the pointer
function zoomAt(factor, px, py) {
  const z = Math.min(MAX_ZOOM, Math.max(1, zoom * factor));
  if (!fit.s || z === zoom) return;
  const ix = (px - view.x) / view.s, iy = (py - view.y) / view.s;
  view.s = fit.s * z;
  view.x = px - ix * view.s;
  view.y = py - iy * view.s;
  zoom = z;
  clampView();
  showZoom();
}

function panBy(dx, dy) {
  view.x += dx;
  view.y += dy;
  clampView();
}

// keep the zoomed focus covering the canvas; centre it on an axis where it fits
function clampView() {
  const r = focusRect(), pad = focusPad();
  const clamp = (pos, size, room) =>
    size <= room ? (room - size) / 2 : Math.min(pad, Math.max(room - size - pad, pos));
  view.x = clamp(view.x + r.x * view.s, r.w * view.s, cv.clientWidth) - r.x * view.s;
  view.y = clamp(view.y + r.y * view.s, r.h * view.s, cv.clientHeight) - r.y * view.s;
}

function showZoom() {
  const btn = $("zoom-btn");
  btn.textContent = zoom === 1 ? "fit" : `${zoom.toFixed(1)}×`;
  btn.disabled = zoom === 1;
}

// canvas mouse event -> image px (not clamped)
function toImage(e) {
  const r = cv.getBoundingClientRect();
  return {
    x: (e.clientX - r.left - view.x) / view.s,
    y: (e.clientY - r.top - view.y) / view.s,
  };
}

// --- drawing ---------------------------------------------------------------

const X = (px) => view.x + px * view.s;
const Y = (py) => view.y + py * view.s;

function draw() {
  const W = cv.clientWidth, H = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (!img.naturalWidth || !state.grid) return;

  const css = getComputedStyle(document.documentElement);
  const c = {
    ink: css.getPropertyValue("--ink").trim(),
    accent: css.getPropertyValue("--accent").trim(),
    mat: css.getPropertyValue("--mat").trim(),
    dim: css.getPropertyValue("--ink-3").trim(),
  };

  // image with a hard ink frame and offset shadow
  const iw = img.naturalWidth * view.s, ih = img.naturalHeight * view.s;
  ctx.fillStyle = c.ink;
  ctx.fillRect(view.x + 4, view.y + 4, iw, ih);
  ctx.imageSmoothingEnabled = view.s < 1;
  ctx.drawImage(img, view.x, view.y, iw, ih);
  ctx.strokeStyle = c.ink;
  ctx.lineWidth = 1;
  ctx.strokeRect(view.x - 0.5, view.y - 0.5, iw + 1, ih + 1);

  ctx.font = "11px " + css.getPropertyValue("--mono");
  ctx.textBaseline = "top";
  if (state.mode === "tile") drawTileView(c);
  else drawImageView(c);
}

// the whole image: overlap bands, tile edges, label boxes, hovered and selected tile
function drawImageView(c) {
  const g = state.grid, W = img.naturalWidth, H = img.naturalHeight;

  // where neighbouring tiles overlap; crossings get the band twice and show darker
  ctx.fillStyle = c.accent + "30";
  overlaps(g.xs, g.tw).forEach(([a, b]) => ctx.fillRect(X(a), Y(0), (b - a) * view.s, H * view.s));
  overlaps(g.ys, g.th).forEach(([a, b]) => ctx.fillRect(X(0), Y(a), W * view.s, (b - a) * view.s));

  // tile edges, a dark line under a light one so they read on any image
  const vx = new Set(g.xs.flatMap((x) => [x, x + g.tw])), hy = new Set(g.ys.flatMap((y) => [y, y + g.th]));
  if (vx.size < cv.clientWidth / 3 && hy.size < cv.clientHeight / 3) {
    ctx.beginPath();
    for (const x of vx) { ctx.moveTo(px(X(x)), Y(0)); ctx.lineTo(px(X(x)), Y(H)); }
    for (const y of hy) { ctx.moveTo(X(0), px(Y(y))); ctx.lineTo(X(W), px(Y(y))); }
    ctx.strokeStyle = "rgba(0,0,0,.5)";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = c.accent;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  for (const b of state.boxes) {
    const r = { x0: (b.cx - b.w / 2) * W, y0: (b.cy - b.h / 2) * H, x1: (b.cx + b.w / 2) * W, y1: (b.cy + b.h / 2) * H };
    drawRect(r, colorFor(b.cls), { width: 1.5, label: className(b.cls), top: Y(0) });
  }

  const hover = g.tiles[state.hover];
  if (hover && state.hover !== state.tile) {
    ctx.setLineDash([6, 4]);
    drawRect(tileRect(hover), "#ffffff", { width: 2 });
    ctx.setLineDash([]);
  }
  const t = selTile();
  drawRect(tileRect(t), c.accent, { width: 3, label: tileName(t), top: -Infinity, outside: true });
}

// the selected tile close up: the rest of the image dimmed, its overlap with the neighbours,
// every box cut to the tile (kept solid, dropped dashed grey) with its original extent
function drawTileView(c) {
  const g = state.grid, t = selTile();
  const tx = X(t.x), ty = Y(t.y), tw = t.w * view.s, th = t.h * view.s;

  ctx.fillStyle = c.mat + "c8";
  ctx.beginPath();
  ctx.rect(0, 0, cv.clientWidth, cv.clientHeight);
  ctx.rect(tx, ty, tw, th);
  ctx.fill("evenodd");

  // the strips this tile shares with its neighbours
  const cols = g.xs.length, rows = g.ys.length;
  const strips = [];
  if (t.c > 0) strips.push([t.x, t.y, g.xs[t.c - 1] + g.tw - t.x, t.h]);
  if (t.c < cols - 1) strips.push([g.xs[t.c + 1], t.y, t.x + t.w - g.xs[t.c + 1], t.h]);
  if (t.r > 0) strips.push([t.x, t.y, t.w, g.ys[t.r - 1] + g.th - t.y]);
  if (t.r < rows - 1) strips.push([t.x, g.ys[t.r + 1], t.w, t.y + t.h - g.ys[t.r + 1]]);
  ctx.fillStyle = c.accent + "24";
  ctx.strokeStyle = c.accent;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  for (const [x, y, w, h] of strips) {
    if (w <= 0 || h <= 0) continue;
    ctx.fillRect(X(x), Y(y), w * view.s, h * view.s);
    ctx.strokeRect(X(x) + 0.5, Y(y) + 0.5, w * view.s - 1, h * view.s - 1);
  }
  ctx.setLineDash([]);

  const W = img.naturalWidth, H = img.naturalHeight;
  for (const b of state.boxes) {
    const k = clipBox(b, t, W, H, state.settings.minVis);
    if (!k) continue;
    const color = k.keep ? colorFor(b.cls) : c.dim;
    if (k.vis < 1) {   // the part of the box that falls outside the tile
      ctx.setLineDash([3, 3]);
      const r = { x0: (b.cx - b.w / 2) * W, y0: (b.cy - b.h / 2) * H, x1: (b.cx + b.w / 2) * W, y1: (b.cy + b.h / 2) * H };
      drawRect(r, color + "99", { width: 1 });
      ctx.setLineDash([]);
    }
    const pct = k.vis < 1 ? ` ${Math.round(k.vis * 100)}%` : "";
    if (!k.keep) ctx.setLineDash([6, 4]);
    drawRect(k, color, { width: k.keep ? 2 : 1.5, label: (k.keep ? className(b.cls) : "dropped") + pct, top: ty, fill: k.keep });
    ctx.setLineDash([]);
  }

  drawRect(tileRect(t), c.accent, { width: 2, label: tileName(t), top: -Infinity, outside: true });
}

// [start, end] of every overlap between consecutive tiles along one axis
function overlaps(starts, size) {
  const out = [];
  for (let i = 0; i + 1 < starts.length; i++) {
    if (starts[i] + size > starts[i + 1]) out.push([starts[i + 1], starts[i] + size]);
  }
  return out;
}

const px = (v) => Math.round(v) + 0.5;   // crisp 1px lines
const tileRect = (t) => ({ x0: t.x, y0: t.y, x1: t.x + t.w, y1: t.y + t.h });
const tileName = (t) => `r${t.r} c${t.c} · ${t.x},${t.y}`;

// rectangle {x0, y0, x1, y1} in image px with an optional label tab; the tab sits above the
// rectangle unless that would cross `top` (canvas y), then inside; `outside` keeps it above
function drawRect(r, color, { width = 2, label = null, top = 0, fill = false, outside = false } = {}) {
  const x = X(r.x0), y = Y(r.y0), w = (r.x1 - r.x0) * view.s, h = (r.y1 - r.y0) * view.s;
  if (fill) {
    ctx.fillStyle = color + "26";
    ctx.fillRect(x, y, w, h);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(x, y, w, h);
  if (label) {
    const dash = ctx.getLineDash();
    ctx.setLineDash([]);
    const tw = ctx.measureText(label).width + 8;
    const ty = outside || y - 15 >= top ? y - 15 - (outside ? width / 2 : 0) : y;
    const solid = color.slice(0, 7);
    ctx.fillStyle = solid;
    ctx.fillRect(x - width / 2, ty, tw, 15);
    ctx.fillStyle = textOn(solid);
    ctx.fillText(label, x - width / 2 + 4, ty + 2);
    ctx.setLineDash(dash);
  }
}

// black or white label text, whichever reads better on a #rrggbb background
function textOn(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#000" : "#fff";
}

// --- pointer ---------------------------------------------------------------

// left or middle drag pans, a left click selects the tile under the pointer
cv.addEventListener("mousedown", (e) => {
  if (!img.naturalWidth || (e.button !== 0 && e.button !== 1)) return;
  e.preventDefault();
  press = { x: e.clientX, y: e.clientY, moved: false };
});

cv.addEventListener("mousemove", (e) => {
  if (!img.naturalWidth) return;
  mouse = toImage(e);
  const h = state.mode === "image" ? tileAt(mouse) : -1;
  if (h !== state.hover) { state.hover = h; draw(); }
});

cv.addEventListener("mouseleave", () => {
  mouse = null;
  if (state.hover >= 0) { state.hover = -1; draw(); }
});

window.addEventListener("mousemove", (e) => {
  if (!press) return;
  const dx = e.clientX - press.x, dy = e.clientY - press.y;
  if (!press.moved && Math.hypot(dx, dy) < CLICK_PX) return;
  press.moved = true;
  cv.classList.add("grab");
  panBy(dx, dy);
  press.x = e.clientX;
  press.y = e.clientY;
  draw();
});

window.addEventListener("mouseup", (e) => {
  if (!press) return;
  const p = press;
  press = null;
  cv.classList.remove("grab");
  if (p.moved || e.button !== 0 || e.target !== cv) return;
  const i = tileAt(toImage(e));
  if (i >= 0 && i !== state.tile) selectTile(i);
});

// double-click a tile to look at it close up, double-click again to go back
cv.addEventListener("dblclick", (e) => {
  if (state.mode === "tile") { setMode("image"); return; }
  const i = tileAt(toImage(e));
  if (i < 0) return;
  state.tile = i;
  setMode("tile");
});

// Touchpad pinch arrives as a wheel event with ctrlKey set (as does Ctrl + mouse wheel): zoom.
// A plain wheel event (two-finger scroll) pans the zoomed image.
cv.addEventListener("wheel", (e) => {
  if (!img.naturalWidth) return;
  e.preventDefault();
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? cv.clientHeight : 1;
  const dx = e.deltaX * unit, dy = e.deltaY * unit;
  const r = cv.getBoundingClientRect();
  if (e.ctrlKey || e.metaKey) {
    const step = Math.max(-50, Math.min(50, dy));   // tame big mouse-wheel notches
    zoomAt(Math.exp(-step * 0.01), e.clientX - r.left, e.clientY - r.top);
  } else {
    panBy(-dx, -dy);
  }
  mouse = toImage(e);
  draw();
}, { passive: false });

window.addEventListener("resize", () => { layout(); draw(); });

// --- side panels -----------------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function render() {
  draw();
  renderTiles();
  renderTileBoxes();
  renderHead();
}

function renderFiles() {
  const s = state.settings;
  const list = $("file-list");
  list.innerHTML = state.images.map((im, i) =>
    `<li data-i="${i}" class="${im.empty ? "blank" : im.done ? "done" : ""} ${i === state.idx ? "active" : ""}" ` +
    `title="${esc(im.name)}${im.w ? ` (${im.w}×${im.h})` : ""}${im.done ? "" : ", no label file"}">` +
    `<span>${esc(im.name)}</span><span class="count">${tileCount(im.w, im.h, s) || ""}</span></li>`
  ).join("");
  list.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  const tiles = state.images.reduce((n, im) => n + tileCount(im.w, im.h, s), 0);
  $("progress").textContent = state.images.length
    ? `${state.images.length} images · ${tiles.toLocaleString()} tiles` : "";
}

function renderHead() {
  const g = state.grid;
  $("tile-counter").textContent = g ? `${state.tile + 1} / ${g.tiles.length}` : "—";
  $("mode-image").classList.toggle("on", state.mode === "image");
  $("mode-tile").classList.toggle("on", state.mode === "tile");
  const s = state.settings;
  $("overlap-pct").textContent = `${Math.round((100 * s.ox) / s.w)}% × ${Math.round((100 * s.oy) / s.h)}%`;
  for (const b of document.querySelectorAll(".presets button")) {
    b.classList.toggle("on", s.w === s.h && s.w === +b.dataset.size);
  }

  if (!g) { $("tile-summary").textContent = ""; return; }
  // the last tile is pulled back to the image edge, so its overlap can be larger than asked
  const edge = (starts, size, asked) => {
    const n = starts.length;
    return n > 1 ? starts[n - 2] + size - starts[n - 1] : asked;
  };
  const ex = edge(g.xs, g.tw, s.ox), ey = edge(g.ys, g.th, s.oy);
  const lines = [`${g.xs.length} × ${g.ys.length} = ${g.tiles.length} tiles of ${g.tw}×${g.th}`];
  if (g.tw < s.w || g.th < s.h) lines.push(`image (${img.naturalWidth}×${img.naturalHeight}) is smaller than the tile`);
  if ((g.xs.length > 1 && ex !== s.ox) || (g.ys.length > 1 && ey !== s.oy)) {
    lines.push(`edge tiles overlap ${g.xs.length > 1 ? ex : 0} × ${g.ys.length > 1 ? ey : 0} px`);
  }
  $("tile-summary").innerHTML = lines.map(esc).join("<br>");
}

function renderTiles() {
  const g = state.grid;
  if (!g) { $("tile-list").innerHTML = `<li class="none">no image</li>`; return; }
  $("tile-list").innerHTML = g.tiles.map((t, i) => {
    const k = clipsIn(t), kept = k.filter((x) => x.keep).length, dropped = k.length - kept;
    return `<li data-t="${i}" class="${i === state.tile ? "active" : ""}" ` +
      `title="row ${t.r}, column ${t.c}: ${kept} box${kept === 1 ? "" : "es"} kept, ${dropped} dropped">` +
      `<span class="idx">${i + 1}</span>` +
      `<span class="pos">${t.x}, ${t.y}</span>` +
      `<span class="count">${kept ? `<b>${kept}</b>` : "–"}${dropped ? ` +${dropped}` : ""}</span></li>`;
  }).join("");
  $("tile-list").querySelector(".active")?.scrollIntoView({ block: "nearest" });
}

function renderTileBoxes() {
  const t = selTile();
  const k = t ? clipsIn(t) : [];
  $("box-list").innerHTML = k.length ? k.map((b) =>
    `<li class="${b.keep ? "" : "dropped"}" title="${b.keep ? "kept" : "dropped: less than " + state.settings.minVis + "% visible"}">` +
    `<span class="swatch" style="background:${colorFor(b.cls)}"></span>` +
    `<span class="cname">${esc(className(b.cls))}</span>` +
    `<span class="dims">${Math.round(b.x1 - b.x0)}×${Math.round(b.y1 - b.y0)}</span>` +
    `<span class="vis">${Math.round(b.vis * 100)}%</span></li>`
  ).join("") : `<li class="none">${t ? "no boxes in this tile" : "no tile"}</li>`;
}

$("file-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li) go(+li.dataset.i);
});

$("tile-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-t]");
  if (li) selectTile(+li.dataset.t);
});

$("tile-list").addEventListener("dblclick", (e) => {
  const li = e.target.closest("li[data-t]");
  if (li) { state.tile = +li.dataset.t; setMode("tile"); }
});

// --- toolbar ---------------------------------------------------------------

$("dir-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const path = $("dir-input").value.trim();
  if (path) openDir(path);
  $("dir-input").blur();
});
$("prev-btn").onclick = () => go(state.idx - 1);
$("next-btn").onclick = () => go(state.idx + 1);
$("prev-tile-btn").onclick = () => selectTile(state.tile - 1);
$("next-tile-btn").onclick = () => selectTile(state.tile + 1);
$("mode-image").onclick = () => setMode("image");
$("mode-tile").onclick = () => setMode("tile");
$("zoom-btn").onclick = () => { resetZoom(); draw(); };

// --- directory browser -----------------------------------------------------

const browser = { path: null };

async function browse(path) {
  let r;
  try {
    r = await api("/api/browse?path=" + encodeURIComponent(path));
  } catch (err) {
    if (path !== "~") return browse("~");
    alert(err.message);
    return;
  }
  browser.path = r.path;
  $("browser-path").textContent = r.path;
  const rows = [];
  if (r.parent) rows.push(`<li data-path="${esc(r.parent)}"><span>../</span><span></span></li>`);
  for (const d of r.dirs) {
    const p = r.path.replace(/\/$/, "") + "/" + d.name;
    rows.push(`<li data-path="${esc(p)}"><span>${esc(d.name)}/</span>` +
              `<span class="count">${d.images ? d.images + " img" : ""}</span></li>`);
  }
  $("browser-list").innerHTML = rows.join("") || `<li class="none">no sub-directories</li>`;
  $("browser-count").textContent = `${r.images} image${r.images === 1 ? "" : "s"} here`;
}

function openBrowser() {
  $("browser").showModal();
  browse($("dir-input").value.trim() || "~");
}

$("browse-btn").onclick = openBrowser;
$("empty-browse-btn").onclick = openBrowser;
$("browser-close").onclick = () => $("browser").close();
$("browser-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-path]");
  if (li) browse(li.dataset.path);
});
$("browser-open").onclick = () => {
  $("browser").close();
  openDir(browser.path);
};

// --- export ----------------------------------------------------------------

let exporting = null;   // {cancel} while an export runs

const exportImages = () => state.images.filter((im) => im.done || !$("export-labelled").checked);

function settingsPayload() {
  const s = state.settings;
  return { w: s.w, h: s.h, ox: s.ox, oy: s.oy, min_vis: s.minVis };
}

function showExportSummary() {
  const s = state.settings, ims = exportImages();
  const tiles = ims.reduce((n, im) => n + tileCount(im.w, im.h, s), 0);
  $("export-summary").textContent =
    `${ims.length} image${ims.length === 1 ? "" : "s"} → ${tiles.toLocaleString()} tiles` +
    `${$("export-skip-empty").checked ? " at most" : ""} · ${s.w}×${s.h}, overlap ${s.ox}×${s.oy}` +
    ` · boxes kept if ≥ ${s.minVis}% visible`;
}

function openExport() {
  if (!state.dir) { alert("Open an image directory first."); return; }
  if ($("export-out").dataset.dir !== state.dir) {   // default to <dir>/tiles for each new dir
    $("export-out").value = state.dir.replace(/\/$/, "") + "/tiles";
    $("export-out").dataset.dir = state.dir;
  }
  $("export-bar").hidden = true;
  $("export-status").textContent = "";
  showExportSummary();
  $("export").showModal();
}

async function runExport() {
  if (exporting) { exporting.cancel = true; return; }
  const out = $("export-out").value.trim();
  const ims = exportImages();
  if (!out || !ims.length) { $("export-status").textContent = "nothing to export"; return; }
  try {
    const check = await api("/api/export/check?out=" + encodeURIComponent(out));
    if (check.images && !confirm(`${check.out} already contains ${check.images} images.\n` +
                                 `Tiles with the same name are overwritten. Continue?`)) return;
    await api("/api/export/start", { out, settings: settingsPayload() });
  } catch (err) {
    $("export-status").textContent = err.message;
    return;
  }

  exporting = { cancel: false };
  $("export-run").textContent = "cancel";
  $("export-run").classList.remove("primary");
  $("export-bar").hidden = false;
  const bar = $("export-bar").firstElementChild;
  const skip = $("export-skip-empty").checked;
  let tiles = 0, withBoxes = 0, done = 0, error = "";
  for (const im of ims) {
    if (exporting.cancel) break;
    $("export-status").textContent = `${done + 1} / ${ims.length} · ${im.name}`;
    try {
      const r = await api("/api/export/image", { out, name: im.name, settings: settingsPayload(), skip_empty: skip });
      tiles += r.tiles;
      withBoxes += r.with_boxes;
    } catch (err) {
      error = `${im.name}: ${err.message}`;
      break;
    }
    done++;
    bar.style.width = `${(100 * done) / ims.length}%`;
  }
  const stopped = error ? `stopped at ${error} · ` : exporting.cancel ? "cancelled · " : "";
  $("export-status").textContent = `${stopped}wrote ${tiles.toLocaleString()} tiles ` +
    `(${withBoxes.toLocaleString()} with boxes) from ${done} image${done === 1 ? "" : "s"} to ${out}`;
  exporting = null;
  $("export-run").textContent = "export";
  $("export-run").classList.add("primary");
}

$("export-btn").onclick = openExport;
$("export-run").onclick = runExport;
$("export-form").addEventListener("submit", (e) => { e.preventDefault(); runExport(); });
$("export-form").addEventListener("change", showExportSummary);
$("export-close").onclick = () => $("export").close();
$("export").addEventListener("close", () => { if (exporting) exporting.cancel = true; });

// --- keyboard --------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || document.querySelector("dialog[open]")) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.shiftKey && e.key.startsWith("Arrow")) {
    e.preventDefault();
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    moveTile(...d);
    return;
  }
  switch (e.key) {
    case "ArrowLeft":  e.preventDefault(); go(state.idx - 1); return;
    case "ArrowRight": e.preventDefault(); go(state.idx + 1); return;
    case "[":          selectTile(state.tile - 1); return;
    case "]":          selectTile(state.tile + 1); return;
    case " ":          e.preventDefault(); setMode(state.mode === "tile" ? "image" : "tile"); return;
    case "Escape":     setMode("image"); return;
    case "f":
    case "F":          resetZoom(); draw(); return;
    case "+":
    case "=":          zoomAt(1.25, cv.clientWidth / 2, cv.clientHeight / 2); draw(); return;
    case "-":          zoomAt(0.8, cv.clientWidth / 2, cv.clientHeight / 2); draw(); return;
  }
});

// --- boot ------------------------------------------------------------------

async function boot() {
  state.settings = loadSettings();
  showSettings();
  const s = await api("/api/state");
  applyState(s);
  layout();
  render();
  if (s.dir) {
    remember(s.dir);
    if (s.images.length) {
      const firstDone = s.images.findIndex((x) => x.done && !x.empty);
      await loadImage(firstDone < 0 ? 0 : firstDone);
    }
  } else if (recall()) {
    await openDir(recall(), true);   // reopen last session's directory if it still exists
  }
}

boot();
