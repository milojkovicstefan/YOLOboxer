// YOLOboxer front-end. Plain JS, no build step.
//
// Boxes are kept in YOLO form {cls, cx, cy, w, h}, all normalised 0-1
// relative to the image, exactly as they are written to the label file.

// keyboard shortcut for class id i is KEYS[i]
const KEYS = "1234567890abcdefghijklmnopqrstuvwxyz";
const COLORS = ["#f5c818", "#e0431c", "#2f9e57", "#c99a06", "#8a3fc7", "#0f9fa8",
                "#d6336c", "#6b8e23", "#b8551f", "#4054b2", "#7a5c3e", "#c2185b"];
const LAST_DIR_KEY = "yoloboxer.lastDir";
const MIN_BOX_PX = 4;    // a drag smaller than this on screen counts as a click
const MAX_ZOOM = 32;     // relative to the fitted size

const state = {
  dir: null,
  images: [],      // [{name, done, empty}]
  classes: [],     // ["cat", "dog", ...]
  idx: 0,          // index into images
  boxes: [],       // boxes of the current image
  sel: -1,         // selected box index
  cur: 0,          // class used for the next drawn box
  dirty: false,
};

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let img = new Image();
let view = { x: 0, y: 0, w: 0, h: 0 };   // where the image sits on the canvas (css px)
let fit = { x: 0, y: 0, w: 0, h: 0 };    // the same, with the whole image fitted (zoom 1)
let zoom = 1;                            // view size relative to fit
let pan = null;                          // last pointer position while middle-drag panning
let mouse = null;                        // last pointer position, normalised
let drag = null;                         // {x0, y0, x1, y1} while drawing

const colorFor = (c) => COLORS[c % COLORS.length];
const keyFor = (c) => KEYS[c] ?? "";
const className = (c) => state.classes[c] ?? `#${c}`;
const current = () => state.images[state.idx];

// --- api -------------------------------------------------------------------

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || r.statusText);
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
  state.cur = Math.min(state.cur, Math.max(0, state.classes.length - 1));
  $("dir-input").value = s.dir ?? "";
  $("empty").hidden = !!s.dir && s.images.length > 0;
  if (s.dir && !s.images.length) {
    $("empty").innerHTML = "<p>No images in this directory.</p>";
  }
  renderFiles();
  renderClasses();
}

async function openDir(path, quiet = false) {
  if (state.dirty && !confirm("Discard unsaved boxes?")) return;
  try {
    applyState(await api("/api/open", { path }));
  } catch (err) {
    if (!quiet) alert(err.message);
    return;
  }
  remember(state.dir);
  if (!state.images.length) { clearStage(); return; }
  const firstTodo = state.images.findIndex((x) => !x.done);
  await loadImage(firstTodo < 0 ? 0 : firstTodo);
}

async function loadImage(i) {
  state.idx = i;
  const name = current().name;
  state.boxes = (await api("/api/label/" + encodeURIComponent(name))).boxes;
  state.sel = -1;
  setDirty(false);
  img = new Image();
  img.onload = () => { layout(); draw(); };
  img.src = "/image/" + encodeURIComponent(name) + "?t=" + Date.now();
  $("image-name").textContent = name;
  $("counter").textContent = `${i + 1} / ${state.images.length}`;
  renderFiles();
  renderBoxes();
  renderClasses();
}

function clearStage() {
  img = new Image();
  state.boxes = [];
  $("image-name").textContent = "";
  $("counter").textContent = "—";
  renderBoxes();
  draw();
}

// --- navigation & saving ---------------------------------------------------

async function go(i) {
  if (!state.images.length || i < 0 || i >= state.images.length || i === state.idx) return;
  if (state.dirty && !confirm("Discard unsaved boxes?")) return;
  await loadImage(i);
}

async function save(advance) {
  if (!state.images.length) return;
  const name = current().name;
  await api("/api/label/" + encodeURIComponent(name), { boxes: state.boxes });
  current().done = true;
  current().empty = state.boxes.length === 0;
  setDirty(false);
  renderFiles();
  if (advance) await go(state.idx + 1);
}

function setDirty(d) {
  state.dirty = d;
  $("dirty").hidden = !d;
}

async function saveClasses(classes) {
  const r = await api("/api/classes", { classes });
  state.classes = r.classes;
  state.cur = Math.min(state.cur, Math.max(0, state.classes.length - 1));
  renderClasses();
  renderBoxes();
  draw();
}

// --- box editing -----------------------------------------------------------

function addBox(b) {
  state.boxes.push(b);
  state.sel = state.boxes.length - 1;
  changed();
}

function deleteBox(i) {
  state.boxes.splice(i, 1);
  state.sel = Math.min(state.sel, state.boxes.length - 1);
  changed();
}

function selectBox(i) {
  state.sel = i;
  if (i >= 0) state.cur = state.boxes[i].cls;
  renderBoxes();
  renderClasses();
  draw();
}

function pickClass(c) {
  if (c < 0 || c >= state.classes.length) return;
  state.cur = c;
  if (state.sel >= 0) {
    state.boxes[state.sel].cls = c;
    changed();
  } else {
    renderClasses();
  }
}

function changed() {
  setDirty(true);
  renderBoxes();
  renderClasses();
  draw();
}

// topmost (smallest) box containing point p, or -1
function boxAt(p) {
  let best = -1, bestArea = Infinity;
  state.boxes.forEach((b, i) => {
    const inside = Math.abs(p.x - b.cx) <= b.w / 2 && Math.abs(p.y - b.cy) <= b.h / 2;
    if (inside && b.w * b.h < bestArea) { best = i; bestArea = b.w * b.h; }
  });
  return best;
}

// --- canvas ----------------------------------------------------------------

// size the canvas to its container and fit the image inside it
function layout() {
  const wrap = $("canvas-wrap");
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!img.naturalWidth) {
    fit = { x: 0, y: 0, w: 0, h: 0 };
  } else {
    const pad = 16;
    const s = Math.min((W - 2 * pad) / img.naturalWidth, (H - 2 * pad) / img.naturalHeight);
    const w = img.naturalWidth * s, h = img.naturalHeight * s;
    fit = { x: (W - w) / 2, y: (H - h) / 2, w, h };
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
  if (!fit.w || z === zoom) return;
  const u = (px - view.x) / view.w, v = (py - view.y) / view.h;
  view.w = fit.w * z;
  view.h = fit.h * z;
  view.x = px - u * view.w;
  view.y = py - v * view.h;
  zoom = z;
  clampView();
  showZoom();
}

function panBy(dx, dy) {
  view.x += dx;
  view.y += dy;
  clampView();
}

// keep the zoomed image covering the canvas; centre it on an axis where it fits
function clampView() {
  const pad = 16;
  const W = cv.clientWidth, H = cv.clientHeight;
  view.x = view.w <= W ? (W - view.w) / 2 : Math.min(pad, Math.max(W - view.w - pad, view.x));
  view.y = view.h <= H ? (H - view.h) / 2 : Math.min(pad, Math.max(H - view.h - pad, view.y));
}

function showZoom() {
  const btn = $("zoom-btn");
  btn.textContent = zoom === 1 ? "fit" : `${zoom.toFixed(1)}×`;
  btn.disabled = zoom === 1;
}

// canvas mouse event -> normalised image coords (clamped to the image)
function toImage(e) {
  const r = cv.getBoundingClientRect();
  const clamp = (v) => Math.min(1, Math.max(0, v));
  return {
    x: clamp((e.clientX - r.left - view.x) / view.w),
    y: clamp((e.clientY - r.top - view.y) / view.h),
  };
}

function draw() {
  const W = cv.clientWidth, H = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if (!img.naturalWidth) return;

  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ink").trim();

  // image with a hard ink frame and offset shadow
  ctx.fillStyle = ink;
  ctx.fillRect(view.x + 4, view.y + 4, view.w, view.h);
  ctx.imageSmoothingEnabled = view.w < img.naturalWidth;
  ctx.drawImage(img, view.x, view.y, view.w, view.h);
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1;
  ctx.strokeRect(view.x - 0.5, view.y - 0.5, view.w + 1, view.h + 1);

  // existing boxes
  ctx.font = "11px " + css.getPropertyValue("--mono");
  ctx.textBaseline = "top";
  state.boxes.forEach((b, i) => drawBox(b, colorFor(b.cls), i === state.sel, className(b.cls)));

  // box being drawn
  if (drag && drag.x1 !== undefined) {
    const b = rectToBox(drag);
    ctx.setLineDash([6, 4]);
    drawBox(b, colorFor(state.cur), false, null);
    ctx.setLineDash([]);
  }

  // crosshair guides
  if (mouse) {
    const x = view.x + mouse.x * view.w, y = view.y + mouse.y * view.h;
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(view.x, Math.round(y) + 0.5); ctx.lineTo(view.x + view.w, Math.round(y) + 0.5);
    ctx.moveTo(Math.round(x) + 0.5, view.y); ctx.lineTo(Math.round(x) + 0.5, view.y + view.h);
    ctx.stroke();
    ctx.strokeStyle = "rgba(0,0,0,.6)";
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawBox(b, color, selected, label) {
  const x = view.x + (b.cx - b.w / 2) * view.w;
  const y = view.y + (b.cy - b.h / 2) * view.h;
  const w = b.w * view.w, h = b.h * view.h;
  if (selected) {
    ctx.fillStyle = color + "33";
    ctx.fillRect(x, y, w, h);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeRect(x, y, w, h);
  if (label) {
    const tw = ctx.measureText(label).width + 8;
    const ty = y - 15 >= view.y ? y - 15 : y;   // flip inside if no room above
    ctx.fillStyle = color;
    ctx.fillRect(x - (selected ? 1.5 : 1), ty, tw, 15);
    ctx.fillStyle = textOn(color);
    ctx.fillText(label, x + 3, ty + 2);
  }
}

// black or white label text, whichever reads better on a #rrggbb background
function textOn(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#000" : "#fff";
}

function rectToBox(d) {
  const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1);
  const y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
  return { cls: state.cur, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 };
}

cv.addEventListener("mousedown", (e) => {
  if (!img.naturalWidth) return;
  if (e.button === 1) {               // middle button: pan
    e.preventDefault();
    pan = { x: e.clientX, y: e.clientY };
    cv.style.cursor = "grabbing";
    return;
  }
  if (e.button !== 0) return;
  const p = toImage(e);
  drag = { x0: p.x, y0: p.y };
});

cv.addEventListener("mousemove", (e) => {
  if (!img.naturalWidth) return;
  mouse = toImage(e);
  if (drag) { drag.x1 = mouse.x; drag.y1 = mouse.y; }
  draw();
});

cv.addEventListener("mouseleave", () => { mouse = null; draw(); });

window.addEventListener("mousemove", (e) => {
  if (!pan) return;
  panBy(e.clientX - pan.x, e.clientY - pan.y);
  pan = { x: e.clientX, y: e.clientY };
  draw();
});

window.addEventListener("mouseup", (e) => {
  if (pan) { pan = null; cv.style.cursor = ""; return; }
  if (!drag) return;
  const d = drag;
  drag = null;
  const b = d.x1 === undefined ? null : rectToBox(d);
  if (b && b.w * view.w >= MIN_BOX_PX && b.h * view.h >= MIN_BOX_PX) {
    if (!state.classes.length) {
      alert("Add a class first (right panel).");
      $("add-class-input").focus();
      draw();
      return;
    }
    addBox(b);
  } else {
    selectBox(boxAt({ x: d.x0, y: d.y0 }));   // a click selects (or deselects)
  }
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

cv.addEventListener("dblclick", () => { resetZoom(); draw(); });

window.addEventListener("resize", () => { layout(); draw(); });

// --- side panels -----------------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderFiles() {
  const list = $("file-list");
  list.innerHTML = state.images.map((im, i) =>
    `<li data-i="${i}" class="${im.empty ? "blank" : im.done ? "done" : ""} ${i === state.idx ? "active" : ""}" ` +
    `title="${esc(im.name)}${im.empty ? " (saved, no boxes)" : ""}">` +
    `<span>${esc(im.name)}</span></li>`
  ).join("");
  list.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  const done = state.images.filter((x) => x.done).length;
  $("progress").textContent = state.images.length ? `${done} / ${state.images.length} labelled` : "";
}

function renderClasses() {
  const counts = state.classes.map((_, c) => state.boxes.filter((b) => b.cls === c).length);
  const last = state.classes.length - 1;
  $("class-list").innerHTML = state.classes.length ? state.classes.map((name, c) =>
    `<li data-c="${c}" class="${c === state.cur ? "active" : ""}">` +
    `<span class="key">${esc(keyFor(c))}</span>` +
    `<span class="swatch" style="background:${colorFor(c)}"></span>` +
    `<span class="cname">${esc(name)}</span>` +
    `<span class="count">${counts[c] || ""}</span>` +
    (c === last ? `<button class="x" data-remove title="remove last class">&times;</button>` : `<span></span>`) +
    `</li>`
  ).join("") : `<li class="none">no classes yet</li>`;
}

function renderBoxes() {
  const W = img.naturalWidth, H = img.naturalHeight;
  $("box-list").innerHTML = state.boxes.length ? state.boxes.map((b, i) =>
    `<li data-i="${i}" class="${i === state.sel ? "active" : ""}">` +
    `<span class="idx">${i + 1}</span>` +
    `<span class="swatch" style="background:${colorFor(b.cls)}"></span>` +
    `<span class="cname">${esc(className(b.cls))}</span>` +
    `<span class="dims">${W ? `${Math.round(b.w * W)}×${Math.round(b.h * H)}` : ""}</span>` +
    `<button class="x" data-remove title="delete box">&times;</button>` +
    `</li>`
  ).join("") : `<li class="none">no boxes</li>`;
}

$("file-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li) go(+li.dataset.i);
});

$("class-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-c]");
  if (!li) return;
  const c = +li.dataset.c;
  if (e.target.closest("[data-remove]")) {
    if (confirm(`Remove class "${state.classes[c]}"?\nBoxes already saved with id ${c} keep that id.`)) {
      saveClasses(state.classes.slice(0, -1));
    }
    return;
  }
  pickClass(c);
});

// double-click a class name to rename it in place
$("class-list").addEventListener("dblclick", (e) => {
  const li = e.target.closest("li[data-c]");
  if (!li) return;
  const c = +li.dataset.c;
  const cell = li.querySelector(".cname");
  const input = document.createElement("input");
  input.value = state.classes[c];
  cell.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = (commit) => {
    if (finished) return;
    finished = true;
    const v = input.value.trim();
    if (commit && v && v !== state.classes[c]) {
      const next = [...state.classes];
      next[c] = v;
      saveClasses(next);
    } else {
      renderClasses();
    }
  };
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") finish(true);
    if (ev.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
});

$("add-class-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("add-class-input");
  const name = input.value.trim();
  if (!name) return;
  if (!state.dir) { alert("Open an image directory first."); return; }
  input.value = "";
  await saveClasses([...state.classes, name]);
  state.cur = state.classes.length - 1;
  renderClasses();
});

$("box-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (!li) return;
  const i = +li.dataset.i;
  if (e.target.closest("[data-remove]")) deleteBox(i);
  else selectBox(i);
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
$("save-btn").onclick = () => save(false);
$("zoom-btn").onclick = () => { resetZoom(); draw(); };
$("save-next-btn").onclick = () => save(true);
$("clear-btn").onclick = () => {
  if (!state.boxes.length) return;
  state.boxes = [];
  state.sel = -1;
  changed();
};

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

// --- statistics ------------------------------------------------------------

async function openStats() {
  if (!state.dir) { alert("Open an image directory first."); return; }
  $("stats-body").innerHTML = statsHtml(await api("/api/stats"));
  $("stats").showModal();
}

function statsHtml(s) {
  const im = s.images;
  const pct = (n, total) => (total ? Math.round((100 * n) / total) : 0);
  const groups = [
    ["with boxes", im.with_boxes, "done"],
    ["saved, no boxes", im.empty, "blank"],
    ["unlabelled", im.unlabelled, "todo"],
  ];

  const tiles = groups.map(([label, n, cls]) =>
    `<div class="tile"><span class="tile-label"><span class="marker ${cls}"></span>${label}</span>` +
    `<b>${n}</b><span class="pct">${pct(n, im.total)}% of ${im.total}</span></div>`
  ).join("");
  const split = groups.filter(([, n]) => n > 0).map(([label, n, cls]) =>
    `<span class="seg ${cls}" style="flex:${n}" title="${label}: ${n} (${pct(n, im.total)}%)"></span>`
  ).join("");

  const max = Math.max(1, ...s.classes.map((c) => c.instances));
  const rows = s.classes.map((c) => {
    const name = c.name === null
      ? `<span class="unknown">#${c.id} (not in classes.txt)</span>`
      : esc(c.name);
    const tip = `${c.name ?? "#" + c.id}: ${c.instances} instance${c.instances === 1 ? "" : "s"}` +
                ` (${pct(c.instances, s.boxes)}% of boxes) in ${c.images} image${c.images === 1 ? "" : "s"}` +
                (c.avg_px === null ? "" : `, ${pxArea(c.avg_px)} per object`);
    return `<tr title="${esc(tip)}">` +
      `<td class="key">${c.name === null ? "" : esc(keyFor(c.id))}</td>` +
      `<td><span class="name"><span class="swatch" style="background:${colorFor(c.id)}"></span>${name}</span></td>` +
      `<td class="num">${c.images}</td>` +
      `<td class="num">${c.avg_px === null ? "–" : c.avg_px.toLocaleString()}</td>` +
      `<td class="inst-cell"><div class="inst-row"><span class="inst-track">` +
      `<span class="inst-bar${c.instances ? "" : " zero"}" style="width:${(100 * c.instances) / max}%"></span>` +
      `</span><span class="num">${c.instances}</span></div></td></tr>`;
  }).join("");

  return `<h3>images <span>${im.total} total</span></h3>` +
    `<div class="tiles">${tiles}</div><div class="split">${split}</div>` +
    `<h3>instances per class <span>${s.boxes} boxes total` +
    `${s.avg_px === null ? "" : ` · ${pxArea(s.avg_px)} per object on average`}</span></h3>` +
    (s.classes.length
      ? `<table class="class-stats"><thead><tr><th>key</th><th>class</th>` +
        `<th class="num">images</th><th class="num" title="average box area in pixels">avg px</th>` +
        `<th>instances</th></tr></thead><tbody>${rows}</tbody></table>`
      : `<p class="note">no classes yet</p>`);
}

// a pixel area with the side of the square of the same area, e.g. "1,600 px (≈ 40×40)"
function pxArea(n) {
  const side = Math.round(Math.sqrt(n));
  return `${n.toLocaleString()} px (≈ ${side}×${side})`;
}

$("stats-btn").onclick = openStats;
$("stats-close").onclick = () => $("stats").close();

// --- keyboard --------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || document.querySelector("dialog[open]")) return;

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    save(false);
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  switch (e.key) {
    case "ArrowLeft":  e.preventDefault(); go(state.idx - 1); return;
    case "ArrowRight": e.preventDefault(); go(state.idx + 1); return;
    case "Enter":      e.preventDefault(); save(true); return;
    case "Escape":     selectBox(-1); return;
    case "+":
    case "=":          zoomAt(1.25, cv.clientWidth / 2, cv.clientHeight / 2); draw(); return;
    case "-":          zoomAt(0.8, cv.clientWidth / 2, cv.clientHeight / 2); draw(); return;
    case "Delete":
    case "Backspace":  e.preventDefault(); if (state.sel >= 0) deleteBox(state.sel); return;
  }
  const c = KEYS.indexOf(e.key.toLowerCase());
  if (c >= 0) pickClass(c);
});

window.addEventListener("beforeunload", (e) => {
  if (state.dirty) e.preventDefault();
});

// --- boot ------------------------------------------------------------------

async function boot() {
  const s = await api("/api/state");
  applyState(s);
  layout();
  if (s.dir) {
    remember(s.dir);
    if (s.images.length) {
      const firstTodo = s.images.findIndex((x) => !x.done);
      await loadImage(firstTodo < 0 ? 0 : firstTodo);
    }
  } else if (recall()) {
    await openDir(recall(), true);   // reopen last session's directory if it still exists
  }
}

boot();
