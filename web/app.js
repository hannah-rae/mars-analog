import { EmbeddingGrid, quantiles, findPeaks, backgroundLSE, combine } from "./search.js?v=4";

const ol = window.ol;
const $ = id => document.getElementById(id);

// matplotlib "inferno", 11 stops -> 256-entry LUT
const INFERNO = ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754",
  "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"];
const LUT = (() => {
  const stops = INFERNO.map(h => [1, 3, 5].map(o => parseInt(h.slice(o, o + 2), 16)));
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1), a = Math.floor(t), f = t - a;
    const b = Math.min(a + 1, stops.length - 1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[a][c] * (1 - f) + stops[b][c] * f);
  }
  return lut;
})();

const state = {
  config: null, meta: null, grid: null, R: 3396190,
  pos: [], neg: [],            // examples: { indices, ring (map coords) | null, bw, bh }
  tool: "point",               // point | box | polygon
  win: [1, 1],                 // sliding-window size (tokens) of the current example search
  text: null,                  // active text query (vocab phrase)
  vocab: null, lse: null,      // dino.txt vocabulary + background log-sum-exp per token
  vals: null,                  // normalised scores in [0,1] (NaN = invalid)
  raw: null,                   // raw score
  threshold: 0.5, view: "heat",
  peaks: [],
};

// ------------------------------------------------------------------ helpers
const toLonLat = ([x, y]) => [x / state.R * 180 / Math.PI, y / state.R * 180 / Math.PI];
const fromLonLat = ([lon, lat]) => [lon * Math.PI / 180 * state.R, lat * Math.PI / 180 * state.R];
const fmtLL = ([lon, lat]) => `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;

function gridExtent() {
  const m = state.meta, [left, , , top] = m.bounds_m;
  return [left, top - m.grid.height * m.token_m, left + m.grid.width * m.token_m, top];
}
function coordToIndex([x, y]) {
  const m = state.meta, [left, , , top] = m.bounds_m;
  const c = Math.floor((x - left) / m.token_m), r = Math.floor((top - y) / m.token_m);
  if (c < 0 || r < 0 || c >= m.grid.width || r >= m.grid.height) return -1;
  return r * m.grid.width + c;
}
function cellExtent(i) {
  const m = state.meta, [left, , , top] = m.bounds_m;
  const c = i % m.grid.width, r = Math.floor(i / m.grid.width);
  return [left + c * m.token_m, top - (r + 1) * m.token_m, left + (c + 1) * m.token_m, top - r * m.token_m];
}
const cellPolygon = i => ol.geom.Polygon.fromExtent(cellExtent(i));
function indexToCoord(i) {
  const m = state.meta, [left, , , top] = m.bounds_m;
  const c = i % m.grid.width, r = Math.floor(i / m.grid.width);
  return [left + (c + 0.5) * m.token_m, top - (r + 0.5) * m.token_m];
}

// ---- examples: a single token (point) or every token whose centre is inside a drawn shape
const pointExample = i => ({ indices: [i], ring: null, bw: 1, bh: 1 });
function regionExample(geom) {
  const m = state.meta, [left, , , top] = m.bounds_m, tm = m.token_m;
  const [x0, y0, x1, y1] = geom.getExtent();
  const c0 = Math.max(0, Math.floor((x0 - left) / tm)), c1 = Math.min(m.grid.width - 1, Math.floor((x1 - left) / tm));
  const r0 = Math.max(0, Math.floor((top - y1) / tm)), r1 = Math.min(m.grid.height - 1, Math.floor((top - y0) / tm));
  const indices = [];
  let cmin = Infinity, cmax = -1, rmin = Infinity, rmax = -1;
  for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) {
    const i = r * m.grid.width + c;
    if (!state.grid.valid[i] || !geom.intersectsCoordinate(indexToCoord(i))) continue;
    indices.push(i); cmin = Math.min(cmin, c); cmax = Math.max(cmax, c); rmin = Math.min(rmin, r); rmax = Math.max(rmax, r);
  }
  if (!indices.length) {                  // shape smaller than a cell: use the cell under its centre
    const i = coordToIndex(ol.extent.getCenter(geom.getExtent()));
    return i >= 0 ? pointExample(i) : null;
  }
  return { indices, ring: geom.getCoordinates()[0], bw: cmax - cmin + 1, bh: rmax - rmin + 1 };
}
function exampleCenter(ex) {
  if (!ex.ring) return ex.indices[0];
  return coordToIndex(ol.extent.getCenter(new ol.geom.Polygon([ex.ring]).getExtent()));
}
/** window polygon of bw x bh tokens centred on cell i (same convention as search.js boxSum) */
function windowPolygon(i, bw, bh) {
  const m = state.meta, [left, , , top] = m.bounds_m, tm = m.token_m;
  const c = i % m.grid.width - ((bw - 1) >> 1), r = Math.floor(i / m.grid.width) - ((bh - 1) >> 1);
  return ol.geom.Polygon.fromExtent([left + c * tm, top - (r + bh) * tm, left + (c + bw) * tm, top - r * tm]);
}
const exampleLabel = ex => ex.ring
  ? `▭ ${ex.bw}×${ex.bh} tokens · ${(ex.bw * state.meta.token_m).toFixed(0)}×${(ex.bh * state.meta.token_m).toFixed(0)} m`
  : fmtLL(toLonLat(indexToCoord(ex.indices[0])));

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) { $("progressBar").style.width = `${100 * got / total}%`; status(`${label} ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`); }
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  $("progressBar").style.width = "0";
  return out;
}
const status = t => { $("status").textContent = t; };

// ------------------------------------------------------------------ map
let map, proj, heatSource, heatLayer, pcaLayer, markerSource, peakSource, landingLayer, gridLayer, hoverSource, drawInteraction;
let shiftDown = false, drawing = false;
const MIN_CELL_PX = 7;  // draw cells as squares / grid lines once a token is at least this many screen px
const heatCanvas = document.createElement("canvas");

function buildMap() {
  const ext = state.meta.bounds_m;
  proj = new ol.proj.Projection({ code: "MARS2000-EQC", units: "m", extent: ext });

  const base = new ol.layer.Image({
    source: new ol.source.ImageStatic({ url: state.config.basemap, imageExtent: ext, projection: proj }),
  });

  heatCanvas.width = state.meta.grid.width;
  heatCanvas.height = state.meta.grid.height;
  heatSource = new ol.source.ImageCanvas({
    projection: proj, ratio: 1,
    canvasFunction: (extent, resolution, pixelRatio, size) => {
      const c = document.createElement("canvas");
      c.width = size[0]; c.height = size[1];
      const ctx = c.getContext("2d");
      ctx.imageSmoothingEnabled = false;
      const g = gridExtent(), sx = size[0] / (extent[2] - extent[0]), sy = size[1] / (extent[3] - extent[1]);
      ctx.drawImage(heatCanvas, (g[0] - extent[0]) * sx, (extent[3] - g[3]) * sy, (g[2] - g[0]) * sx, (g[3] - g[1]) * sy);
      return c;
    },
  });
  heatLayer = new ol.layer.Image({ source: heatSource, opacity: +$("opacity").value });

  pcaLayer = new ol.layer.Image({
    visible: false, opacity: +$("opacity").value,
    source: new ol.source.ImageStatic({ url: `${state.datasetPath}/pca_rgb.png`, imageExtent: gridExtent(), projection: proj }),
  });

  markerSource = new ol.source.Vector();
  peakSource = new ol.source.Vector();
  // examples: a dot when zoomed out, the actual token cell once it is big enough to see
  const markerStyle = (f, res) => {
    const color = f.get("neg") ? "#ef5d5d" : "#7fd18b";
    const fill = new ol.style.Fill({ color: f.get("neg") ? "rgba(239,93,93,0.18)" : "rgba(127,209,139,0.18)" });
    if (f.get("region")) return [
      new ol.style.Style({ stroke: new ol.style.Stroke({ color: "#000", width: 4 }) }),
      new ol.style.Style({ stroke: new ol.style.Stroke({ color, width: 2 }), fill }),
    ];
    if (state.meta.token_m / res < MIN_CELL_PX) return new ol.style.Style({
      image: new ol.style.Circle({ radius: 6, fill: new ol.style.Fill({ color }), stroke: new ol.style.Stroke({ color: "#000", width: 2 }) }),
    });
    return [
      new ol.style.Style({ geometry: cellPolygon(f.get("index")), stroke: new ol.style.Stroke({ color: "#000", width: 4 }) }),
      new ol.style.Style({ geometry: cellPolygon(f.get("index")), stroke: new ol.style.Stroke({ color, width: 2 }),
        fill: new ol.style.Fill({ color: f.get("neg") ? "rgba(239,93,93,0.25)" : "rgba(127,209,139,0.25)" }) }),
    ];
  };

  // token grid: cell boundaries in view, only when cells are large enough on screen
  gridLayer = new ol.layer.Image({
    visible: false,
    source: new ol.source.ImageCanvas({
      projection: proj, ratio: 1,
      canvasFunction: (extent, resolution, pixelRatio, size) => {
        const c = document.createElement("canvas");
        c.width = size[0]; c.height = size[1];
        const m = state.meta, tm = m.token_m, g = gridExtent();
        const sx = size[0] / (extent[2] - extent[0]), sy = size[1] / (extent[3] - extent[1]);
        if (tm * sx < MIN_CELL_PX * pixelRatio) return c;
        const ctx = c.getContext("2d");
        ctx.strokeStyle = "rgba(255,255,255,0.28)"; ctx.lineWidth = pixelRatio;
        const x0 = Math.max(g[0], extent[0]), x1 = Math.min(g[2], extent[2]);
        const y0 = Math.max(g[1], extent[1]), y1 = Math.min(g[3], extent[3]);
        ctx.beginPath();
        for (let x = g[0] + Math.ceil((x0 - g[0]) / tm) * tm; x <= x1; x += tm) {
          const px = Math.round((x - extent[0]) * sx) + 0.5; ctx.moveTo(px, (extent[3] - y1) * sy); ctx.lineTo(px, (extent[3] - y0) * sy);
        }
        for (let y = g[3] - Math.ceil((g[3] - y1) / tm) * tm; y >= y0; y -= tm) {
          const py = Math.round((extent[3] - y) * sy) + 0.5; ctx.moveTo((x0 - extent[0]) * sx, py); ctx.lineTo((x1 - extent[0]) * sx, py);
        }
        ctx.stroke();
        return c;
      },
    }),
  });
  // the cell under the cursor (what a click would select)
  hoverSource = new ol.source.Vector();
  const hoverLayer = new ol.layer.Vector({
    source: hoverSource,
    style: [new ol.style.Style({ stroke: new ol.style.Stroke({ color: "rgba(0,0,0,0.7)", width: 3.5 }) }),
            new ol.style.Style({ stroke: new ol.style.Stroke({ color: "#fff", width: 1.5 }) })],
  });
  const peakLabel = f => new ol.style.Style({
    geometry: f.get("center") ? new ol.geom.Point(f.get("center")) : undefined,
    image: new ol.style.Circle({ radius: 10, fill: new ol.style.Fill({ color: "rgba(20,14,12,0.8)" }),
      stroke: new ol.style.Stroke({ color: "#e37148", width: 2 }) }),
    text: new ol.style.Text({ text: String(f.get("rank")), font: "600 11px JetBrains Mono, monospace",
      fill: new ol.style.Fill({ color: "#fff" }) }),
  });
  // region matches: the matched window outline plus the rank label at its centre
  const peakStyle = f => f.get("center") ? [
    new ol.style.Style({ stroke: new ol.style.Stroke({ color: "rgba(0,0,0,0.7)", width: 4 }) }),
    new ol.style.Style({ stroke: new ol.style.Stroke({ color: "#e37148", width: 2, lineDash: [6, 4] }) }),
    peakLabel(f),
  ] : peakLabel(f);

  const ls = state.config.landing_site;
  landingLayer = new ol.layer.Vector({
    source: new ol.source.Vector({ features: ls ? [new ol.Feature(new ol.geom.Point(fromLonLat(ls.lonlat)))] : [] }),
    style: new ol.style.Style({
      image: new ol.style.RegularShape({ points: 5, radius: 9, radius2: 4, fill: new ol.style.Fill({ color: "#fff" }),
        stroke: new ol.style.Stroke({ color: "#000", width: 1.5 }) }),
      text: ls ? new ol.style.Text({ text: ls.label, offsetY: -18, font: "500 12px Inter, sans-serif",
        fill: new ol.style.Fill({ color: "#fff" }), stroke: new ol.style.Stroke({ color: "#000", width: 3 }) }) : undefined,
    }),
  });

  map = new ol.Map({
    target: "map",
    layers: [base, heatLayer, pcaLayer, gridLayer, landingLayer, hoverLayer,
      new ol.layer.Vector({ source: peakSource, style: peakStyle }),
      new ol.layer.Vector({ source: markerSource, style: markerStyle })],
    view: new ol.View({ projection: proj, extent: ext, constrainOnlyCenter: true, maxZoom: 22 }),
    controls: ol.control.defaults.defaults({ attribution: false, rotate: false }).extend([new ol.control.ScaleLine()]),
  });
  map.getView().fit(ext, { padding: [20, 20, 20, window.innerWidth > 640 ? 360 : 20] });

  map.on("singleclick", e => {
    const i = coordToIndex(e.coordinate);
    if (i < 0 || !state.grid.valid[i]) return;
    if (e.originalEvent.altKey && state.vocab) { describe(i); return; }
    if (state.tool !== "point") return;   // the draw interaction handles clicks
    state.text = null;
    (e.originalEvent.shiftKey ? state.neg : state.pos).push(pointExample(i));
    update();
  });
  document.addEventListener("keydown", e => { if (e.key === "Shift") shiftDown = true; });
  document.addEventListener("keyup", e => { if (e.key === "Shift") shiftDown = false; });
  window.addEventListener("blur", () => { shiftDown = false; });
  map.on("pointermove", e => {
    if (e.dragging) return;
    const i = coordToIndex(e.coordinate);
    hoverSource.clear();
    if (gridLayer.getVisible() && i >= 0) hoverSource.addFeature(new ol.Feature(cellPolygon(i)));
    let t = fmtLL(toLonLat(e.coordinate));
    if (gridLayer.getVisible() && i >= 0) t += `  cell ${Math.floor(i / state.meta.grid.width)},${i % state.meta.grid.width}`;
    if (i >= 0 && state.vals && !Number.isNaN(state.vals[i])) t += `  score ${state.raw[i].toFixed(3)}`;
    $("readout").textContent = t;
  });
}

// ------------------------------------------------------------------ render
function drawHeat() {
  const { width: w, height: h } = state.meta.grid;
  const ctx = heatCanvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const d = img.data, v = state.vals, thr = state.threshold;
  if (v) {
    for (let i = 0; i < w * h; i++) {
      const x = v[i];
      if (!(x >= thr)) continue;
      const li = Math.round(x * 255) * 3, o = i * 4;
      d[o] = LUT[li]; d[o + 1] = LUT[li + 1]; d[o + 2] = LUT[li + 2];
      d[o + 3] = 255 * Math.min(1, 0.35 + 0.65 * (x - thr) / Math.max(1e-6, 1 - thr));
    }
  }
  ctx.putImageData(img, 0, 0);
  heatSource.changed();
}

function drawMarkers() {
  markerSource.clear();
  for (const [list, neg] of [[state.pos, false], [state.neg, true]])
    for (const ex of list) {
      const f = new ol.Feature(ex.ring ? new ol.geom.Polygon([ex.ring]) : new ol.geom.Point(indexToCoord(ex.indices[0])));
      f.set("neg", neg); f.set("index", ex.indices[0]); f.set("region", !!ex.ring);
      markerSource.addFeature(f);
    }
  const ul = $("examples"); ul.innerHTML = "";
  for (const [list, cls] of [[state.pos, "pos"], [state.neg, "neg"]])
    list.forEach((ex, k) => {
      const li = document.createElement("li"); li.className = cls;
      li.textContent = `${cls === "pos" ? "+" : "−"} ${exampleLabel(ex)}`;
      li.title = "Remove";
      li.onclick = () => { list.splice(k, 1); update(); };
      ul.appendChild(li);
    });
}

function drawPeaks() {
  peakSource.clear();
  const ol_ = $("peaks"); ol_.innerHTML = "";
  if (!state.vals) { ol_.innerHTML = '<li class="empty">Add an example to search.</li>'; return; }
  const [bw, bh] = state.text ? [1, 1] : state.win, region = bw > 1 || bh > 1;
  // region matches should not overlap much; point matches at least 250 m apart
  const spacing = Math.max(3, Math.round(250 / state.meta.token_m), Math.ceil(0.8 * Math.max(bw, bh)));
  // rank on raw scores: normalised values saturate at 1 and would tie
  const [lo, hi] = state.stretch;
  state.peaks = findPeaks(state.raw, state.meta.grid.width, state.meta.grid.height,
    { count: 10, minValue: lo + Math.max(0.5, state.threshold) * (hi - lo), spacing,
      exclude: state.text ? [] : state.pos.map(exampleCenter).filter(i => i >= 0) });
  if (!state.peaks.length) { ol_.innerHTML = '<li class="empty">No matches above threshold.</li>'; return; }
  state.peaks.forEach((p, r) => {
    const c = indexToCoord(p.index);
    const box = region ? windowPolygon(p.index, bw, bh) : null;
    if ($("showPeaks").checked) {
      const f = new ol.Feature(box || new ol.geom.Point(c)); f.set("rank", r + 1);
      if (box) f.set("center", c);
      peakSource.addFeature(f);
    }
    const li = document.createElement("li");
    li.innerHTML = `<span class="n">${r + 1}</span><span>${fmtLL(toLonLat(c))}</span><span>${state.raw[p.index].toFixed(3)}</span>`;
    li.onclick = () => box
      ? map.getView().fit(box.getExtent(), { padding: [120, 120, 120, window.innerWidth > 640 ? 460 : 120], maxZoom: 22, duration: 400 })
      : map.getView().animate({ center: c, zoom: Math.max(map.getView().getZoom(), 4), duration: 400 });
    ol_.appendChild(li);
  });
}

function update() {
  drawMarkers();
  writeHash();
  syncTextUI();
  if (state.text) return updateText();
  if (!state.pos.length && !state.neg.length) {
    state.vals = state.raw = null; drawHeat(); drawPeaks(); status(readyText()); return;
  }
  const t0 = performance.now();
  // each example counts once: its tokens are averaged into one unit vector
  const vec = ex => state.grid.meanVector(ex.indices);
  const qp = combine(state.pos.map(vec)), qn = combine(state.neg.map(vec));
  // window size = average size of the positive examples (1x1 for clicked points)
  const sized = state.pos.length ? state.pos : state.neg;
  const bw = Math.max(1, Math.round(sized.reduce((a, e) => a + e.bw, 0) / sized.length));
  const bh = Math.max(1, Math.round(sized.reduce((a, e) => a + e.bh, 0) / sized.length));
  state.win = [bw, bh];
  $("legTitle").textContent = bw > 1 || bh > 1
    ? `similarity of ${bw}×${bh}-token windows (${(bw * state.meta.token_m).toFixed(0)}×${(bh * state.meta.token_m).toFixed(0)} m) to the examples`
    : "similarity to examples";
  const raw = state.grid.windowScore(qp, qn, bw, bh);
  const [lo, hi] = quantiles(raw, [0.5, 0.9995]);
  const vals = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) vals[i] = Number.isNaN(raw[i]) ? NaN : Math.min(1, Math.max(0, (raw[i] - lo) / (hi - lo)));
  state.raw = raw; state.vals = vals; state.stretch = [lo, hi];
  drawHeat(); drawPeaks();
  status(`${readyText()} · searched in ${(performance.now() - t0).toFixed(0)} ms`);
  $("legLo").textContent = lo.toFixed(2); $("legHi").textContent = hi.toFixed(2);
}

const readyText = () => `${(state.grid.n / 1e6).toFixed(2)}M tokens · ${state.meta.token_m.toFixed(0)} m · ${state.meta.model.split(".").pop()}`;

function setView(v) {
  state.view = v;
  document.querySelectorAll(".seg button[data-view]").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  heatLayer.setVisible(v === "heat");
  pcaLayer.setVisible(v === "pca");
}

// ------------------------------------------------------------------ drawing tools
const TOOL_HINTS = {
  point: "<kbd>Click</kbd> a token to add a positive example · <kbd>Shift</kbd>+<kbd>Click</kbd> a negative",
  box: "<kbd>Click</kbd> two corners to draw a box (hold <kbd>Shift</kbd> on the second click for a negative). Finds places whose same-size window looks like the box as a whole.",
  polygon: "<kbd>Click</kbd> to add vertices, <kbd>Double-click</kbd> to finish (hold <kbd>Shift</kbd> to finish as a negative). Finds places whose window of the same size looks like the shape.",
};
function boxGeometry(coords, geom) {
  const [[x0, y0], [x1, y1]] = [coords[0], coords[coords.length - 1]];
  const ring = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]];
  if (geom) geom.setCoordinates(ring); else geom = new ol.geom.Polygon(ring);
  return geom;
}
function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll("#tools button").forEach(b => b.classList.toggle("on", b.dataset.tool === tool));
  $("toolHint").innerHTML = TOOL_HINTS[tool];
  if (drawInteraction) { map.removeInteraction(drawInteraction); drawInteraction = null; drawing = false; }
  if (tool === "point") return;
  drawInteraction = new ol.interaction.Draw({
    source: new ol.source.Vector(),
    type: tool === "box" ? "Circle" : "Polygon",
    geometryFunction: tool === "box" ? boxGeometry : undefined,
    freehandCondition: () => false,
    condition: e => !e.originalEvent.altKey,
    style: [new ol.style.Style({ stroke: new ol.style.Stroke({ color: "#000", width: 4 }) }),
            new ol.style.Style({ stroke: new ol.style.Stroke({ color: "#fff", width: 2, lineDash: [6, 4] }),
              fill: new ol.style.Fill({ color: "rgba(255,255,255,0.12)" }),
              image: new ol.style.Circle({ radius: 4, fill: new ol.style.Fill({ color: "#fff" }) }) })],
  });
  drawInteraction.on("drawstart", () => { drawing = true; });
  drawInteraction.on("drawabort", () => { drawing = false; });
  drawInteraction.on("drawend", e => {
    drawing = false;
    const ex = regionExample(e.feature.getGeometry());
    if (!ex) return;
    state.text = null;
    (shiftDown ? state.neg : state.pos).push(ex);
    update();
  });
  map.addInteraction(drawInteraction);
}

// ------------------------------------------------------------------ url state
function writeHash() {
  const ll = c => toLonLat(c).map(x => x.toFixed(5)).join(",");
  const enc = (ex, sign) => ex.ring ? sign.toUpperCase() + ex.ring.slice(0, -1).map(ll).join("_") : sign + ll(indexToCoord(ex.indices[0]));
  const parts = state.text ? ["t" + encodeURIComponent(state.text)]
    : [...state.pos.map(ex => enc(ex, "p")), ...state.neg.map(ex => enc(ex, "n"))];
  history.replaceState(null, "", parts.length ? "#" + parts.join(";") : location.pathname + location.search);
}
function readHash() {
  for (const p of location.hash.slice(1).split(";").filter(Boolean)) {
    if (p[0] === "t") { state.text = decodeURIComponent(p.slice(1)); continue; }
    if (p[0] === "P" || p[0] === "N") {      // region: lon,lat_lon,lat_...
      const ring = p.slice(1).split("_").map(s => fromLonLat(s.split(",").map(Number)));
      if (ring.length < 3 || ring.some(c => c.some(Number.isNaN))) continue;
      const ex = regionExample(new ol.geom.Polygon([[...ring, ring[0]]]));
      if (ex) (p[0] === "N" ? state.neg : state.pos).push(ex);
      continue;
    }
    const [lon, lat] = p.slice(1).split(",").map(Number);
    const i = coordToIndex(fromLonLat([lon, lat]));
    if (i >= 0) (p[0] === "n" ? state.neg : state.pos).push(pointExample(i));
  }
}

// ------------------------------------------------------------------ init
function drawLegend() {
  const c = $("legendCanvas"), ctx = c.getContext("2d"), img = ctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) { img.data.set([LUT[i * 3], LUT[i * 3 + 1], LUT[i * 3 + 2], 255], i * 4); }
  ctx.putImageData(img, 0, 0);
}

async function loadDataset(ds) {
  state.datasetPath = ds.path;
  state.meta = await (await fetch(`${ds.path}/meta.json`)).json();
  state.R = state.meta.mars_radius_m;
  const [codes, valid] = await Promise.all([
    fetchBytes(`${ds.path}/embeddings.u8`, "embeddings"),
    fetchBytes(`${ds.path}/valid.bits`, "mask"),
  ]);
  status("Preparing…");
  await new Promise(r => setTimeout(r));
  state.grid = new EmbeddingGrid(state.meta, codes, valid);
  if (ds.text) {
    state.vocab = await (await fetch(`${ds.path}/vocab.json`)).json();
    state.vocab.byText = new Map(state.vocab.phrases.map(p => [p.text.toLowerCase(), p]));
    const bg = state.vocab.background.map(t => state.vocab.byText.get(t.toLowerCase()));
    state.lse = backgroundLSE(state.grid, bg, state.vocab.logit_scale);
  }
}

async function main() {
  drawLegend();
  state.config = await (await fetch("config.json")).json();
  if (state.config.repo) $("repoLink").href = state.config.repo;
  const dsList = state.config.datasets;
  const params = new URLSearchParams(location.search);
  const ds = dsList.find(d => d.id === params.get("emb")) || dsList[0];
  if (dsList.length > 1) {
    $("datasetRow").hidden = false;
    for (const d of dsList) $("dataset").add(new Option(d.label, d.id, false, d === ds));
    $("dataset").onchange = e => { params.set("emb", e.target.value); location.search = params.toString(); };
  }
  await loadDataset(ds);
  buildMap();
  readHash();

  for (const p of state.config.presets || []) {
    const b = document.createElement("button");
    b.textContent = p.label; b.title = p.note || "";
    b.onclick = () => {
      const pts = list => (list || []).map(ll => coordToIndex(fromLonLat(ll))).filter(i => i >= 0).map(pointExample);
      const boxes = list => (list || []).map(([lon0, lat0, lon1, lat1]) =>
        regionExample(ol.geom.Polygon.fromExtent([...fromLonLat([lon0, lat0]), ...fromLonLat([lon1, lat1])]))).filter(Boolean);
      state.text = null;
      state.pos = [...pts(p.pos), ...boxes(p.pos_boxes)];
      state.neg = [...pts(p.neg), ...boxes(p.neg_boxes)];
      update();
      if (state.pos.length) map.getView().animate({ center: indexToCoord(exampleCenter(state.pos[0])), zoom: Math.max(map.getView().getZoom(), 3), duration: 500 });
    };
    $("presets").appendChild(b);
  }

  $("threshold").oninput = e => { state.threshold = +e.target.value; $("thrVal").value = state.threshold.toFixed(2); drawHeat(); drawPeaks(); };
  $("opacity").oninput = e => { const v = +e.target.value; $("opVal").value = v.toFixed(2); heatLayer.setOpacity(v); pcaLayer.setOpacity(v); };
  $("clear").onclick = () => { state.pos = []; state.neg = []; state.text = null; update(); };
  $("showPeaks").onchange = drawPeaks;
  $("showLanding").onchange = e => landingLayer.setVisible(e.target.checked);
  const setGrid = on => {
    gridLayer.setVisible(on); $("showGrid").checked = on;
    if (!on) hoverSource.clear();
    try { localStorage.setItem("mars-analog:grid", on ? "1" : "0"); } catch {}
  };
  $("showGrid").onchange = e => setGrid(e.target.checked);
  let gridPref = null;
  try { gridPref = localStorage.getItem("mars-analog:grid"); } catch {}
  setGrid(gridPref === "1");
  $("gridLabel").textContent = `Show token grid (${state.meta.token_m.toFixed(0)} m cells)`;
  document.querySelectorAll(".seg button[data-view]").forEach(b => b.onclick = () => setView(b.dataset.view));
  document.querySelectorAll("#tools button").forEach(b => b.onclick = () => setTool(b.dataset.tool));
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "Escape") {
      if (drawing) { drawInteraction.abortDrawing(); drawing = false; return; }
      state.pos = []; state.neg = []; state.text = null; update();
    }
    if (e.key === "p") setTool("point");
    if (e.key === "b") setTool("box");
    if (e.key === "o") setTool("polygon");
    if (e.key === "g") setGrid(!gridLayer.getVisible());
    if (e.key === " ") { e.preventDefault(); setView(state.view === "heat" ? "none" : "heat"); }
  });
  if (state.vocab) initTextUI();
  setTool("point");
  update();
}

// ------------------------------------------------------------------ text search
function initTextUI() {
  $("textCard").hidden = false;
  const v = state.vocab;
  for (const p of v.phrases) $("vocabList").appendChild(new Option(p.text));
  $("textHint").innerHTML = `${v.phrases.length} precomputed phrases (dino.txt) · <kbd>Alt</kbd>+<kbd>Click</kbd> the map to see which phrases match a spot`;
  for (const t of state.config.text_examples || []) {
    const b = document.createElement("button"); b.textContent = t; b.dataset.text = t;
    b.onclick = () => runText(t);
    $("textChips").appendChild(b);
  }
  $("textForm").onsubmit = e => { e.preventDefault(); runText($("textInput").value); };
  $("describeClose").onclick = () => { $("describe").hidden = true; };
}

function runText(t) {
  t = (t || "").trim();
  if (!t) return;
  const hit = state.vocab.byText.get(t.toLowerCase());
  if (!hit) {
    const near = state.vocab.phrases.filter(p => p.text.includes(t.toLowerCase().split(" ").pop())).slice(0, 4).map(p => `"${p.text}"`);
    $("textHint").innerHTML = `<span class="warn">"${t}" isn't in the precomputed vocabulary.</span> ${near.length ? "Try " + near.join(", ") + "." : "Pick a phrase from the list."}`;
    return;
  }
  state.text = hit.text; state.pos = []; state.neg = [];
  update();
}

function syncTextUI() {
  if (!state.vocab) return;
  $("textInput").value = state.text || "";
  document.querySelectorAll("#textChips button").forEach(b => b.classList.toggle("on", b.dataset.text === state.text));
}

function updateText() {
  const t0 = performance.now();
  const p = state.vocab.byText.get(state.text.toLowerCase());
  if (!p) { state.text = null; return update(); }
  const cos = state.grid.textCosine(p.q, p.b);
  const isBg = state.vocab.background.some(b => b.toLowerCase() === p.text.toLowerCase());
  const vals = new Float32Array(cos.length);
  if (isBg) {
    // relative score against itself is meaningless: stretch raw cosine instead
    const [lo, hi] = quantiles(cos, [0.5, 0.999]);
    for (let i = 0; i < cos.length; i++) vals[i] = Number.isNaN(cos[i]) ? NaN : Math.min(1, Math.max(0, (cos[i] - lo) / (hi - lo)));
    state.raw = cos; state.stretch = [lo, hi];
    $("legTitle").textContent = "cosine (background phrase)"; $("legLo").textContent = lo.toFixed(3); $("legHi").textContent = hi.toFixed(3);
  } else {
    const s = state.vocab.logit_scale, lse = state.lse;
    for (let i = 0; i < cos.length; i++) vals[i] = Number.isNaN(cos[i]) ? NaN : 1 / (1 + Math.exp(lse[i] - s * cos[i]));
    state.raw = vals; state.stretch = [0, 1];
    $("legTitle").textContent = "P(query vs background)"; $("legLo").textContent = "0"; $("legHi").textContent = "1";
  }
  state.vals = vals;
  drawHeat(); drawPeaks();
  status(`${readyText()} · "${p.text}" in ${(performance.now() - t0).toFixed(0)} ms`);
}

function describe(i) {
  const v = state.vocab, s = v.logit_scale;
  const cos = state.grid.textCosineAt(i, v.phrases);
  const m = Math.max(...cos), ex = cos.map(c => Math.exp(s * (c - m))), z = ex.reduce((a, b) => a + b, 0);
  const ranked = v.phrases.map((p, k) => ({ text: p.text, cos: cos[k], prob: ex[k] / z })).sort((a, b) => b.cos - a.cos).slice(0, 8);
  const ol_ = $("describeList"); ol_.innerHTML = "";
  for (const r of ranked) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${r.text}</span><span>${r.cos.toFixed(3)} · ${(100 * r.prob).toFixed(0)}%</span>`;
    li.onclick = () => runText(r.text);
    ol_.appendChild(li);
  }
  $("describe").hidden = false;
}

main().catch(err => { console.error(err); status(`Error: ${err.message}`); });
