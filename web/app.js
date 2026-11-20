import { EmbeddingGrid, quantiles, findPeaks } from "./search.js";

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
  pos: [], neg: [],            // token indices
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
function indexToCoord(i) {
  const m = state.meta, [left, , , top] = m.bounds_m;
  const c = i % m.grid.width, r = Math.floor(i / m.grid.width);
  return [left + (c + 0.5) * m.token_m, top - (r + 0.5) * m.token_m];
}

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
let map, proj, heatSource, heatLayer, pcaLayer, markerSource, peakSource, landingLayer;
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
  const markerStyle = f => new ol.style.Style({
    image: new ol.style.Circle({ radius: 6, fill: new ol.style.Fill({ color: f.get("neg") ? "#ef5d5d" : "#7fd18b" }),
      stroke: new ol.style.Stroke({ color: "#000", width: 2 }) }),
  });
  const peakStyle = f => new ol.style.Style({
    image: new ol.style.Circle({ radius: 10, fill: new ol.style.Fill({ color: "rgba(20,14,12,0.8)" }),
      stroke: new ol.style.Stroke({ color: "#e37148", width: 2 }) }),
    text: new ol.style.Text({ text: String(f.get("rank")), font: "600 11px JetBrains Mono, monospace",
      fill: new ol.style.Fill({ color: "#fff" }) }),
  });

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
    layers: [base, heatLayer, pcaLayer, landingLayer,
      new ol.layer.Vector({ source: peakSource, style: peakStyle }),
      new ol.layer.Vector({ source: markerSource, style: markerStyle })],
    view: new ol.View({ projection: proj, extent: ext, constrainOnlyCenter: true, maxZoom: 22 }),
    controls: ol.control.defaults.defaults({ attribution: false, rotate: false }).extend([new ol.control.ScaleLine()]),
  });
  map.getView().fit(ext, { padding: [20, 20, 20, window.innerWidth > 640 ? 360 : 20] });

  map.on("singleclick", e => {
    const i = coordToIndex(e.coordinate);
    if (i < 0 || !state.grid.valid[i]) return;
    (e.originalEvent.shiftKey ? state.neg : state.pos).push(i);
    update();
  });
  map.on("pointermove", e => {
    if (e.dragging) return;
    const i = coordToIndex(e.coordinate);
    let t = fmtLL(toLonLat(e.coordinate));
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
    for (const i of list) { const f = new ol.Feature(new ol.geom.Point(indexToCoord(i))); f.set("neg", neg); markerSource.addFeature(f); }
  const ul = $("examples"); ul.innerHTML = "";
  for (const [list, cls] of [[state.pos, "pos"], [state.neg, "neg"]])
    list.forEach((i, k) => {
      const li = document.createElement("li"); li.className = cls;
      li.textContent = `${cls === "pos" ? "+" : "−"} ${fmtLL(toLonLat(indexToCoord(i)))}`;
      li.title = "Remove";
      li.onclick = () => { list.splice(k, 1); update(); };
      ul.appendChild(li);
    });
}

function drawPeaks() {
  peakSource.clear();
  const ol_ = $("peaks"); ol_.innerHTML = "";
  if (!state.vals) { ol_.innerHTML = '<li class="empty">Add an example to search.</li>'; return; }
  const spacing = Math.max(3, Math.round(250 / state.meta.token_m));
  // rank on raw scores: normalised values saturate at 1 and would tie
  const [lo, hi] = state.stretch;
  state.peaks = findPeaks(state.raw, state.meta.grid.width, state.meta.grid.height,
    { count: 10, minValue: lo + Math.max(0.5, state.threshold) * (hi - lo), spacing, exclude: state.pos });
  if (!state.peaks.length) { ol_.innerHTML = '<li class="empty">No matches above threshold.</li>'; return; }
  state.peaks.forEach((p, r) => {
    const c = indexToCoord(p.index);
    if ($("showPeaks").checked) { const f = new ol.Feature(new ol.geom.Point(c)); f.set("rank", r + 1); peakSource.addFeature(f); }
    const li = document.createElement("li");
    li.innerHTML = `<span class="n">${r + 1}</span><span>${fmtLL(toLonLat(c))}</span><span>${state.raw[p.index].toFixed(3)}</span>`;
    li.onclick = () => map.getView().animate({ center: c, zoom: Math.max(map.getView().getZoom(), 4), duration: 400 });
    ol_.appendChild(li);
  });
}

function update() {
  drawMarkers();
  writeHash();
  if (!state.pos.length && !state.neg.length) {
    state.vals = state.raw = null; drawHeat(); drawPeaks(); status(readyText()); return;
  }
  const t0 = performance.now();
  const raw = state.grid.score(state.pos, state.neg);
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
  document.querySelectorAll(".seg button").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  heatLayer.setVisible(v === "heat");
  pcaLayer.setVisible(v === "pca");
}

// ------------------------------------------------------------------ url state
function writeHash() {
  const ll = i => toLonLat(indexToCoord(i)).map(x => x.toFixed(5)).join(",");
  const parts = [...state.pos.map(i => "p" + ll(i)), ...state.neg.map(i => "n" + ll(i))];
  history.replaceState(null, "", parts.length ? "#" + parts.join(";") : location.pathname + location.search);
}
function readHash() {
  for (const p of decodeURIComponent(location.hash.slice(1)).split(";").filter(Boolean)) {
    const [lon, lat] = p.slice(1).split(",").map(Number);
    const i = coordToIndex(fromLonLat([lon, lat]));
    if (i >= 0) (p[0] === "n" ? state.neg : state.pos).push(i);
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
      state.pos = (p.pos || []).map(ll => coordToIndex(fromLonLat(ll))).filter(i => i >= 0);
      state.neg = (p.neg || []).map(ll => coordToIndex(fromLonLat(ll))).filter(i => i >= 0);
      update();
      map.getView().animate({ center: indexToCoord(state.pos[0]), zoom: Math.max(map.getView().getZoom(), 3), duration: 500 });
    };
    $("presets").appendChild(b);
  }

  $("threshold").oninput = e => { state.threshold = +e.target.value; $("thrVal").value = state.threshold.toFixed(2); drawHeat(); drawPeaks(); };
  $("opacity").oninput = e => { const v = +e.target.value; $("opVal").value = v.toFixed(2); heatLayer.setOpacity(v); pcaLayer.setOpacity(v); };
  $("clear").onclick = () => { state.pos = []; state.neg = []; update(); };
  $("showPeaks").onchange = drawPeaks;
  $("showLanding").onchange = e => landingLayer.setVisible(e.target.checked);
  document.querySelectorAll(".seg button").forEach(b => b.onclick = () => setView(b.dataset.view));
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "Escape") { state.pos = []; state.neg = []; update(); }
    if (e.key === " ") { e.preventDefault(); setView(state.view === "heat" ? "none" : "heat"); }
  });
  update();
}

main().catch(err => { console.error(err); status(`Error: ${err.message}`); });
