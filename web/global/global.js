import { EmbeddingGrid, quantiles, backgroundLSE } from "../search.js?v=5";

const ol = window.ol;
const $ = id => document.getElementById(id);
const EXTENT = [-180, -90, 180, 90];
const SRC = "https://github.com/hannah-rae/mars-analog";
const DINO = 'Features: Meta DINOv3 + dino.txt. Inspired by <a href="https://calebrob.com/dinosats/" target="_blank" rel="noopener">DINO-Sats</a>. ' +
  `<a href="${SRC}" target="_blank" rel="noopener">Source</a>`;

// approximate centres of well-known features (planetocentric, east longitude in -180..180)
const BODIES = {
  mars: {
    name: "Global Mars", data: "../data/themis",
    lede: "All of Mars in THEMIS daytime infrared, one DINOv3 embedding per ~22 km chip. Search by text, or click a chip to find others like it.",
    tiles: { url: "https://trek.nasa.gov/tiles/Mars/EQ/THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018/1.0.0/default/default028mm", ext: "png", levels: 10, extent: [-180, -65, 180, 65], credit: "THEMIS Day IR: NASA Mars Trek",
      // the same v12 mosaic the embeddings came from, whole globe but coarser: used beyond +-65 deg
      global: { url: "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12_clon0_ly/1.0.0/default/default028mm", ext: "png", levels: 9, extent: EXTENT } },
    credits: 'Imagery: THEMIS Day IR 100 m global mosaic v12 (NASA/JPL/ASU, USGS Astrogeology); zoomed-in tiles from <a href="https://trek.nasa.gov/mars/" target="_blank" rel="noopener">NASA Mars Trek</a>. ',
    examples: ["impact crater", "sand dunes", "volcano", "canyon", "layered rock", "smooth plains", "river channel", "ice"],
    other: '<a href="?body=moon">Search the Moon →</a>',
    places: [
      ["Olympus Mons", -133.8, 18.65], ["Ascraeus Mons", -104.1, 11.9], ["Pavonis Mons", -113.0, 1.5],
      ["Arsia Mons", -120.1, -8.3], ["Alba Mons", -109.6, 40.5], ["Valles Marineris", -59.2, -13.9],
      ["Noctis Labyrinthus", -101.6, -6.8], ["Hellas Planitia", 70.5, -42.4], ["Argyre Planitia", -44.0, -49.7],
      ["Utopia Planitia", 117.5, 46.7], ["Elysium Mons", 147.2, 25.0], ["Isidis Planitia", 87.0, 12.9],
      ["Syrtis Major", 69.5, 8.4], ["Arabia Terra", 5.0, 22.8], ["Meridiani Planum", -2.5, -0.2],
      ["Gale", 137.8, -5.4], ["Jezero", 77.6, 18.4], ["Gusev", 175.4, -14.5], ["Chryse Planitia", -40.3, 28.4],
      ["Acidalia Planitia", -22.0, 46.7], ["Planum Boreum", 15.0, 87.0], ["Planum Australe", 160.0, -84.0],
    ],
  },
  moon: {
    name: "Global Moon", data: "../data/wac",
    lede: "The whole Moon in the LROC WAC 100 m mosaic, one DINOv3 embedding per ~22 km chip. Search by text, or click a chip to find others like it.",
    tiles: { url: "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm", ext: "jpg", levels: 9, extent: EXTENT, credit: "LROC WAC: NASA Moon Trek" },
    credits: 'Imagery: LROC WAC global morphology mosaic 100 m (NASA/GSFC/ASU, USGS Astrogeology); zoomed-in tiles from <a href="https://trek.nasa.gov/moon/" target="_blank" rel="noopener">NASA Moon Trek</a>. ',
    examples: ["impact crater", "small impact craters", "fresh impact crater", "smooth plains", "mountains", "lava flow", "crater rim", "rugged terrain"],
    other: '<a href="?body=mars">Search Mars →</a>',
    places: [
      ["Tycho", -11.4, -43.3], ["Copernicus", -20.1, 9.6], ["Mare Imbrium", -15.6, 32.8], ["Mare Tranquillitatis", 31.4, 8.5],
      ["Mare Serenitatis", 17.5, 28.0], ["Oceanus Procellarum", -56.7, 20.7], ["Mare Crisium", 59.1, 17.0],
      ["Mare Orientale", -92.8, -19.4], ["South Pole–Aitken", -169.0, -53.0], ["Aristarchus", -47.4, 23.7],
      ["Plato", -9.4, 51.6], ["Clavius", -14.4, -58.4], ["Mare Moscoviense", 147.9, 27.3], ["Tsiolkovskiy", 129.1, -20.4],
      ["Apollo 11", 23.5, 0.7], ["Apollo 15", 3.6, 26.1], ["Apollo 17", 30.8, 20.2], ["Chang'e 4", 177.6, -45.4],
      ["Shackleton", 129.2, -89.7],
    ],
  },
};
const BODY = BODIES[new URLSearchParams(location.search).get("body")] || BODIES.mars;
const DATA = BODY.data;
const HEAT_W = 4096, HEAT_H = 2048;   // global heat raster, ~5 km per pixel at the equator


const INFERNO = ["#000004", "#160b39", "#420a68", "#6a176e", "#932667", "#bc3754",
  "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"];
const LUT = (() => {
  const stops = INFERNO.map(h => [1, 3, 5].map(o => parseInt(h.slice(o, o + 2), 16)));
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1), a = Math.floor(t), f = t - a, b = Math.min(a + 1, stops.length - 1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[a][c] * (1 - f) + stops[b][c] * f);
  }
  return lut;
})();

const state = { meta: null, L: null, grid: null, vocab: null, lse: null, pos: [], neg: [], text: null,
  raw: null, vals: null, stretch: [0, 1], threshold: 0.5, view: "heat", pixChip: null };
const status = t => { $("status").textContent = t; };
const fmtLL = ([lon, lat]) => `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
const wrap180 = lon => ((lon + 540) % 360) - 180;

// ------------------------------------------------------------------ chip layout
// bands: { r0, lat_top, lat_bottom, w, s, n, offset }; chip k of a band starts at x = k*s (source px), width w
function bandOfLat(lat) {
  const L = state.L;
  const rowc = (90 - lat) / 180 * L.height;                       // fractional source row
  const b = Math.round((rowc - L.chip / 2) / L.stride);
  return Math.min(L.bands.length - 1, Math.max(0, b));
}
function chipAt(lon, lat) {
  const L = state.L, band = L.bands[bandOfLat(lat)];
  const x = ((((lon - L.lon_of_x0) % 360) + 360) % 360) / 360 * L.width; // source column
  // nearest chip centre (k*s + w/2), measured around the wrap: the last chip of a band crosses 360 deg
  const k0 = Math.round((x - band.w / 2) / band.s), W = L.width;
  let best = 0, bestD = Infinity;
  for (let k = k0 - 1; k <= k0 + 1; k++) {
    const kk = ((k % band.n) + band.n) % band.n;
    const dx = Math.abs(x - (kk * band.s + band.w / 2)) % W, d = Math.min(dx, W - dx);
    if (d < bestD) { bestD = d; best = kk; }
  }
  return band.offset + best;
}
function bandIndexOfChip(i) {
  const bands = state.L.bands;
  let lo = 0, hi = bands.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (bands[m].offset <= i) lo = m; else hi = m - 1; }
  return lo;
}
function chipBounds(i) {   // [lon0, lat_bottom, lon1, lat_top]; lon0 in -180..180, lon1 may exceed 180
  const L = state.L, b = L.bands[bandIndexOfChip(i)], k = i - b.offset;
  const lon0 = wrap180(L.lon_of_x0 + (k * b.s) / L.width * 360);
  return [lon0, b.lat_bottom, lon0 + b.w / L.width * 360, b.lat_top];
}
const chipCenter = i => { const [a, b, c, d] = chipBounds(i); return [wrap180((a + c) / 2), (b + d) / 2]; };
const chipPolygon = i => ol.geom.Polygon.fromExtent(chipBounds(i));

function angDist([lon1, lat1], [lon2, lat2]) {   // great-circle angle in radians
  const r = Math.PI / 180, dlat = (lat2 - lat1) * r, dlon = (lon2 - lon1) * r;
  const a = Math.sin(dlat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dlon / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(a)));
}

// ------------------------------------------------------------------ data
async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get("content-length") || 0, reader = res.body.getReader(), chunks = [];
  let got = 0;
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

async function load() {
  state.meta = await (await fetch(`${DATA}/meta.json`)).json();
  state.L = state.meta.layout;
  const [codes, valid] = await Promise.all([fetchBytes(`${DATA}/embeddings.u8`, "embeddings"), fetchBytes(`${DATA}/valid.bits`, "mask")]);
  status("Preparing…"); await new Promise(r => setTimeout(r));
  state.grid = new EmbeddingGrid(state.meta, codes, valid);
  state.vocab = await (await fetch(`${DATA}/vocab.json`)).json();
  state.vocab.byText = new Map(state.vocab.phrases.map(p => [p.text.toLowerCase(), p]));
  state.lse = backgroundLSE(state.grid, state.vocab.background.map(t => state.vocab.byText.get(t.toLowerCase())), state.vocab.logit_scale);
  // which chip each heat-raster pixel shows (nearest chip centre in its band)
  state.pixChip = new Int32Array(HEAT_W * HEAT_H);
  for (let y = 0; y < HEAT_H; y++) {
    const lat = 90 - (y + 0.5) * 180 / HEAT_H;
    for (let x = 0; x < HEAT_W; x++) state.pixChip[y * HEAT_W + x] = chipAt(-180 + (x + 0.5) * 360 / HEAT_W, lat);
  }
}

// ------------------------------------------------------------------ map
let map, heatSource, heatLayer, markerSource, peakSource, hoverSource, namesLayer;
const heatCanvas = document.createElement("canvas");

function buildMap() {
  const mPerDeg = 2 * Math.PI * state.L.mars_radius_m / 360;
  const proj = new ol.proj.Projection({ code: "MARS-LONLAT", units: "degrees", extent: EXTENT, global: true,
    metersPerUnit: mPerDeg, getPointResolution: res => res });   // degrees at the equator; OL converts with metersPerUnit
  const overview = new ol.layer.Image({ source: new ol.source.ImageStatic({ url: `${DATA}/overview.jpg`, imageExtent: EXTENT, projection: proj }) });
  const T = BODY.tiles;
  const trekLayer = (L, credit) => new ol.layer.Tile({
    extent: L.extent, minZoom: 3,
    source: new ol.source.XYZ({
      projection: proj, crossOrigin: "anonymous", wrapX: false,
      tileGrid: new ol.tilegrid.TileGrid({ extent: EXTENT, origin: [-180, 90], tileSize: 256,
        resolutions: Array.from({ length: L.levels }, (_, z) => 180 / 256 / 2 ** z) }),
      tileUrlFunction: ([z, x, y]) => `${L.url}/${z}/${y}/${x}.${L.ext}`,
      attributions: credit,
    }),
  });
  const trek = trekLayer(T, T.credit);
  const trekGlobal = T.global ? trekLayer(T.global) : null;
  heatCanvas.width = HEAT_W; heatCanvas.height = HEAT_H;
  heatSource = new ol.source.ImageCanvas({
    projection: proj, ratio: 1,
    canvasFunction: (extent, resolution, pixelRatio, size) => {
      const c = document.createElement("canvas"); c.width = size[0]; c.height = size[1];
      const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = false;
      const sx = size[0] / (extent[2] - extent[0]), sy = size[1] / (extent[3] - extent[1]);
      ctx.drawImage(heatCanvas, (EXTENT[0] - extent[0]) * sx, (extent[3] - EXTENT[3]) * sy, 360 * sx, 180 * sy);
      return c;
    },
  });
  heatLayer = new ol.layer.Image({ source: heatSource, opacity: +$("opacity").value });

  markerSource = new ol.source.Vector({ wrapX: false });
  peakSource = new ol.source.Vector({ wrapX: false });
  hoverSource = new ol.source.Vector({ wrapX: false });
  const outline = (color, width, dash) => [
    new ol.style.Style({ stroke: new ol.style.Stroke({ color: "rgba(0,0,0,0.7)", width: width + 2 }) }),
    new ol.style.Style({ stroke: new ol.style.Stroke({ color, width, lineDash: dash }) }),
  ];
  const markerStyle = f => [...outline(f.get("neg") ? "#ef5d5d" : "#7fd18b", 2),
    new ol.style.Style({ fill: new ol.style.Fill({ color: f.get("neg") ? "rgba(239,93,93,0.2)" : "rgba(127,209,139,0.2)" }) })];
  const peakStyle = f => [...outline("#e37148", 2, [5, 3]), new ol.style.Style({
    geometry: new ol.geom.Point(f.get("center")),
    image: new ol.style.Circle({ radius: 10, fill: new ol.style.Fill({ color: "rgba(20,14,12,0.85)" }), stroke: new ol.style.Stroke({ color: "#e37148", width: 2 }) }),
    text: new ol.style.Text({ text: String(f.get("rank")), font: "600 11px JetBrains Mono, monospace", fill: new ol.style.Fill({ color: "#fff" }) }),
  })];
  namesLayer = new ol.layer.Vector({
    source: new ol.source.Vector({ wrapX: false, features: BODY.places.map(([name, lon, lat]) => { const f = new ol.Feature(new ol.geom.Point([lon, lat])); f.set("name", name); return f; }) }),
    style: f => new ol.style.Style({ text: new ol.style.Text({ text: f.get("name"), font: "500 11px Inter, sans-serif",
      fill: new ol.style.Fill({ color: "rgba(255,255,255,0.85)" }), stroke: new ol.style.Stroke({ color: "rgba(0,0,0,0.8)", width: 3 }) }) }),
    declutter: true,
  });

  map = new ol.Map({
    target: "map",
    layers: [overview, ...(trekGlobal ? [trekGlobal] : []), trek, heatLayer, namesLayer,
      new ol.layer.Vector({ source: hoverSource, style: outline("#fff", 1.5) }),
      new ol.layer.Vector({ source: peakSource, style: peakStyle }),
      new ol.layer.Vector({ source: markerSource, style: markerStyle })],
    view: new ol.View({ projection: proj, extent: EXTENT, constrainOnlyCenter: true, center: [0, 0], zoom: 1, maxZoom: T.levels + 1 }),
    controls: ol.control.defaults.defaults({ rotate: false, attributionOptions: { collapsible: true } }).extend([new ol.control.ScaleLine()]),
  });
  map.getView().fit(EXTENT, { padding: [20, 20, 20, window.innerWidth > 640 ? 360 : 20] });

  map.on("singleclick", e => {
    const [lon, lat] = e.coordinate;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
    const i = chipAt(lon, lat);
    if (!state.grid.valid[i]) return;
    if (e.originalEvent.altKey) { describe(i); return; }
    state.text = null;
    (e.originalEvent.shiftKey ? state.neg : state.pos).push(i);
    update();
  });
  map.on("pointermove", e => {
    if (e.dragging) return;
    const [lon, lat] = e.coordinate;
    hoverSource.clear();
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) { $("readout").textContent = ""; return; }
    const i = chipAt(lon, lat);
    if ($("showHover").checked) hoverSource.addFeature(new ol.Feature(chipPolygon(i)));
    let t = `${fmtLL([lon, lat])}  chip ${i}`;
    if (state.raw && !Number.isNaN(state.raw[i])) t += `  score ${state.raw[i].toFixed(3)}`;
    $("readout").textContent = t;
  });
}

// ------------------------------------------------------------------ render
function drawHeat() {
  const ctx = heatCanvas.getContext("2d"), img = ctx.createImageData(HEAT_W, HEAT_H), d = img.data;
  const v = state.vals, thr = state.threshold, pc = state.pixChip;
  if (v) for (let p = 0; p < pc.length; p++) {
    const x = v[pc[p]];
    if (!(x >= thr)) continue;
    const li = Math.round(x * 255) * 3, o = p * 4;
    d[o] = LUT[li]; d[o + 1] = LUT[li + 1]; d[o + 2] = LUT[li + 2];
    d[o + 3] = 255 * Math.min(1, 0.35 + 0.65 * (x - thr) / Math.max(1e-6, 1 - thr));
  }
  ctx.putImageData(img, 0, 0);
  heatSource.changed();
}

function drawMarkers() {
  markerSource.clear();
  const ul = $("examples"); ul.innerHTML = "";
  for (const [list, neg] of [[state.pos, false], [state.neg, true]]) list.forEach((i, k) => {
    const f = new ol.Feature(chipPolygon(i)); f.set("neg", neg); markerSource.addFeature(f);
    const li = document.createElement("li"); li.className = neg ? "neg" : "pos";
    li.textContent = `${neg ? "−" : "+"} ${fmtLL(chipCenter(i))}`; li.title = "Remove";
    li.onclick = () => { list.splice(k, 1); update(); };
    ul.appendChild(li);
  });
}

function drawPeaks() {
  peakSource.clear();
  const ol_ = $("peaks"); ol_.innerHTML = "";
  if (!state.raw) { ol_.innerHTML = '<li class="empty">Search to see matches.</li>'; renderPreview([]); return; }
  const [lo, hi] = state.stretch, minRaw = lo + Math.max(0.5, state.threshold) * (hi - lo);
  const cand = [];
  for (let i = 0; i < state.raw.length; i++) if (state.raw[i] >= minRaw) cand.push(i);
  cand.sort((a, b) => state.raw[b] - state.raw[a]);
  const minSep = 1.5 * state.meta.chip_km * 1000 / state.L.mars_radius_m;   // radians
  const taken = state.text ? [] : state.pos.map(chipCenter), peaks = [];
  for (const i of cand) {
    const c = chipCenter(i);
    if (taken.some(t => angDist(t, c) < minSep)) continue;
    taken.push(c); peaks.push(i);
    if (peaks.length >= 20) break;
  }
  renderPreview(peaks.slice(0, 10));
  if (!peaks.length) { ol_.innerHTML = '<li class="empty">No matches above threshold.</li>'; return; }
  peaks.forEach((i, r) => {
    const c = chipCenter(i);
    if ($("showPeaks").checked) { const f = new ol.Feature(chipPolygon(i)); f.set("rank", r + 1); f.set("center", c); peakSource.addFeature(f); }
    const li = document.createElement("li");
    li.innerHTML = `<span class="n">${r + 1}</span><span>${fmtLL(c)}</span><span>${state.raw[i].toFixed(3)}</span>`;
    li.onclick = () => zoomToChip(i);
    ol_.appendChild(li);
  });
}

function zoomToChip(i) {
  map.getView().fit(chipPolygon(i).getExtent(), { padding: [150, 150, 150, window.innerWidth > 640 ? 500 : 150], duration: 400, maxZoom: 7 });
}

// ------------------------------------------------------------------ preview thumbnails
// Each thumbnail is the chip's lon/lat box resampled to a square, i.e. what the model saw (224 x 224 px).
// Drawn from the overview basemap at once, then from Trek tiles where they exist (CORS-enabled).
const tileCache = new Map();
let overviewImg = null, previewKey = "";
function loadImage(url, cors) {
  return new Promise(res => {
    const img = new Image(); if (cors) img.crossOrigin = "anonymous";
    img.onload = () => res(img); img.onerror = () => res(null); img.src = url;
  });
}
function tileImage(T, z, x, y) {
  const url = `${T.url}/${z}/${y}/${x}.${T.ext}`;
  if (!tileCache.has(url)) tileCache.set(url, loadImage(url, true));
  return tileCache.get(url);
}
function drawOverviewCrop(ctx, img, [lon0, lat0, lon1, lat1], size) {
  const ppd = img.naturalWidth / 360, sx = size / (lon1 - lon0), sy = size / (lat1 - lat0);
  for (const shift of [0, -360]) {      // a chip may cross +180: draw its wrapped part too
    const a = lon0 + shift, b = Math.min(lon1 + shift, 180);
    if (b <= -180 || a >= 180) continue;
    const ca = Math.max(a, -180);
    ctx.drawImage(img, (ca + 180) * ppd, (90 - lat1) * ppd, (b - ca) * ppd, (lat1 - lat0) * ppd,
      (ca - a) * sx, 0, (b - ca) * sx, size);
  }
}
async function drawChipThumb(canvas, i) {
  const size = canvas.width, ctx = canvas.getContext("2d"), bb = chipBounds(i), [lon0, lat0, lon1, lat1] = bb;
  ctx.fillStyle = "#111"; ctx.fillRect(0, 0, size, size);
  overviewImg ??= loadImage(`${DATA}/overview.jpg`, false);
  const ov = await overviewImg;
  if (ov) drawOverviewCrop(ctx, ov, bb, size);
  const inside = L => L && lat0 >= L.extent[1] && lat1 <= L.extent[3];
  const T = inside(BODY.tiles) ? BODY.tiles : inside(BODY.tiles.global) ? BODY.tiles.global : null;
  if (!T) return;                                            // no tiles here: overview only
  // finest level with at least `size` px across the chip height, and no more than 16 tiles
  let z = Math.min(T.levels - 1, Math.max(0, Math.ceil(Math.log2(180 * size / (256 * (lat1 - lat0))))));
  const span = z => 180 / 2 ** z, count = z => (Math.floor((lon1 + 180) / span(z)) - Math.floor((lon0 + 180) / span(z)) + 1) *
    (Math.floor((90 - lat0) / span(z)) - Math.floor((90 - lat1) / span(z)) + 1);
  while (z > 0 && count(z) > 16) z--;
  const d = span(z), cols = 2 ** (z + 1), sx = size / (lon1 - lon0), sy = size / (lat1 - lat0);
  const x0 = Math.floor((lon0 + 180) / d), x1 = Math.floor((lon1 + 180 - 1e-9) / d);
  const y0 = Math.floor((90 - lat1) / d), y1 = Math.floor((90 - lat0 - 1e-9) / d);
  const jobs = [];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    jobs.push(tileImage(T, z, ((x % cols) + cols) % cols, y).then(img => {
      if (!img) return;
      const tl = x * d - 180, tt = 90 - y * d;                 // tile's top-left lon/lat (unwrapped)
      ctx.drawImage(img, (tl - lon0) * sx, (lat1 - tt) * sy, d * sx, d * sy);
    }));
  }
  await Promise.all(jobs);
}
function renderPreview(peaks) {
  const box = $("preview");
  if (!box) return;
  const query = state.text ? "t:" + state.text : "p:" + state.pos.join(",");
  const key = query + "|" + peaks.join(",");
  if (key === previewKey) return;          // threshold slider moves often redraw the same list
  previewKey = key;
  box.hidden = !peaks.length;
  if (!peaks.length) return;
  const grid = $("previewGrid"); grid.innerHTML = "";
  const q = document.createElement("div"); q.className = "thumb query";
  if (state.text) {
    q.innerHTML = `<div class="qtext"><span class="small">text</span><b></b></div>`;
    q.querySelector("b").textContent = `“${state.text}”`;
  } else if (state.pos.length) {
    const c = document.createElement("canvas"); c.width = c.height = 224; q.appendChild(c);
    q.insertAdjacentHTML("beforeend", `<span class="tag">query${state.pos.length > 1 ? ` +${state.pos.length - 1}` : ""}</span>`);
    q.title = fmtLL(chipCenter(state.pos[0])); q.onclick = () => zoomToChip(state.pos[0]);
    drawChipThumb(c, state.pos[0]);
  } else {
    q.innerHTML = `<div class="qtext"><span class="small">negatives only</span></div>`;
  }
  grid.appendChild(q);
  peaks.forEach((i, r) => {
    const t = document.createElement("button"); t.className = "thumb"; t.title = `#${r + 1} · ${fmtLL(chipCenter(i))} · score ${state.raw[i].toFixed(3)}`;
    const c = document.createElement("canvas"); c.width = c.height = 112; t.appendChild(c);
    t.insertAdjacentHTML("beforeend", `<span class="rank">${r + 1}</span><span class="score">${state.raw[i].toFixed(2)}</span>`);
    t.onclick = () => zoomToChip(i);
    t.onmouseenter = () => { hoverSource.clear(); hoverSource.addFeature(new ol.Feature(chipPolygon(i))); };
    t.onmouseleave = () => hoverSource.clear();
    grid.appendChild(t);
    drawChipThumb(c, i);
  });
}

function setLegend(title, lo, hi) { $("legTitle").textContent = title; $("legLo").textContent = lo; $("legHi").textContent = hi; }

function update() {
  drawMarkers(); writeHash(); syncTextUI();
  const t0 = performance.now();
  if (state.text) {
    const p = state.vocab.byText.get(state.text.toLowerCase());
    const cos = state.grid.textCosine(p.q, p.b);
    const isBg = state.vocab.background.some(b => b.toLowerCase() === p.text.toLowerCase());
    const vals = new Float32Array(cos.length);
    if (isBg) {
      const [lo, hi] = quantiles(cos, [0.5, 0.999]);
      for (let i = 0; i < cos.length; i++) vals[i] = Number.isNaN(cos[i]) ? NaN : Math.min(1, Math.max(0, (cos[i] - lo) / (hi - lo)));
      state.raw = cos; state.stretch = [lo, hi]; setLegend("cosine (background phrase)", lo.toFixed(3), hi.toFixed(3));
    } else {
      const s = state.vocab.logit_scale, lse = state.lse;
      const P = new Float32Array(cos.length);
      for (let i = 0; i < cos.length; i++) P[i] = Number.isNaN(cos[i]) ? NaN : 1 / (1 + Math.exp(lse[i] - s * cos[i]));
      // P saturates over most of a planet, so stretch it between quantiles for contrast
      const [lo, hi] = quantiles(P, [0.5, 0.999]), span = Math.max(1e-6, hi - lo);
      for (let i = 0; i < P.length; i++) vals[i] = Number.isNaN(P[i]) ? NaN : Math.min(1, Math.max(0, (P[i] - lo) / span));
      state.raw = P; state.stretch = [lo, hi]; setLegend("P(query vs background)", lo.toFixed(3), hi.toFixed(3));
    }
    state.vals = vals;
  } else if (state.pos.length || state.neg.length) {
    const raw = state.grid.score(state.pos, state.neg);
    const [lo, hi] = quantiles(raw, [0.5, 0.9995]);
    const vals = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) vals[i] = Number.isNaN(raw[i]) ? NaN : Math.min(1, Math.max(0, (raw[i] - lo) / (hi - lo)));
    state.raw = raw; state.vals = vals; state.stretch = [lo, hi];
    setLegend("similarity to example chips", lo.toFixed(2), hi.toFixed(2));
  } else {
    state.raw = state.vals = null;
  }
  drawHeat(); drawPeaks();
  const n = `${(state.grid.n / 1e3).toFixed(0)}k chips · ${state.meta.chip_km.toFixed(1)} km`;
  status(state.raw ? `${n} · searched in ${(performance.now() - t0).toFixed(0)} ms` : n);
}

// ------------------------------------------------------------------ text
function initText() {
  $("textCard").hidden = false;
  const v = state.vocab;
  for (const p of v.phrases) $("vocabList").appendChild(new Option(p.text));
  $("textHint").innerHTML = `${v.phrases.length} precomputed phrases (dino.txt, image-level) · <kbd>Alt</kbd>+<kbd>Click</kbd> a chip to see which phrases match it`;
  for (const t of BODY.examples) {
    if (!v.byText.has(t)) continue;
    const b = document.createElement("button"); b.textContent = t; b.dataset.text = t; b.onclick = () => runText(t);
    $("textChips").appendChild(b);
  }
  $("textForm").onsubmit = e => { e.preventDefault(); runText($("textInput").value); };
  $("describeClose").onclick = () => { $("describe").hidden = true; };
}
function runText(t) {
  t = (t || "").trim(); if (!t) return;
  const hit = state.vocab.byText.get(t.toLowerCase());
  if (!hit) { $("textHint").innerHTML = `<span class="warn">"${t}" isn't in the precomputed vocabulary.</span> Pick a phrase from the list.`; return; }
  state.text = hit.text; state.pos = []; state.neg = []; update();
}
function syncTextUI() {
  $("textInput").value = state.text || "";
  document.querySelectorAll("#textChips button").forEach(b => b.classList.toggle("on", b.dataset.text === state.text));
}
function describe(i) {
  const v = state.vocab, s = v.logit_scale, cos = state.grid.textCosineAt(i, v.phrases);
  const m = Math.max(...cos), ex = cos.map(c => Math.exp(s * (c - m))), z = ex.reduce((a, b) => a + b, 0);
  const ranked = v.phrases.map((p, k) => ({ text: p.text, cos: cos[k], prob: ex[k] / z })).sort((a, b) => b.cos - a.cos).slice(0, 8);
  const ol_ = $("describeList"); ol_.innerHTML = "";
  for (const r of ranked) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${r.text}</span><span>${r.cos.toFixed(3)} · ${(100 * r.prob).toFixed(0)}%</span>`;
    li.onclick = () => runText(r.text); ol_.appendChild(li);
  }
  $("describe").hidden = false;
}

// ------------------------------------------------------------------ url state
function writeHash() {
  const ll = i => chipCenter(i).map(x => x.toFixed(3)).join(",");
  const parts = state.text ? ["t" + encodeURIComponent(state.text)] : [...state.pos.map(i => "p" + ll(i)), ...state.neg.map(i => "n" + ll(i))];
  history.replaceState(null, "", parts.length ? "#" + parts.join(";") : location.pathname + location.search);
}
function readHash() {
  for (const p of location.hash.slice(1).split(";").filter(Boolean)) {
    if (p[0] === "t") {
      let t = null; try { t = decodeURIComponent(p.slice(1)); } catch { continue; }
      const hit = state.vocab.byText.get(t.toLowerCase()); if (hit) state.text = hit.text; continue;
    }
    if (p[0] !== "p" && p[0] !== "n") continue;
    const [lon, lat] = p.slice(1).split(",").map(Number);
    if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lat) > 90) continue;
    const i = chipAt(lon, lat);
    if (state.grid.valid[i]) (p[0] === "n" ? state.neg : state.pos).push(i);
  }
}

// ------------------------------------------------------------------ init
async function main() {
  document.title = `Mars Analog · ${BODY.name}`;
  $("bodyName").textContent = BODY.name; $("lede").textContent = BODY.lede;
  $("credits").innerHTML = BODY.credits + DINO;
  $("navLinks").insertAdjacentHTML("beforeend", ` · ${BODY.other}`);
  const c = $("legendCanvas"), lctx = c.getContext("2d"), limg = lctx.createImageData(256, 1);
  for (let i = 0; i < 256; i++) limg.data.set([LUT[i * 3], LUT[i * 3 + 1], LUT[i * 3 + 2], 255], i * 4);
  lctx.putImageData(limg, 0, 0);
  await load();
  buildMap(); initText(); readHash();
  $("threshold").oninput = e => { state.threshold = +e.target.value; $("thrVal").value = state.threshold.toFixed(2); drawHeat(); drawPeaks(); };
  $("opacity").oninput = e => { const v = +e.target.value; $("opVal").value = v.toFixed(2); heatLayer.setOpacity(v); };
  $("clear").onclick = () => { state.pos = []; state.neg = []; state.text = null; update(); };
  $("showPeaks").onchange = drawPeaks;
  $("showHover").onchange = () => hoverSource.clear();
  const setCollapsed = v => { $("preview").classList.toggle("collapsed", v); $("previewToggle").textContent = v ? "+" : "–";
    $("previewToggle").title = v ? "Show previews" : "Hide previews"; try { localStorage.setItem("previewCollapsed", v ? "1" : ""); } catch {} };
  let collapsed = false; try { collapsed = !!localStorage.getItem("previewCollapsed"); } catch {}
  setCollapsed(collapsed);
  $("previewToggle").onclick = () => setCollapsed(!$("preview").classList.contains("collapsed"));
  $("showNames").onchange = e => namesLayer.setVisible(e.target.checked);
  document.querySelectorAll(".seg button[data-view]").forEach(b => b.onclick = () => {
    state.view = b.dataset.view;
    document.querySelectorAll(".seg button[data-view]").forEach(x => x.classList.toggle("on", x === b));
    heatLayer.setVisible(state.view === "heat");
  });
  document.addEventListener("keydown", e => {
    if (["INPUT", "BUTTON", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.key === "Escape") { state.pos = []; state.neg = []; state.text = null; update(); }
    if (e.key === " ") { e.preventDefault(); $$view(); }
  });
  update();
}
function $$view() { const b = document.querySelector(`.seg button[data-view="${state.view === "heat" ? "none" : "heat"}"]`); b && b.click(); }

main().catch(err => { console.error(err); status(`Error: ${err.message}`); });
