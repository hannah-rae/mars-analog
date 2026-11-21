// Query-by-example similarity over quantised PCA'd DINOv3 tokens.
//
// Each token is stored as k uint8 codes; dequantised value is
//   x_j = vmin_j + code_j * step_j,   step_j = (vmax_j - vmin_j) / 255
// For a query direction q the cosine is
//   cos = (sum_j q_j x_j) / |x| = (bias + sum_j w_j code_j) / |x|
// with w_j = q_j * step_j and bias = sum_j q_j vmin_j. Positive and negative
// queries combine linearly (cos(x, qp) - cos(x, qn)), so one pass suffices.

export class EmbeddingGrid {
  constructor(meta, codes, validBits) {
    this.meta = meta;
    this.codes = codes;
    this.k = meta.pca.n_components;
    this.w = meta.grid.width;
    this.h = meta.grid.height;
    this.n = this.w * this.h;
    this.vmin = Float32Array.from(meta.pca.vmin);
    this.step = Float32Array.from(meta.pca.vmax.map((v, j) => (v - meta.pca.vmin[j]) / 255));
    this.valid = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) this.valid[i] = (validBits[i >> 3] >> (7 - (i & 7))) & 1;
    this.invNorm = new Float32Array(this.n);
    const { k, vmin, step } = this;
    for (let i = 0, b = 0; i < this.n; i++, b += k) {
      let s = 0;
      for (let j = 0; j < k; j++) { const x = vmin[j] + codes[b + j] * step[j]; s += x * x; }
      this.invNorm[i] = s > 0 ? 1 / Math.sqrt(s) : 0;
    }
  }

  /** unit-length dequantised vector for token i */
  vector(i) {
    const v = new Float32Array(this.k);
    const b = i * this.k;
    for (let j = 0; j < this.k; j++) v[j] = (this.vmin[j] + this.codes[b + j] * this.step[j]) * this.invNorm[i];
    return v;
  }

  /** normalised mean of the unit vectors of the given token indices */
  meanVector(indices) {
    if (!indices.length) return null;
    const m = new Float32Array(this.k);
    for (const i of indices) { const v = this.vector(i); for (let j = 0; j < this.k; j++) m[j] += v[j]; }
    let s = 0; for (let j = 0; j < this.k; j++) s += m[j] * m[j];
    s = 1 / Math.sqrt(s || 1);
    for (let j = 0; j < this.k; j++) m[j] *= s;
    return m;
  }

  /** score = cos(x, mean(pos)) - cos(x, mean(neg)); NaN where invalid */
  score(pos, neg) {
    const qp = this.meanVector(pos), qn = this.meanVector(neg);
    const { k, n, codes, invNorm, valid, vmin, step } = this;
    const w = new Float32Array(k);
    let bias = 0;
    for (let j = 0; j < k; j++) {
      const q = (qp ? qp[j] : 0) - (qn ? qn[j] : 0);
      w[j] = q * step[j];
      bias += q * vmin[j];
    }
    const out = new Float32Array(n);
    for (let i = 0, b = 0; i < n; i++, b += k) {
      if (!valid[i]) { out[i] = NaN; continue; }
      let s = bias;
      for (let j = 0; j < k; j++) s += w[j] * codes[b + j];
      out[i] = s * invNorm[i];
    }
    return out;
  }
}

/**
 * Text scoring. Tokens were L2-normalised before PCA, so for a unit text vector t
 *   cos(t, x) ~= t.mu + (C t) . p,   p = PCA coords = vmin + code * step
 * vocab.json stores q = C t and b = t.mu per phrase.
 */
EmbeddingGrid.prototype.textCosine = function (q, b, out = new Float32Array(this.n)) {
  const { k, n, codes, valid, vmin, step } = this;
  const w = new Float32Array(k);
  let bias = b;
  for (let j = 0; j < k; j++) { w[j] = q[j] * step[j]; bias += q[j] * vmin[j]; }
  for (let i = 0, o = 0; i < n; i++, o += k) {
    if (!valid[i]) { out[i] = NaN; continue; }
    let s = bias;
    for (let j = 0; j < k; j++) s += w[j] * codes[o + j];
    out[i] = s;
  }
  return out;
};

/** cosine of every vocab phrase at token i */
EmbeddingGrid.prototype.textCosineAt = function (i, phrases) {
  const o = i * this.k;
  return phrases.map(p => {
    let s = p.b;
    for (let j = 0; j < this.k; j++) s += p.q[j] * (this.vmin[j] + this.codes[o + j] * this.step[j]);
    return s;
  });
};

/** per-token log-sum-exp of scale * cos over background phrases */
export function backgroundLSE(grid, bgPhrases, scale) {
  const n = grid.n, mx = new Float32Array(n).fill(-Infinity), sum = new Float32Array(n);
  const cos = bgPhrases.map(p => grid.textCosine(p.q, p.b));
  for (const c of cos) for (let i = 0; i < n; i++) if (scale * c[i] > mx[i]) mx[i] = scale * c[i];
  for (const c of cos) for (let i = 0; i < n; i++) sum[i] += Math.exp(scale * c[i] - mx[i]);
  const lse = new Float32Array(n);
  for (let i = 0; i < n; i++) lse[i] = mx[i] + Math.log(sum[i]);
  return lse;
}

/** normalised sum of unit vectors (null if none) */
export function combine(vectors) {
  if (!vectors.length) return null;
  const m = new Float32Array(vectors[0].length);
  for (const v of vectors) for (let j = 0; j < m.length; j++) m[j] += v[j];
  let s = 0; for (let j = 0; j < m.length; j++) s += m[j] * m[j];
  s = 1 / Math.sqrt(s || 1);
  for (let j = 0; j < m.length; j++) m[j] *= s;
  return m;
}

/** a_i = u_i . d for every token (u_i = unit token vector); 0 where invalid */
EmbeddingGrid.prototype.dirDot = function (d, out = new Float32Array(this.n)) {
  const { k, n, codes, invNorm, valid, vmin, step } = this;
  const w = new Float32Array(k);
  let bias = 0;
  for (let j = 0; j < k; j++) { w[j] = d[j] * step[j]; bias += d[j] * vmin[j]; }
  for (let i = 0, b = 0; i < n; i++, b += k) {
    if (!valid[i]) { out[i] = 0; continue; }
    let s = bias;
    for (let j = 0; j < k; j++) s += w[j] * codes[b + j];
    out[i] = s * invNorm[i];
  }
  return out;
};

/**
 * Centred box sums of a W x H field over bw x bh windows, clipped at the edges.
 * Window for cell (r, c) spans rows r-top..r-top+bh-1 and cols c-left..c-left+bw-1,
 * with top = floor((bh-1)/2), left = floor((bw-1)/2).
 */
export function boxSum(src, W, H, bw, bh, out = new Float32Array(W * H), tmp = new Float32Array(W * H)) {
  const left = (bw - 1) >> 1, right = bw - 1 - left, top = (bh - 1) >> 1, bottom = bh - 1 - top;
  for (let r = 0; r < H; r++) {          // horizontal running sums
    const o = r * W;
    let s = 0;
    for (let c = 0; c <= Math.min(right, W - 1); c++) s += src[o + c];
    for (let c = 0; c < W; c++) {
      tmp[o + c] = s;
      const add = c + right + 1, sub = c - left;
      if (add < W) s += src[o + add];
      if (sub >= 0) s -= src[o + sub];
    }
  }
  const acc = new Float64Array(W);       // vertical running sums
  for (let r = 0; r <= Math.min(bottom, H - 1); r++) for (let c = 0; c < W; c++) acc[c] += tmp[r * W + c];
  for (let r = 0; r < H; r++) {
    const o = r * W, add = (r + bottom + 1) * W, sub = (r - top) * W;
    for (let c = 0; c < W; c++) {
      out[o + c] = acc[c];
      if (r + bottom + 1 < H) acc[c] += tmp[add + c];
      if (r - top >= 0) acc[c] -= tmp[sub + c];
    }
  }
  return out;
}

/** |sum of unit token vectors| over every bw x bh window (cached per size) */
EmbeddingGrid.prototype.windowNorms = function (bw, bh) {
  this._normCache ??= new Map();
  const key = `${bw}x${bh}`;
  if (this._normCache.has(key)) return this._normCache.get(key);
  const { k, n, w: W, h: H, codes, invNorm, valid, vmin, step } = this;
  const f = new Float32Array(n), s = new Float32Array(n), tmp = new Float32Array(n), sq = new Float32Array(n);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < n; i++) f[i] = valid[i] ? (vmin[j] + codes[i * k + j] * step[j]) * invNorm[i] : 0;
    boxSum(f, W, H, bw, bh, s, tmp);
    for (let i = 0; i < n; i++) sq[i] += s[i] * s[i];
  }
  for (let i = 0; i < n; i++) sq[i] = Math.sqrt(sq[i]);
  if (this._normCache.size > 4) this._normCache.clear();
  this._normCache.set(key, sq);
  return sq;
};

/**
 * Region score for every window position (value stored at the window's centre cell):
 *   cos(mean of window tokens, qp) - cos(mean of window tokens, qn)
 * = (sum_window u) . (qp - qn) / |sum_window u|
 * With bw = bh = 1 this is the per-token score.
 */
EmbeddingGrid.prototype.windowScore = function (qp, qn, bw, bh) {
  const d = new Float32Array(this.k);
  for (let j = 0; j < this.k; j++) d[j] = (qp ? qp[j] : 0) - (qn ? qn[j] : 0);
  const a = this.dirDot(d);
  const out = (bw === 1 && bh === 1) ? a : boxSum(a, this.w, this.h, bw, bh);
  const norms = (bw === 1 && bh === 1) ? null : this.windowNorms(bw, bh);
  for (let i = 0; i < this.n; i++) {
    if (!this.valid[i]) { out[i] = NaN; continue; }
    if (norms) out[i] = norms[i] > 0 ? out[i] / norms[i] : NaN;
  }
  return out;
};

/** approximate quantiles from a strided sample */
export function quantiles(arr, qs, sampleSize = 100000) {
  const stride = Math.max(1, Math.floor(arr.length / sampleSize));
  const s = [];
  for (let i = 0; i < arr.length; i += stride) if (!Number.isNaN(arr[i])) s.push(arr[i]);
  s.sort((a, b) => a - b);
  return qs.map(q => s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]);
}

/** greedy non-maximum suppression on a w x h grid of scores (NaN ignored) */
export function findPeaks(vals, w, h, { count = 10, minValue = 0.5, spacing = 8, exclude = [] } = {}) {
  const cand = [];
  for (let i = 0; i < vals.length; i++) if (vals[i] >= minValue) cand.push(i);
  cand.sort((a, b) => vals[b] - vals[a]);
  const taken = exclude.map(i => [i % w, Math.floor(i / w)]);
  const peaks = [];
  const s2 = spacing * spacing;
  for (const i of cand) {
    const x = i % w, y = Math.floor(i / w);
    if (taken.some(([tx, ty]) => (tx - x) ** 2 + (ty - y) ** 2 < s2)) continue;
    taken.push([x, y]);
    peaks.push({ index: i, x, y, value: vals[i] });
    if (peaks.length >= count) break;
  }
  return peaks;
}
