// Query-by-example and text similarity over quantised PCA'd DINOv3 chip embeddings.
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

  /** unit-length dequantised vector for chip i */
  vector(i) {
    const v = new Float32Array(this.k);
    const b = i * this.k;
    for (let j = 0; j < this.k; j++) v[j] = (this.vmin[j] + this.codes[b + j] * this.step[j]) * this.invNorm[i];
    return v;
  }

  /** normalised mean of the unit vectors of the given chips */
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

/** approximate quantiles from a strided sample */
export function quantiles(arr, qs, sampleSize = 100000) {
  const stride = Math.max(1, Math.floor(arr.length / sampleSize));
  const s = [];
  for (let i = 0; i < arr.length; i += stride) if (!Number.isNaN(arr[i])) s.push(arr[i]);
  s.sort((a, b) => a - b);
  return qs.map(q => s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]);
}
