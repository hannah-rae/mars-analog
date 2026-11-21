"""Feasibility test for global chip-level search.

1. Benchmarks DINOv3 ViT-L/16 chip throughput on this machine.
2. Embeds overlapping 224 px chips of the Jezero CTX mosaic (one vector per chip):
     - backbone CLS token (1024-d)
     - dino.txt image embedding (2048-d = head CLS + mean of head patch tokens; text-alignable)
3. Compresses the vectors several ways and measures how well each preserves
   full-precision nearest neighbours (recall@10 vs fp32, excluding overlapping chips).

Run (two passes; torch and faiss conflict over OpenMP in one process on macOS):
  .venv/bin/python experiments/chip_compression.py embed
  .venv/bin/python experiments/chip_compression.py eval
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

CHIP, STRIDE, NQ, K = 224, 64, 600, 10
CACHE = Path("data/jezero_chip_embeddings.npz")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "embed"

if STAGE == "embed":
  import rasterio
  import torch
  import torch.nn.functional as F
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
  from dinotxt import IMAGENET_MEAN, IMAGENET_STD, load_dinotxt  # noqa: E402
  dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model, _ = load_dinotxt("weights", "../third_party/dinov3")
  model.to(dev).eval()
  vm = model.visual_model
  mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
  std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)


  @torch.no_grad()
  def embed(batch):  # (B, 1, H, W) float in [0,1] on device
      x = (batch.expand(-1, 3, -1, -1) - mean) / std
      with torch.autocast(dev.type, dtype=torch.float16):
          cls, patches, regs = vm.get_backbone_features(x)
          tok = vm.head(torch.cat([cls.unsqueeze(1), regs, patches], dim=1))
      head_cls, head_patch = tok[:, 0], tok[:, regs.shape[1] + 1:]
      txt = torch.cat([head_cls, head_patch.mean(1)], dim=1)
      return cls.float().cpu(), txt.float().cpu()


  # ---------------------------------------------------------------- throughput
  bench = {}
  for size, bs in [(224, 64), (448, 16)]:
      x = torch.rand(bs, 1, size, size, device=dev)
      embed(x)
      if dev.type == "mps":
          torch.mps.synchronize()
      t = time.time()
      for _ in range(3):
          embed(x)
      if dev.type == "mps":
          torch.mps.synchronize()
      bench[size] = 3 * bs / (time.time() - t)
      print(f"throughput {size}px: {bench[size]:.0f} chips/s (backbone + dino.txt head, fp16, {dev.type})")

  # ---------------------------------------------------------------- chips
  img = rasterio.open("data/jezero_ctx_6m.tif").read(1)
  H, W = img.shape
  ys = list(range(0, H - CHIP + 1, STRIDE))
  xs = list(range(0, W - CHIP + 1, STRIDE))
  pos = np.array([(y, x) for y in ys for x in xs])
  timg = torch.from_numpy(img).float() / 255
  cls_all, txt_all = [], []
  t = time.time()
  for i in range(0, len(pos), 64):
      b = torch.stack([timg[y:y + CHIP, x:x + CHIP] for y, x in pos[i:i + 64]])[:, None].to(dev)
      c, tx = embed(b)
      cls_all.append(c), txt_all.append(tx)
  print(f"embedded {len(pos)} chips in {time.time() - t:.0f}s")
  np.savez(CACHE, pos=pos, cls=F.normalize(torch.cat(cls_all), dim=1).numpy(),
           txt=F.normalize(torch.cat(txt_all), dim=1).numpy(), bench=json.dumps(bench))
  print(f"saved {CACHE}")
  sys.exit(0)


# ---------------------------------------------------------------- evaluation
import faiss  # noqa: E402

faiss.omp_set_num_threads(8)

z = np.load(CACHE)
pos, bench = z["pos"], json.loads(str(z["bench"]))
def center(X):  # retrieval uses mean-centred cosine (as in the demo); GT and all methods share it
    X = X - X.mean(0)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


embs = {"backbone CLS (1024-d)": center(z["cls"]), "dino.txt image emb (2048-d)": center(z["txt"])}
rng = np.random.default_rng(0)
qidx = rng.choice(len(pos), NQ, replace=False)
# exclude chips that overlap the query (centres closer than one chip width)
dist = np.abs(pos[qidx, None, :] - pos[None, :, :]).max(-1)
excl = dist < CHIP


def topk(sim):
    sim = sim.copy()
    sim[excl] = -np.inf
    return np.argsort(-sim, axis=1)[:, :100]


def recall(gt, pred, k=K, depth=K):
    return float(np.mean([len(set(g[:k]) & set(p[:depth])) / k for g, p in zip(gt, pred)]))


def pca(X, d):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:d]


def quant_u8(Z):
    lo, hi = np.percentile(Z, 0.05, axis=0), np.percentile(Z, 99.95, axis=0)
    q = np.clip(np.round((Z - lo) / (hi - lo) * 255), 0, 255)
    return lo + q / 255 * (hi - lo)


def cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


results = {"throughput_chips_per_s": bench, "n_chips": len(pos), "n_queries": NQ, "methods": {}}
for name, X in embs.items():
    D = X.shape[1]
    gt = topk(X[qidx] @ X.T)
    rows = []

    def add(label, bytes_, pred):
        rows.append((label, bytes_, recall(gt, pred), recall(gt, pred, depth=100)))
        print(f"  {name[:12]} {label:30s} {bytes_:5d} B  R@10 {rows[-1][2]:.3f}", flush=True)

    add("fp16", 2 * D, topk(cos(X[qidx].astype(np.float16).astype(np.float32), X.astype(np.float16).astype(np.float32))))
    s = np.abs(X).max(0)
    Xi8 = np.round(X / s * 127) * s / 127
    add("int8 scalar", D, topk(cos(Xi8[qidx], Xi8)))
    for d in [256, 128, 64, 32, 16]:
        mu, V = pca(X, d)
        Zq = quant_u8((X - mu) @ V.T)
        Xr = Zq @ V + mu  # reconstruct (add the mean back) so the metric matches fp32 cosine
        add(f"PCA-{d} + uint8", d, topk(cos(Xr[qidx], Xr)))
    for bits in [1024, 512, 256, 128]:
        mu, V = pca(X, min(bits, D))
        R, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(V.shape[0], V.shape[0])))
        B = (((X - mu) @ V.T) @ R) > 0  # random rotation balances variance across bits
        Bs = B.astype(np.float32) * 2 - 1
        ham = Bs[qidx] @ Bs.T  # = bits - 2*hamming
        add(f"binary {bits} bits (PCA+rot sign)", bits // 8, topk(ham))
    for m in [64, 32, 16, 8]:
        opq = faiss.OPQMatrix(D, m)
        opq.niter = 10
        pq = faiss.IndexPQ(D, m, 8, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexPreTransform(opq, pq)
        index.train(X)
        index.add(X)
        Xr = index.sa_decode(index.sa_encode(X))
        add(f"OPQ+PQ {m} bytes", m, topk(cos(Xr[qidx], Xr)))
    print(f"\n{name}: recall of fp32 top-{K} (excluding overlapping chips)")
    print(f"{'method':32s} {'bytes':>6s} {'R@10':>6s} {'R@10 in top100':>15s}")
    for r in rows:
        print(f"{r[0]:32s} {r[1]:6d} {r[2]:6.3f} {r[3]:15.3f}")
    results["methods"][name] = [dict(method=r[0], bytes=r[1], recall10=r[2], recall10_at100=r[3]) for r in rows]

out = Path("experiments/chip_compression_results.json")
out.write_text(json.dumps(results, indent=1))
print(f"\nwrote {out}")
