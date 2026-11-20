"""Compute a DINOv3 patch-embedding grid for a single-band Mars CTX GeoTIFF.

Pipeline (after DINO-Sats, adapted to grayscale CTX):
  1. Read the CTX image, upsample it by --factor so DINOv3's 16 px patches
     cover 16/factor source pixels (e.g. factor 3 on 6 m CTX -> 32 m tokens).
  2. Run DINOv3 ViT-L/16 over overlapping windows and blend the patch tokens
     with a 2D Hann weight to suppress window-edge/positional artifacts.
  3. L2-normalise tokens, fit PCA, keep --n_components, quantise each
     component to uint8 over its 0.05-99.95 percentile range.
  4. Write everything the static web app needs to web/data/<name>/.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

MARS_RADIUS_M = 3396190.0


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def window_starts(total, win, stride):
    if total <= win:
        return [0]
    starts = list(range(0, total - win + 1, stride))
    if starts[-1] != total - win:
        starts.append(total - win)
    return starts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/jezero_ctx_6m.tif")
    p.add_argument("--out_dir", default="web/data")
    p.add_argument("--name", default=None, help="output sub-folder (default: derived from model + factor)")
    p.add_argument("--model", default="vit_large_patch16_dinov3.sat493m",
                   help="timm model name, e.g. vit_large_patch16_dinov3.sat493m or vit_large_patch16_dinov3.lvd1689m")
    p.add_argument("--factor", type=int, default=3, help="upsampling factor applied before the 16 px patch model")
    p.add_argument("--window", type=int, default=1024, help="window size in upsampled pixels (multiple of 16)")
    p.add_argument("--stride", type=int, default=256, help="window stride in upsampled pixels (multiple of 16)")
    p.add_argument("--n_components", type=int, default=64)
    p.add_argument("--pca_sample", type=int, default=300_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    assert args.window % 16 == 0 and args.stride % 16 == 0
    device = pick_device(args.device)
    name = args.name or f"{args.model.split('.')[-1]}_x{args.factor}"
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- image
    with rasterio.open(args.input) as ds:
        img = ds.read(1)
        transform = ds.transform
        crs_wkt = ds.crs.to_wkt()
        nodata = ds.nodata
    H, W = img.shape
    valid = img != nodata if nodata is not None else np.ones_like(img, bool)
    print(f"image {W}x{H}, pixel {transform.a} m, valid {valid.mean():.3f}")

    f = args.factor
    tok_src_px = 16 / f  # source pixels per token edge
    th, tw = math.ceil(H / tok_src_px), math.ceil(W / tok_src_px)  # token grid
    uh, uw = max(th * 16, args.window), max(tw * 16, args.window)  # padded upsampled size

    x = torch.from_numpy(img).float()[None, None] / 255.0
    x = F.interpolate(x, scale_factor=f, mode="bicubic", align_corners=False).clamp_(0, 1)
    x = F.pad(x, (0, uw - x.shape[-1], 0, uh - x.shape[-2]), mode="reflect")[0, 0]
    print(f"upsampled+padded to {uw}x{uh}; token grid {tw}x{th} ({tw * th:,} tokens, "
          f"{transform.a * tok_src_px:.1f} m each)")

    # ---------------------------------------------------------------- model
    model = timm.create_model(args.model, pretrained=True).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    n_prefix = model.num_prefix_tokens
    dim = model.num_features

    wt = args.window // 16
    hann = torch.hann_window(wt + 2, periodic=False)[1:-1]
    weight = (hann[:, None] * hann[None, :]).clamp_min(1e-3)  # (wt, wt)

    gh, gw = uh // 16, uw // 16
    acc = torch.zeros(gh, gw, dim, dtype=torch.float32)
    wsum = torch.zeros(gh, gw, 1, dtype=torch.float32)

    ys = window_starts(uh, args.window, args.stride)
    xs = window_starts(uw, args.window, args.stride)
    jobs = [(y, x0) for y in ys for x0 in xs]
    print(f"{len(jobs)} windows of {args.window}px, stride {args.stride}")

    amp = device.type in ("cuda", "mps")
    with torch.no_grad():
        for i in tqdm(range(0, len(jobs), args.batch)):
            batch = jobs[i:i + args.batch]
            tiles = torch.stack([x[y:y + args.window, x0:x0 + args.window] for y, x0 in batch])
            tiles = tiles[:, None].expand(-1, 3, -1, -1).to(device)
            tiles = (tiles - mean) / std
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                feats = model.forward_features(tiles)[:, n_prefix:]
            feats = feats.float().cpu().view(len(batch), wt, wt, dim)
            for (y, x0), ft in zip(batch, feats):
                ty, tx = y // 16, x0 // 16
                acc[ty:ty + wt, tx:tx + wt] += ft * weight[..., None]
                wsum[ty:ty + wt, tx:tx + wt] += weight[..., None]

    tokens = (acc / wsum)[:th, :tw].reshape(-1, dim)
    del acc, wsum
    tokens = F.normalize(tokens, dim=1)

    # token validity: token is valid if >50% of its source pixels are valid
    vt = torch.from_numpy(valid.astype(np.float32))[None, None]
    vt = F.interpolate(vt, size=(th, tw), mode="area")[0, 0].numpy() > 0.5

    # ---------------------------------------------------------------- PCA
    rng = np.random.default_rng(0)
    vidx = np.flatnonzero(vt.ravel())
    sample = tokens[rng.choice(vidx, size=min(args.pca_sample, len(vidx)), replace=False)]
    mu = sample.mean(0)
    cov = ((sample - mu).T @ (sample - mu)) / (len(sample) - 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    k = args.n_components
    comps = evecs[:, :k].float().T.contiguous()  # (k, dim)
    explained = float(evals[:k].sum() / evals.sum())
    print(f"PCA {k} components explain {explained:.1%} of variance")

    proj = ((tokens - mu) @ comps.T).numpy()  # (N, k)
    del tokens
    lo = np.percentile(proj[vidx], 0.05, axis=0)
    hi = np.percentile(proj[vidx], 99.95, axis=0)
    codes = np.clip(np.round((proj - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)

    # sanity: how well do quantised PCA cosines match full PCA cosines?
    deq = lo + codes.astype(np.float32) / 255 * (hi - lo)
    q = rng.choice(vidx, 20)
    a = proj[q] / np.linalg.norm(proj[q], axis=1, keepdims=True)
    b = deq[q] / np.linalg.norm(deq[q], axis=1, keepdims=True)
    print(f"quantisation self-cosine (mean): {(a * b).sum(1).mean():.4f}")

    codes.tofile(out / "embeddings.u8")
    np.packbits(vt.ravel()).tofile(out / "valid.bits")

    # PCA false-colour view (components 1-3)
    rgb = codes[:, :3].reshape(th, tw, 3).copy()
    rgb[~vt] = 0
    Image.fromarray(rgb).save(out / "pca_rgb.png")

    # ---------------------------------------------------------------- meta
    def lonlat(xm, ym):
        return math.degrees(xm / MARS_RADIUS_M), math.degrees(ym / MARS_RADIUS_M)

    left, top = transform.c, transform.f
    right, bottom = left + W * transform.a, top + H * transform.e
    meta = {
        "name": name,
        "model": args.model,
        "factor": f,
        "window": args.window,
        "stride": args.stride,
        "dim": dim,
        "grid": {"width": tw, "height": th},
        "image": {"width": W, "height": H, "pixel_m": transform.a},
        "token_src_px": tok_src_px,
        "token_m": transform.a * tok_src_px,
        "bounds_m": [left, bottom, right, top],
        "bounds_lonlat": [*lonlat(left, bottom), *lonlat(right, top)],
        "crs_wkt": crs_wkt,
        "mars_radius_m": MARS_RADIUS_M,
        "pca": {
            "n_components": k,
            "explained_variance": explained,
            "vmin": lo.astype(float).tolist(),
            "vmax": hi.astype(float).tolist(),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/ ({codes.nbytes / 1e6:.1f} MB embeddings)")


if __name__ == "__main__":
    main()
