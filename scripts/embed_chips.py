"""Global chip-level embeddings for an equirectangular planetary mosaic
(e.g. Mars THEMIS Day IR 100 m, Moon LROC WAC 100 m).

One dino.txt image-level embedding (2048-d, text-alignable) per chip, on an
approximately equal-area chip layout:

  * Latitude bands of --chip rows (e.g. 224 px = 22.4 km at 100 m/px), stepped by --stride rows.
  * Within a band centred at latitude phi, chips are --chip / cos(phi) px wide (so each covers
    about the same ground area), stepped by the same fraction of their width, and wrap at 360 deg.
  * Each chip is resampled to --input x --input px before the model.

Progress is checkpointed after every band (resume by re-running the same command).
After all bands: PCA -> uint8 codes + metadata for the web viewer, written to web/data/<name>/.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

from dinotxt import IMAGENET_MEAN, IMAGENET_STD, image_embeddings, load_dinotxt

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--input", default="data/themis_dayir_100m_v12.tif")
p.add_argument("--name", default="themis")
p.add_argument("--body", default="Mars")
p.add_argument("--chip", type=int, default=224, help="chip height in source px")
p.add_argument("--stride", type=int, default=150, help="band step in source px (chip widths step by the same fraction)")
p.add_argument("--input_size", type=int, default=224, help="model input size in px")
p.add_argument("--max_nodata", type=float, default=0.5)
p.add_argument("--n_components", type=int, default=64)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--work_dir", default="data/chips")
p.add_argument("--out_dir", default="web/data")
p.add_argument("--overview_width", type=int, default=8192)
p.add_argument("--max_bands", type=int, default=0, help="testing: only embed the first N bands")
p.add_argument("--no_overview", action="store_true", help="testing: skip the overview basemap")
p.add_argument("--weights_dir", default="weights")
p.add_argument("--dinov3_repo", default="../third_party/dinov3")
args = p.parse_args()

work = Path(args.work_dir) / args.name
work.mkdir(parents=True, exist_ok=True)
out = Path(args.out_dir) / args.name
out.mkdir(parents=True, exist_ok=True)

ds = rasterio.open(args.input)
H, W = ds.height, ds.width
crs = ds.crs.to_dict()
assert crs.get("proj") == "eqc" and "R" in crs, f"expected a spherical equirectangular mosaic, got {crs}"
R = float(crs["R"])
px = ds.transform.a
top_y = ds.transform.f
lon_of_x0 = float(crs.get("lon_0", 0.0)) + math.degrees(ds.transform.c / R)  # longitude of the left edge
C, S = args.chip, args.stride


def row_lat(r):  # latitude of the top edge of row r
    return math.degrees((top_y - r * px) / R)


# ---------------------------------------------------------------- chip layout
bands, offset = [], 0
starts = list(range(0, H - C + 1, S))
if starts[-1] != H - C:
    starts.append(H - C)
if args.max_bands:
    starts = starts[: args.max_bands]
for r0 in starts:
    lat_c = row_lat(r0 + C / 2)
    w = min(W, int(round(C / max(math.cos(math.radians(lat_c)), 1e-6))))
    s = max(1, int(round(w * S / C)))
    n = max(1, math.ceil(W / s))
    bands.append(dict(r0=r0, lat_top=row_lat(r0), lat_bottom=row_lat(r0 + C), w=w, s=s, n=n, offset=offset))
    offset += n
N = offset
print(f"{W}x{H} px at {px:.0f} m; {len(bands)} bands, {N:,} chips "
      f"({C * px / 1000:.1f} km, stride {S * px / 1000:.1f} km)")

layout = dict(width=W, height=H, pixel_m=px, chip=C, stride=S, lon_of_x0=lon_of_x0, mars_radius_m=R, bands=bands)
(work / "layout.json").write_text(json.dumps(layout))

# ---------------------------------------------------------------- embeddings (resumable)
emb_path, valid_path, prog_path = work / "emb_f16.npy", work / "valid.npy", work / "progress.json"
if emb_path.exists():
    emb = np.load(emb_path, mmap_mode="r+")
    valid = np.load(valid_path, mmap_mode="r+")
    done = json.loads(prog_path.read_text())["bands_done"]
else:
    emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(N, 2048))
    valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.uint8, shape=(N,))
    done = 0

if done < len(bands):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_dinotxt(args.weights_dir, args.dinov3_repo)
    model.to(dev).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    amp = dev.type in ("mps", "cuda")
    remaining = sum(b["n"] for b in bands[done:])
    bar = tqdm(total=remaining, unit="chip", smoothing=0.05)
    for bi in range(done, len(bands)):
        b = bands[bi]
        band = ds.read(1, window=Window(0, b["r0"], W, C))
        band = np.concatenate([band, band[:, : b["w"]]], axis=1)  # wrap at 360 deg
        xs = [k * b["s"] for k in range(b["n"])]
        for i in range(0, len(xs), args.batch):
            xb = xs[i:i + args.batch]
            chips = np.stack([band[:, x:x + b["w"]] for x in xb])  # (B, C, w) uint8
            nod = (chips == 0).mean(axis=(1, 2))
            t = torch.from_numpy(chips).float().div_(255)[:, None]
            if b["w"] != args.input_size or C != args.input_size:  # area resize on CPU (MPS needs divisible sizes)
                t = F.interpolate(t, size=(args.input_size, args.input_size), mode="area")
            t = t.to(dev)
            t = (t.expand(-1, 3, -1, -1) - mean) / std
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                e = image_embeddings(model, t)
            e = F.normalize(e.float(), dim=1).cpu().numpy().astype(np.float16)
            j0 = b["offset"] + i
            emb[j0:j0 + len(xb)] = e
            valid[j0:j0 + len(xb)] = (nod <= args.max_nodata).astype(np.uint8)
            bar.update(len(xb))
        emb.flush(); valid.flush()
        prog_path.write_text(json.dumps({"bands_done": bi + 1}))
    bar.close()

# ---------------------------------------------------------------- PCA + quantisation
X = np.asarray(emb, dtype=np.float32)
v = np.asarray(valid).astype(bool)
print(f"valid chips: {v.sum():,} / {N:,}")
rng = np.random.default_rng(0)
sidx = rng.choice(np.flatnonzero(v), size=min(300_000, int(v.sum())), replace=False)
mu = X[sidx].mean(0)
cov = np.cov((X[sidx] - mu).T)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
k = args.n_components
comps = evecs[:, order[:k]].T.astype(np.float32)
explained = float(evals[order[:k]].sum() / evals.sum())
print(f"PCA {k}: {explained:.1%} of variance")
proj = (X - mu) @ comps.T
lo = np.percentile(proj[v], 0.05, axis=0)
hi = np.percentile(proj[v], 99.95, axis=0)
codes = np.clip(np.round((proj - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
codes.tofile(out / "embeddings.u8")
np.packbits(v).tofile(out / "valid.bits")
np.savez(Path("data") / f"{args.name}_pca.npz", mean=mu, components=comps, vmin=lo, vmax=hi)

# ---------------------------------------------------------------- overview basemap (lon -180..180)
ow = args.overview_width
ov = None if args.no_overview else ds.read(1, out_shape=(ow // 2, ow), resampling=rasterio.enums.Resampling.average)
if ov is not None:
    shift = int(round(((lon_of_x0 + 180.0) % 360.0) / 360.0 * ow))  # viewer extent starts at lon -180
    ov = np.roll(ov, shift, axis=1)
    Image.fromarray(ov).save(out / "overview.jpg", quality=85, optimize=True, progressive=True)

meta = dict(
    name=args.name, body=args.body, source=Path(args.input).name, model="dinotxt (DINOv3 ViT-L/16 LVD + dino.txt)",
    embedding="image-level: [head CLS, mean head patch tokens], 2048-d",
    grid=dict(width=N, height=1), n_chips=N, layout=layout, input_size=args.input_size,
    chip_km=C * px / 1000, stride_km=S * px / 1000,
    pca=dict(n_components=k, explained_variance=explained, vmin=lo.tolist(), vmax=hi.tolist()),
)
(out / "meta.json").write_text(json.dumps(meta))
print(f"wrote {out}/ ({codes.nbytes / 1e6:.1f} MB codes)")
