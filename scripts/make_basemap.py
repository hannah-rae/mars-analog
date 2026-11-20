"""Export the CTX GeoTIFF as a JPEG basemap for the web viewer."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/jezero_ctx_6m.tif")
p.add_argument("--out", default="web/data/ctx.jpg")
p.add_argument("--quality", type=int, default=85)
args = p.parse_args()

with rasterio.open(args.input) as ds:
    img = ds.read(1)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(img.astype(np.uint8), mode="L").save(args.out, quality=args.quality, optimize=True, progressive=True)
print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
