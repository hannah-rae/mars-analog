# Mars Analog: search Jezero Crater with DINOv3

Click a spot on a Mars Reconnaissance Orbiter CTX image of Jezero Crater, and
the page highlights every other place in the scene that DINOv3 sees as
similar. Everything runs in the browser: no server, no GPU at query time.

This is a Mars version of Caleb Robinson's
[DINO-Sats](https://calebrob.com/dinosats/)
([post](https://geospatialml.substack.com/p/searching-aerial-imagery-with-dinosats),
[code](https://github.com/calebrob6/dinosats)), which searches NAIP aerial
imagery with DINOv3 and dino.txt.

## How it works

1. **Imagery.** The USGS Mars 2020 Terrain Relative Navigation CTX
   orthomosaic of Jezero (`JEZ_ctx_B_soc_008_orthoMosaic_6m`): 5322 × 5040
   px at 6 m/px (about 32 × 30 km), single band, Mars 2000 equirectangular.
2. **Features.** The image is upsampled 3× and run through DINOv3 ViT-L/16
   (`vit_large_patch16_dinov3.sat493m` from timm by default), so each patch
   token covers 32 m on the ground (943k tokens). Grayscale is copied to
   all three channels. The model runs over 1024 px windows at a 256 px
   stride, and the overlapping windows are blended with a 2D Hann weight to
   remove window-edge artifacts.
3. **Compression.** Tokens are L2-normalized and reduced to 64 PCA
   components. Each component is quantized to uint8 over its 0.05–99.95
   percentile range, so each token takes 64 bytes and the whole scene is
   about 60 MB.
4. **Search.** The browser downloads the token grid once. A query is the
   mean of the clicked tokens (positives) minus the mean of the
   shift-clicked tokens (negatives). The score for every token is
   `cos(x, q⁺) − cos(x, q⁻)`. PCA projection and dequantization fold into a
   single 64-term dot product per token (see [`web/search.js`](web/search.js)),
   so a full-scene query takes tens of milliseconds in plain JavaScript.
5. **Display.** Scores are stretched between the median and the 99.95th
   percentile and colored with the inferno colormap. Greedy peak picking
   lists the top 10 matches, at least 250 m apart.

## How this differs from DINO-Sats

DINO-Sats is mainly *text* search: dino.txt aligns DINOv3 with a text
encoder trained on web photos. That alignment is unlikely to hold for
grayscale orbital images of Mars, so this demo uses **query-by-example**
(positive and negative clicks) on the visual features alone. Adding
dino.txt text search would be a natural experiment.

Other differences: a single small scene, so embeddings ship as one flat
binary file instead of a multi-level COG; a 3× upsample (32 m tokens) to
suit 6 m CTX; and no positional-bias projection (Hann blending at stride
W/4 handles most of the window ripple).

## Run it locally

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
mkdir -p data
curl -L -o data/jezero_ctx_6m.tif \
  https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif
.venv/bin/python scripts/make_basemap.py
.venv/bin/python scripts/embed.py            # ~15 min on an Apple M-series GPU (MPS)
python3 -m http.server -d web 8000           # open http://localhost:8000
```

`embed.py` options: `--model vit_large_patch16_dinov3.lvd1689m` uses the
web-image DINOv3 instead of the satellite one; `--factor` sets the
upsampling (token size = 96 m / factor); `--device cuda|mps|cpu`. Each run
writes to `web/data/<name>/`. Add an entry to `web/config.json` and the UI
shows a dropdown to switch between embedding sets.

Controls: **click** adds a positive example, **shift+click** adds a
negative, **Esc** clears, **Space** toggles the heatmap. Click an example
chip to remove it. The query is saved in the URL hash, so you can share a
link to a search.

## Deploy

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push
to `main`. The generated files in `web/data/` are about 65 MB in total, so
they can be committed directly (each file is under GitHub's 100 MB limit).

## Licenses

- **Code:** MIT (see [LICENSE](LICENSE)).
- **DINOv3 weights and everything derived from them**, including
  `web/data/*/embeddings.u8` and `pca_rgb.png`: the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **CTX imagery:** NASA/JPL-Caltech/MSSS. The orthomosaic is from USGS
  Astrogeology (Fergason et al., Mars 2020 TRN products). U.S. government
  work, public domain.
- **OpenLayers** (BSD-2-Clause), loaded from jsDelivr.

## Acknowledgments

Design, colormap, and scoring approach adapted from
[DINO-Sats](https://github.com/calebrob6/dinosats) by Caleb Robinson (MIT).
