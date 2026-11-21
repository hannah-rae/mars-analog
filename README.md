# Mars Analog: search Mars and the Moon with DINOv3

**Live:** https://hannah-rae.github.io/mars-analog/ (Mars) ·
[Moon](https://hannah-rae.github.io/mars-analog/?body=moon)

Search the whole surface of Mars (THEMIS daytime infrared) or the Moon (LROC
WAC) two ways: type a phrase such as "sand dunes" or "impact crater", or click
a spot and see every other place on the planet that DINOv3 sees as similar.
Everything runs in the browser: no server, no GPU at query time.

This started as a Mars version of Caleb Robinson's
[DINO-Sats](https://calebrob.com/dinosats/)
([post](https://geospatialml.substack.com/p/searching-aerial-imagery-with-dinosats),
[code](https://github.com/calebrob6/dinosats)), which searches NAIP aerial
imagery with DINOv3 and dino.txt.

## How it works

1. **Imagery.** The USGS global mosaics at 100 m/px: THEMIS Day IR v12 for Mars
   (213,390 × 106,696 px) and LROC WAC (June 2013) for the Moon
   (109,164 × 54,582 px). Both are single-band, equirectangular.
2. **Chips** ([`scripts/embed_chips.py`](scripts/embed_chips.py)). Latitude
   bands 224 px (22.4 km) tall, stepped by 150 px (15 km). Within a band,
   chips are 224 / cos(latitude) px wide, so every chip covers roughly the same
   ground area, and they wrap at 360°. Each chip is area-resampled to
   224 × 224. Chips that are mostly nodata or mostly shadow (the WAC mosaic is
   largely in shadow above about 83°) are marked invalid. Mars has 644,631
   chips and the Moon 168,781.
3. **Features.** Each chip is embedded with DINOv3 ViT-L/16 (the standard
   web-image `lvd1689m` backbone) plus the
   [dino.txt](https://arxiv.org/abs/2412.16334) vision head. The embedding is
   dino.txt's *image-level* vector (head CLS + mean of head patch tokens,
   2048-d), which is the one aligned with text. Grayscale is copied to all
   three channels. The run is checkpointed after every band, so it can be
   resumed; Mars takes about 1.8 h and the Moon about 30 min on an M5 Pro
   (MPS).
4. **Compression.** Embeddings are reduced to 64 PCA components (84–86% of
   the variance), and each component is quantized to uint8 over its
   0.05–99.95 percentile range: 64 bytes per chip, 41 MB for Mars and 11 MB
   for the Moon. [`experiments/chip_compression.py`](experiments/chip_compression.py)
   compares PCA, product quantization and binary codes on 224 px chips of a
   6 m CTX mosaic of Jezero Crater (an earlier version of this demo): PCA-64 keeps about
   85% of the full-precision top-10 neighbours, and all of them are within
   the compressed top 100, so a re-ranking stage could recover exact results.
5. **Click search.** A query is the mean of the clicked chips (positives)
   minus the mean of the shift-clicked chips (negatives). Every chip is scored
   with `cos(x, q⁺) − cos(x, q⁻)`. PCA projection and dequantization fold into
   a single 64-term dot product per chip (see [`web/search.js`](web/search.js)),
   so a whole-planet query takes under 100 ms in plain JavaScript.
6. **Text search.** Phrases from [`scripts/vocab_mars.txt`](scripts/vocab_mars.txt)
   and [`scripts/vocab_moon.txt`](scripts/vocab_moon.txt) are embedded ahead of
   time with the dino.txt text encoder (full 2048-d), averaged over four prompt
   templates ("a satellite image of {}", …), and projected into each body's
   PCA space ([`scripts/embed_vocab.py`](scripts/embed_vocab.py)), so the
   browser needs no text model. As in DINO-Sats, each chip's score is relative
   to a set of background prompts ("the surface of Mars", "rocky terrain", …):
   `P = 1 / (1 + exp(logsumexp_bg(100·cos) − 100·cos_query))`. The heatmap
   stretches P between its median and 99.9th percentile. **Alt+click** shows
   which phrases best match a chip.
7. **Viewer** ([`web/app.js`](web/app.js), OpenLayers). The heatmap is a
   4096 × 2048 global raster; each pixel shows the chip whose centre is
   nearest, computed directly from the band layout, so no lookup table is
   downloaded. The top 20 matches are picked greedily at least 1.5 chips apart
   (great-circle distance). The **top-10 preview** panel shows each match as
   the model saw it (the chip's lon/lat box resampled to a square), next to
   the query. Background and preview imagery come from NASA
   [Mars Trek](https://trek.nasa.gov/mars/) and [Moon Trek](https://trek.nasa.gov/moon/)
   tiles over a low-resolution overview.

### How well does text search work?

dino.txt learned from web photos with captions, not grayscale orbital images,
so expect broad terms to work and fine geologic units not to. In testing,
"sand dunes" finds the north polar erg on Mars and "canyon" finds Valles
Marineris, while "impact crater" lights up much of both bodies (at 22 km,
craters are everywhere). Click search is more reliable for specific features.

## Run it locally

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git ../third_party/dinov3

# DINOv3 + dino.txt weights (DINOv3 License). Official links come from Meta's
# download form; this repo was built from a Hugging Face mirror of the same
# files. scripts/dinotxt.py checks the SHA-256 of each file before loading.
mkdir -p weights && cd weights
M=https://huggingface.co/PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder/resolve/main
for f in bpe_simple_vocab_16e6.txt.gz dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
         dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth; do curl -LO $M/$f; done
cd ..

# Mars
mkdir -p data
curl -L -o data/themis_dayir_100m_v12.tif \
  https://planetarymaps.usgs.gov/mosaic/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12.tif   # 22.8 GB
.venv/bin/python scripts/embed_chips.py
.venv/bin/python scripts/embed_vocab.py --name themis --image_level

# Moon
curl -L -o data/lroc_wac_global_100m.tif \
  https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif  # 6.0 GB
.venv/bin/python scripts/embed_chips.py --input data/lroc_wac_global_100m.tif --name wac --body Moon
.venv/bin/python scripts/embed_vocab.py --name wac --image_level --vocab scripts/vocab_moon.txt

python3 -m http.server -d web 8000   # open http://localhost:8000
```

Each run writes to `web/data/<name>/` and keeps its full-precision embeddings
in `data/chips/<name>/`. To change which chips count as valid without
re-running the model, pass `--revalidate` (with `--max_nodata`, `--dark_level`
or `--max_dark`); it redoes PCA, so re-run `embed_vocab.py` afterwards. To add
search phrases, edit the vocabulary file and re-run `embed_vocab.py`.

Controls: type or pick a phrase to search by text. **Click** a chip to add a
positive example and **Shift+click** a negative. **Alt+click** shows matching
phrases, **Esc** clears, **Space** toggles the heatmap. Click a match (in the
list or the preview panel) to zoom to it. The query is saved in the URL hash,
so you can share a link to a search.

## Deploy

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push
to `main`. The generated files in `web/data/` are about 70 MB in total, so
they are committed directly (each file is under GitHub's 100 MB limit).

## Licenses

- **Code:** MIT (see [LICENSE](LICENSE)).
- **DINOv3 weights and everything derived from them**, including
  `web/data/*/embeddings.u8` and `vocab.json`: the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **Imagery:** THEMIS Day IR global mosaic (NASA/JPL/ASU; USGS Astrogeology)
  and LROC WAC global mosaic (NASA/GSFC/ASU; USGS Astrogeology), U.S.
  government works in the public domain. Map tiles from NASA Mars Trek and
  Moon Trek.
- **OpenLayers** (BSD-2-Clause), loaded from jsDelivr.

## Acknowledgments

Design, colormap, and scoring approach (including relative text scoring
against background prompts) adapted from
[DINO-Sats](https://github.com/calebrob6/dinosats) by Caleb Robinson (MIT).
dino.txt and DINOv3 are from Meta AI
([facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)).
