# Mars Analog: search Jezero Crater with DINOv3

Search a Mars Reconnaissance Orbiter CTX image of Jezero Crater two ways:
type a phrase such as "sand dunes" or "impact crater", or click a spot and see
every other place in the scene that DINOv3 sees as similar. Everything runs in
the browser: no server, no GPU at query time.

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
   (the standard web-image `lvd1689m` backbone) plus the
   [dino.txt](https://arxiv.org/abs/2412.16334) vision head, whose patch
   tokens are aligned with text. Each token covers 32 m on the ground (943k
   tokens). Grayscale is copied to all three channels. The model runs over
   1024 px windows at a 256 px stride, and the overlapping windows are
   blended with a 2D Hann weight to remove window-edge artifacts.
3. **Compression.** Tokens are L2-normalized and reduced to 64 PCA
   components (85% of the variance). Each component is quantized to uint8
   over its 0.05–99.95 percentile range, so each token takes 64 bytes and
   the whole scene is about 60 MB.
4. **Click search.** A query is the mean of the clicked tokens (positives)
   minus the mean of the shift-clicked tokens (negatives). Every token is
   scored with `cos(x, q⁺) − cos(x, q⁻)`. PCA projection and dequantization
   fold into a single 64-term dot product per token (see
   [`web/search.js`](web/search.js)), so a full-scene query takes tens of
   milliseconds in plain JavaScript.
5. **Text search.** Phrases from [`scripts/vocab_mars.txt`](scripts/vocab_mars.txt)
   are embedded ahead of time with the dino.txt text encoder, averaged over
   four prompt templates ("a satellite image of {}", …), and projected into
   the scene's PCA space ([`scripts/embed_vocab.py`](scripts/embed_vocab.py)),
   so the browser needs no text model. As in DINO-Sats, each token's score is
   relative to a set of background prompts ("the surface of Mars", "rocky
   terrain", …): `P = 1 / (1 + exp(logsumexp_bg(100·cos) − 100·cos_query))`.
   **Alt+click** shows which phrases best match a spot.
6. **Display.** Inferno colormap, threshold and opacity sliders, and a list
   of the top 10 matches (greedy peak picking, at least 250 m apart).

### How well does text search work on Mars?

dino.txt learned from web photos with captions, not grayscale orbital
images, so expect broad terms to work and fine geologic units not to. In
testing, "sand dunes" finds the rippled sand on the floor of the inlet
channel, and "impact crater" finds craters. Terms like "river delta" barely
register. Click search is more reliable for specific features.

## How this differs from DINO-Sats

- Text search uses a fixed, precomputed vocabulary (about 120 Mars phrases)
  instead of running the text encoder in the browser.
- Click search with positive and negative examples.
- A single small scene, so embeddings ship as one flat binary file instead
  of a multi-level COG.
- A 3× upsample (32 m tokens) to suit 6 m CTX.
- No positional-bias projection; Hann blending at stride W/4 handles most of
  the window ripple.

## Run it locally

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git ../third_party/dinov3

# Imagery
mkdir -p data
curl -L -o data/jezero_ctx_6m.tif \
  https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif

# DINOv3 + dino.txt weights (DINOv3 License). Official links come from Meta's
# download form; this repo was built from a Hugging Face mirror of the same
# files. scripts/dinotxt.py checks the SHA-256 of each file before loading.
mkdir -p weights && cd weights
M=https://huggingface.co/PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder/resolve/main
for f in bpe_simple_vocab_16e6.txt.gz dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
         dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth; do curl -LO $M/$f; done
cd ..

.venv/bin/python scripts/make_basemap.py
.venv/bin/python scripts/embed.py --model dinotxt   # ~20 min on an Apple M-series GPU (MPS)
.venv/bin/python scripts/embed_vocab.py             # text embeddings for the vocabulary
python3 -m http.server -d web 8000                  # open http://localhost:8000
```

To add search phrases, edit `scripts/vocab_mars.txt` and re-run
`embed_vocab.py`. `embed.py` also accepts any timm DINOv3 model (for
example `--model vit_large_patch16_dinov3.sat493m`, the satellite-trained
backbone) for click-only search; `--factor` sets the upsampling (token size =
96 m / factor). Each run writes to `web/data/<name>/`; list it in
`web/config.json` (with `"text": true` if it has a `vocab.json`) and the UI
shows a dropdown to switch between embedding sets.

Controls: type or pick a phrase to search by text. **Click** adds a positive
example, **Shift+click** a negative, **Alt+click** shows matching phrases,
**Esc** clears, **Space** toggles the heatmap. The query is saved in the URL
hash, so you can share a link to a search.

## Deploy

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push
to `main`. The generated files in `web/data/` are about 65 MB in total, so
they can be committed directly (each file is under GitHub's 100 MB limit).

## Licenses

- **Code:** MIT (see [LICENSE](LICENSE)).
- **DINOv3 weights and everything derived from them**, including
  `web/data/*/embeddings.u8`, `pca_rgb.png` and `vocab.json`: the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **CTX imagery:** NASA/JPL-Caltech/MSSS. The orthomosaic is from USGS
  Astrogeology (Fergason et al., Mars 2020 TRN products). U.S. government
  work, public domain.
- **OpenLayers** (BSD-2-Clause), loaded from jsDelivr.

## Acknowledgments

Design, colormap, and scoring approach (including relative text scoring
against background prompts) adapted from
[DINO-Sats](https://github.com/calebrob6/dinosats) by Caleb Robinson (MIT).
dino.txt and DINOv3 are from Meta AI
([facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)).
