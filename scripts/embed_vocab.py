"""Precompute dino.txt text embeddings for the search vocabulary.

Each phrase is embedded with a prompt ensemble (see dinotxt.TEMPLATES), then
projected into the scene's PCA space so the browser can score it with one
64-term dot product per token:  cos(t, x) ~= t.mu + (C t) . p
Writes web/data/<name>/vocab.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dinotxt import TEMPLATES, load_dinotxt, text_embeddings

p = argparse.ArgumentParser()
p.add_argument("--name", default="dinotxt_x3")
p.add_argument("--vocab", default="scripts/vocab_mars.txt")
p.add_argument("--out_dir", default="web/data")
p.add_argument("--pca_dir", default="data")
p.add_argument("--weights_dir", default="weights")
p.add_argument("--dinov3_repo", default="../third_party/dinov3")
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = p.parse_args()

phrases, background = [], []
for line in Path(args.vocab).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("bg:"):
        background.append(line[3:].strip())
    else:
        phrases.append(line)
allp = list(dict.fromkeys(background + phrases))

pca = np.load(Path(args.pca_dir) / f"{args.name}_pca.npz")
mu, C = pca["mean"], pca["components"]  # (D,), (k, D)

model, tok = load_dinotxt(args.weights_dir, args.dinov3_repo)
model.to(args.device)
T = text_embeddings(model, tok, allp, torch.device(args.device)).numpy()  # (N, D)
logit_scale = float(model.logit_scale.exp())

entries = [{"text": t, "q": np.round(C @ v, 5).tolist(), "b": round(float(v @ mu), 6)} for t, v in zip(allp, T)]
out = {
    "templates": list(TEMPLATES),
    "logit_scale": logit_scale,
    "background": background,
    "phrases": entries,
}
dst = Path(args.out_dir) / args.name / "vocab.json"
dst.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {dst}: {len(entries)} phrases ({len(background)} background), logit scale {logit_scale:.1f}, "
      f"{dst.stat().st_size / 1e3:.0f} kB")
