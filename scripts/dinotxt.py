"""Load DINOv3 ViT-L/16 + dino.txt from local checkpoint files.

Needs a clone of https://github.com/facebookresearch/dinov3 (for the model code)
and the three files listed in WEIGHT_FILES in --weights_dir. Checkpoints are
loaded with torch.load(weights_only=True), so a tampered file cannot run code.
"""

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WEIGHT_FILES = {
    "backbone": ("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                 "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"),
    "dinotxt": ("dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
                "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"),
    "bpe": ("bpe_simple_vocab_16e6.txt.gz",
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP-style prompt ensemble, adapted to orbital imagery
TEMPLATES = (
    "a satellite image of {}.",
    "an aerial photo of {}.",
    "a black and white satellite photo of {}.",
    "a photo of {}.",
)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_dinotxt(weights_dir="weights", dinov3_repo="../third_party/dinov3", verify=True):
    weights_dir = Path(weights_dir)
    paths = {}
    for key, (name, digest) in WEIGHT_FILES.items():
        p = weights_dir / name
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
        if verify and sha256(p) != digest:
            raise ValueError(f"sha256 mismatch for {p}")
        paths[key] = p

    sys.path.insert(0, str(Path(dinov3_repo).resolve()))
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
    from dinov3.eval.text.tokenizer import get_tokenizer

    model, _ = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=False, bpe_path_or_url=str(paths["bpe"]))
    bb = torch.load(paths["backbone"], map_location="cpu", weights_only=True)
    model.visual_model.backbone.load_state_dict(bb, strict=True)
    sd = torch.load(paths["dinotxt"], map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("visual_model.backbone.")]
    if missing or unexpected:
        raise RuntimeError(f"dino.txt load: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    tokenizer = get_tokenizer(bpe_path_or_url=str(paths["bpe"]))
    return model, tokenizer


@torch.no_grad()
def text_embeddings(model, tokenizer, phrases, device, templates=TEMPLATES, patch_aligned=True):
    """Prompt-ensembled unit text embedding per phrase.

    patch_aligned=True: 2nd half (N, 1024), aligned with dino.txt patch tokens.
    patch_aligned=False: full (N, 2048), aligned with image-level embeddings (head CLS + mean patch).
    """
    out = []
    for phrase in phrases:
        tokens = tokenizer.tokenize([t.format(phrase) for t in templates]).to(device)
        f = model.encode_text(tokens).float()
        if patch_aligned:
            f = f[:, f.shape[1] // 2:]  # 2nd half aligns with patch tokens
        f = F.normalize(f, dim=-1).mean(0)
        out.append(F.normalize(f, dim=-1))
    return torch.stack(out).cpu()


@torch.no_grad()
def image_embeddings(model, images):
    """dino.txt image-level embedding (B, 2048) = [head CLS, mean of head patch tokens], unnormalised."""
    vm = model.visual_model
    cls, patches, regs = vm.get_backbone_features(images)
    tok = vm.head(torch.cat([cls.unsqueeze(1), regs, patches], dim=1))
    return torch.cat([tok[:, 0], tok[:, regs.shape[1] + 1:].mean(1)], dim=1)


@torch.no_grad()
def patch_tokens(model, images):
    """dino.txt head patch tokens for normalised images (B,3,H,W) -> (B, H/16, W/16, 1024)."""
    _, patches, _ = model.visual_model.get_class_and_patch_tokens(images)
    B, _, H, W = images.shape
    return patches.reshape(B, H // 16, W // 16, -1)
