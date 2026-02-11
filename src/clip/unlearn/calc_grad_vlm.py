#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, json, logging, argparse, random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Optional: webdataset support if your data is in .tar shards (LAION-like)
try:
    import webdataset as wds
    HAS_WDS = True
except Exception:
    HAS_WDS = False

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)

# -----------------------------
# Datasets
# -----------------------------

class FolderImageText(Dataset):
    """
    Very simple folder dataset:
      - images under {root}/images/**/*.jpg|png|jpeg
      - a single text prompt per dataset (e.g., "{celeb_name.replace('_',' ')}.")
    If you instead have per-image captions, you can put a .txt next to each image with the same stem.
    """
    def __init__(self, root: str, text_prompt: str, exts=(".jpg",".jpeg",".png",".webp")):
        self.root = Path(root)
        self.exts = exts
        self.files = []
        for ext in exts:
            self.files += list(self.root.rglob(f"*{ext}"))
        self.files = sorted(self.files)
        self.text_prompt = text_prompt

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        # If a sidecar .txt exists, use it; otherwise use the constant text_prompt
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            txt = sidecar.read_text().strip()
        else:
            txt = self.text_prompt
        return img, txt


def build_loader(split_path: str, text_prompt: str, batch_size: int, num_workers: int, processor: CLIPProcessor):
    """
    Auto-detect: if path endswith .tar -> webdataset, else -> folder dataset.
    """
    if split_path.endswith(".tar"):
        if not HAS_WDS:
            raise RuntimeError("webdataset is not installed; pip install webdataset")
        # expected samples: {'jpg' or 'png', 'txt' (optional)}
        def _preproc(sample):
            # sample like: {'__key__':..., 'jpg': bytes, 'txt': str}
            img_bytes = sample.get("jpg", sample.get("png"))
            img = Image.open(img_bytes).convert("RGB")
            txt = sample.get("txt")
            if txt is None or (isinstance(txt, (list, tuple)) and len(txt) == 0):
                txt = text_prompt
            elif isinstance(txt, (list, tuple)):
                txt = txt[0]
            return img, txt

        dataset = (
            wds.WebDataset(split_path, shardshuffle=True)
              .decode("pil")
              .to_tuple("jpg;png", "txt")
              .map(lambda x: (x[0], x[1] if isinstance(x[1], str) else (x[1][0] if len(x[1])>0 else text_prompt)))
              .map(lambda x: (x[0].convert("RGB"), x[1]))
        )
        # We'll collate via processor inside the loop (no pretokenization here)
        loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None).batched(batch_size)
        return loader, True
    else:
        ds = FolderImageText(split_path, text_prompt=text_prompt)
        # We'll return PILs and raw strings; processing happens in the loop
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return loader, False

# -----------------------------
# CLIP Loss (InfoNCE)
# -----------------------------

def clip_contrastive_loss(image_embeds: torch.Tensor,
                          text_embeds: torch.Tensor,
                          logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Standard CLIP loss exactly like OpenAI / HF:
    - normalize features
    - compute logits = scale * cosine_sim
    - InfoNCE both ways and average
    """
    # (N, D) normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text  = logits_per_image.t()
    targets = torch.arange(len(image_embeds), device=image_embeds.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text,  targets)
    return (loss_i + loss_t) / 2.0

# -----------------------------
# Grad collection
# -----------------------------

def average_gradients(named_params, grads_accum, denom: float):
    for name, p in named_params:
        if name in grads_accum and grads_accum[name] is not None:
            grads_accum[name] /= max(1.0, denom)

def zero_like_params(model: nn.Module, device) -> dict:
    return {k: torch.zeros_like(p, device=device) for k, p in model.named_parameters()}

def add_gradients_inplace(grads_dict: dict, model: nn.Module, norm: Optional[str]):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if norm is None:
            grads_dict[name] += p.grad
        elif norm == "l2":
            grads_dict[name] += p.grad.pow(2)

def save_grads(grads_dict: dict, save_root: Path, split: str, unlearn_suffix: str, norm: Optional[str]):
    save_root.mkdir(parents=True, exist_ok=True)
    if norm is None:
        fname = f"{split}_grads{unlearn_suffix}.pt"
    else:
        fname = f"{split}_importance{unlearn_suffix}.pt"
    path = save_root / fname
    torch.save(grads_dict, path)
    logging.info(f"Saved {split} gradients to {path}")

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train-data", type=str, required=True, help="retain set: folder or .tar")
    ap.add_argument("--forget-data", type=str, required=True, help="forget set: folder or .tar")
    ap.add_argument("--celeb-name", type=str, required=True, help="e.g. Elon_Musk")

    # model
    ap.add_argument("--model-id", type=str, default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32","bf16","fp16"])

    # optimization-ish
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)

    # misc / output
    ap.add_argument("--result-dir", type=str, default="results")
    ap.add_argument("--norm", type=str, default=None, choices=[None, "l2"], help="if set, accumulates L2(grad) as 'importance'")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + processor
    model = CLIPModel.from_pretrained(args.model-id if hasattr(args, "model-id") else args.model_id)
    processor = CLIPProcessor.from_pretrained(args.model-id if hasattr(args, "model-id") else args.model_id)
    model = model.to(device)

    # Precision
    if args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None  # fp32

    # Data
    celeb_text = args.celeb_name.replace("_", " ") + "."
    dl_train, train_is_wds   = build_loader(args.train_data,  celeb_text, args.batch_size, args.workers, processor)
    dl_forget, forget_is_wds = build_loader(args.forget_data, celeb_text, args.batch_size, args.workers, processor)

    # Where to save
    model_repo = "openai"
    model_name = args.model_id.split("/")[-1] if "/" in args.model_id else args.model_id
    save_root = Path(args.result_dir) / "grads" / f"{args.celeb_name}_{model_repo}_{model_name}"

    # ---------- PASS 1: FORGET ----------
    model.train()
    grads_forget = zero_like_params(model, device=device)
    n_steps = 0

    for batch in dl_forget:
        # batch can be list of tuples (folder) or dict (wds batched)
        if forget_is_wds:
            # webdataset returns a list/tuple of (images, texts) batched already
            images, texts = batch
        else:
            images, texts = batch  # list of PILs and list of strings

        # Build processor inputs (we do this every step so we can mix sources)
        inputs = processor(text=texts, images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward
        with torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device.type=="cuda") else torch.enable_grad():
            out = model(**inputs)
            # Pooled embeddings (already normalized by HF; we re-normalize defensively)
            img_feat = out.image_embeds
            txt_feat = out.text_embeds
            # CosineEmbeddingLoss (same as your notebook), then NEGATE for forget
            target = torch.ones(img_feat.size(0), device=device)
            f_loss = nn.CosineEmbeddingLoss()(img_feat, txt_feat, target)
            total = f_loss  # NEGATE for forget split

        model.zero_grad(set_to_none=True)
        total.backward()
        add_gradients_inplace(grads_forget, model, norm=args.norm)
        n_steps += 1

    average_gradients(model.named_parameters(), grads_forget, denom=n_steps)
    unlearn_suffix = ""  # keep filenames compatible with your script; add "_o" if you use the *_o variant
    save_grads(grads_forget, save_root, split="forget", unlearn_suffix=unlearn_suffix, norm=args.norm)

    # ---------- PASS 2: TRAIN/RETAIN ----------
    model.train()
    grads_train = zero_like_params(model, device=device)
    n_steps = 0

    for batch in dl_train:
        if train_is_wds:
            images, texts = batch
        else:
            images, texts = batch

        inputs = processor(text=texts, images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device.type=="cuda") else torch.enable_grad():
            out = model(**inputs)
            img_feat = out.image_embeds
            txt_feat = out.text_embeds
            logit_scale = model.logit_scale.exp()
            t_loss = clip_contrastive_loss(img_feat, txt_feat, logit_scale)

        model.zero_grad(set_to_none=True)
        t_loss.backward()
        add_gradients_inplace(grads_train, model, norm=args.norm)
        n_steps += 1

    average_gradients(model.named_parameters(), grads_train, denom=n_steps)
    save_grads(grads_train, save_root, split="train", unlearn_suffix=unlearn_suffix, norm=args.norm)

    logging.info("All done.")

if __name__ == "__main__":
    # Allow either --model-id or --model_id on CLI
    import sys
    sys.argv = [a.replace("--model-id","--model_id") for a in sys.argv]
    main()
