#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute CLIP (HF) gradients for 'forget' (celebrity) and 'train/retain' splits from WebDataset tars.

Outputs:
  <result_dir>/grads/{CELEB}_{repo}_{name}/forget_grads.pt
  <result_dir>/grads/{CELEB}_{repo}_{name}/train_grads.pt

Notes:
- 'forget' uses NEGATIVE cosine-embedding loss between image/text embeddings (your convention)
- 'train' uses HF CLIP contrastive loss (return_loss=True)
- No intermediate checkpoints; only final averaged gradients are saved
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import webdataset as wds

from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
import argparse


# -------------------- WebDataset helpers --------------------

def log_and_continue(exn):
    logging.warning(f'WebDataset warning: {repr(exn)} — skipping.')
    return True

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image

def create_wds(input_shards: str, bs: int = 16):
    pipeline = [wds.SimpleShardList(input_shards)]
    pipeline.extend([
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.to_tuple("image", "text"),
        wds.batched(bs, partial=True),
    ])
    dataset = wds.DataPipeline(*pipeline)
    loader = wds.WebLoader(
        dataset,
        batch_size=None,          # batching is done in pipeline
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )
    return loader


# -------------------- gradient utils --------------------

def initialize_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(param, device=param.device)
            for name, param in model.named_parameters()}

def accumulate_gradients(model: nn.Module, gradients: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if p.grad is not None:
            gradients[name] += p.grad

def average_gradients(gradients: Dict[str, torch.Tensor], denom: int) -> None:
    denom = float(max(denom, 1))
    for name in gradients:
        gradients[name] /= denom

def save_gradients_final(gradients: Dict[str, torch.Tensor], save_dir: Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cpu_grads = {k: v.detach().cpu() for k, v in gradients.items()}
    path = save_dir / filename
    torch.save(cpu_grads, path)
    logging.info(f"Saved gradients to {path}")


# -------------------- core split pass --------------------

def process_split(model_clip: CLIPModel,
                  processor_clip: CLIPProcessor,
                  dataloader,
                  celeb_name: str,
                  split: str,
                  device: torch.device,
                  save_root: Path) -> Dict[str, torch.Tensor]:

    gradients = initialize_gradients(model_clip)
    batch_count = 0

    logging.info(f"Processing '{split}' split...")
    model_clip.zero_grad(set_to_none=True)
    model_clip.train(True)

    amp_enabled = (device.type == "cuda")

    for images, texts in tqdm(dataloader, desc=f"{split}"):
        # For 'forget', override captions with celeb name (space instead of underscore)
        if split == 'forget':
            texts = [celeb_name.replace('_', ' ')] * len(texts)

        inputs = processor_clip(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            outputs = model_clip(**inputs, return_loss=True)

            if split == 'forget':
                # Negative cosine (your convention for forget set)
                image_features = outputs.image_embeds
                text_features  = outputs.text_embeds
                total_loss = nn.CosineEmbeddingLoss()(
                    image_features,
                    text_features,
                    torch.ones(len(images), device=device)
                )
                total_loss = -total_loss
            else:
                # Standard CLIP contrastive loss
                total_loss = outputs.loss

        total_loss.float().backward()
        accumulate_gradients(model_clip, gradients)
        batch_count += 1

        model_clip.zero_grad(set_to_none=True)

    # Average by number of batches
    average_gradients(gradients, batch_count)

    # Save final only
    final_fname = "forget_grads.pt" if split == "forget" else "train_grads.pt"
    save_gradients_final(gradients, save_root, final_fname)

    logging.info(f"Done '{split}': {batch_count} batches.")
    return gradients


# -------------------- CLI / main --------------------

def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="Compute final averaged forget/train gradients for a HF CLIP model (WebDataset).")
    p.add_argument("--celeb-name", required=True, help="e.g., Elon_Musk")
    p.add_argument("--clip-model-id", required=True, help='e.g., "openai/clip-vit-large-patch14-336"')
    p.add_argument("--result-dir", default="/home/rania/SLUG/results", help="Root for results/")
    p.add_argument("--forget-data", required=True, help="Path to forget .tar (WebDataset)")
    p.add_argument("--train-data",  required=True, help="Path to retain/train .tar (WebDataset)")
    p.add_argument("--batch-size", type=int, default=16, help="WebDataset microbatch")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_cli(argv)

    # env niceties
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    accelerator = Accelerator()
    device = accelerator.device

    celeb_name    = args.celeb_name
    clip_model_id = args.clip_model_id

    # Load HF CLIP
    model_clip = CLIPModel.from_pretrained(clip_model_id)
    processor_clip = CLIPProcessor.from_pretrained(clip_model_id)
    model_clip.to(device)

    # (Optional) try to enable gradient checkpointing to reduce VRAM
    try:
        model_clip.gradient_checkpointing_enable()
    except Exception:
        try: model_clip.vision_model.gradient_checkpointing = True
        except Exception: pass
        try: model_clip.text_model.gradient_checkpointing = True
        except Exception: pass

    # Output directory convention
    repo, name = clip_model_id.split('/')
    save_root = Path(args.result_dir) / f"grads/{celeb_name}_{repo}_{name}"
    save_root.mkdir(parents=True, exist_ok=True)

    # Build loaders
    forget_loader = create_wds(str(args.forget_data), bs=args.batch_size)
    train_loader  = create_wds(str(args.train_data),  bs=args.batch_size)

    # Collect grads — NO optimizer step, only backward/accumulate/average
    _ = process_split(
        model_clip=model_clip,
        processor_clip=processor_clip,
        dataloader=forget_loader,
        celeb_name=celeb_name,
        split="forget",
        device=device,
        save_root=save_root,
    )

    _ = process_split(
        model_clip=model_clip,
        processor_clip=processor_clip,
        dataloader=train_loader,
        celeb_name=celeb_name,
        split="train",
        device=device,
        save_root=save_root,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(sys.argv[1:])
