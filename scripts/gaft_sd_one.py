#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Unlearned Checkpoint -> SD2.1 end-to-end (with correct ImageNet label-index alignment):

- Load unlearned checkpoint (GAFT/FT/SSD/Salun/… ) into an OpenCLIP model.
- Optional quantitative CLIP eval:
    • CelebA faces (--eval-mode celeba, --eval-target-name "Elon_Musk")
    • ImageNet objects (--eval-mode imagenet, --eval-target-name "orange")
      ✅ FIX: classifier & targets are aligned to ImageFolder’s class order.
- Load Stable Diffusion 2.1; port OpenCLIP text weights into the HF text encoder.
- Save before/after SD images.

Examples
CelebA:
  python scripts/evaluate_unlearned_sd.py \
    --ckpt /path/to/checkpoint.pt \
    --model ViT-H-14 --pretrained laion2B-s32B-b79K \
    --prompt "A portrait photo of {concept}" \
    --eval-mode celeba --eval-target-name "Elon_Musk"

ImageNet:
  python scripts/evaluate_unlearned_sd.py \
    --ckpt /path/to/checkpoint.pt \
    --model ViT-H-14 --pretrained laion2B-s32B-b79K \
    --prompt "a photo of a {concept}" \
    --eval-mode imagenet --eval-target-name "orange" \
    --imagenet-val /path/to/ImageNet/val \
    --imagenet-map-json /path/to/imagenet_class_index.json
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Union, Dict, Tuple
from itertools import islice

# ---------- Make your repo's src importable ----------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]      # .../SLUG
SRC_ROOT  = REPO_ROOT / "src"         # .../SLUG/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# -----------------------------------------------------

os.environ["DIFFUSERS_OFFLOAD_STATE_DICT"] = "0"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPProcessor

# OpenCLIP (vendored in your repo)
from clip import open_clip


# ========= CONFIGS =========
CELEBA_ROOT = REPO_ROOT / "data/celeba"
CELEBA_EVAL_NAMES =["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Kanye West", "Chris Brown", "Bruno Mars", "Kim Kardashian", "Taylor Swift", "Ariana Grande"]
# A small, diverse retain subset for ImageNet (you can edit this)
# ["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Kanye West", "Chris Brown", "Bruno Mars", "Kim Kardashian", "Taylor Swift", "Ariana Grande"]
# A small, diverse retain subset for ImageNet (you can edit this)
IMAGENET_RETAIN_CLASSES = [
    "banana", "orange", "strawberry", "broccoli", "cucumber",
    "golden retriever", "tabby", "tiger", "bald eagle", "penguin",
    "sports car", "motor scooter", "container ship", "airliner",
    "laptop", "backpack", "umbrella", "dining table", "television", "football", "baseball", 
]
# ==========================


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate unlearned OpenCLIP checkpoint on SD and optional classification.")
    # --- Checkpoint / OpenCLIP ---
    p.add_argument("--ckpt", type=str, default="/home/rania/SLUG/src/clip/ckpt/gaft_mark_1e-6/checkpoints/epoch_10.pt")
    p.add_argument("--model", type=str, default="ViT-H-14", help="OpenCLIP model name (e.g., ViT-H-14)")
    p.add_argument("--pretrained", type=str, default="laion2B-s32B-b79K", help="OpenCLIP pretrained tag")

    # --- Stable Diffusion ---
    p.add_argument("--sd-id", type=str, default="stabilityai/stable-diffusion-2-1", help="Stable Diffusion model id")
    p.add_argument("--prompt", type=str, default="A portrait photo of {concept}", help="Prompt; '{concept}' is replaced")
    p.add_argument("--steps", type=int, default=30, help="Sampling steps")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    p.add_argument("--seed", type=int, default=51, help="Seed for SD generation")

    # --- Evaluation Mode ---
    p.add_argument("--eval-mode", type=str, default="celeba", choices=["none", "celeba", "imagenet"],
                   help="Run quantitative eval or skip.")
    p.add_argument("--eval-target-name", type=str, default="Mark_Zuckerberg",
                   help="Concept name for Forget Acc (e.g., 'Elon_Musk' or 'apple').")

    # --- CelebA Args ---
    p.add_argument("--celeba-root", type=str, default=str(CELEBA_ROOT),
                   help="CelebA root with img_align_celeba/ and list_identity_celeba.txt")

    # --- ImageNet Args ---
    p.add_argument("--imagenet-val", type=str, default=str(REPO_ROOT / "data/ImageNet/val"),
                   help="ImageNet val directory (contains nXXXXXXXX subfolders).")
    p.add_argument("--imagenet-map-json", type=str, default=str(REPO_ROOT / "data/ImageNet/imagenet_class_index.json"),
                   help="JSON mapping like {'0': ['n01440764','tench'], ...}")
    p.add_argument("--imagenet-batch-size", type=int, default=64)
    p.add_argument("--imagenet-workers", type=int, default=4)

    # --- Output ---
    p.add_argument("--outdir", type=str, default=str(REPO_ROOT / "results/Visual_Results/mark/GAFT/seed51"),
                   help="Base directory to save outputs (subdirs created automatically).")

    # --- System ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    # Validate & prepare paths/labels
    args.ckpt = Path(args.ckpt)
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    if args.eval_mode != "none" and not args.eval_target_name:
        p.error("--eval-target-name is required for celeba/imagenet eval.")

    if args.eval_mode == "celeba":
        args.celeba_root = Path(args.celeba_root)
        if not (args.celeba_root / "list_identity_celeba.txt").exists():
            raise FileNotFoundError(f"list_identity_celeba.txt not found in {args.celeba_root}")
        args.celeb_name = args.eval_target_name.replace(' ', '_')
        args.eval_target_name_display = args.eval_target_name.replace('_', ' ')
        if args.eval_target_name_display not in CELEBA_EVAL_NAMES:
            print(f"[Warn] '{args.eval_target_name_display}' not in default CELEBA_EVAL_NAMES (OK if you know it exists).")

    elif args.eval_mode == "imagenet":
        args.imagenet_val = Path(args.imagenet_val)
        args.imagenet_map_json = Path(args.imagenet_map_json)
        if not args.imagenet_val.is_dir():
            raise FileNotFoundError(f"ImageNet val directory not found: {args.imagenet_val}")
        if not args.imagenet_map_json.is_file():
            raise FileNotFoundError(f"ImageNet class map JSON not found: {args.imagenet_map_json}")
        args.eval_target_name_display = args.eval_target_name.lower().replace('_', ' ')
    else:
        args.eval_target_name_display = (args.eval_target_name or "concept").replace('_', ' ')
        if args.prompt == "a photo of a {concept}":
            print("[Warn] --eval-mode none with default prompt; '{concept}' will be left as given.")

    # Output dir per concept
    concept_token = (args.eval_target_name or "general").replace(' ', '_')
    args.outdir = Path(args.outdir) / f"{args.model}_{args.pretrained.replace('/', '_')}" / concept_token
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Format prompt
    args.prompt = args.prompt.format(concept=args.eval_target_name_display)
    return args


# ----------------- Helpers & Porting -----------------

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def render(pipe: StableDiffusionPipeline, prompt: str, steps: int, guidance: float, generator) -> Image.Image:
    return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]

@torch.no_grad()
def encode_text(model, tokenizer, text: str, device):
    toks = tokenizer([text]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0]

def _to_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return src.to(device=ref.device, dtype=ref.dtype, non_blocking=True)

@torch.no_grad()
def port_openclip_text_to_hf(openclip_model, te_hf):
    oc_state = dict(openclip_model.named_parameters())
    oc_state.update(dict(openclip_model.named_buffers()))
    mapped_params = 0

    # Embeddings
    if "token_embedding.weight" in oc_state:
        te_w = te_hf.embeddings.token_embedding.weight
        if te_w.shape == oc_state["token_embedding.weight"].shape:
            te_w.data.copy_(_to_like(oc_state["token_embedding.weight"], te_w)); mapped_params += te_w.numel()
        else:
            print(f"[WARN] token_embedding mismatch: {te_w.shape} vs {oc_state['token_embedding.weight'].shape}")
    else:
        print("[WARN] token_embedding.weight not found in OpenCLIP.")

    if "positional_embedding" in oc_state:
        pe_w = te_hf.embeddings.position_embedding.weight
        if pe_w.shape == oc_state["positional_embedding"].shape:
            pe_w.data.copy_(_to_like(oc_state["positional_embedding"], pe_w)); mapped_params += pe_w.numel()
        else:
            print(f"[WARN] positional_embedding mismatch: {pe_w.shape} vs {oc_state['positional_embedding'].shape}")
    else:
        print("[WARN] positional_embedding not found in OpenCLIP.")

    # Transformer blocks
    oc_blocks = openclip_model.transformer.resblocks
    hf_layers = te_hf.encoder.layers
    n_oc, n_hf, n_map = len(oc_blocks), len(hf_layers), min(len(oc_blocks), len(hf_layers))
    if n_oc != n_hf:
        print(f"[INFO] Layer-count mismatch: OpenCLIP={n_oc}, HF={n_hf}. Mapping first {n_map}.")

    for i in range(n_map):
        try:
            # LN1
            ln1_w = hf_layers[i].layer_norm1.weight; ln1_b = hf_layers[i].layer_norm1.bias
            ln1_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_1.weight"], ln1_w))
            ln1_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_1.bias"],  ln1_b)); mapped_params += ln1_w.numel()+ln1_b.numel()

            # QKV
            attn_prefix = f"transformer.resblocks.{i}.attn"
            has_in_proj = (attn_prefix + ".in_proj_weight") in oc_state
            if has_in_proj:
                in_w = oc_state[attn_prefix + ".in_proj_weight"]; in_b = oc_state[attn_prefix + ".in_proj_bias"]
                q_w, k_w, v_w = in_w.chunk(3, dim=0); q_b, k_b, v_b = in_b.chunk(3, dim=0)
            else:
                q_w = oc_state[attn_prefix + ".q_proj.weight"]; q_b = oc_state[attn_prefix + ".q_proj.bias"]
                k_w = oc_state[attn_prefix + ".k_proj.weight"]; k_b = oc_state[attn_prefix + ".k_proj.bias"]
                v_w = oc_state[attn_prefix + ".v_proj.weight"]; v_b = oc_state[attn_prefix + ".v_proj.bias"]

            hf_q_w = hf_layers[i].self_attn.q_proj.weight; hf_q_b = hf_layers[i].self_attn.q_proj.bias
            hf_k_w = hf_layers[i].self_attn.k_proj.weight; hf_k_b = hf_layers[i].self_attn.k_proj.bias
            hf_v_w = hf_layers[i].self_attn.v_proj.weight; hf_v_b = hf_layers[i].self_attn.v_proj.bias
            hf_q_w.data.copy_(_to_like(q_w, hf_q_w)); hf_q_b.data.copy_(_to_like(q_b, hf_q_b))
            hf_k_w.data.copy_(_to_like(k_w, hf_k_w)); hf_k_b.data.copy_(_to_like(k_b, hf_k_b))
            hf_v_w.data.copy_(_to_like(v_w, hf_v_w)); hf_v_b.data.copy_(_to_like(v_b, hf_v_b))
            mapped_params += q_w.numel()+q_b.numel()+k_w.numel()+k_b.numel()+v_w.numel()+v_b.numel()

            # Out proj
            out_w = oc_state[attn_prefix + ".out_proj.weight"]; out_b = oc_state[attn_prefix + ".out_proj.bias"]
            hf_out_w = hf_layers[i].self_attn.out_proj.weight; hf_out_b = hf_layers[i].self_attn.out_proj.bias
            hf_out_w.data.copy_(_to_like(out_w, hf_out_w)); hf_out_b.data.copy_(_to_like(out_b, hf_out_b)); mapped_params += out_w.numel()+out_b.numel()

            # LN2
            ln2_w = hf_layers[i].layer_norm2.weight; ln2_b = hf_layers[i].layer_norm2.bias
            ln2_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_2.weight"], ln2_w))
            ln2_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_2.bias"],  ln2_b)); mapped_params += ln2_w.numel()+ln2_b.numel()

            # MLP
            fc1_w = hf_layers[i].mlp.fc1.weight; fc1_b = hf_layers[i].mlp.fc1.bias
            fc2_w = hf_layers[i].mlp.fc2.weight; fc2_b = hf_layers[i].mlp.fc2.bias
            fc1_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_fc.weight"],  fc1_w))
            fc1_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_fc.bias"],   fc1_b))
            fc2_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_proj.weight"], fc2_w))
            fc2_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_proj.bias"],  fc2_b))
            mapped_params += fc1_w.numel()+fc1_b.numel()+fc2_w.numel()+fc2_b.numel()

        except KeyError as e:
            raise KeyError(f"[Porting] Missing key at layer {i}: {e}") from e

    # Final LN
    try:
        fln_w = te_hf.final_layer_norm.weight; fln_b = te_hf.final_layer_norm.bias
        fln_w.data.copy_(_to_like(oc_state["ln_final.weight"], fln_w))
        fln_b.data.copy_(_to_like(oc_state["ln_final.bias"],   fln_b)); mapped_params += fln_w.numel()+fln_b.numel()
    except KeyError as e:
        raise KeyError(f"[Porting] Missing final LN key: {e}") from e

    print(f"[DEBUG] Total elements mapped into HF text encoder: {mapped_params:,}")
    return n_map, n_oc, n_hf


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def build_zero_shot_classifier(
    model,
    tokenizer,
    classnames: Sequence[str],
    templates: Sequence[Union[Callable, str]],
    num_classes_per_batch: Optional[int] = 50,
    device: Union[str, torch.device] = "cpu",
    use_tqdm: bool = False,
):
    """Builds zeroshot classifier weights in the class order provided (CRITICAL!)."""
    assert len(classnames) > 0
    assert len(templates) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)

    def _process_batch(batch_classnames):
        texts_local = [tpl.format(c) if use_format else tpl(c)
                       for c in batch_classnames for tpl in templates]
        toks = tokenizer(texts_local).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(isinstance(device, torch.device) and device.type == 'cuda')):
            class_embeds = model.encode_text(toks)
        class_embeds = class_embeds.reshape(len(batch_classnames), num_templates, -1).mean(dim=1)
        class_embeds = F.normalize(class_embeds, dim=-1)
        return class_embeds.T  # [D, num_classes_in_batch]

    chunks = (batched(classnames, num_classes_per_batch)
              if num_classes_per_batch else [classnames])
    parts = []
    for batch in (tqdm(chunks, desc="ZS classifier") if use_tqdm else chunks):
        parts.append(_process_batch(list(batch)))
    return torch.cat(parts, dim=1)


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]


# ----------------- CELEBA EVAL -----------------

def run_celeba_eval(model_openclip, classifier_celeba, args, hf_processor, device, jpg_dict, name_list):
    print("[INFO] Evaluating CelebA subset...")
    test_top1, test_top5 = [], []
    forget_t1, forget_t5 = 0.0, 0.0

    for name_with_space in CELEBA_EVAL_NAMES:
        name_us = name_with_space.replace(' ', '_')
        if name_us not in name_list:
            print(f"[Warn] '{name_us}' not found in CelebA. Skipping.")
            continue
        label = name_list.index(name_us)
        top1 = top5 = n = 0
        for image_id in jpg_dict.get(name_us, []):
            img_path = args.celeba_root / "img_align_celeba" / image_id
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                continue

            target = torch.tensor([label], device=device)
            inputs = hf_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                image_feats = model_openclip.encode_image(inputs['pixel_values'])
                image_feats = F.normalize(image_feats, dim=-1)
                logits = (model_openclip.logit_scale.exp() * image_feats @ classifier_celeba)
            a1, a5 = accuracy(logits, target, topk=(1, 5))
            top1 += a1; top5 += a5; n += 1

        if n > 0:
            t1 = top1 / n; t5 = top5 / n
            print(f"  - {name_us}: top1={t1*100:.2f}% top5={t5*100:.2f}% ({int(top1)}/{n})")
            if name_us == args.celeb_name:
                forget_t1, forget_t5 = t1, t5
            else:
                test_top1.append(t1); test_top5.append(t5)

    retain_t1 = float(np.mean(test_top1)) if test_top1 else 0.0
    retain_t5 = float(np.mean(test_top5)) if test_top5 else 0.0
    return forget_t1, forget_t5, retain_t1, retain_t5, len(test_top1)


# ----------------- IMAGENET EVAL (FIXED) -----------------

def _normalize_name(s: str) -> str:
    return s.lower().replace('_', ' ').strip()

def run_imagenet_eval(
    model_openclip,
    args,
    preprocess_fn,           # from create_model_and_transforms
    imagenet_map,            # dict with id/index/name maps (built below)
    device,
):
    """
    Correct, label-aligned ImageNet evaluation.

    - Build dataset with ImageFolder (targets are DATASET indices in alphabetical WNID order).
    - Build the zeroshot classifier using CLASSNAMES ALIGNED TO dataset.classes.
    - Tally per-class results by DATASET index.
    - Convert concept names to DATASET indices via (name -> imagenet idx -> WNID -> dataset idx).
    """
    print("[INFO] Loading ImageNet dataset...")
    dataset = datasets.ImageFolder(root=str(args.imagenet_val), transform=preprocess_fn)
    loader = DataLoader(dataset, batch_size=args.imagenet_batch_size, shuffle=False,
                        num_workers=args.imagenet_workers, pin_memory=True, drop_last=False)
    print(f"[INFO] ImageNet val images: {len(dataset)} | classes: {len(dataset.classes)}")

    # Build classnames in the SAME order as dataset.classes (alphabetical WNIDs)
    ordered_wnids = dataset.classes  # list of WNIDs in dataset target order

    def wnid_to_primary_name(wnid: str) -> str:
        idx = imagenet_map['id_to_index'][wnid]              # 0..999 ImageNet index
        name_str = imagenet_map['index_to_name'][str(idx)][1]  # "label1, label2, ..."
        return name_str.split(',')[0].strip()

    classnames_aligned = [wnid_to_primary_name(w) for w in ordered_wnids]
    templates = (lambda c: f"a photo of a {c}.",)

    print("[INFO] Building ImageNet zero-shot classifier (aligned to dataset indices)...")
    tokenizer = open_clip.get_tokenizer(args.model)
    classifier = build_zero_shot_classifier(model_openclip, tokenizer, classnames_aligned, templates,
                                            device=device, use_tqdm=True)

    # ---- Eval loop (per-class tallies in DATASET index space) ----
    k = 5
    per_class = defaultdict(lambda: [0, 0, 0])  # ds_idx -> [correct@1, correct@5, total]
    total = 0
    model_openclip.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating ImageNet"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)  # DATASET indices!

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                image_feats = model_openclip.encode_image(images)
                image_feats = F.normalize(image_feats, dim=-1)
                logits = (model_openclip.logit_scale.exp() * image_feats @ classifier)

            preds = logits.topk(k, 1, True, True)[1].t()  # indices in classifier == DATASET indices
            correct = preds.eq(targets.view(1, -1).expand_as(preds))  # SAME index space

            # Update per-class tallies
            for b in range(targets.size(0)):
                ds_idx = int(targets[b].item())
                per_class[ds_idx][0] += int(correct[0, b].item())           # top-1
                per_class[ds_idx][1] += int(correct[:k, b].any().item())    # top-5
                per_class[ds_idx][2] += 1
            total += images.size(0)

    print(f"[INFO] Processed {total} images across {len(per_class)} classes.")

    # ---- Resolve forget & retain classes via DATASET index ----
    def name_to_dataset_index(name: str) -> Optional[int]:
        nm = _normalize_name(name)
        imgnet_idx = imagenet_map['norm_name_to_index'].get(nm, None)  # 0..999 index
        if imgnet_idx is None:
            return None
        wnid = imagenet_map['index_to_name'][str(imgnet_idx)][0]  # WNID
        return dataset.class_to_idx.get(wnid, None)               # DATASET index

    # Forget class
    forget_ds_idx = name_to_dataset_index(args.eval_target_name_display)
    forget_acc1 = forget_acc5 = 0.0
    if forget_ds_idx is None:
        print(f"[ERROR] Could not resolve forget concept '{args.eval_target_name_display}' to dataset index.")
    else:
        c1, c5, tot = per_class.get(forget_ds_idx, (0, 0, 0))
        if tot > 0:
            forget_acc1 = c1 / tot
            forget_acc5 = c5 / tot
        label_str = classnames_aligned[forget_ds_idx] if 0 <= forget_ds_idx < len(classnames_aligned) else "unknown"
        print(f"  - Forget Class ({label_str}): Top-1: {forget_acc1*100:.2f}% ({c1}/{tot}), Top-5: {forget_acc5*100:.2f}% ({c5}/{tot})")

    # Retain classes
    retain_acc1_list, retain_acc5_list, found = [], [], 0
    for nm in IMAGENET_RETAIN_CLASSES:
        ds_idx = name_to_dataset_index(nm)
        if ds_idx is None:
            continue
        c1, c5, tot = per_class.get(ds_idx, (0, 0, 0))
        if tot > 0:
            retain_acc1_list.append(c1 / tot)
            retain_acc5_list.append(c5 / tot)
            found += 1
    avg_retain_acc1 = float(np.mean(retain_acc1_list)) if retain_acc1_list else 0.0
    avg_retain_acc5 = float(np.mean(retain_acc5_list)) if retain_acc5_list else 0.0

    print(f"[INFO] Calculated retain average over {found} classes.")
    return forget_acc1, forget_acc5, avg_retain_acc1, avg_retain_acc5, found


# ----------------- MAIN -----------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Saving outputs to: {args.outdir}")

    # 1) Load OpenCLIP model + transforms
    print(f"[INFO] Creating OpenCLIP model: {args.model} ({args.pretrained})")
    model_unlearned, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model_unlearned.eval()

    # 2) Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        print("[INFO] Stripping 'module.' prefix.")
        sd = {k[len("module."):]: v for k, v in sd.items()}
    load_res = model_unlearned.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict: missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}")

    # 3) Optional quantitative eval
    if args.eval_mode != "none":
        print(f"\n[INFO] Running quantitative CLIP evaluation (mode: {args.eval_mode})...")
        if args.eval_mode == "celeba":
            # Load CelebA identities
            jpg_dict = defaultdict(list)
            id_file = Path(args.celeba_root) / "list_identity_celeba.txt"
            with open(id_file, "r") as f:
                lines = f.readlines()
            for line in lines[2:]:
                img_id, identity = line.strip().split()
                jpg_dict[identity].append(img_id)
            name_list = tuple(sorted(jpg_dict.keys()))
            CELEB_NAMES_ALL = [n.replace('_', ' ') for n in name_list]
            CELEBA_TEMPLATES = (lambda c: f"a photo of {c}.",)
            print("[INFO] Building CelebA classifier...")
            classifier_celeba = build_zero_shot_classifier(
                model_unlearned, tokenizer, CELEB_NAMES_ALL, CELEBA_TEMPLATES, device=device, use_tqdm=True
            )
            hf_processor = CLIPProcessor.from_pretrained(f"laion/CLIP-{args.model}-{args.pretrained}")
            f1, f5, r1, r5, nret = run_celeba_eval(
                model_unlearned, classifier_celeba, args, hf_processor, device, jpg_dict, name_list
            )

        elif args.eval_mode == "imagenet":
            # Build ImageNet name maps
            with open(args.imagenet_map_json, "r") as f:
                idx_to = json.load(f)  # {"0": ["n01440764","tench"], ...}
            imagenet_map = {
                "index_to_name": idx_to,                                   # str idx -> [wnid, names str]
                "id_to_index": {v[0]: int(k) for k, v in idx_to.items()},  # wnid -> idx
                "index_to_id": {int(k): v[0] for k, v in idx_to.items()},  # idx -> wnid
                "norm_name_to_index": {}
            }
            # Build normalized name -> idx using all synonyms
            for k, (wnid, names) in idx_to.items():
                idx = int(k)
                for nm in [t.strip() for t in names.split(",")]:
                    imagenet_map["norm_name_to_index"][_normalize_name(nm)] = idx

            f1, f5, r1, r5, nret = run_imagenet_eval(
                model_unlearned, args, preprocess_val, imagenet_map, device
            )
        else:
            f1 = f5 = r1 = r5 = nret = 0.0

        print("\n--- Quantitative Results ---")
        print(f"Forget Accuracy ({args.eval_target_name_display}):")
        print(f"  Top-1: {f1*100:.2f}%")
        print(f"  Top-5: {f5*100:.2f}%")
        print(f"Retain Accuracy (Avg. over {nret} classes):")
        print(f"  Top-1: {r1*100:.2f}%")
        print(f"  Top-5: {r5*100:.2f}%")
        print("----------------------------\n")

    # 4) Free VRAM before SD
    print("[INFO] Moving OpenCLIP to CPU before SD load...")
    model_unlearned.to("cpu"); torch.cuda.empty_cache()
    try: del classifier_celeba
    except: pass
    try: del f1, f5, r1, r5, nret
    except: pass
    import gc; gc.collect(); torch.cuda.empty_cache()

    # 5) Load SD 2.1
    print(f"[INFO] Loading SD pipeline: {args.sd_id}")
    te_hf_base = CLIPTextModel.from_pretrained(args.sd_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_id, torch_dtype=torch.float16, text_encoder=te_hf_base, low_cpu_mem_usage=False, device_map=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    extra_prompts = [
        
        ("A portrait photo of a Elon Musk", "musk"),
        ("A portrait photo of a Jeff Bezos", "bezos"),
        ("A portrait photo of a Sundar Pichai", "sundar"),
        ("A portrait photo of a Bill Gates", "gates"),
        ("A portrait photo of a Tim Cook", "cook"),
        ("A Facebook login page on a laptop screen", "facebook_login"),
        ("A person scrolling through their Facebook news feed", "facebook_feed"),
        ("A Meta Quest VR headset on a table", "meta_quest"),
        ("A simple virtual reality room with floating menus", "vr_room"),
        ("A software engineer working on code at a desk", "tech_engineer"),
        ("A team meeting inside a modern tech office", "tech_meeting"),






        
    ]

    # 6) Baseline render
    print("[INFO] Rendering baseline (original SD)...")
    gen_base = torch.Generator(device=device).manual_seed(args.seed)
    img_before = render(pipe, args.prompt, args.steps, args.guidance, gen_base)
    img_before.save(args.outdir / "sd_before_swap.png")
    print(f"[INFO] Saved: {args.outdir / 'sd_before_swap.png'}")
    print("[INFO] Rendering extra prompts (baseline)…")
    for i, (ptext, tag) in enumerate(extra_prompts):
        gen_extra_base = torch.Generator(device=device).manual_seed(args.seed + 1 + i)
        img_base = render(pipe, ptext, args.steps, args.guidance, gen_extra_base)
        img_base.save(args.outdir / f"sd_before_swap_{tag}.png")
        outp = args.outdir / f"sd_before_swap_{tag}.png"
        img_base.save(outp)
        print(f"  - Saved: {outp}")

    # 7) Port text weights
    print("[INFO] Porting OpenCLIP text weights -> SD text encoder...")
    n_map, n_oc, n_hf = port_openclip_text_to_hf(model_unlearned, pipe.text_encoder.text_model)
    print(f"[INFO] Port complete. Mapped {n_map} layers (OpenCLIP={n_oc}, HF={n_hf}).")

    # 8) Swapped render(s)
    print(f"[INFO] Rendering swapped ('{args.prompt}')...")
    gen_swap = torch.Generator(device=device).manual_seed(args.seed)
    img_after = render(pipe, args.prompt, args.steps, args.guidance, gen_swap)
    swap_name = f"sd_after_swap_{args.eval_target_name_display.replace(' ', '_')}.png"
    img_after.save(args.outdir / swap_name)
    print(f"[INFO] Saved: {args.outdir / swap_name}")

    print("[INFO] Rendering extra prompts (swapped)...")
    for i, (ptext, tag) in enumerate(extra_prompts):
        gen_extra = torch.Generator(device=device).manual_seed(args.seed + 1 + i)
        img = render(pipe, ptext, args.steps, args.guidance, gen_extra)
        fname = f"sd_after_swap_{tag}.png"
        img.save(args.outdir / fname)
        print(f"  - Saved: {args.outdir / fname}")

    print("\n[RESULT]")
    print(f"Saved images under: {args.outdir}")
    print("Done.")

if __name__ == "__main__":
    main()
