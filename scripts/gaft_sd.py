#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAFT → SD2.1 end-to-end:
- Load GAFT checkpoint into an OpenCLIP model (same pair as training).
- **ADDED: Perform quantitative CLIP evaluation (Forget/Retain Acc).**
- Load Stable Diffusion 2.1 and PORT OpenCLIP text weights into the HF text encoder.
- Save before/after images for multiple prompts to visualize unlearning effect.

Usage (from ~/SLUG):
  conda activate mu
  python scripts/gaft_sd.py \
       --ckpt src/clip/ckpt/YOUR_RUN_NAME/checkpoints/epoch_XX.pt \
       --model ViT-H-14 --pretrained laion2B-s32B-b79K \
       --prompt "A portrait photo of Elon Musk" \
       --celeb-name "Elon_Musk" # ADDED: Name used for Forget Acc calculation
"""

import argparse
import os
import sys
from pathlib import Path
from io import BytesIO
from collections import defaultdict # ADDED
from typing import Callable, List, Optional, Sequence, Union, Dict, Tuple # ADDED
from itertools import islice # ADDED


# ---------- Make your repo's src importable ----------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]      # .../SLUG
SRC_ROOT  = REPO_ROOT / "src"         # .../SLUG/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# -----------------------------------------------------

# Diffusers sometimes passes offload flags older transformers can't handle
os.environ["DIFFUSERS_OFFLOAD_STATE_DICT"] = "0"

import torch
import torch.nn.functional as F # ADDED
from torch import nn # ADDED
from PIL import Image
import requests
import numpy as np # ADDED

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPProcessor # ADDED CLIPProcessor

# Import OpenCLIP from your vendored package: src/clip/open_clip
from clip import open_clip


# ========= ADDED: CONFIG FOR EVALUATION =========
# Must contain img_align_celeba/ and list_identity_celeba.txt
CELEBA_ROOT = REPO_ROOT / "data/celeba"
# Names used for quick celeb evals (Retain set)
# Make sure CELEB_NAME (from args) is the first element for easy splitting later
EVAL_TEXTS = ["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Taylor Swift", "Kim Kardashian"]
# ================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="/home/rania/SLUG/src/clip/ckpt/salun_1e-6_elon/checkpoints/epoch_10.pt",
                   help="Path to GAFT checkpoint (epoch_*.pt or epoch_latest.pt). If empty, auto-find latest.")
    p.add_argument("--model", type=str, default="ViT-H-14",
                   help="OpenCLIP model name used in training (e.g., ViT-H-14)")
    p.add_argument("--pretrained", type=str, default="laion2B-s32B-b79K",
                   help="OpenCLIP pretrained tag used in training")
    p.add_argument("--sd-id", type=str, default="stabilityai/stable-diffusion-2-1",
                   help="Stable Diffusion model id")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--prompt", type=str, default="A portrait photo of Elon Musk",
                   help="Main prompt to render before/after")
    p.add_argument("--outdir", type=str, default=str(REPO_ROOT / "results/output-SD2-GAFT_salun"),
                   help="Directory to save outputs")
    p.add_argument("--steps", type=int, default=30, help="Sampling steps")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    # --- ADDED ARGS ---
    p.add_argument("--celeb-name", type=str, default="Elon_Musk",
                   help="Celebrity name used for Forget Acc (e.g., Elon_Musk). Must match EVAL_TEXTS[0] after replacing '_' with ' '.")
    p.add_argument("--celeba-root", type=str, default=str(CELEBA_ROOT),
                   help="Path to CelebA dataset root (contains img_align_celeba/ and list_identity_celeba.txt)")
    # --- END ADDED ARGS ---
    args = p.parse_args()

    # --- ADDED VALIDATION ---
    args.celeba_root = Path(args.celeba_root)
    if not (args.celeba_root / "list_identity_celeba.txt").exists():
         raise FileNotFoundError(f"list_identity_celeba.txt not found in {args.celeba_root}")
    if args.celeb_name.replace('_', ' ') != EVAL_TEXTS[0]:
         print(f"[Warning] --celeb-name '{args.celeb_name}' does not match first element of EVAL_TEXTS ('{EVAL_TEXTS[0]}'). Forget accuracy calculation might be mislabeled.")
    # --- END ADDED VALIDATION ---

    return args


def set_seed(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def find_latest_ckpt() -> Path:
    ckpt_root = SRC_ROOT / "clip" / "ckpt"
    candidates = sorted(
        ckpt_root.rglob("checkpoints/epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}/**/checkpoints/epoch_*.pt")
    print(f"[INFO] Auto-found latest checkpoint: {candidates[0]}")
    return candidates[0]


def render(pipe: StableDiffusionPipeline, prompt: str, steps: int, guidance: float) -> Image.Image:
    return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]


@torch.no_grad()
def encode_text(model, tokenizer, text: str, device):
    toks = tokenizer([text])
    # Use model's text tower directly for embedding
    # Assuming model is an OpenCLIP model instance
    with torch.no_grad(), torch.cuda.amp.autocast(): # Use autocast for potential speedup/memory
         feats = model.encode_text(toks.to(device))
         feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0]  # [D]


# ---------------- MAPPING: OpenCLIP text → HF CLIPTextModel (SD2.1) ----------------
# (This section is unchanged)
def _to_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast & move src to ref's dtype and device (no grad)."""
    return src.to(device=ref.device, dtype=ref.dtype, non_blocking=True)

@torch.no_grad()
def port_openclip_text_to_hf(openclip_model, te_hf):
    """
    Copy text weights from an OpenCLIP model (open_clip.CLIP) into a
    transformers.CLIPTextModel's inner module (te_hf = pipe.text_encoder.text_model).
    Handles known differences. Tolerates layer count mismatch.
    """

    oc = openclip_model
    oc_state = dict(oc.named_parameters())
    oc_state.update(dict(oc.named_buffers())) # Include buffers like positional_embedding

    mapped_params = 0

    # ---- Embeddings ----
    if "token_embedding.weight" in oc_state:
        te_w = te_hf.embeddings.token_embedding.weight
        if te_w.shape == oc_state["token_embedding.weight"].shape:
             te_w.data.copy_(_to_like(oc_state["token_embedding.weight"], te_w))
             mapped_params += te_w.numel()
        else: print(f"[WARN] Skipping token_embedding: shape mismatch {te_w.shape} vs {oc_state['token_embedding.weight'].shape}")
    else: print("[WARN] OpenCLIP token_embedding.weight not found.")

    if "positional_embedding" in oc_state:
        pe_w = te_hf.embeddings.position_embedding.weight
        if pe_w.shape == oc_state["positional_embedding"].shape:
             pe_w.data.copy_(_to_like(oc_state["positional_embedding"], pe_w))
             mapped_params += pe_w.numel()
        else: print(f"[WARN] Skipping positional_embedding: shape mismatch {pe_w.shape} vs {oc_state['positional_embedding'].shape}")

    else: print("[WARN] OpenCLIP positional_embedding not found.")

    # ---- Blocks ----
    oc_blocks = getattr(oc, "transformer").resblocks
    hf_layers = te_hf.encoder.layers

    n_oc  = len(oc_blocks)
    n_hf  = len(hf_layers)
    n_map = min(n_oc, n_hf)

    if n_oc != n_hf:
        print(f"[INFO] Layer-count mismatch: OpenCLIP has {n_oc}, HF has {n_hf}. Mapping first {n_map} layers.")

    for i in range(n_map):
        try:
            # LN1
            ln1_w = hf_layers[i].layer_norm1.weight
            ln1_b = hf_layers[i].layer_norm1.bias
            ln1_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_1.weight"], ln1_w))
            ln1_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_1.bias"],  ln1_b))
            mapped_params += ln1_w.numel() + ln1_b.numel()

            # Attention Projections (handle in_proj vs q/k/v)
            attn_prefix = f"transformer.resblocks.{i}.attn"
            has_in_proj = (attn_prefix + ".in_proj_weight") in oc_state

            if has_in_proj:
                in_w = oc_state[attn_prefix + ".in_proj_weight"]
                in_b = oc_state[attn_prefix + ".in_proj_bias"]
                q_w, k_w, v_w = in_w.chunk(3, dim=0)
                q_b, k_b, v_b = in_b.chunk(3, dim=0)
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
            mapped_params += q_w.numel() + q_b.numel() + k_w.numel() + k_b.numel() + v_w.numel() + v_b.numel()

            # Attention out_proj
            out_w = oc_state[attn_prefix + ".out_proj.weight"]
            out_b = oc_state[attn_prefix + ".out_proj.bias"]
            hf_out_w = hf_layers[i].self_attn.out_proj.weight
            hf_out_b = hf_layers[i].self_attn.out_proj.bias
            hf_out_w.data.copy_(_to_like(out_w, hf_out_w))
            hf_out_b.data.copy_(_to_like(out_b, hf_out_b))
            mapped_params += out_w.numel() + out_b.numel()

            # LN2
            ln2_w = hf_layers[i].layer_norm2.weight
            ln2_b = hf_layers[i].layer_norm2.bias
            ln2_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_2.weight"], ln2_w))
            ln2_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.ln_2.bias"],  ln2_b))
            mapped_params += ln2_w.numel() + ln2_b.numel()

            # MLP (c_fc/c_proj -> fc1/fc2)
            fc1_w = hf_layers[i].mlp.fc1.weight; fc1_b = hf_layers[i].mlp.fc1.bias
            fc2_w = hf_layers[i].mlp.fc2.weight; fc2_b = hf_layers[i].mlp.fc2.bias

            fc1_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_fc.weight"],  fc1_w))
            fc1_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_fc.bias"],   fc1_b))
            fc2_w.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_proj.weight"], fc2_w))
            fc2_b.data.copy_(_to_like(oc_state[f"transformer.resblocks.{i}.mlp.c_proj.bias"],  fc2_b))
            mapped_params += fc1_w.numel() + fc1_b.numel() + fc2_w.numel() + fc2_b.numel()

        except KeyError as e:
            raise KeyError(f"Error mapping layer {i}: Missing key {e}. "
                           "Ensure OpenCLIP model structure matches expectations.") from e
        except RuntimeError as e:
            raise RuntimeError(f"Error mapping layer {i}: Tensor shape/dtype mismatch? {e}") from e


    # Final LN
    try:
        fln_w = te_hf.final_layer_norm.weight
        fln_b = te_hf.final_layer_norm.bias
        fln_w.data.copy_(_to_like(oc_state["ln_final.weight"], fln_w))
        fln_b.data.copy_(_to_like(oc_state["ln_final.bias"],   fln_b))
        mapped_params += fln_w.numel() + fln_b.numel()
    except KeyError as e:
        raise KeyError(f"Error mapping final layer norm: Missing key {e}") from e
    except RuntimeError as e:
        raise RuntimeError(f"Error mapping final layer norm: Tensor shape/dtype mismatch? {e}") from e


    print(f"[DEBUG] Total elements mapped: {mapped_params:,}")
    return n_map, n_oc, n_hf

# ----------------- ADDED: HELPER FUNCTIONS FOR EVALUATION -----------------
def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def build_zero_shot_classifier(
        model, # Expects OpenCLIP model here
        tokenizer, # Expects OpenCLIP tokenizer
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """Builds zero-shot classifier weights using OpenCLIP model."""
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = lambda it: tqdm.tqdm(it, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = lambda it: it

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts_local = [template.format(c) if use_format else template(c)
                       for c in batch_classnames for template in templates]
        # Use OpenCLIP tokenizer and text encoder
        toks = tokenizer(texts_local).to(device)
        class_embeddings = model.encode_text(toks) # model.get_text_features(**inputs) -> model.encode_text(toks)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = F.normalize(class_embeddings, dim=-1) # Normalize
        class_embeddings = class_embeddings.T # Transpose for matrix multiplication
        return class_embeddings

    with torch.no_grad(), torch.cuda.amp.autocast(): # Use autocast
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Slightly modified to handle OpenCLIP's logit scale if needed, though usually applied outside
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run_name(model, classifier, name_to_run, processor, device,
             jpg_dict: Dict[str, List[str]], name_list: Sequence[str], celeba_root: Path):
    """Evaluates CLIP accuracy for a specific name on CelebA images."""
    # This function uses the *full* OpenCLIP model (model) and the pre-built classifier weights
    if name_to_run not in name_list:
        print(f"[Warning] Name '{name_to_run}' not found in CelebA identity list. Skipping eval.")
        return 0.0, 0.0

    label = name_list.index(name_to_run)
    top1, top5, n = 0., 0., 0.

    image_ids = jpg_dict.get(name_to_run, [])
    if not image_ids:
        print(f"[Warning] No images found for name '{name_to_run}' in jpg_dict. Skipping eval.")
        return 0.0, 0.0

    # Get the image preprocessing function from the specific OpenCLIP model instance
    # (Assumes model tuple was (model, train_preprocess, val_preprocess))
    # We need a way to get the preprocess fn. Let's load it separately for now.
    # OR pass it in. For simplicity, we load the processor.
    # Note: Using HF Processor here for image loading consistency.
    # Could also use open_clip's preprocess if passed correctly.
    hf_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


    for image_id in image_ids:
        image_path = celeba_root / "img_align_celeba" / image_id
        try:
             image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
             print(f"[Warning] Image not found: {image_path}. Skipping.")
             continue

        target = torch.tensor([label]).to(device)

        # Preprocess image using HF Processor's image part
        image_input = hf_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(): # Use autocast
            # Get image features using the OpenCLIP model's visual tower
            image_features = model.encode_image(image_input['pixel_values'])
            image_features = F.normalize(image_features, dim=-1)

            # Calculate logits using the pre-built classifier weights
            # OpenCLIP applies logit scale during loss, HF applies it here.
            # We mimic HF here for direct comparison. Classifier is already normalized.
            logits = (model.logit_scale.exp() * image_features @ classifier)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += 1

    if n == 0:
        print(f"[Warning] No valid images processed for '{name_to_run}'. Returning 0 accuracy.")
        return 0.0, 0.0

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5
# ----------------- END HELPER FUNCTIONS ---------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    # Adjust output dir to include model/pretrained info for clarity
    outdir = Path(args.outdir) / f"{args.model}_{args.pretrained.replace('/', '_')}" / args.celeb_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving outputs to: {outdir}")

    # 1) Resolve checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    # 2) Build base + GAFT OpenCLIP models (using args.model, args.pretrained)
    print(f"[INFO] Loading OpenCLIP base model: {args.model} ({args.pretrained})")
    model_base, _, preprocess_val_base = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    # Get tokenizer associated with the OpenCLIP model
    tokenizer = open_clip.get_tokenizer(args.model)

    print(f"[INFO] Creating OpenCLIP GAFT model structure: {args.model} ({args.pretrained})")
    model_gaft, _, preprocess_val_gaft = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device # Load structure, weights will be replaced
    )

    model_base.eval()
    model_gaft.eval()

    # 3) Load GAFT weights into model_gaft
    print(f"[INFO] Loading weights from checkpoint into GAFT model structure...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        print("[INFO] Stripping 'module.' prefix from checkpoint state_dict.")
        sd = {k[len("module."):]: v for k, v in sd.items()}

    load_res = model_gaft.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict: missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}")

    # Sanity check: ensure *some* weights were actually loaded
    matched_elems = 0
    gaft_state = model_gaft.state_dict()
    for k, v in sd.items():
        if k in gaft_state and gaft_state[k].shape == v.shape:
             matched_elems += v.numel()
    print(f"[DEBUG] Matched parameter elements loaded: {matched_elems:,}")
    if matched_elems == 0:
        raise RuntimeError(
            "No parameters matched. Your GAFT weights were NOT applied. "
            "Double-check --model/--pretrained match your training pair."
        )


    # 4) Cosine sanity check (unchanged)
    with torch.no_grad():
        t_base = encode_text(model_base, tokenizer, args.celeb_name.replace('_', ' '), device)
        t_gaft = encode_text(model_gaft, tokenizer, args.celeb_name.replace('_', ' '), device)
        cos = torch.cosine_similarity(t_base.unsqueeze(0), t_gaft.unsqueeze(0)).item()
    print(f"[SANITY] cos(text_base, text_gaft) for '{args.celeb_name.replace('_', ' ')}' = {cos:.6f} "
          f"(should be < 0.999 if GAFT changed weights)")

    # ---- ADDED: QUANTITATIVE EVALUATION ----
    print("\n[INFO] Running Quantitative CLIP Evaluation on GAFT Model...")

    # Load CelebA identities
    celeba_identity_file = args.celeba_root / "list_identity_celeba.txt"
    print(f"[INFO] Loading CelebA identities from: {celeba_identity_file}")
    jpg_dict = defaultdict(list)
    with open(celeba_identity_file, 'r') as f:
         lines = f.readlines()
    for line in lines[2:]: # Skip header lines
         try:
             image_id, identity_id_str = line.strip().split()
             # Map identity ID to name (assuming list_attr_celeba.txt isn't needed here, just the ID mapping)
             # We need the actual names, let's assume identity_id maps to a name list later
             # For now, let's parse identity names directly if available, or load mapping
             # The original SLUG code assumes `list_identity_celeba.txt` contains NAMEs, let's check format...
             # Ah, it seems `list_identity_celeba.txt` maps image_id -> identity_NAME. Good.
             identity_name = identity_id_str # Assuming the second column IS the name string
             jpg_dict[identity_name].append(image_id)
         except ValueError:
             print(f"[Warning] Skipping malformed line in identity file: {line.strip()}")

    name_set = set(jpg_dict.keys())
    name_list = tuple(sorted(name_set)) # This is the full list of celebs in CelebA
    print(f"[INFO] Found {len(name_list)} unique identities in CelebA.")

    # Define class names and templates for the classifier (using ALL CelebA names)
    CELEB_NAMES_ALL = [name.replace('_', ' ') for name in name_list]
    CELEB_TEMPLATES = (lambda c: f'a photo of {c}.',) # Standard template

    # Build the zero-shot classifier using the GAFT model
    print("[INFO] Building CelebA zero-shot classifier...")
    classifier_celeb_gaft = build_zero_shot_classifier(
        model_gaft, # Use the unlearned OpenCLIP model
        tokenizer,
        classnames=CELEB_NAMES_ALL,
        templates=CELEB_TEMPLATES,
        num_classes_per_batch=50, # Adjust batch size based on VRAM
        device=device,
        use_tqdm=True,
    )
    print("[INFO] Classifier built.")

    # Evaluate Forget and Retain Accuracy
    test_top1_unlearned, test_top5_unlearned = [], []
    forget_acc1_unlearned, forget_acc5_unlearned = 0.0, 0.0

    # Ensure EVAL_TEXTS names exist in CelebA dataset
    eval_names_with_underscore = [name.replace(' ', '_') for name in EVAL_TEXTS]

    # Use the Hugging Face processor for image loading in run_name
    hf_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


    print("[INFO] Evaluating performance on selected names...")
    for name_with_space in EVAL_TEXTS:
        name_with_underscore = name_with_space.replace(' ', '_')

        if name_with_underscore not in name_list:
             print(f"[Warning] Evaluation name '{name_with_underscore}' not found in CelebA list. Skipping.")
             continue

        # run_name expects OpenCLIP model, pre-built classifier, name, HF processor, etc.
        t1, t5 = run_name(
             model_gaft, classifier_celeb_gaft, name_with_underscore,
             hf_processor, device, jpg_dict, name_list, args.celeba_root
        )

        print(f"  - {name_with_underscore}: top1: {t1*100:.2f}%, top5: {t5*100:.2f}%")

        if name_with_underscore == args.celeb_name:
            forget_acc1_unlearned, forget_acc5_unlearned = t1, t5
        else:
            test_top1_unlearned.append(t1)
            test_top5_unlearned.append(t5)

    # Calculate average retain accuracy
    if test_top1_unlearned:
         avg_test_top1 = float(np.mean(test_top1_unlearned))
         avg_test_top5 = float(np.mean(test_top5_unlearned))
    else:
         avg_test_top1, avg_test_top5 = 0.0, 0.0


    print("\n--- Quantitative Results ---")
    print(f"Forget Accuracy ({args.celeb_name}):")
    print(f"  Top-1: {forget_acc1_unlearned*100:.2f}%")
    print(f"  Top-5: {forget_acc5_unlearned*100:.2f}%")
    print(f"Retain Accuracy (Avg. over {len(test_top1_unlearned)} others):")
    print(f"  Top-1: {avg_test_top1*100:.2f}%")
    print(f"  Top-5: {avg_test_top5*100:.2f}%")
    print("----------------------------\n")

    # ---- END ADDED EVALUATION ----


    # 5) Free VRAM before SD
    print("[INFO] Moving CLIP models to CPU before loading SD...")
    # Keep model_gaft on CPU, we need its weights
    model_gaft.to("cpu")
    # Base model no longer needed
    try:
        del model_base
    except NameError: pass
    try:
        del classifier_celeb_gaft # Free classifier weights from GPU
    except NameError: pass
    torch.cuda.empty_cache()
    import gc; gc.collect(); torch.cuda.empty_cache() # Force cleanup


    # 6) Load Stable Diffusion 2.1
    print(f"[INFO] Loading SD pipeline: {args.sd_id}")
    # Load HF text encoder separately first to ensure correct class/dtype
    te_hf_base = CLIPTextModel.from_pretrained(
        args.sd_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_id,
        torch_dtype=torch.float16,
        text_encoder=te_hf_base, # Pass the pre-loaded encoder
        low_cpu_mem_usage=False,
        device_map=None, # Ensure loading onto the specified device
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device) # Move entire pipe to device


    # 7) Render baseline
    print("[INFO] Rendering baseline (original SD)...")
    img_before = render(pipe, args.prompt, args.steps, args.guidance)
    img_before.save(outdir / "sd_before_swap.png")
    print(f"[INFO] Saved baseline image to {outdir / 'sd_before_swap.png'}")


    # 8) Swap in GAFT text weights using the robust porting function
    print("[INFO] Porting OpenCLIP GAFT text weights -> SD text encoder...")
    # Target the inner '.text_model' which contains the actual transformer layers
    te_target_hf = pipe.text_encoder.text_model
    num_mapped, num_oc_layers, num_hf_layers = port_openclip_text_to_hf(
         openclip_model=model_gaft, # Pass the GAFT model (now on CPU)
         te_hf=te_target_hf       # Pass the inner text model on GPU
    )
    print(f"[INFO] Port complete. Mapped {num_mapped} layers (OpenCLIP={num_oc_layers}, HF={num_hf_layers}).")


    # 9) Render after-swap (Main Prompt)
    print(f"[INFO] Rendering swapped ('{args.prompt}')...")
    img_after_main = render(pipe, args.prompt, args.steps, args.guidance)
    img_after_main.save(outdir / f"sd_after_swap_GAFT_{args.celeb_name}.png")
    print(f"[INFO] Saved swapped image to {outdir / f'sd_after_swap_GAFT_{args.celeb_name}.png'}")

    # ---- ADDED: RENDER MORE PROMPTS ----
    print("[INFO] Rendering additional prompts with swapped weights...")
    extra_prompts = [
        ("A space rocket", "rocket"),
        ("A portrait photo of Jeff Bezos", "bezos"),
        ("A painting in the style of Picasso", "picasso"), # Example abstract concept
    ]

    for extra_prompt, name_suffix in extra_prompts:
        print(f"  - Rendering swapped ('{extra_prompt}')...")
        img_after_extra = render(pipe, extra_prompt, args.steps, args.guidance)
        fname = f"sd_after_swap_GAFT_{name_suffix}.png"
        img_after_extra.save(outdir / fname)
        print(f"    Saved swapped image to {outdir / fname}")
    # ---- END ADDED PROMPTS ----


    print("\n[RESULT]")
    print(f"Saved images to: {outdir}")
    print(f" - {'sd_before_swap.png':<35} (original SD text encoder)")
    print(f" - {'sd_after_swap_GAFT_' + args.celeb_name + '.png':<35} (with GAFT text encoder)")
    for _, name_suffix in extra_prompts:
        print(f" - {'sd_after_swap_GAFT_' + name_suffix + '.png':<35} (with GAFT text encoder)")

    print("\nNotes:")
    print("• Compare 'before' and 'after' images for the main prompt to see unlearning.")
    print("• Check other 'after' images to assess impact on unrelated concepts.")
    print("• If quantitative results (Forget Acc) are high but images changed,")
    print("  it suggests GAFT affected SD generation more than zero-shot classification.")


if __name__ == "__main__":
    main()