#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neuron-wise unlearning for SD-2.1 text encoder with proper best-step selection.

Usage (with UES step search on a CelebA subset):
python scripts/sd_unlearn.py \
  --celeb Elon_Musk \
  --model-name ViT-H-14 \
  --pretrained laion2B-s32B-b79K \
  --result-dir /home/rania/SLUG/results \
  --celeba-root /path/to/celeba \
  --ues-search --ues-refine \
  --prompts "A portrait photo of Elon Musk" "A sea turtle in the ocean"
"""

# --- env toggles (set before imports) ---
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import json, math, argparse, copy, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ---------- basic utilities ----------
def per_neuron_norm(G: torch.Tensor) -> Optional[torch.Tensor]:
    if G is None:
        return None
    if G.ndim == 2:   # Linear: [out, in]
        return G.norm(p=2, dim=1)
    if G.ndim == 4:   # Conv: [Cout, Cin, kH, kW]
        return G.flatten(1).norm(p=2, dim=1)
    return None

def rowwise_cosine(Gf: torch.Tensor, Gr: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    if Gf is None or Gr is None or Gf.shape != Gr.shape:
        return None
    a, b = (Gf, Gr)
    if a.ndim == 4:
        a = a.flatten(1); b = b.flatten(1)
    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1).clamp(-1.0, 1.0)

def mask_from_indices(indices: List[int], length: int, device) -> torch.Tensor:
    m = torch.zeros(length, dtype=torch.bool, device=device)
    if not indices:
        return m
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    idx = idx[(idx >= 0) & (idx < length)]
    if idx.numel():
        m[idx] = True
    return m

def build_row_weights(
    forget_G: torch.Tensor, retain_G: torch.Tensor, mask: torch.Tensor,
    delta: float = 1.0, gamma: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """
    Per-row scalar weights: ((1 - max(0, cos_i))^delta) * ((eps + Rr_i^2)^-gamma)
    """
    Rr  = per_neuron_norm(retain_G)
    cos = rowwise_cosine(forget_G, retain_G, eps=eps)
    if Rr is None or cos is None:
        return torch.ones(int(mask.sum().item()), device=forget_G.device, dtype=torch.float32)
    align_pen = (1.0 - cos.clamp(min=0.0)).pow(float(delta))
    shield    = (eps + Rr.pow(2)).pow(-float(gamma))
    w_full    = align_pen * shield
    w_sel     = w_full[mask].to(torch.float32)
    return torch.clamp(w_sel, 0.1, 10.0)

# ---------- name mapping: open_clip → HF CLIPTextModel inner ----------
def _parse_block_idx(oc_name: str) -> Optional[int]:
    parts = oc_name.split(".")
    try:
        return int(parts[2]) if parts[0] == "transformer" and parts[1] == "resblocks" else None
    except Exception:
        return None

def oc_to_hf_simple(oc_name: str) -> Optional[str]:
    if not oc_name.startswith("transformer.resblocks."):
        return None
    i = _parse_block_idx(oc_name)
    if i is None:
        return None
    if oc_name.endswith(".mlp.c_fc.weight"):
        return f"encoder.layers.{i}.mlp.fc1.weight"
    if oc_name.endswith(".mlp.c_proj.weight"):
        return f"encoder.layers.{i}.mlp.fc2.weight"
    if oc_name.endswith(".attn.out_proj.weight"):
        return f"encoder.layers.{i}.self_attn.out_proj.weight"
    return None

def is_in_proj_weight(oc_name: str) -> bool:
    return oc_name.startswith("transformer.resblocks.") and oc_name.endswith(".attn.in_proj_weight")

# ---------- Si JSON loader ----------
def load_selected_indices(si_json_path: Path) -> Dict[str, List[int]]:
    d = json.loads(si_json_path.read_text())
    if "Si" in d and "language" in d["Si"]:
        lang_ranked = d["Si"]["language"].get("ranked", [])
    elif "language" in d:
        lang_ranked = d["language"].get("ranked", [])
    else:
        raise RuntimeError(f"Unexpected JSON format in {si_json_path}")
    out: Dict[str, List[int]] = {}
    for item in lang_ranked:
        lname = item.get("layer", "")
        idxs  = item.get("important_idx", [])
        if lname and isinstance(idxs, list) and len(idxs) > 0:
            out[lname] = idxs
    return out

# ---------- UES helpers ----------
def _rel_drop(curr: float, base: float, eps: float = 1e-6) -> float:
    if base <= eps:
        return 0.0 if curr <= base else min(1.0, (curr - base) / max(eps, curr))
    return min(1.0, max(0.0, (base - curr) / max(eps, base)))

def _ues_score(
    f1: float, f5: float, t1: float, t5: float, c1: float, c5: float,
    base_f1: float, base_f5: float, base_t1: float, base_t5: float, base_c1: float, base_c5: float,
    alpha: float
) -> float:
    forget_gain = 0.5 * (_rel_drop(f1, base_f1) + _rel_drop(f5, base_f5))
    retain_loss = 0.25 * (
        _rel_drop(t1, base_t1) +
        _rel_drop(t5, base_t5) +
        _rel_drop(c1, base_c1) +
        _rel_drop(c5, base_c5)
    )
    return alpha * forget_gain - (1.0 - alpha) * retain_loss

# ---------- minimal CelebA evaluator for UES step search ----------
def load_celeba_index(celeba_root: Path):
    """
    Returns dict: name (with underscores) -> list of filenames (e.g., 000001.jpg)
    Expects:
      celeba_root / "list_identity_celeba.txt"
      celeba_root / "img_align_celeba" / <images>
    """
    idx_file = celeba_root / "list_identity_celeba.txt"
    if not idx_file.exists():
        raise FileNotFoundError(f"Missing {idx_file}")
    with idx_file.open("r") as f:
        lines = f.readlines()
    name_to_files: Dict[str, List[str]] = {}
    for line in lines[2:]:
        img, name = line.strip().split()
        name_to_files.setdefault(name, []).append(img)
    return name_to_files

def pick_eval_subset(name_to_files: Dict[str, List[str]],
                     forget_name_us: str,
                     num_negatives: int = 4,
                     imgs_per_class: int = 10,
                     rng: Optional[random.Random] = None):
    rng = rng or random.Random(123)
    all_names = sorted(name_to_files.keys())
    if forget_name_us not in name_to_files:
        raise RuntimeError(f"Forget identity '{forget_name_us}' not found in CelebA index.")

    negatives = [n for n in all_names if n != forget_name_us]
    rng.shuffle(negatives)
    negatives = negatives[:num_negatives]

    def sample_imgs(name):
        files = name_to_files[name]
        if len(files) <= imgs_per_class:
            return files
        return rng.sample(files, imgs_per_class)

    subset = {forget_name_us: sample_imgs(forget_name_us)}
    for n in negatives:
        subset[n] = sample_imgs(n)
    return subset, negatives

def build_cached_image_feats(clip_model: CLIPModel,
                             processor: CLIPProcessor,
                             celeba_root: Path,
                             subset: Dict[str, List[str]],
                             device: str):
    """
    Returns dict: name_us -> (N_i, D) image features tensor (normalized).
    We compute with the *base* (unedited) CLIP, since image tower is unchanged.
    """
    clip_model.eval()
    out = {}
    with torch.no_grad():
        for name, files in subset.items():
            feats = []
            for fn in files:
                img = Image.open(celeba_root/"img_align_celeba"/fn).convert("RGB")
                inputs = processor(text=["."], images=img, return_tensors="pt", padding=True,
                                   truncation=True, max_length=77).to(device)
                # Only need image feats, dummy text is ignored for image tower
                image_features = clip_model.get_image_features(**{"pixel_values": inputs["pixel_values"]})
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                feats.append(image_features)
            out[name] = torch.cat(feats, dim=0)  # (Ni, D)
    return out

def text_features_for_names(clip_model: CLIPModel,
                            processor: CLIPProcessor,
                            names_human: List[str],
                            device: str):
    """
    names_human: e.g., ["Elon Musk", "Mark Zuckerberg", ...]
    Returns (C,D) normalized text features (single simple template "NAME.").
    """
    clip_model.eval()
    with torch.no_grad():
        texts = [f"{n}." for n in names_human]
        tok = processor(text=texts, images=[Image.new("RGB",(1,1))]*len(texts),  # dummy image list to satisfy processor
                        return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        # Only need text tokens
        text_features = clip_model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features  # (C, D)

def eval_ues_on_subset(clip_model: CLIPModel,
                       processor: CLIPProcessor,
                       img_feats_cache: Dict[str, torch.Tensor],
                       forget_name_us: str,
                       retain_names_us: List[str],
                       alpha: float,
                       device: str):
    """
    Computes top-1/top-5 on forget vs retain sets using zero-shot classification
    with the current clip_model (text tower edited). Image tower unchanged.

    Returns: (fgt1,fgt5, c1,c5, t1,t5) where we set celeb metrics == retain metrics (proxy).
    """
    # Map underscore names -> human form
    def de_us(s): return s.replace("_", " ")

    classnames_human = [de_us(forget_name_us)] + [de_us(n) for n in retain_names_us]
    text_feats = text_features_for_names(clip_model, processor, classnames_human, device)  # (C,D)
    C = text_feats.shape[0]

    def acc_for_name(name_us: str):
        imgs = img_feats_cache[name_us]  # (N,D)
        # logits: (N,C)
        logits = 100.0 * (imgs @ text_feats.T)
        # targets: forget is class 0; negatives map to their index in class list
        target_idx = 0 if name_us == forget_name_us else (1 + retain_names_us.index(name_us))
        target = torch.full((logits.shape[0],), target_idx, dtype=torch.long, device=logits.device)
        # topk
        topk = logits.topk(k=min(5, C), dim=1).indices
        top1 = (topk[:, :1] == target.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
        top5 = (topk == target.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
        return top1, top5

    f1, f5 = acc_for_name(forget_name_us)
    retain_t1 = []; retain_t5 = []
    for n in retain_names_us:
        a1, a5 = acc_for_name(n)
        retain_t1.append(a1); retain_t5.append(a5)
    t1 = float(np.mean(retain_t1)) if retain_t1 else 0.0
    t5 = float(np.mean(retain_t5)) if retain_t5 else 0.0
    # Proxy celeb metrics with retain (same set)
    c1, c5 = t1, t5
    return f1, f5, c1, c5, t1, t5

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Apply neuron-wise edit to SD-2.1 with best-step selection")
    ap.add_argument("--celeb", default="Elon_Musk")
    ap.add_argument("--result-dir", dest="result_dir", default="/home/rania/SLUG/results")
    ap.add_argument("--model-name", dest="model_name", default="ViT-H-14")
    ap.add_argument("--pretrained",  default="laion2B-s32B-b79K")
    ap.add_argument("--clip-id", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--sd-id",   default="stabilityai/stable-diffusion-2-1")

    # IO
    ap.add_argument("--grads-root", default=None,
                    help="Override grads dir. Default: results/grads/<celeb>_<model_name>_<pretrained>")
    ap.add_argument("--si-json", default=None,
                    help="Override Si JSON path. Default: results/neuron importance_global_sd/<celeb>/<model>_<pretrained>_Si.json")
    ap.add_argument("--summary-json", default=None,
                    help="If set, read best_step from this summary.json.")

    # Step options
    ap.add_argument("--step", type=float, default=None,
                    help="Explicit step to use.")
    ap.add_argument("--step-mult", type=float, default=1.0,
                    help="Multiply the final step by this factor.")
    ap.add_argument("--initial-div", type=float, default=10.0,
                    help="Heuristic divider if no step/summary given.")

    # Weight hyperparams
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)

    # UES search on CelebA subset
    ap.add_argument("--ues-search", action="store_true",
                    help="Run UES-based step search on a small CelebA subset.")
    ap.add_argument("--celeba-root", type=str, default="data/celeba",
                    help="Path to CelebA (must contain list_identity_celeba.txt and img_align_celeba/).")
    ap.add_argument("--ues-images-per-class", type=int, default=10)
    ap.add_argument("--ues-num-negatives", type=int, default=4)
    ap.add_argument("--ues-alpha", type=float, default=0.5)
    ap.add_argument("--ues-max-iters", type=int, default=12)
    ap.add_argument("--ues-refine", action="store_true")

    # Prompts + out
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="Prompts to render after edit.")
    ap.add_argument("--outdir", default=None,
                    help="Where to save images and edited text-encoder. Default results/sd2_edits/<celeb>")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    random.seed(123)

    # ---- load grads (open_clip names) ----
    if args.grads_root:
        grads_dir = Path(args.grads_root)
    else:
        grads_dir = Path(args.result_dir) / "grads" / f"{args.celeb}_{args.model_name}_{args.pretrained}"
    fg_path = grads_dir / "forget_grads.pt"
    rg_path = grads_dir / "train_grads.pt"
    if not fg_path.exists() or not rg_path.exists():
        raise FileNotFoundError(f"Missing grads:\n{fg_path}\n{rg_path}")
    forget_grads = torch.load(fg_path, map_location="cpu")
    retain_grads = torch.load(rg_path, map_location="cpu")

    # ---- load Si JSON (selected neurons) ----
    si_path = Path(args.si_json) if args.si_json else (
        Path(args.result_dir) / "neuron importance_global_sd" / args.celeb / f"{args.model_name}_{args.pretrained}_Si.json"
    )
    if not si_path.exists():
        raise FileNotFoundError(f"Si JSON not found: {si_path}")
    selected = load_selected_indices(si_path)

    # ---- build masked update plan in open_clip names ----
    plan_oc: Dict[str, torch.Tensor] = {}
    grad_sel_sq_sum = 0.0
    cpu = torch.device("cpu")

    for oc_name, idxs in selected.items():
        Gf = forget_grads.get(oc_name, None)
        Gr = retain_grads.get(oc_name, None)
        if Gf is None or Gr is None:
            continue
        if Gf.ndim not in (2, 4):
            continue  # only row-wise masking supported
        mask = mask_from_indices(idxs, Gf.shape[0], device=cpu)
        if not mask.any():
            continue
        w = build_row_weights(Gf.float(), Gr.float(), mask, delta=args.delta, gamma=args.gamma)
        U = torch.zeros_like(Gf, dtype=torch.float32)
        if Gf.ndim == 2:
            U[mask, :] = w[:, None] * Gf[mask, :].float()
            grad_sel_sq_sum += float((U[mask, :] ** 2).sum().item())
        else:
            U[mask, :, :, :] = w[:, None, None, None] * Gf[mask, :, :, :].float()
            grad_sel_sq_sum += float((U[mask, :, :, :] ** 2).sum().item())
        plan_oc[oc_name] = U

    if not plan_oc or grad_sel_sq_sum <= 0:
        raise RuntimeError("No effective neuron updates. Check JSON layer names vs grads content.")

    # ---- load HF CLIP and map to HF inner text names ----
    clip = CLIPModel.from_pretrained(args.clip_id)
    processor = CLIPProcessor.from_pretrained(args.clip_id)
    clip = clip.to(device).eval()
    inner_params = dict(clip.text_model.named_parameters())
    plan_hf: Dict[str, torch.Tensor] = {}
    param_sel_sq_sum = 0.0

    for oc_name, U in plan_oc.items():
        hf_simple = oc_to_hf_simple(oc_name)
        if hf_simple is not None and hf_simple in inner_params:
            Uh = U.to(device)
            plan_hf[hf_simple] = Uh
            mask_rows = torch.any(Uh != 0, dim=tuple(range(1, Uh.ndim)))
            p = inner_params[hf_simple]
            if p.ndim == 2:
                param_sel_sq_sum += float((p.data[mask_rows, :].float() ** 2).sum().item())
            elif p.ndim == 4:
                param_sel_sq_sum += float((p.data[mask_rows, :, :, :].float() ** 2).sum().item())
            continue

        if is_in_proj_weight(oc_name):
            i = _parse_block_idx(oc_name)
            if i is None:
                continue
            Uh = U.to(device)
            if Uh.ndim != 2 or Uh.shape[0] % 3 != 0:
                continue
            E = Uh.shape[1]
            rows = Uh.shape[0] // 3
            if rows != E:
                continue
            splits: List[Tuple[str, torch.Tensor]] = [
                (f"encoder.layers.{i}.self_attn.q_proj.weight", Uh[0:E, :]),
                (f"encoder.layers.{i}.self_attn.k_proj.weight", Uh[E:2*E, :]),
                (f"encoder.layers.{i}.self_attn.v_proj.weight", Uh[2*E:3*E, :]),
            ]
            for name_h, Uh_part in splits:
                if name_h not in inner_params:
                    continue
                plan_hf[name_h] = Uh_part
                mask_rows = torch.any(Uh_part != 0, dim=1)
                p = inner_params[name_h]
                param_sel_sq_sum += float((p.data[mask_rows, :].float() ** 2).sum().item())
            continue

    if not plan_hf:
        raise RuntimeError("Nothing mapped into HF CLIP text encoder. Add mappings for your selected layers.")

    # ---- choose the step ----
    step = args.step
    if step is None and args.summary_json:
        try:
            best = json.loads(Path(args.summary_json).read_text()).get("best_step", None)
            if best is not None:
                step = float(best)
        except Exception:
            step = None

    params_norm = math.sqrt(max(param_sel_sq_sum, 1e-12))
    grads_norm  = math.sqrt(max(grad_sel_sq_sum,  1e-12))
    if step is None:
        if grads_norm < 1e-9:
            raise RuntimeError("Selected grads nearly zero; cannot pick a stable step.")
        step = -(params_norm / grads_norm) / float(args.initial_div)
    seed_step = step

    # ---- optional: UES search on CelebA subset (self-contained) ----
    if args.ues_search:
        celeba_root = Path(args.celeba_root)
        name_to_files = load_celeba_index(celeba_root)
        forget_name_us = args.celeb  # e.g., Elon_Musk
        subset, negs = pick_eval_subset(
            name_to_files,
            forget_name_us=forget_name_us,
            num_negatives=args.ues_num_negatives,
            imgs_per_class=args.ues_images_per_class,
        )
        # Cache image feats with base (unedited) CLIP; image tower unchanged
        img_feats_cache = build_cached_image_feats(
            clip_model=clip, processor=processor, celeba_root=celeba_root,
            subset=subset, device=device
        )

        # Baseline metrics
        (fgt1_b, fgt5_b, c1_b, c5_b, t1_b, t5_b) = eval_ues_on_subset(
            clip_model=clip, processor=processor, img_feats_cache=img_feats_cache,
            forget_name_us=forget_name_us, retain_names_us=negs, alpha=args.ues_alpha, device=device
        )

        def _apply_step_to_clip(base_clip: CLIPModel, plan: Dict[str, torch.Tensor], s: float) -> CLIPModel:
            model = copy.deepcopy(base_clip).to(device).eval()
            with torch.no_grad():
                named = dict(model.text_model.named_parameters())
                for name, U in plan.items():
                    if name in named:
                        p = named[name]
                        upd = (s * U.to(p.device, dtype=torch.float32)).to(p.dtype)
                        p.add_(upd)
            return model

        def _measure(model_clip: CLIPModel):
            return eval_ues_on_subset(
                model_clip, processor, img_feats_cache,
                forget_name_us=forget_name_us, retain_names_us=negs,
                alpha=args.ues_alpha, device=device
            )

        # Coarse bracketing
        best = dict(ues=-1e9, step=seed_step, metr=None)
        lo, hi = 0.0, float("inf")
        step_cand = seed_step

        for it in range(1, int(args.ues_max_iters) + 1):
            cand_clip = _apply_step_to_clip(clip, plan_hf, step_cand)
            (f1, f5, c1, c5, t1, t5) = _measure(cand_clip)
            ues = _ues_score(f1, f5, t1, t5, c1, c5,
                             fgt1_b, fgt5_b, t1_b, t5_b, c1_b, c5_b, args.ues_alpha)
            if ues > best["ues"]:
                best.update(ues=ues, step=step_cand, metr=(f1, f5, c1, c5, t1, t5))

            forget_zero = (f5 <= 0.0 or f1 <= 0.0)
            if forget_zero:
                hi = step_cand
                step_cand = (lo + hi) / 2.0
            else:
                lo = step_cand
                step_cand = (step_cand * 2.0) if not math.isfinite(hi) else (lo + hi) / 2.0

        step = best["step"]

        # Golden-section refine (optional)
        if args.ues_refine:
            def _eval_step(s):
                m = _apply_step_to_clip(clip, plan_hf, s)
                (f1,f5,c1,c5,t1,t5) = _measure(m)
                return _ues_score(f1,f5,t1,t5,c1,c5, fgt1_b,fgt5_b,t1_b,t5_b,c1_b,c5_b, args.ues_alpha)

            m_lo, m_hi = 0.5, 2.0
            phi = (1 + 5**0.5)/2
            resphi = 2 - phi
            ax = math.log(m_lo); bx = math.log(m_hi)
            x1 = bx - resphi*(bx-ax); x2 = ax + resphi*(bx-ax)
            s1 = step * math.exp(x1); s2 = step * math.exp(x2)
            u1 = _eval_step(s1); u2 = _eval_step(s2)
            best_step = s1 if u1 >= u2 else s2
            best_ues  = max(u1,u2)
            for _ in range(14):
                if u1 >= u2:
                    bx, x2, u2 = x2, x1, u1
                    x1 = bx - resphi*(bx-ax)
                    s1 = step * math.exp(x1)
                    u1 = _eval_step(s1)
                    if u1 > best_ues: best_step, best_ues = s1, u1
                else:
                    ax, x1, u1 = x1, x2, u2
                    x2 = ax + resphi*(bx-ax)
                    s2 = step * math.exp(x2)
                    u2 = _eval_step(s2)
                    if u2 > best_ues: best_step, best_ues = s2, u2
            step = best_step

    # Final scaling
    step *= float(args.step_mult)
    print(f"[STEP] using step={step:+.6e}")

    # ---- apply updates to the HF CLIP text encoder (transformer-only) ----
    with torch.no_grad():
        for name, Uadd in plan_hf.items():
            if name not in inner_params:
                continue
            p = inner_params[name]
            upd = (step * Uadd.to(p.device, dtype=torch.float32)).to(p.dtype)
            p.add_(upd)
    print("[OK] Applied neuron-wise edit to CLIP text encoder.")

    # ---- prepare output dir; save edited HF transformer for reuse ----
    outdir = Path(args.outdir or (Path(args.result_dir) / "sd2_edits" / args.celeb))
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(clip.text_model.state_dict(), outdir / "text_encoder_hf_clip-transformer_only.pt")

    # ---- build the CLIPTextModel wrapper that SD-2.1 expects, load our edited transformer into it ----
    te_sd = CLIPTextModel.from_pretrained(
        args.sd_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    ).to(device)

    sd_state = {k: v.to(dtype=te_sd.dtype, device=te_sd.device) for k, v in clip.text_model.state_dict().items()}
    with torch.no_grad():
        te_sd.text_model.load_state_dict(sd_state, strict=False)
    torch.save(te_sd.state_dict(), outdir / "text_encoder_sd21_CLIPTextModel.fp16.pt")

    # ---- create SD-2.1 pipeline with our wrapped, edited text encoder ----
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_id,
        torch_dtype=torch.float16,
        text_encoder=te_sd,
        low_cpu_mem_usage=False,
        offload_state_dict=False,
        device_map=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # ---- render prompts ----
    prompts = args.prompts or [
        f"A portrait photo of {args.celeb.replace('_',' ')}",
        "A photo of Mark Zuckerberg",
        "A sea turtle in the ocean",
    ]
    for i, prompt in enumerate(prompts, 1):
        image = pipe(prompt).images[0]
        fp = outdir / f"{i:02d}.png"
        image.save(fp)
        print(f"[IMG] {fp}  ←  '{prompt}'")

    print("[DONE] Outputs:", outdir)

if __name__ == "__main__":
    main()
