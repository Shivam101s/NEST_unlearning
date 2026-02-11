#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neuron-based CLIP text-tower unlearning + Stable Diffusion visualization with UES-based step search.

Fixes included (vs your pasted script):
1) ImageNet mapping is robust everywhere (step-search + final eval):
   - stores both "school bus" and "school_bus" aliases
   - also handles punctuation/whitespace normalization
2) Si JSON layer mapping correctly loads BOTH:
   - transformer.resblocks.*.mlp.c_fc.weight  -> HF text_model.encoder.layers.*.mlp.fc1.weight
   - transformer.resblocks.*.mlp.c_proj.weight -> HF text_model.encoder.layers.*.mlp.fc2.weight
3) Final ImageNet map builder is consistent with step-search (aliases added).
4) Minor safety/clarity fixes: deterministic seeds, cleaner device autocast handling, better warnings.

Usage (typical):
  python unlearn.py --eval-target-name school_bus --eval-mode imagenet

Optional:
  --step-sign -1  (default: -1, matches your previous)
  --step-sign +1  (if you find sign should be flipped for forgetting)
"""

import os, warnings, re, json, time, argparse, sys, gc
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union, Dict, Tuple
from itertools import islice
from collections import defaultdict, Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFUSERS_OFFLOAD_STATE_DICT"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

warnings.filterwarnings("ignore")

# ========= PATHS (edit if your layout differs) =========
REPO_ROOT         = Path(__file__).resolve().parents[1]
DATA_ROOT         = REPO_ROOT / "data"
CELEBA_ROOT       = DATA_ROOT / "celeba"
IMAGENET_ROOT     = DATA_ROOT / "ImageNet"
IMAGENET_MAP_JSON = IMAGENET_ROOT / "imagenet_class_index.json"

CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PAIR          = "ViT-H-14 laion2B-s32B-b79K"
SD_MODEL_ID   = "stabilityai/stable-diffusion-2-1"
# =======================================================

# ========= EVAL CONFIG =========
CELEBA_EVAL_NAMES = [
    "Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Kanye West",
    "Chris Brown", "Bruno Mars", "Kim Kardashian", "Taylor Swift",
    "Ariana Grande"
]

IMAGENET_RETAIN_CLASSES = [
    "banana", "orange", "strawberry", "broccoli", "cucumber",
    "golden retriever", "tabby", "tiger", "bald eagle", "penguin",
    "sports car", "motor scooter", "container ship",
    "laptop", "backpack", "umbrella", "dining table", "television",
    "football", "baseball",
]
# ==============================

accelerator = Accelerator()
DEFAULT_DEVICE = accelerator.device


# ----------------- normalization helpers -----------------
def _norm_key(s: str) -> str:
    """
    Normalize keys for robust matching:
    - lowercase
    - remove most punctuation
    - collapse whitespace
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s_]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _add_aliases(name_to_index: Dict[str, int], raw_name: str, idx: int):
    nm = _norm_key(raw_name)
    if not nm:
        return
    name_to_index[nm] = idx
    name_to_index[nm.replace(" ", "_")] = idx
    name_to_index[nm.replace("_", " ")] = idx


# ----------------- Argument Parser -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Neuron unlearning evaluation for SD with CelebA/ImageNet options.")
    # --- Input Concept & Model ---
    p.add_argument("--eval-target-name", type=str, default="school_bus",
                   help="Concept name being unlearned (e.g., 'Elon_Musk', 'apple', 'school_bus').")
    p.add_argument("--clip-model-id", type=str, default=CLIP_MODEL_ID, help="Hugging Face CLIP model ID.")
    p.add_argument("--clip-pair-tag", type=str, default=PAIR.replace(" ", "_"),
                   help="Tag matching OpenCLIP format for finding grads/neurons.")
    # --- Neuron Importance ---
    p.add_argument("--neuron-json", type=str, default="",
                   help="Path to important neurons JSON file. If empty, searches defaults.")
    # --- Gradients ---
    p.add_argument("--grads-root", type=str, default=str(REPO_ROOT / "results/grads"),
                   help="Base directory for precomputed gradients.")
    # --- Stable Diffusion ---
    p.add_argument("--sd-id", type=str, default=SD_MODEL_ID, help="Stable Diffusion model ID.")
    p.add_argument("--prompt", type=str, default="A photo of {concept}",
                   help="Main prompt. '{concept}' replaced by --eval-target-name.")
    p.add_argument("--steps", type=int, default=30, help="Sampling steps.")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale.")
    p.add_argument("--seed", type=int, default=47, help="Seed for SD generation.")
    # --- Step sign (important) ---
    p.add_argument("--step-sign", type=int, default=-1, choices=[-1, 1],
                   help="Sign for step_init. Default -1 matches your previous code. Try +1 if forgetting goes the wrong way.")
    # --- Evaluation Mode ---
    p.add_argument("--eval-mode", type=str, default="imagenet",
                   choices=["none", "celeba", "imagenet"],
                   help="Run quantitative evaluation ('celeba', 'imagenet') or skip ('none').")
    # --- CelebA Args ---
    p.add_argument("--celeba-root", type=str, default=str(CELEBA_ROOT), help="Path to CelebA dataset root.")
    # --- ImageNet Args ---
    p.add_argument("--imagenet-val", type=str, default=str(IMAGENET_ROOT / "val"), help="Path to ImageNet val root.")
    p.add_argument("--imagenet-map-json", type=str, default=str(IMAGENET_MAP_JSON),
                   help="Path to ImageNet class index JSON file.")
    p.add_argument("--imagenet-batch-size", type=int, default=64)
    p.add_argument("--imagenet-workers", type=int, default=4)
    # --- Output ---
    p.add_argument("--outdir-base", type=str,
                   default=str(REPO_ROOT / "results/Visual_Results/Domain/NEST/seed47"),
                   help="Base directory for outputs.")
    # --- System ---
    p.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))

    args = p.parse_args()

    # --- Derived names ---
    args.eval_target_name_display    = args.eval_target_name.replace("_", " ")
    args.eval_target_name_underscore = args.eval_target_name.replace(" ", "_")
    args.grads_dir = Path(args.grads_root) / f"{args.eval_target_name_underscore}_{args.clip_pair_tag}"

    if not args.grads_dir.exists():
        raise FileNotFoundError(f"Gradient directory not found: {args.grads_dir}")

    if not args.neuron_json:
        search_paths = [
            Path(f"{REPO_ROOT}/results/neuron importance_global_sd/{args.eval_target_name_underscore}/{args.clip_pair_tag}_Si.json"),
            Path(f"{REPO_ROOT}/results/neuron_importance_global/{args.eval_target_name_underscore}/{args.clip_pair_tag}_Si.json"),
            Path(f"{REPO_ROOT}/results/neuron_importance/{args.eval_target_name_underscore}/{args.clip_pair_tag}_Si.json"),
        ]
        args.neuron_json = next((pp for pp in search_paths if pp.exists()), None)
        if args.neuron_json is None:
            raise FileNotFoundError(f"Neuron JSON not found for {args.eval_target_name_underscore}. Specify with --neuron-json.")
    else:
        args.neuron_json = Path(args.neuron_json)
        assert args.neuron_json.exists(), f"Neuron JSON not found: {args.neuron_json}"

    if args.eval_mode == "celeba":
        args.celeba_root = Path(args.celeba_root)
        assert (args.celeba_root / "list_identity_celeba.txt").exists(), f"CelebA identity file missing: {args.celeba_root}"
        if args.eval_target_name_display not in CELEBA_EVAL_NAMES:
            print(f"[WARN] Target '{args.eval_target_name_display}' not in CELEBA_EVAL_NAMES.")
    elif args.eval_mode == "imagenet":
        args.imagenet_val      = Path(args.imagenet_val)
        args.imagenet_map_json = Path(args.imagenet_map_json)
        assert args.imagenet_val.is_dir(), f"ImageNet val dir not found: {args.imagenet_val}"
        assert args.imagenet_map_json.is_file(), f"ImageNet map JSON not found: {args.imagenet_map_json}"

    args.outdir = Path(args.outdir_base) / args.clip_pair_tag / args.eval_target_name_underscore
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.prompt = args.prompt.format(concept=args.eval_target_name_display)
    return args


# ----------------- Helpers -----------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _autocast_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"

def render(pipe: StableDiffusionPipeline, prompt: str, steps: int, guidance: float, generator) -> Image.Image:
    return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]


# --- CelebA eval helpers ---
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    if batch_size == 0:
        return [0.0] * len(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).item() for k in topk]

def run_name_celeba(model_hf, classifier_celeba, name_to_run_underscore,
                    processor, device, jpg_dict, name_list_celeba, celeba_root):
    if name_to_run_underscore not in name_list_celeba:
        return 0.0, 0.0, 0
    label = name_list_celeba.index(name_to_run_underscore)
    top1_count, top5_count, n = 0.0, 0.0, 0
    image_ids = jpg_dict.get(name_to_run_underscore, [])
    if not image_ids:
        return 0.0, 0.0, 0

    for image_id in image_ids:
        image_path = Path(celeba_root) / "img_align_celeba" / image_id
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        target = torch.tensor([label], device=device)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast(_autocast_device(device)):
            img_feats = model_hf.get_image_features(pixel_values=inputs["pixel_values"])
            img_feats = F.normalize(img_feats, dim=-1)
            logits = (model_hf.logit_scale.exp() * img_feats @ classifier_celeba)

        acc1_img_count, acc5_img_count = accuracy(logits, target, topk=(1, 5))
        top1_count += acc1_img_count
        top5_count += acc5_img_count
        n += 1

    if n == 0:
        return 0.0, 0.0, 0
    return top1_count / n, top5_count / n, n

def run_celeba_eval(model_hf, classifier_celeba, args, processor, device, jpg_dict, name_list_celeba):
    print("[INFO] Evaluating performance on selected CelebA names...")
    test_top1_acc_list, test_top5_acc_list = [], []
    forget_acc1, forget_acc5 = 0.0, 0.0
    num_retain_classes_evaluated = 0
    celeba_target_name_underscore = args.eval_target_name_underscore

    for name_with_space in CELEBA_EVAL_NAMES:
        name_us = name_with_space.replace(" ", "_")
        t1_acc, t5_acc, count = run_name_celeba(
            model_hf, classifier_celeba, name_us,
            processor, device, jpg_dict, name_list_celeba, Path(args.celeba_root)
        )
        if count > 0:
            print(f"  - {name_us}: top1={t1_acc*100:.2f}%, top5={t5_acc*100:.2f}% ({int(t1_acc*count)}/{count})")
            if name_us == celeba_target_name_underscore:
                forget_acc1, forget_acc5 = t1_acc, t5_acc
            else:
                test_top1_acc_list.append(t1_acc)
                test_top5_acc_list.append(t5_acc)
                num_retain_classes_evaluated += 1
        else:
            print(f"  - {name_us}: No images processed.")
            if name_us == celeba_target_name_underscore:
                forget_acc1, forget_acc5 = 0.0, 0.0

    avg_test_top1 = float(np.mean(test_top1_acc_list)) if test_top1_acc_list else 0.0
    avg_test_top5 = float(np.mean(test_top5_acc_list)) if test_top5_acc_list else 0.0
    return forget_acc1, forget_acc5, avg_test_top1, avg_test_top5, num_retain_classes_evaluated


# --- ZS classifier helpers ---
def batched(iterable, n):
    if n <= 0:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    return iter(lambda: list(islice(it, n)), [])

def build_zero_shot_classifier(model, tokenizer, classnames: Sequence[str],
                               templates: Sequence[Union[Callable, str]],
                               num_classes_per_batch: Optional[int] = 50,
                               device: Union[str, torch.device] = "cpu",
                               use_tqdm: bool = False):
    num_templates = len(templates)
    num_classes   = len(classnames)
    iter_wrap = lambda it, **kw: tqdm(it, **kw) if use_tqdm else it
    ac_dev = _autocast_device(device if isinstance(device, torch.device) else torch.device(str(device)))

    @torch.no_grad()
    def _process_batch(batch_classnames):
        texts_local = [
            tpl.format(c) if isinstance(tpl, str) else tpl(c)
            for c in batch_classnames for tpl in templates
        ]
        inputs = tokenizer(texts_local, padding=True, return_tensors="pt", truncation=True, max_length=77).to(device)
        with torch.amp.autocast(ac_dev):
            class_embeds = model.get_text_features(**inputs)
        class_embeds = class_embeds.reshape(len(batch_classnames), num_templates, -1).mean(dim=1)
        return F.normalize(class_embeds, dim=-1).T

    with torch.no_grad():
        if num_classes_per_batch and num_classes > num_classes_per_batch:
            num_iter = (num_classes + num_classes_per_batch - 1) // num_classes_per_batch
            batches = batched(classnames, num_classes_per_batch)
            all_embeds = [_process_batch(batch) for batch in iter_wrap(batches, total=num_iter, desc="Building ZS classifier")]
            zeroshot_weights = torch.cat(all_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


# --- ImageNet helpers ---
def load_imagenet_map(map_json_path: Path):
    """
    Loads imagenet_class_index.json and builds robust name mappings (space + underscore + normalized).
    """
    with open(map_json_path, "r") as f:
        idx_to_details = json.load(f)  # { "0": ["n01440764","tench, Tinca tinca"], ... }

    id_to_index: Dict[str, int] = {v[0]: int(k) for k, v in idx_to_details.items()}
    name_to_index: Dict[str, int] = {}
    index_to_mainname: Dict[int, str] = {}

    for idx_str, (wnid, names_str) in idx_to_details.items():
        idx = int(idx_str)
        names = [s.strip() for s in names_str.split(",")]
        main = names[0]
        index_to_mainname[idx] = main

        for n in names:
            _add_aliases(name_to_index, n, idx)

    return {
        "index_to_name_tuple": idx_to_details,
        "id_to_index": id_to_index,
        "name_to_index": name_to_index,
        "index_to_mainname": index_to_mainname,
    }

def build_imagenet_preprocess(processor, model_hf):
    try:
        img_size = model_hf.config.vision_config.image_size
    except Exception:
        img_size = 224
        print(f"[WARN] Using default img_size={img_size}")
    normalize = transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
    preprocess_eval = transforms.Compose([
        transforms.Resize(processor.image_processor.size["shortest_edge"], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess_eval

def make_imagenet_subset_for_step_search(args, processor, model_hf, imagenet_map,
                                         forget_name_display: str,
                                         retain_names: List[str]):
    CAP_PER_CLASS = None
    preprocess_eval = build_imagenet_preprocess(processor, model_hf)
    full_ds = datasets.ImageFolder(root=str(args.imagenet_val), transform=preprocess_eval)

    def name_to_wnid(nm: str) -> Optional[str]:
        nm_norm = nm.strip()
        nm_key  = _norm_key(nm_norm)

        # 1) folder/wnid direct
        if nm_norm in full_ds.class_to_idx:
            return nm_norm
        if nm_key in full_ds.class_to_idx:
            return nm_key

        # 2) lookup name->idx (robust)
        idx = imagenet_map["name_to_index"].get(nm_key, -1)
        if idx == -1:
            idx = imagenet_map["name_to_index"].get(nm_key.replace("_", " "), -1)
        if idx == -1:
            idx = imagenet_map["name_to_index"].get(nm_key.replace(" ", "_"), -1)
        if idx == -1:
            return None

        wnid = imagenet_map["index_to_name_tuple"][str(idx)][0]
        return wnid if wnid in full_ds.class_to_idx else None

    target_wnid  = name_to_wnid(forget_name_display)
    retain_wnids = [name_to_wnid(n) for n in retain_names]

    wnids_wanted: List[str] = []
    if target_wnid is None:
        print(f"[WARN] Forget class '{forget_name_display}' not present in ImageNet val directory.")
        print("       Step search will proceed using retain classes only; forget metrics will be 0.")
        forget_subset_idx = None
    else:
        wnids_wanted.append(target_wnid)
        forget_subset_idx = 0  # temporarily; will reassign after ordering

    for w in retain_wnids:
        if w is not None and w in full_ds.class_to_idx and w not in wnids_wanted:
            wnids_wanted.append(w)

    if not wnids_wanted:
        raise RuntimeError("No valid ImageNet classes found in val directory for step-search.")

    wanted_class_indices = {full_ds.class_to_idx[w]: w for w in wnids_wanted}
    per_class_counter = Counter()
    chosen_indices: List[int] = []

    for idx, (_, y) in enumerate(full_ds.samples):
        if y in wanted_class_indices and (CAP_PER_CLASS is None or per_class_counter[y] < CAP_PER_CLASS):
            chosen_indices.append(idx)
            per_class_counter[y] += 1

    subset = Subset(full_ds, chosen_indices)
    loader = DataLoader(
        subset,
        batch_size=args.imagenet_batch_size,
        shuffle=False,
        num_workers=args.imagenet_workers,
        pin_memory=True,
        drop_last=False,
    )

    subset_wnids_order = wnids_wanted
    wnid_to_subset_idx = {w: i for i, w in enumerate(subset_wnids_order)}
    if target_wnid is not None and target_wnid in wnid_to_subset_idx:
        forget_subset_idx = wnid_to_subset_idx[target_wnid]
    else:
        forget_subset_idx = None

    subset_classnames: List[str] = []
    for w in subset_wnids_order:
        idx = imagenet_map["id_to_index"][w]
        subset_classnames.append(imagenet_map["index_to_mainname"][idx])

    print(f"[INFO] ImageNet subset for step-search: {len(subset)} images from {len(subset_wnids_order)} classes.")
    if forget_subset_idx is None:
        print("[WARN] No forget class in ImageNet subset; treating all as retain for step search.")
    return loader, subset, wnid_to_subset_idx, subset_classnames, forget_subset_idx

@torch.no_grad()
def eval_imagenet_subset_step(model_hf, classifier_sub, loader, subset_ds, wnid_to_subset_idx,
                              device, k=5, forget_subset_idx: Optional[int] = 0):
    per_class_counts: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    ac_dev = _autocast_device(device)
    model_hf.eval()
    base_ds: datasets.ImageFolder = subset_ds.dataset if isinstance(subset_ds, Subset) else subset_ds

    for images, target_indices_in_base in tqdm(loader, desc="Step-search eval (ImageNet subset)", leave=False):
        images = images.to(device, non_blocking=True)
        target_indices_in_base = target_indices_in_base.to(device, non_blocking=True)

        with torch.amp.autocast(ac_dev):
            img_feats = model_hf.get_image_features(pixel_values=images)
            img_feats = F.normalize(img_feats, dim=-1)
            logits = (model_hf.logit_scale.exp() * img_feats @ classifier_sub)

        pred_topk = logits.topk(k, 1, True, True)[1].t()

        batch_subset_targets = []
        for y in target_indices_in_base.tolist():
            wnid = base_ds.classes[y]
            batch_subset_targets.append(wnid_to_subset_idx.get(wnid, -1))

        batch_subset_targets = torch.tensor(batch_subset_targets, device=device, dtype=torch.long)
        valid_mask = batch_subset_targets.ge(0)
        if not valid_mask.any():
            continue

        valid_targets = batch_subset_targets[valid_mask]
        correct = pred_topk[:, valid_mask].eq(valid_targets.view(1, -1).expand_as(pred_topk[:, valid_mask]))

        for b_idx in range(valid_targets.numel()):
            cls = int(valid_targets[b_idx].item())
            per_class_counts[cls][0] += float(correct[0, b_idx].item())
            per_class_counts[cls][1] += float(correct[:k, b_idx].any().item())
            per_class_counts[cls][2] += 1

    f1 = f5 = 0.0
    if forget_subset_idx is not None and forget_subset_idx in per_class_counts and per_class_counts[forget_subset_idx][2] > 0:
        c1, c5, total = per_class_counts[forget_subset_idx]
        f1 = c1 / total
        f5 = c5 / total

    retain_keys = [
        cls for cls, (_, _, cnt) in per_class_counts.items()
        if cnt > 0 and (forget_subset_idx is None or cls != forget_subset_idx)
    ]
    if retain_keys:
        r1 = np.mean([per_class_counts[cls][0] / per_class_counts[cls][2] for cls in retain_keys])
        r5 = np.mean([per_class_counts[cls][1] / per_class_counts[cls][2] for cls in retain_keys])
    else:
        r1 = r5 = 0.0

    return float(f1), float(f5), float(r1), float(r5)


def _resolve_imagenet_class_idx(imagenet_map: dict, concept: str) -> Tuple[int, Optional[str]]:
    """
    Resolve "school bus"/"school_bus" robustly using name_to_index aliases.
    """
    target = _norm_key(concept)
    if not target:
        return -1, None

    for key in (target, target.replace(" ", "_"), target.replace("_", " ")):
        idx = imagenet_map["name_to_index"].get(key, -1)
        if idx != -1:
            return idx, key

    # soft fallback: substring
    for k, v in imagenet_map["name_to_index"].items():
        if target == k or target in k or k in target:
            return v, k
    return -1, None


# --- Full ImageNet FINAL eval ---
def run_imagenet_eval(model_hf, classifier_imagenet, args, processor, imagenet_map, device):
    print("[INFO] Evaluating performance on ImageNet validation set...")
    start_time = time.time()

    preprocess_eval = build_imagenet_preprocess(processor, model_hf)

    try:
        imagenet_dataset = datasets.ImageFolder(root=str(args.imagenet_val), transform=preprocess_eval)
        wnids = imagenet_dataset.classes
        imagenet_loader = DataLoader(
            imagenet_dataset,
            batch_size=args.imagenet_batch_size,
            shuffle=False,
            num_workers=args.imagenet_workers,
            pin_memory=True,
            drop_last=False,
        )
        print(f"[INFO] Loaded ImageNet dataset: {len(imagenet_dataset)} images, {len(wnids)} classes.")
    except Exception as e:
        print(f"[ERROR] Failed to load ImageNet dataset: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0

    k = 5
    per_class_results = defaultdict(lambda: [0.0, 0.0, 0])
    total_processed = 0
    ac_dev = _autocast_device(device)
    model_hf.eval()

    with torch.no_grad():
        for images, class_labels in tqdm(imagenet_loader, desc="Evaluating ImageNet"):
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            with torch.amp.autocast(ac_dev):
                img_feats = model_hf.get_image_features(pixel_values=images)
                img_feats = F.normalize(img_feats, dim=-1)
                logits = (model_hf.logit_scale.exp() * img_feats @ classifier_imagenet)

            pred_indices = logits.topk(k, 1, True, True)[1].t()

            batch_wnids = [wnids[idx.item()] for idx in class_labels.cpu()]
            target_classifier_indices = torch.tensor(
                [imagenet_map["id_to_index"].get(wnid, -1) for wnid in batch_wnids],
                device=device, dtype=torch.long
            )

            valid_mask = target_classifier_indices.ne(-1)
            if not valid_mask.any():
                continue

            correct = pred_indices.eq(target_classifier_indices.view(1, -1).expand_as(pred_indices))[:, valid_mask]
            valid_tci = target_classifier_indices[valid_mask]

            for b in range(valid_tci.size(0)):
                tci = int(valid_tci[b].item())
                per_class_results[tci][0] += float(correct[0, b].item())
                per_class_results[tci][1] += float(correct[:k, b].any().item())
                per_class_results[tci][2] += 1
            total_processed += valid_tci.size(0)

    print(f"[INFO] ImageNet evaluation finished in {time.time() - start_time:.2f}s ({total_processed} images).")

    # Forget class
    forget_acc1 = forget_acc5 = 0.0
    forget_idx, matched_key = _resolve_imagenet_class_idx(imagenet_map, args.eval_target_name_display)
    if forget_idx == -1:
        print(f"[ERROR] Concept '{args.eval_target_name_display}' not in ImageNet map (robust resolve failed).")
        return 0.0, 0.0, 0.0, 0.0, 0

    if matched_key:
        print(f"[INFO] Matched forget concept '{args.eval_target_name_display}' -> '{matched_key}' (idx={forget_idx})")

    if forget_idx in per_class_results:
        c1, c5, total = per_class_results[forget_idx]
        if total > 0:
            forget_acc1, forget_acc5 = c1 / total, c5 / total
        name_str = imagenet_map["index_to_name_tuple"][str(forget_idx)][1]
        print(f"  - Forget Class ({name_str}): Top-1={forget_acc1*100:.2f}% ({int(c1)}/{total}), "
              f"Top-5={forget_acc5*100:.2f}% ({int(c5)}/{total})")

    # Retain classes
    retain_acc1_list, retain_acc5_list = [], []
    num_retain_found = 0
    for r_name in IMAGENET_RETAIN_CLASSES:
        r_idx, _ = _resolve_imagenet_class_idx(imagenet_map, r_name)
        if r_idx != -1 and r_idx in per_class_results:
            c1, c5, total = per_class_results[r_idx]
            if total > 0:
                retain_acc1_list.append(c1 / total)
                retain_acc5_list.append(c5 / total)
                num_retain_found += 1

    avg_retain_acc1 = float(np.mean(retain_acc1_list)) if retain_acc1_list else 0.0
    avg_retain_acc5 = float(np.mean(retain_acc5_list)) if retain_acc5_list else 0.0
    print(f"[INFO] Calculated avg retain accuracy over {num_retain_found} classes.")
    return forget_acc1, forget_acc5, avg_retain_acc1, avg_retain_acc5, num_retain_found


# ----------------- OpenCLIP->HF remap + Si JSON loader + masking + UES -----------------
def _split_qkv(mat_w: torch.Tensor) -> Dict[str, torch.Tensor]:
    d = mat_w.size(0) // 3
    return {"q": mat_w[:d, :], "k": mat_w[d:2*d, :], "v": mat_w[2*d:, :]}

def _split_qkv_bias(bias: torch.Tensor) -> Dict[str, torch.Tensor]:
    d = bias.size(0) // 3
    return {"q": bias[:d], "k": bias[d:2*d], "v": bias[2*d:]}

def remap_openclip_text_grads_to_hf(grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Supports OpenCLIP keys like:
      transformer.resblocks.{L}.mlp.c_fc.weight / c_proj.weight
      *.attn.in_proj_weight/bias, out_proj.*
    Maps to HF CLIPTextModel names under:
      text_model.encoder.layers.{L}.*
    """
    if any(k.startswith("text_model.encoder.layers.") for k in grads.keys()):
        return grads

    mapped: Dict[str, torch.Tensor] = {}
    block_re = re.compile(r"(?:^|\.)(?:transformer\.resblocks|text\.transformer\.resblocks)\.(\d+)\.")

    for k, v in grads.items():
        m = block_re.search(k)
        L = int(m.group(1)) if m else -1
        if L == -1:
            continue

        hf_prefix = f"text_model.encoder.layers.{L}."

        if k.endswith(".attn.in_proj_weight"):
            qkv = _split_qkv(v)
            mapped[hf_prefix + "self_attn.q_proj.weight"] = qkv["q"]
            mapped[hf_prefix + "self_attn.k_proj.weight"] = qkv["k"]
            mapped[hf_prefix + "self_attn.v_proj.weight"] = qkv["v"]
            continue
        if k.endswith(".attn.in_proj_bias"):
            qkvb = _split_qkv_bias(v)
            mapped[hf_prefix + "self_attn.q_proj.bias"] = qkvb["q"]
            mapped[hf_prefix + "self_attn.k_proj.bias"] = qkvb["k"]
            mapped[hf_prefix + "self_attn.v_proj.bias"] = qkvb["v"]
            continue
        if k.endswith(".attn.out_proj.weight"):
            mapped[hf_prefix + "self_attn.out_proj.weight"] = v
            continue
        if k.endswith(".attn.out_proj.bias"):
            mapped[hf_prefix + "self_attn.out_proj.bias"] = v
            continue
        if k.endswith(".mlp.c_fc.weight"):
            mapped[hf_prefix + "mlp.fc1.weight"] = v
            continue
        if k.endswith(".mlp.c_fc.bias"):
            mapped[hf_prefix + "mlp.fc1.bias"] = v
            continue
        if k.endswith(".mlp.c_proj.weight"):
            mapped[hf_prefix + "mlp.fc2.weight"] = v
            continue
        if k.endswith(".mlp.c_proj.bias"):
            mapped[hf_prefix + "mlp.fc2.bias"] = v
            continue

    return mapped if mapped else grads

def map_json_layer_to_hf(name_openclip: str) -> Optional[str]:
    """
    FIXED: supports both c_fc and c_proj weights.
    """
    m = re.match(r"transformer\.resblocks\.(\d+)\.mlp\.(c_fc|c_proj)\.weight$", name_openclip)
    if not m:
        return None
    L = int(m.group(1))
    which = m.group(2)
    return f"text_model.encoder.layers.{L}.mlp.{'fc1' if which=='c_fc' else 'fc2'}.weight"

def load_neuron_indices_json(json_path: Path, variant: str = "Si") -> Dict[str, List[int]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    section = data.get(variant, data)

    out: Dict[str, List[int]] = {}
    ranked = section.get("language", {}).get("ranked", [])
    for item in ranked:
        lname = item.get("layer", "")
        idxs  = item.get("important_idx", [])
        if not lname or not isinstance(idxs, list):
            continue
        hf = map_json_layer_to_hf(lname)
        if hf is not None:
            out[hf] = idxs
    return out

def expand_index_list_to_mask(index_list: List[int], length: int, device=None) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if not index_list:
        return mask
    idx = torch.as_tensor(index_list, dtype=torch.long, device=device)
    idx = idx[(idx >= 0) & (idx < length)]
    if idx.numel():
        mask[idx] = True
    return mask

def per_neuron_norm(G: torch.Tensor) -> Optional[torch.Tensor]:
    if G is None:
        return None
    Gf = G.flatten(1) if G.ndim == 4 else G
    return Gf.norm(p=2, dim=1) if Gf.ndim == 2 else None

def rowwise_cosine(Gf: torch.Tensor, Gr: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    if Gf is None or Gr is None or Gf.shape != Gr.shape:
        return None
    a = Gf.detach().to(dtype=torch.float32)
    b = Gr.detach().to(dtype=torch.float32)
    if a.ndim == 4:
        a = a.flatten(1)
        b = b.flatten(1)
    if a.ndim != 2:
        return None
    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1).clamp(-1.0, 1.0)

def build_row_scale_vector(forget_G: torch.Tensor, retain_G: torch.Tensor,
                           mask_bool: torch.Tensor,
                           delta: float = 1.0, gamma: float = 1.0,
                           eps: float = 1e-8) -> torch.Tensor:
    Rr  = per_neuron_norm(retain_G)
    cos = rowwise_cosine(forget_G, retain_G, eps=eps)
    if Rr is None or cos is None or not mask_bool.any():
        return torch.ones(int(mask_bool.sum()), dtype=torch.float32, device=mask_bool.device)
    align_pen = (1.0 - cos.clamp(min=0.0)).pow(float(delta))
    shield    = (eps + (Rr**2)).pow(-float(gamma))
    w_full    = align_pen * shield
    w_masked  = w_full[mask_bool.to(w_full.device)]
    return torch.clamp(w_masked.to(torch.float32), min=0.05, max=10.0)

def _rel_drop(curr: float, base: float, eps: float = 1e-6) -> float:
    if base <= eps:
        return 0.0 if curr <= base else min(1.0, (curr - base) / max(eps, curr))
    return min(1.0, max(0.0, (base - curr) / max(eps, base)))

def ues(fgt1, fgt5, test1, test5, base_f1, base_f5, base_t1, base_t5, alpha=0.5):
    fg = 0.5 * (_rel_drop(fgt1, base_f1) + _rel_drop(fgt5, base_f5))
    rl = 0.5 * (_rel_drop(test1, base_t1) + _rel_drop(test5, base_t5))
    return alpha * fg - (1.0 - alpha) * rl


# ----------------- Main -----------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Target concept: {args.eval_target_name_display} (underscore={args.eval_target_name_underscore})")
    print(f"[INFO] Outputs -> {args.outdir}")
    print(f"[INFO] Neuron JSON -> {args.neuron_json}")

    # 1) Load CLIP
    print(f"[INFO] Loading CLIP model: {args.clip_model_id}")
    model_clip     = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    processor_clip = CLIPProcessor.from_pretrained(args.clip_model_id)
    model_clip.eval()

    # 2) Load gradients
    print(f"[INFO] Loading gradients from: {args.grads_dir}")
    try:
        forget_grads = torch.load(args.grads_dir / "forget_grads.pt", map_location="cpu")
        retain_grads = torch.load(args.grads_dir / "train_grads.pt",  map_location="cpu")
        forget_grads = remap_openclip_text_grads_to_hf(forget_grads)
        retain_grads = remap_openclip_text_grads_to_hf(retain_grads)
        print("[INFO] Gradients loaded and mapped.")
    except Exception as e:
        print(f"[ERROR] Failed to load/map gradients: {e}")
        sys.exit(1)

    # 3) Load neuron indices
    try:
        idx_map = load_neuron_indices_json(Path(args.neuron_json), variant="Si")
    except Exception as e:
        print(f"[ERROR] Failed to load neuron JSON: {e}")
        sys.exit(1)

    if not idx_map:
        raise RuntimeError("Neuron JSON loaded but produced empty idx_map. Check JSON structure / variant key.")

    # 4) Build neuron masks (only for layers present in grads)
    masks_dict: Dict[str, torch.Tensor] = {}
    for ln_hf, idxs in idx_map.items():
        if ln_hf in forget_grads and isinstance(forget_grads[ln_hf], torch.Tensor) and forget_grads[ln_hf].ndim == 2:
            out_len = forget_grads[ln_hf].shape[0]
            masks_dict[ln_hf] = expand_index_list_to_mask(idxs, out_len, device="cpu")

    if not masks_dict:
        missing = sorted(set(idx_map.keys()) - set(forget_grads.keys()))
        print("[DEBUG] Example missing keys (json->hf) not found in forget_grads:", missing[:10])
        raise RuntimeError("No usable neuron masks created (layer-name mismatch between JSON and gradients).")

    # 5) Baseline for STEP SEARCH
    if args.eval_mode == "celeba":
        print("\n[INFO] Running BASELINE CelebA evaluation for step search...")
        celeba_identity_file = Path(args.celeba_root) / "list_identity_celeba.txt"
        jpg_dict_celeba = defaultdict(list)
        with open(celeba_identity_file, "r") as f:
            lines = f.readlines()[2:]
        for line in lines:
            image_id, identity_name = line.strip().split()
            jpg_dict_celeba[identity_name].append(image_id)
        name_list_celeba = tuple(sorted(jpg_dict_celeba.keys()))
        CELEB_NAMES_ALL  = [name.replace("_", " ") for name in name_list_celeba]
        CELEB_TEMPLATES  = (lambda c: f"a photo of {c}.",)

        classifier_celeba_base = build_zero_shot_classifier(
            model_clip, processor_clip.tokenizer, CELEB_NAMES_ALL,
            CELEB_TEMPLATES, device=device, use_tqdm=True
        )

        test_top1_base, test_top5_base = [], []
        target_us = args.eval_target_name_underscore
        forget1_base = forget5_base = 0.0

        for name_sp in CELEBA_EVAL_NAMES:
            name_us = name_sp.replace(" ", "_")
            if name_us not in name_list_celeba:
                continue
            tt1, tt5, count = run_name_celeba(
                model_clip, classifier_celeba_base, name_us,
                processor_clip, device, jpg_dict_celeba,
                name_list_celeba, Path(args.celeba_root)
            )
            if name_us == target_us:
                forget1_base, forget5_base = tt1, tt5
            else:
                test_top1_base.append(tt1)
                test_top5_base.append(tt5)

        test1_base = float(np.mean(test_top1_base)) if test_top1_base else 0.0
        test5_base = float(np.mean(test_top5_base)) if test_top5_base else 0.0

        del classifier_celeba_base
        torch.cuda.empty_cache()

        print(f"Baseline Forget Acc ({target_us}): Top-1={forget1_base*100:.2f}%, Top-5={forget5_base*100:.2f}%")
        print(f"Baseline Retain Acc (CelebA Avg): Top-1={test1_base*100:.2f}%, Top-5={test5_base*100:.2f}%")
        print("--------------------------------------------------")

    elif args.eval_mode == "imagenet":
        print("\n[INFO] Running BASELINE ImageNet (subset) evaluation for step search...")
        imagenet_map = load_imagenet_map(Path(args.imagenet_map_json))
        (loader_subset, subset_ds, wnid_to_subset_idx, subset_classnames, forget_subset_idx) = make_imagenet_subset_for_step_search(
            args, processor_clip, model_clip, imagenet_map, args.eval_target_name_display, IMAGENET_RETAIN_CLASSES
        )
        IMAGENET_TEMPLATES = (lambda c: f"a photo of a {c}.",)
        classifier_subset  = build_zero_shot_classifier(
            model_clip, processor_clip.tokenizer, subset_classnames, IMAGENET_TEMPLATES, device=device, use_tqdm=False
        )
        f1b, f5b, t1b, t5b = eval_imagenet_subset_step(
            model_clip, classifier_subset, loader_subset, subset_ds, wnid_to_subset_idx,
            device, k=5, forget_subset_idx=forget_subset_idx
        )
        forget1_base, forget5_base, test1_base, test5_base = f1b, f5b, t1b, t5b

        print(f"Baseline Forget Acc ({args.eval_target_name_display}): Top-1={forget1_base*100:.2f}%, Top-5={forget5_base*100:.2f}%")
        print(f"Baseline Retain Acc (ImageNet subset Avg): Top-1={test1_base*100:.2f}%, Top-5={test5_base*100:.2f}%")
        print("--------------------------------------------------")
    else:
        print("\n[WARN] eval-mode=none: step-search baseline metrics will be set to 0.")
        forget1_base = forget5_base = test1_base = test5_base = 0.0

    # 6) Plan neuron update vector U
    param_sq_sum = 0.0
    grad_sq_sum  = 0.0
    plan_updates: Dict[str, torch.Tensor] = {}
    print("[INFO] Planning neuron update vector (U)...")
    with torch.no_grad():
        for ln, mask_cpu in masks_dict.items():
            if ln not in forget_grads or ln not in retain_grads:
                continue
            Gf = forget_grads[ln].float()
            Gr = retain_grads[ln].float()
            if Gf.ndim != 2 or not mask_cpu.any():
                continue

            mask_gpu = mask_cpu.to(device)
            w = build_row_scale_vector(Gf.to(device), Gr.to(device), mask_gpu, delta=1.0, gamma=1.0, eps=1e-8)

            Gf_dev = Gf.to(device)
            Gf_norm = Gf_dev / (Gf_dev.norm(p=2, dim=1, keepdim=True) + 1e-8)

            U = torch.zeros_like(Gf_norm)
            U[mask_gpu, :] = w[:, None] * Gf_norm[mask_gpu, :]

            p_cpu = model_clip.get_parameter(ln).detach().float().cpu()
            param_sq_sum += float(p_cpu[mask_cpu, :].pow(2).sum().item())
            grad_sq_sum  += float(U.pow(2).sum().item())

            plan_updates[ln] = U.half().cpu()

            del Gf, Gr, mask_gpu, w, Gf_dev, Gf_norm, U, p_cpu

    if (not plan_updates) or grad_sq_sum <= 1e-12:
        raise RuntimeError("No effective neuron updates planned. Check masks/grads alignment.")

    params_norm = float(np.sqrt(param_sq_sum))
    grad_norm   = float(np.sqrt(grad_sq_sum))

    step_init_mag = (params_norm / (grad_norm + 1e-12)) / 10.0
    step_init     = args.step_sign * step_init_mag

    print(f"[NEURON] Target ||P_mask||={params_norm:.4f}, Update ||U||={grad_norm:.4f}, step_init={step_init:+.6f} (sign={args.step_sign})")
    print(f"[NEURON] Updating {len(plan_updates)} parameter tensors:")
    for ln, m in masks_dict.items():
        print(f"  - {ln}: {int(m.sum().item())} neurons targeted")

    # 7) Cache original params
    original_params = {ln: model_clip.get_parameter(ln).data.detach().cpu().clone() for ln in plan_updates}
    torch.cuda.empty_cache()

    # 8) Apply/restore helpers
    def apply_step_inplace(step_val: float):
        with torch.no_grad():
            for ln, U_cpu in plan_updates.items():
                p    = model_clip.get_parameter(ln)
                base = original_params[ln].to(p.device, p.dtype)
                U    = U_cpu.to(p.device, p.dtype)
                p.data.copy_(base)
                p.data.add_(U, alpha=float(step_val))

    def restore_original_inplace():
        with torch.no_grad():
            for ln in plan_updates.keys():
                p = model_clip.get_parameter(ln)
                p.data.copy_(original_params[ln].to(p.device, p.dtype))

    # 9) Step search loop
    if args.eval_mode == "none":
        print("[INFO] eval-mode=none: skipping step-search; using step_init.")
        best_step = step_init
        best_tuple = (forget1_base, forget5_base, test1_base, test5_base)
    else:
        MAX_ITERS = 10
        alpha     = 0.5
        best_step = step_init
        best_ues  = -float("inf")
        step      = step_init

        # We assume "more negative" and "more positive" both might be useful depending on sign.
        # We'll do a simple bracket search around step, expanding magnitude if needed.
        step_low = 0.0
        step_high = None  # unknown

        print("\n[INFO] Starting step search using UES...")

        best_tuple = (forget1_base, forget5_base, test1_base, test5_base)

        for it in range(MAX_ITERS):
            apply_step_inplace(step)
            try:
                if args.eval_mode == "celeba":
                    classifier_mod = build_zero_shot_classifier(
                        model_clip, processor_clip.tokenizer, CELEB_NAMES_ALL,
                        CELEB_TEMPLATES, num_classes_per_batch=10, device=device, use_tqdm=False
                    )

                    test_top1, test_top5 = [], []
                    f1 = f5 = 0.0
                    for name_sp in CELEBA_EVAL_NAMES:
                        name_us = name_sp.replace(" ", "_")
                        if name_us not in name_list_celeba:
                            continue
                        tt1, tt5, count = run_name_celeba(
                            model_clip, classifier_mod, name_us,
                            processor_clip, device, jpg_dict_celeba,
                            name_list_celeba, Path(args.celeba_root)
                        )
                        if count > 0:
                            if name_us == args.eval_target_name_underscore:
                                f1, f5 = tt1, tt5
                            else:
                                test_top1.append(tt1)
                                test_top5.append(tt5)

                    t1 = float(np.mean(test_top1)) if test_top1 else 0.0
                    t5 = float(np.mean(test_top5)) if test_top5 else 0.0
                    del classifier_mod

                else:
                    IMAGENET_TEMPLATES = (lambda c: f"a photo of a {c}.",)
                    classifier_subset = build_zero_shot_classifier(
                        model_clip, processor_clip.tokenizer, subset_classnames,
                        IMAGENET_TEMPLATES, device=device, use_tqdm=False
                    )
                    f1, f5, t1, t5 = eval_imagenet_subset_step(
                        model_clip, classifier_subset, loader_subset, subset_ds,
                        wnid_to_subset_idx, device, k=5, forget_subset_idx=forget_subset_idx
                    )
                    del classifier_subset

                score = ues(f1, f5, t1, t5, forget1_base, forget5_base, test1_base, test5_base, alpha=alpha)
                print(f"[Step {it}] step={step:+.6f} | fgt@1={f1:.3f} fgt@5={f5:.3f} | test@1={t1:.3f} test@5={t5:.3f} | UES={score:+.4f}")

                if score >= best_ues:
                    best_ues = score
                    best_step = step
                    best_tuple = (f1, f5, t1, t5)

                # Heuristic: if forget@5 is tiny, treat as "good forgetting" and set high bracket.
                if f5 <= 0.01:
                    step_high = step
                else:
                    step_low = step

                # Update step
                if step_high is None:
                    # expand magnitude
                    step = step_low * 2.0 if step_low != 0.0 else step_init * 2.0
                else:
                    step = 0.5 * (step_low + step_high)

                # Convergence
                if step_high is not None and abs(step_high - step_low) < max(1e-9, abs(step_init) * 0.01):
                    print("[INFO] Step search converged.")
                    break

            except Exception as e:
                print(f"[ERROR] Step-search iteration {it} failed at step={step:+.6f}: {e}")
                step *= 0.5
                if abs(step) < 1e-8:
                    print("[ERROR] Step became too small. Aborting search.")
                    break
            finally:
                restore_original_inplace()
                torch.cuda.empty_cache()

        print(f"[INFO] Best step (UES): {best_step:+.6f}")
        print(f"       Best tuple: fgt1={best_tuple[0]:.3f}, fgt5={best_tuple[1]:.3f}, test1={best_tuple[2]:.3f}, test5={best_tuple[3]:.3f}")

    # 10) Apply best step permanently
    print(f"[INFO] Applying best step {best_step:+.6f} permanently.")
    apply_step_inplace(best_step)
    model_clip_modified = model_clip

    # 11) Final quantitative eval
    if args.eval_mode != "none":
        print(f"\n[INFO] Running FINAL Quantitative Evaluation (Mode: {args.eval_mode})...")
        final_f1 = final_f5 = final_r1 = final_r5 = 0.0
        num_retain = 0

        if args.eval_mode == "celeba":
            celeba_identity_file = Path(args.celeba_root) / "list_identity_celeba.txt"
            jpg_dict_celeba = defaultdict(list)
            with open(celeba_identity_file, "r") as f:
                lines = f.readlines()[2:]
            for line in lines:
                image_id, identity_name = line.strip().split()
                jpg_dict_celeba[identity_name].append(image_id)
            name_list_celeba = tuple(sorted(jpg_dict_celeba.keys()))
            CELEB_NAMES_ALL  = [name.replace("_", " ") for name in name_list_celeba]
            CELEB_TEMPLATES  = (lambda c: f"a photo of {c}.",)

            final_classifier_celeba = build_zero_shot_classifier(
                model_clip_modified, processor_clip.tokenizer,
                CELEB_NAMES_ALL, CELEB_TEMPLATES,
                device=device, use_tqdm=True
            )
            final_f1, final_f5, final_r1, final_r5, num_retain = run_celeba_eval(
                model_clip_modified, final_classifier_celeba, args,
                processor_clip, device, jpg_dict_celeba, name_list_celeba
            )
            del final_classifier_celeba

        elif args.eval_mode == "imagenet":
            imagenet_map_final = load_imagenet_map(Path(args.imagenet_map_json))

            # Build 1000-way classnames (main name per class)
            idx_to_details = imagenet_map_final["index_to_name_tuple"]
            imagenet_class_names = []
            for idx_str, (_wnid, names_str) in idx_to_details.items():
                main_name = names_str.split(",")[0].strip()
                imagenet_class_names.append(main_name)

            IMAGENET_TEMPLATES = (lambda c: f"a photo of a {c}.",)
            final_classifier_imagenet = build_zero_shot_classifier(
                model_clip_modified, processor_clip.tokenizer,
                imagenet_class_names, IMAGENET_TEMPLATES,
                device=device, use_tqdm=True
            )
            final_f1, final_f5, final_r1, final_r5, num_retain = run_imagenet_eval(
                model_clip_modified, final_classifier_imagenet, args,
                processor_clip, imagenet_map_final, device
            )
            del final_classifier_imagenet

        print("\n--- FINAL Quantitative Results ---")
        print(f"Forget Accuracy ({args.eval_target_name_display}): Top-1={final_f1*100:.2f}%, Top-5={final_f5*100:.2f}%")
        print(f"Retain Accuracy (Avg. over {num_retain} classes): Top-1={final_r1*100:.2f}%, Top-5={final_r5*100:.2f}%")
        print("--------------------------------\n")

    # 12) SD 2.1 swap-in and renders
    print("[INFO] Preparing for Stable Diffusion...")
    updated_text_params = {n: p.detach().half().cpu() for n, p in model_clip_modified.text_model.named_parameters()}

    model_clip_modified.to("cpu")
    del model_clip_modified
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[INFO] Loading SD pipeline: {args.sd_id}")
    try:
        te_for_pipe = CLIPTextModel.from_pretrained(args.sd_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_id,
            torch_dtype=torch.float16,
            text_encoder=te_for_pipe,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        print("[INFO] SD pipeline loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load SD pipeline: {e}")
        sys.exit(1)

    # Baseline render
    print("[INFO] Rendering baseline (original SD)...")
    generator_base = torch.Generator(device=device).manual_seed(args.seed)
    img_before = render(pipe, args.prompt, args.steps, args.guidance, generator_base)
    img_before.save(args.outdir / "sd_original_concept.png")

    # Swap edited TE weights
    print("[INFO] Swapping unlearned text encoder weights into SD text encoder...")
    te = pipe.text_encoder.text_model
    num_params_swapped = 0
    with torch.no_grad():
        target_sd = te.state_dict()
        for name, source_param_cpu in updated_text_params.items():
            if name in target_sd and target_sd[name].shape == source_param_cpu.shape:
                target_sd[name].copy_(source_param_cpu.to(target_sd[name].device, dtype=target_sd[name].dtype))
                num_params_swapped += 1
    print(f"[INFO] Swapped {num_params_swapped} parameter tensors.")
    if num_params_swapped == 0:
        print("[ERROR] No parameters were swapped! Check name alignment between CLIPModel.text_model and SD text_encoder.")

    # Render main prompt (after swap)
    print(f"[INFO] Rendering swapped (unlearned) for prompt: '{args.prompt}'")
    generator_swap = torch.Generator(device=device).manual_seed(args.seed)
    img_after = render(pipe, args.prompt, args.steps, args.guidance, generator_swap)
    img_after.save(args.outdir / f"sd_unlearned_{args.eval_target_name_underscore}.png")

    # Extra prompts
    extra_prompts = [
        ("A photo of school bus",      "bus"),
        ("A photo of a car", "car"),
        ("A photo of a van",      "van"),
        ("A photo of ambulance",      "ambu"),
        ("A photo of truck", "truck"),
    ]
    print("[INFO] Rendering additional prompts (sanity checks)...")
    for i, (pmt, suffix) in enumerate(extra_prompts):
        gen = torch.Generator(device=device).manual_seed(args.seed + 1 + i)
        im = render(pipe, pmt, args.steps, args.guidance, gen)
        im.save(args.outdir / f"sd_unlearned_{suffix}.png")

    print(f"\n[INFO] Done. Outputs saved under: {args.outdir}")


if __name__ == "__main__":
    main()
