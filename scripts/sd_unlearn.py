#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Neuron-based unlearning + SD visualization with UES-based bracket/binary search (no floors).

import os, warnings, requests, re, json
from pathlib import Path
from io import BytesIO
from typing import Callable, List, Optional, Sequence, Union, Dict, Tuple
from itertools import islice

# Keep diffusers from passing offload_state_dict into old transformers
os.environ["DIFFUSERS_OFFLOAD_STATE_DICT"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt

from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

warnings.filterwarnings('ignore')

# ========= PATHS (edit if your layout differs) =========
REPO_ROOT   = Path(__file__).resolve().parents[1]  # assumes file in ~/SLUG/scripts/
DATA_ROOT   = REPO_ROOT / "/home/rania/SLUG/data"
CELEBA_ROOT = DATA_ROOT / "/home/rania/SLUG/data/celeba"  # needs img_align_celeba/ + list_identity_celeba.txt

CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PAIR          = "ViT-H-14 laion2B-s32B-b79K"
CELEB_NAME    = "elon_musk"

# Grads (already computed)
MASK_ROOT = Path(f"/home/rania/SLUG/results/grads/{CELEB_NAME}_ViT-H-14_laion2B-s32B-b79K")

# Important neurons JSON (first existing is used)
NEURON_JSON_CANDIDATES = [
    Path(f"/home/rania/SLUG/results/neuron importance_global_sd/{CELEB_NAME}/ViT-H-14_laion2B-s32B-b79K_Si.json"),
    Path(f"/home/rania/SLUG/results/neuron_importance_global/{CELEB_NAME}/ViT-H-14_laion2B-s32B-b79K_Si.json"),
    Path(f"/home/rania/SLUG/results/neuron_importance/{CELEB_NAME}/ViT-H-14_laion2B-s32B-b79K_Si.json"),
    Path(f"/home/rania/SLUG/results/important_neurons.json"),
]

SD_MODEL_ID = "stabilityai/stable-diffusion-2-1"
OUT_ROOT    = REPO_ROOT / f"results/output-SD_neuron2/ViT-H-14/{CELEB_NAME}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
# =======================================================

DO_COSINE_VIS = False
texts = ["Elon Musk", "Mark Zuckerberg", "Jeff Bezos", "Taylor Swift", "Lady Gaga", "Chris Brown", "Bruce_Lee"]

accelerator = Accelerator()
device = accelerator.device

# ----------------- helpers -----------------
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
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
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
        inputs = tokenizer(texts_local, padding=True, return_tensors="pt",
                           truncation=True, max_length=77).to(device)
        class_embeddings = model.get_text_features(**inputs)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run_name(model, classifier, name, processor, device,
             jpg_dict: Dict[str, List[str]], name_list: Sequence[str], celeba_root: Path):
    label = name_list.index(name)
    top1, top5, n = 0., 0., 0.
    for image_id in jpg_dict[name]:
        image_path = celeba_root / "img_align_celeba" / image_id
        image = Image.open(image_path).convert("RGB")
        target = torch.tensor([label]).to(device)
        inputs = processor(text=[f"{name.replace('_',' ')}."], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        image_features = outputs.image_embeds
        logits = 100. * image_features @ classifier
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += 1
    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

# --------- name mapping: OpenCLIP -> HF text encoder ---------
def _split_qkv(mat_w: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert mat_w.dim() == 2 and mat_w.size(0) % 3 == 0, "in_proj_weight must be (3*D, D)"
    d = mat_w.size(0) // 3
    q, k, v = mat_w[:d, :], mat_w[d:2*d, :], mat_w[2*d:, :]
    return {"q": q, "k": k, "v": v}

def _split_qkv_bias(bias: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert bias.dim() == 1 and bias.size(0) % 3 == 0, "in_proj_bias must be (3*D)"
    d = bias.size(0) // 3
    q, k, v = bias[:d], bias[d:2*d], bias[2*d:]
    return {"q": q, "k": k, "v": v}

def remap_openclip_text_grads_to_hf(grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    if any(k.startswith("text_model.encoder.layers.") for k in grads.keys()):
        return grads
    block_re = re.compile(r"(?:^|\.)(?:transformer\.resblocks|text\.transformer\.resblocks)\.(\d+)\.")
    for k, v in grads.items():
        m = block_re.search(k)
        if not m:
            continue
        L = int(m.group(1))
        hf_prefix = f"text_model.encoder.layers.{L}."
        if k.endswith(".attn.in_proj_weight"):
            qkv = _split_qkv(v)
            mapped[hf_prefix + "self_attn.q_proj.weight"] = qkv["q"]
            mapped[hf_prefix + "self_attn.k_proj.weight"] = qkv["k"]
            mapped[hf_prefix + "self_attn.v_proj.weight"] = qkv["v"]; continue
        if k.endswith(".attn.in_proj_bias"):
            qkvb = _split_qkv_bias(v)
            mapped[hf_prefix + "self_attn.q_proj.bias"] = qkvb["q"]
            mapped[hf_prefix + "self_attn.k_proj.bias"] = qkvb["k"]
            mapped[hf_prefix + "self_attn.v_proj.bias"] = qkvb["v"]; continue
        if k.endswith(".attn.out_proj.weight"):
            mapped[hf_prefix + "self_attn.out_proj.weight"] = v; continue
        if k.endswith(".attn.out_proj.bias"):
            mapped[hf_prefix + "self_attn.out_proj.bias"] = v; continue
        if k.endswith(".mlp.c_fc.weight"):
            mapped[hf_prefix + "mlp.fc1.weight"] = v; continue
        if k.endswith(".mlp.c_fc.bias"):
            mapped[hf_prefix + "mlp.fc1.bias"] = v; continue
        if k.endswith(".mlp.c_proj.weight"):
            mapped[hf_prefix + "mlp.fc2.weight"] = v; continue
        if k.endswith(".mlp.c_proj.bias"):
            mapped[hf_prefix + "mlp.fc2.bias"] = v; continue
    return mapped if mapped else grads

def map_json_layer_to_hf(name_openclip: str) -> Optional[str]:
    # e.g. "transformer.resblocks.20.mlp.c_fc.weight" -> "text_model.encoder.layers.20.mlp.fc1.weight"
    m = re.match(r"transformer\.resblocks\.(\d+)\.mlp\.c_fc\.weight$", name_openclip)
    if not m:
        return None
    L = int(m.group(1))
    return f"text_model.encoder.layers.{L}.mlp.fc1.weight"

# ----------------- Si JSON loader + masks -----------------
def load_neuron_indices_json(json_path: Path, variant: str = "Si") -> Dict[str, List[int]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    section = data.get(variant, data)
    out: Dict[str, List[int]] = {}
    ranked = section.get("language", {}).get("ranked", [])
    for item in ranked:
        lname = item.get("layer", "")
        idxs  = item.get("important_idx", [])
        if lname and isinstance(idxs, list):
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

# ----------------- per-row weighting -----------------
def per_neuron_norm(G: torch.Tensor) -> Optional[torch.Tensor]:
    if G is None: return None
    if G.ndim == 2: return G.norm(p=2, dim=1)
    if G.ndim == 4: return G.flatten(1).norm(p=2, dim=1)
    return None

def rowwise_cosine(Gf: torch.Tensor, Gr: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    if Gf is None or Gr is None or Gf.shape != Gr.shape: return None
    a = Gf.detach().to(dtype=torch.float32)
    b = Gr.detach().to(dtype=torch.float32)
    if a.ndim == 4:
        a = a.flatten(1); b = b.flatten(1)
    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)
    return (a * b).sum(dim=1).clamp(-1.0, 1.0)

def build_row_scale_vector(forget_G: torch.Tensor,
                           retain_G: torch.Tensor,
                           mask_bool: torch.Tensor,
                           delta: float = 1.0, gamma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    # Slightly stronger retain shielding (gamma=1.0). No floors.
    Rr  = per_neuron_norm(retain_G)              # [out]
    cos = rowwise_cosine(forget_G, retain_G, eps=eps)
    if Rr is None or cos is None:
        return torch.ones_like(mask_bool, dtype=torch.float32)[mask_bool]
    align_pen = (1.0 - cos.clamp(min=0.0)).pow(float(delta))
    shield    = (eps + (Rr ** 2)).pow(-float(gamma))
    w_full    = align_pen * shield
    return torch.clamp(w_full[mask_bool].to(torch.float32), min=0.05, max=10.0)

# ------- UES (no floors) -------
def _rel_drop(curr: float, base: float, eps: float = 1e-6) -> float:
    # relative drop in [0,1]; 0 if improved or equal
    if base <= eps:
        return 0.0 if curr <= base else min(1.0, (curr - base) / max(eps, curr))
    return min(1.0, max(0.0, (base - curr) / max(eps, base)))

def ues(fgt1, fgt5, test1, test5, base_f1, base_f5, base_t1, base_t5, alpha=0.5):
    forget_gain = 0.5 * (_rel_drop(fgt1, base_f1) + _rel_drop(fgt5, base_f5))
    retain_loss = 0.5 * (_rel_drop(test1, base_t1) + _rel_drop(test5, base_t5))
    return alpha * forget_gain - (1.0 - alpha) * retain_loss

# ----------------- main script -----------------

# 1) Load CLIP
model_clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
processor_clip = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

# 2) CelebA identities
from collections import defaultdict
file_image_name = CELEBA_ROOT / "list_identity_celeba.txt"
with open(file_image_name, 'r') as f:
    lines = f.readlines()
jpg_dict = defaultdict(list)
for line in lines[2:]:
    image_id, identity_name = line.strip().split()
    jpg_dict[identity_name].append(image_id)
name_set = set(jpg_dict.keys())
name_list = tuple(sorted(name_set))
CELEB_NAMES = [name.replace('_', ' ') for name in name_list]
CELEB_TEMPLATES = (lambda c: f"a photo of {c}.",)

classifier_celeb = build_zero_shot_classifier(
    model_clip, tokenizer=processor_clip.tokenizer,
    classnames=CELEB_NAMES, templates=CELEB_TEMPLATES,
    num_classes_per_batch=10, device=device, use_tqdm=True,
)

# Quick check
n = "Elon_Musk"
t1, t5 = run_name(model_clip, classifier_celeb, n, processor_clip, device, jpg_dict, name_list, CELEBA_ROOT)
print("Testing CLIP zero-shot classifier ...")
print(f"[{n}] top1: {t1*100:.2f}%, top5: {t5*100:.2f}%")

# 3) Load grads & map names
forget_grads = torch.load(MASK_ROOT / 'forget_grads.pt', map_location='cpu')
retain_grads = torch.load(MASK_ROOT / 'train_grads.pt',   map_location='cpu')  # IMPORTANT: should be non-Elon keep set
forget_grads = remap_openclip_text_grads_to_hf(forget_grads)
retain_grads = remap_openclip_text_grads_to_hf(retain_grads)

# 4) Load Si JSON
neuron_json = next((p for p in NEURON_JSON_CANDIDATES if p.exists()), NEURON_JSON_CANDIDATES[0])
if not neuron_json.exists():
    raise FileNotFoundError("Important-neurons JSON not found at any of:\n" + "\n".join(map(str, NEURON_JSON_CANDIDATES)))
print(f"[Si] Using neuron-importance file: {neuron_json}")
idx_map = load_neuron_indices_json(neuron_json, variant="Si")

# 5) Build row masks (fc1 rows = hidden neurons)
masks_dict: Dict[str, torch.Tensor] = {}
for ln_hf, idxs in idx_map.items():
    if ln_hf not in forget_grads:  # skip any layers not present in grads
        continue
    G = forget_grads[ln_hf]
    if G.ndim != 2:
        continue
    out_len = G.shape[0]
    masks_dict[ln_hf] = expand_index_list_to_mask(idxs, out_len, device="cpu")

if not masks_dict:
    raise RuntimeError("No usable neuron masks from JSON (check names and grads).")

# 6) Baseline metrics (original)
test_top1_base, test_top5_base = [], []
for nm in texts:
    nm_u = nm.replace(' ', '_')
    with torch.no_grad():
        tt1, tt5 = run_name(model_clip, classifier_celeb, nm_u, processor_clip, device, jpg_dict, name_list, CELEBA_ROOT)
    if nm_u == CELEB_NAME:
        forget1_base, forget5_base = tt1, tt5
    else:
        test_top1_base.append(tt1); test_top5_base.append(tt5)
test1_base  = float(np.mean(test_top1_base))
test5_base  = float(np.mean(test_top5_base))

print(f"Original Forget acc: Top-1={forget1_base:.2f}, Top-5={forget5_base:.2f}")
print(f"Original Retain acc: Top-1={test1_base:.2f}, Top-5={test5_base:.2f}")

# 7) Plan masked global update U and initial step
param_sq_sum = 0.0
grad_sq_sum  = 0.0
plan_updates: Dict[str, torch.Tensor] = {}

with torch.no_grad():
    for ln, mask in masks_dict.items():
        if ln not in forget_grads or ln not in retain_grads:
            continue
        Gf = forget_grads[ln].float()
        Gr = retain_grads[ln].float()
        if Gf.ndim != 2 or (not mask.any()):
            continue

        w = build_row_scale_vector(Gf, Gr, mask, delta=1.0, gamma=1.0, eps=1e-8)

        # Optional: row-normalize forget grads to stop huge rows from dominating
        Gf_norm = Gf / (Gf.norm(p=2, dim=1, keepdim=True) + 1e-8)

        U = torch.zeros_like(Gf, dtype=torch.float32)
        U[mask, :] = w[:, None] * Gf_norm[mask, :]

        p_cpu = model_clip.get_parameter(ln).detach().float().cpu()
        param_sq_sum += float((p_cpu[mask, :] ** 2).sum().item())
        grad_sq_sum  += float((U ** 2).sum().item())

        plan_updates[ln] = U.half().cpu()

if (not plan_updates) or grad_sq_sum <= 0:
    raise RuntimeError("No effective neuron-masked updates from Si JSON + grads. Check names/paths.")

params_norm = np.sqrt(param_sq_sum)
grad_norm   = np.sqrt(grad_sq_sum)
step_init   = - (params_norm / (grad_norm + 1e-12)) / 10.0  # conservative start

print(f"[NEURON] ||params_sel||={params_norm:.6f}, ||U||={grad_norm:.6f}, init step={step_init:.5g}")
print("[Si] planned params:", len(plan_updates))
for ln, m in masks_dict.items():
    print(f"  - {ln}: masked_rows={int(m.sum().item())}")

# Cache originals on CPU
with torch.no_grad():
    original_params: Dict[str, torch.Tensor] = {
        ln: model_clip.get_parameter(ln).data.detach().cpu().clone()
        for ln in plan_updates.keys()
    }
torch.cuda.empty_cache()

def apply_step_inplace(step_val: float):
    with torch.no_grad():
        for ln, U_cpu in plan_updates.items():
            p = model_clip.get_parameter(ln)
            base = original_params[ln].to(device, dtype=p.dtype, non_blocking=True)
            U    = U_cpu.to(device, dtype=p.dtype, non_blocking=True)
            p.data.copy_(base)
            p.data.add_(step_val, U)
            del base, U
    torch.cuda.empty_cache()

def restore_original_inplace():
    with torch.no_grad():
        for ln in plan_updates.keys():
            p = model_clip.get_parameter(ln)
            p.data.copy_(original_params[ln].to(device, dtype=p.dtype, non_blocking=True))
    torch.cuda.empty_cache()

# 8) 1-D bracket/binary search with UES selection (no floors)
MAX_ITERS = 10
alpha = 0.5  # UES balance
best_step  = step_init
best_ues   = -1e9
best_tuple = (forget1_base, forget5_base, test1_base, test5_base)

step = step_init
step_low, step_high = 0.0, float('inf')

for it in range(MAX_ITERS):
    apply_step_inplace(step)

    # small-batch classifier to save VRAM
    classifier_mod = build_zero_shot_classifier(
        model_clip,
        tokenizer=processor_clip.tokenizer,
        classnames=CELEB_NAMES,
        templates=CELEB_TEMPLATES,
        num_classes_per_batch=5,
        device=device,
        use_tqdm=False,
    )

    # Evaluate
    test_top1, test_top5 = [], []
    for nm in texts:
        nm_u = nm.replace(' ', '_')
        with torch.no_grad():
            tt1, tt5 = run_name(model_clip, classifier_mod, nm_u, processor_clip, device, jpg_dict, name_list, CELEBA_ROOT)
        if nm_u == CELEB_NAME:
            f1, f5 = tt1, tt5
        else:
            test_top1.append(tt1); test_top5.append(tt5)
    t1 = float(np.mean(test_top1)) if test_top1 else 0.0
    t5 = float(np.mean(test_top5)) if test_top5 else 0.0

    score = ues(f1, f5, t1, t5, base_f1=forget1_base, base_f5=forget5_base, base_t1=test1_base, base_t5=test5_base, alpha=alpha)
    print(f"[it {it}] step={step:+.6f} | fgt@1={f1:.2f} fgt@5={f5:.2f} | test@1={t1:.2f} test@5={t5:.2f} | UES={score:.4f}")

    if score > best_ues:
        best_ues   = score
        best_step  = step
        best_tuple = (f1, f5, t1, t5)

    # bracket/binary rule you requested:
    if f5 <= 0.0:  # too strong; reduce magnitude
        step_high = step
        step = (step_low + step_high) / 2.0
    else:          # not strong enough; increase magnitude
        step_low = step
        if np.isfinite(step_high):
            step = (step_low + step_high) / 2.0
        else:
            step = step * 2.0  # more negative

    restore_original_inplace()
    del classifier_mod
    torch.cuda.empty_cache()

print(f"[NEURON] best_step (by UES): {best_step:+.6f} with (fgt1={best_tuple[0]:.2f}, fgt5={best_tuple[1]:.2f}, test1={best_tuple[2]:.2f}, test5={best_tuple[3]:.2f})")

# 9) Apply the best step permanently
apply_step_inplace(best_step)
model_clip_modified = model_clip

# 10) SD 2.1 swap-in and renders
updated_text_params = {n: p.detach().half().cpu() for n, p in model_clip_modified.text_model.named_parameters()}

try:
    del classifier_celeb
except Exception:
    pass

model_clip_modified.to("cpu")
torch.cuda.empty_cache()
import gc
gc.collect(); torch.cuda.empty_cache()

# load SD with explicit TE to avoid offload_state_dict path
te_for_pipe = CLIPTextModel.from_pretrained(
    SD_MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=torch.float16,
    text_encoder=te_for_pipe,
    low_cpu_mem_usage=False,
    device_map=None,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Baseline render
prompt = "A photo of Elon Musk"
img = pipe(prompt).images[0]
plt.figure(); plt.imshow(img); plt.axis('off')
plt.title(f"Original SD output for: '{prompt}'")
plt.tight_layout(); plt.savefig(OUT_ROOT / "sd_original_musk.png", dpi=150)

# In-place swap of edited TE weights
te = pipe.text_encoder.text_model
orig_te_state = {k: v.detach().half().cpu() for k, v in te.state_dict().items()}
with torch.no_grad():
    dev = next(te.parameters()).device
    for name, p in te.named_parameters():
        if name in updated_text_params:
            p.data.copy_(updated_text_params[name].to(dev, dtype=p.dtype))

# Renders with unlearned TE
prompt = "A photo of Elon Musk"
img = pipe(prompt).images[0]
plt.figure(); plt.imshow(img); plt.axis('off')
plt.title(f"Unlearned SD output for: '{prompt}'")
plt.tight_layout(); plt.savefig(OUT_ROOT / "sd_unlearned_musk.png", dpi=150)

for other_prompt, fname in [
    ("An photo of a mark_zuckerberg", "sd_unlearned_mark.png"),
    ("A photo of a jeff_bezos", "sd_unlearned_bezos.png"),
    ("A photo of a sundar_pichai", "sd_unlearned_sundar.png"),
    ("A photo of a bill_gates", "sd_unlearned_bill.png"),
    ("A photo of a larry_page", "sd_unlearned_larry.png"),


]:
    im = pipe(other_prompt).images[0]
    plt.figure(); plt.imshow(im); plt.axis('off')
    plt.title(f"Unlearned SD output for: '{other_prompt}'")
    plt.tight_layout(); plt.savefig(OUT_ROOT / fname, dpi=150)

# Restore TE
with torch.no_grad():
    dev = next(te.parameters()).device
    for name, p in te.named_parameters():
        p.data.copy_(orig_te_state[name].to(dev, dtype=p.dtype))

print(f"Done. Outputs saved under: {OUT_ROOT}")
