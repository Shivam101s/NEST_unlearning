#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neuron-importance VLM unlearning (vision-only) — CLIP/SD-matched variant

Keeps the exact method you used for CLIP & SD:
- U construction:  U_i = w_i * (Gf_i / ||Gf_i||_2)
- Row weights:     w_i = (1 - clamp_cos)^delta * (eps + ||Gr_i||^2)^(-gamma)
- Shielding:       gamma = 1.0
- Ratio init:      step0 = - (||W_sel|| / ||U||) / init_div
- Search schedule: same bracket/binary loop
- Step selection:  UES (balanced forget/retain), like your CLIP/SD code

Notes:
- Vision-only edits (uses Si JSON "vision" ranked rows mapped to HF names).
- LM quantized to 8-bit; CLIP vision tower in true FP16 on CUDA.
"""

import os, re, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    CLIPModel,
    CLIPVisionModel,
    BitsAndBytesConfig,
)

# ========= Defaults (edit paths/IDs as needed) =========
CELEB_NAME_DEFAULT        = "Tom_Cruise"
RESULTS_ROOT_DEFAULT      = Path("/home/rania/SLUG/results/grads")
SAVE_DIR_DEFAULT          = Path("/media/rania/02B6436D0A908CAC/VLM_weights/vlm_nest_cruise")

CLIP_MODEL_ID_DEFAULT     = "openai/clip-vit-large-patch14-336"   # ViT-L/14 @ 336px
LLAVA_MODEL_ID_DEFAULT    = "llava-hf/llava-1.5-7b-hf"

TARGET_DATASET_DEFAULT    = "ytan-ucr/mu_llava_tom_cruise"
RETAIN_DATASET_DEFAULT    = "ytan-ucr/mu_llava_celeb"
TEST_DATASET_DEFAULT      = None  # optional third set

NEURON_JSON_DEFAULT       = Path("/home/rania/SLUG/results/neuron importance_global_vlm/Tom_Cruise/openai_clip-vit-large-patch14-336_Si.json")

MAX_ITERS_DEFAULT         = 10
INITIAL_DIVISOR_DEFAULT   = 10   # step0 = -(||W_sel||/||U||)/INITIAL_DIVISOR

# Eval (slightly higher retain subset for stabler UES; adjust as you like)
BATCH_SIZE_DEFAULT        = 1
EVAL_MAX_NEW_TOKENS       = 24
SUBSET_TARGET             = 10   # was 10
SUBSET_RETAIN             = 10   # was 10
SUBSET_TEST               = 10
SEED                      = 1234

# Row-weight hyperparams (JSON can override)
DELTA_DEFAULT             = 1.0
EPS_DEFAULT               = 1e-6
GAMMA_DEFAULT             = 1.0  # match SD/CLIP shielding

# UES balance (same as your other scripts)
UES_ALPHA                 = 0.5

# Numeric safety for cosine
COS_EPS                   = 1e-6

# Perf toggles
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========= Utils =========
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset(name: Optional[str], subset_size: Optional[int] = None):
    if name is None:
        return None
    try:
        ds = load_dataset(name, split="test")
    except ValueError:
        print(f"[get_dataset] Split 'test' not found for {name}. Using 'train'.")
        ds = load_dataset(name, split="train")
    if subset_size is not None and subset_size < len(ds):
        ds = ds.select(range(subset_size))
    for col in ("image", "question", "answer"):
        if col not in ds.column_names:
            raise ValueError(f"Dataset {name} must contain column '{col}'. Found: {ds.column_names}")
    return ds

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str) -> set:
    return set(_normalize_text(s).split())

def strong_match(generated: str, truth: str) -> bool:
    g_norm = _normalize_text(generated)
    t_norm = _normalize_text(truth)
    if t_norm and t_norm in g_norm:
        return True
    gt, tt = _token_set(generated), _token_set(truth)
    if not tt:
        return False
    overlap = len(gt & tt) / len(tt)
    return overlap >= 0.6

def build_chat_prompt(processor: AutoProcessor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)

def cache_batches_cpu(processor: AutoProcessor, dataset, batch_size: int = BATCH_SIZE_DEFAULT):
    if dataset is None:
        return None
    cached = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [build_chat_prompt(processor, q) for q in batch["question"]]
        images  = [img.convert("RGB") for img in batch["image"]]
        enc = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        truths = [a for a in batch["answer"]]
        cached.append((enc, truths))
    return cached

@torch.inference_mode()
def eval_cached_cpu2gpu(model: LlavaForConditionalGeneration,
                        processor: AutoProcessor,
                        cached_batches,
                        max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> Optional[float]:
    if cached_batches is None:
        return None
    model.eval()
    try:
        model.generation_config.use_cache = False
    except Exception:
        pass

    correct = total = 0
    for enc, truths in cached_batches:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        try:
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gen = model.generate(
                **enc,
                max_new_tokens=max(8, max_new_tokens // 2),
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )
        outs = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for out, tgt in zip(outs, truths):
            gen_ans = out.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in out else out.strip()
            if strong_match(gen_ans, tgt):
                correct += 1
            total += 1
        del enc
        torch.cuda.empty_cache()
    return float(correct) / float(total) if total > 0 else 0.0


# ========= Vision tower helpers =========
def get_vision_tower(model: LlavaForConditionalGeneration) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "vision_tower") and isinstance(model.model.vision_tower, nn.Module):
        return model.model.vision_tower
    if hasattr(model, "vision_tower") and isinstance(model.vision_tower, nn.Module):
        return model.vision_tower
    raise AttributeError("Could not locate vision tower on the LLaVA model.")

def set_vision_tower(model: LlavaForConditionalGeneration, vt: nn.Module):
    if hasattr(model, "vision_tower"):
        model.vision_tower = vt
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        model.model.vision_tower = vt


# ========= Load LLaVA with 8-bit LM + FP16 vision =========
def load_llava_8bit_lm_fp16_vision(model_id: str,
                                   clip_vision_id: str,
                                   offload_dir: Path | None = None):
    if offload_dir: offload_dir.mkdir(parents=True, exist_ok=True)
    bnb8 = BitsAndBytesConfig(load_in_8bit=True)
    max_memory = {0: "10GiB", "cpu": "120GiB"}  # adjust if needed

    print("[llava] loading 8-bit LM + FP16 vision …")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb8,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=str(offload_dir) if offload_dir else None,
        max_memory=max_memory,
        attn_implementation="eager",
    )

    vt = CLIPVisionModel.from_pretrained(clip_vision_id, dtype=torch.float16).to(device)
    set_vision_tower(model, vt)
    return model


# ========= Si JSON loader + name remap =========
def _remap_openai_to_hf(openai_name: str) -> str:
    s = openai_name
    s = s.replace("visual.transformer.resblocks.", "vision_model.encoder.layers.")
    s = s.replace(".mlp.c_fc.weight", ".mlp.fc1.weight")
    s = s.replace(".mlp.c_proj.weight", ".mlp.fc2.weight")
    s = s.replace(".attn.out_proj.weight", ".self_attn.out_proj.weight")
    s = s.replace(".attn.in_proj_weight", ".self_attn.in_proj_weight")
    s = s.replace(".attn.q_proj.weight", ".self_attn.q_proj.weight")
    s = s.replace(".attn.k_proj.weight", ".self_attn.k_proj.weight")
    s = s.replace(".attn.v_proj.weight", ".self_attn.v_proj.weight")
    return s

def load_neuron_importance_from_ranked(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    cfg = data.get("config", {})
    cfg_delta = float(cfg.get("delta", DELTA_DEFAULT))
    cfg_gamma = float(cfg.get("gamma", GAMMA_DEFAULT))
    cfg_eps   = float(cfg.get("eps",   EPS_DEFAULT))

    if "Si" in data and "vision" in data["Si"] and "ranked" in data["Si"]["vision"]:
        ranked = data["Si"]["vision"]["ranked"]
    elif "vision" in data and "ranked" in data["vision"]:
        ranked = data["vision"]["ranked"]
    else:
        raise ValueError("Could not find 'Si.vision.ranked' in the JSON.")

    rows_by_layer: Dict[str, List[int]] = {}
    for item in ranked:
        ln_raw = item.get("layer")
        rows = item.get("important_idx", [])
        if not ln_raw or not isinstance(rows, list):
            continue
        ln = _remap_openai_to_hf(ln_raw)
        rows_by_layer.setdefault(ln, []).extend(int(i) for i in rows)
    for ln in list(rows_by_layer.keys()):
        rows_by_layer[ln] = sorted(set(rows_by_layer[ln]))

    return rows_by_layer, {"delta": cfg_delta, "gamma": cfg_gamma, "eps": cfg_eps}


# ========= Build masked, weighted update U (vision-only; SD/CLIP form) =========
def build_masked_weighted_update(
    forget_grads: Dict[str, torch.Tensor],
    retain_grads: Dict[str, torch.Tensor],
    neuron_rows: Dict[str, List[int]],
    llava_vision_state: Dict[str, torch.Tensor],
    *,
    delta: float,
    eps: float,
    gamma: float,
) -> Tuple[Dict[str, torch.Tensor], float, float, Dict[str, str]]:
    """
    U_i = w_i * (Gf_i / ||Gf_i||_2) on selected rows
    w_i = (1 - clamp_cos)^delta * (eps + ||Gr_i||^2)^(-gamma)
    """
    cos = nn.CosineSimilarity(dim=-1, eps=COS_EPS)

    U_layers: Dict[str, torch.Tensor] = {}
    wsel_sq_sum = 0.0
    u_sq_sum = 0.0
    used = skipped_name = skipped_shape = dropped_rows = 0

    for ln, rows in neuron_rows.items():
        if ln not in forget_grads or ln not in retain_grads or ln not in llava_vision_state:
            skipped_name += 1
            continue

        gf = forget_grads[ln]
        gr = retain_grads[ln]
        W  = llava_vision_state[ln]
        if W.dim() != 2:
            print(f"[warn] param not 2D for {ln}: shape={tuple(W.shape)}; skipping.")
            skipped_shape += 1
            continue

        out_dim = gf.shape[0]
        gf2 = gf.view(out_dim, -1).to(torch.float32)
        gr2 = gr.view(out_dim, -1).to(torch.float32)
        W2  = W.view(out_dim, -1).to(torch.float32)
        if gf2.shape != W2.shape:
            print(f"[warn] shape mismatch for {ln}: param {tuple(W2.shape)} vs grad {tuple(gf2.shape)}; skipping.")
            skipped_shape += 1
            continue

        # weights
        cos_vals = cos(gr2, gf2)                   # [out]
        clamp_cos = torch.clamp(cos_vals, min=0.0) # max(0, cos)
        gr_row_norm2 = torch.sum(gr2 * gr2, dim=-1)
        w_row = (1.0 - clamp_cos).pow(delta) * (eps + gr_row_norm2).pow(-gamma)

        # build U: selected rows only; normalize Gf rows
        U = torch.zeros_like(W2)
        for r in rows:
            if not (0 <= r < out_dim): 
                dropped_rows += 1; continue
            if not (torch.isfinite(w_row[r]).item() and torch.isfinite(gf2[r, :]).all().item()):
                dropped_rows += 1; continue
            nr = torch.norm(gf2[r, :], p=2)
            if nr <= 0 or not torch.isfinite(nr).item():
                dropped_rows += 1; continue
            U[r, :].copy_(w_row[r] * (gf2[r, :] / (nr + 1e-8)))

        # norms computed on the selected rows only
        sel_mask = torch.zeros(out_dim, dtype=torch.bool)
        for r in rows:
            if 0 <= r < out_dim:
                sel_mask[r] = True
        wsel_sq_sum += float(torch.sum((W2[sel_mask, :].to(torch.float64)) ** 2).item())
        u_sq_sum    += float(torch.sum((U[sel_mask, :].to(torch.float64))  ** 2).item())

        U_layers[ln] = U.to(W.dtype).view_as(W)
        used += 1

    W_sel_norm = math.sqrt(max(wsel_sq_sum, 0.0))
    U_norm     = math.sqrt(max(u_sq_sum,  0.0))
    stats = {
        "used": str(used),
        "skipped_name": str(skipped_name),
        "skipped_shape": str(skipped_shape),
        "dropped_rows": str(dropped_rows),
    }
    return U_layers, W_sel_norm, U_norm, stats


def apply_update_inplace(model: LlavaForConditionalGeneration,
                         U_layers: Dict[str, torch.Tensor],
                         step: float):
    with torch.inference_mode():
        vt = get_vision_tower(model)
        for ln, U in U_layers.items():
            p = vt.get_parameter(ln)
            if p.data.shape != U.shape:
                print(f"[warn] apply: shape mismatch {ln}: p {tuple(p.data.shape)} vs U {tuple(U.shape)}; skipping.")
                continue
            p.data.add_(U.to(dtype=p.dtype, device=p.device), alpha=step)


# ========= UES (same as your CLIP/SD scripts) =========
def _rel_drop(curr: float, base: float, eps: float = 1e-6) -> float:
    if base <= eps:
        return 0.0 if curr <= base else min(1.0, (curr - base) / max(eps, curr))
    return min(1.0, max(0.0, (base - curr) / max(eps, base)))

def ues(fgt1, fgt5, test1, test5, base_f1, base_f5, base_t1, base_t5, alpha=0.5):
    forget_gain = 0.5 * (_rel_drop(fgt1, base_f1) + _rel_drop(fgt5, base_f5))
    retain_loss = 0.5 * (_rel_drop(test1, base_t1) + _rel_drop(test5, base_t5))
    return alpha * forget_gain - (1.0 - alpha) * retain_loss


# ========= Save helpers =========
def save_state_dict_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

def safe_save_pretrained(model,
                         processor,
                         out_dir: str,
                         *,
                         meta: Dict):
    out_dir = str(out_dir)
    try:
        model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
        processor.save_pretrained(out_dir)
        with open(Path(out_dir) / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[save] full model saved to {out_dir}")
        return
    except Exception as e:
        print(f"[save] normal save_pretrained failed: {e}")

    has_meta = any(getattr(p, "is_meta", False) for p in model.parameters())
    if not has_meta:
        try:
            sd = save_state_dict_cpu(model)
            torch.cuda.empty_cache()
            model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True, max_shard_size="2GB")
            processor.save_pretrained(out_dir)
            with open(Path(out_dir) / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[save] saved via CPU state_dict to {out_dir}")
            return
        except Exception as e2:
            print(f"[save] CPU state_dict save failed: {e2}")

    patch_dir = Path(out_dir) / "patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    with open(patch_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    try:
        vt_state = {k: v.detach().cpu() for k, v in get_vision_tower(model).state_dict().items()}
        torch.save(vt_state, patch_dir / "vision_tower_state_dict.pt")
        print("[save] wrote PATCH (vision tower state) + meta")
    except Exception as e3:
        print(f"[save] patch vision save failed: {e3}")
    processor.save_pretrained(out_dir)


# ========= Main =========
def main():
    set_seed(SEED)

    celeb_name      = CELEB_NAME_DEFAULT
    results_root    = RESULTS_ROOT_DEFAULT
    save_dir        = SAVE_DIR_DEFAULT
    clip_model_id   = CLIP_MODEL_ID_DEFAULT
    llava_model_id  = LLAVA_MODEL_ID_DEFAULT
    target_dataset  = TARGET_DATASET_DEFAULT
    retain_dataset  = RETAIN_DATASET_DEFAULT
    test_dataset    = TEST_DATASET_DEFAULT
    neuron_json     = NEURON_JSON_DEFAULT

    max_iters       = MAX_ITERS_DEFAULT
    init_div        = INITIAL_DIVISOR_DEFAULT

    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "unlearning_log.txt"
    with open(log_path, "w") as f:
        pass

    # --- Load grads + Si rows (+ config deltas) ---
    model_repo, model_name = clip_model_id.split("/")
    celeb_root = results_root / f"{celeb_name}_{model_repo}_{model_name}"
    fg_path = celeb_root / "forget_grads.pt"
    rg_path = celeb_root / "train_grads.pt"
    if not fg_path.exists() or not rg_path.exists():
        raise FileNotFoundError(f"Missing grads at:\n  {fg_path}\n  {rg_path}")

    forget_grads = torch.load(fg_path, map_location="cpu")
    retain_grads = torch.load(rg_path, map_location="cpu")
    neuron_rows, cfg = load_neuron_importance_from_ranked(neuron_json)
    delta = float(cfg.get("delta", DELTA_DEFAULT))
    gamma = float(cfg.get("gamma", GAMMA_DEFAULT))   # default 1.0 here
    eps   = float(cfg.get("eps",   EPS_DEFAULT))     # default 1e-6

    # --- Optional CLIP audit (not used further; just ensures ID is valid) ---
    _ = CLIPModel.from_pretrained(clip_model_id, dtype=torch.float16).to(device)

    # --- Load LLaVA (8-bit LM + FP16 vision) ---
    offload_dir = save_dir / "cpu_offload"
    llava = load_llava_8bit_lm_fp16_vision(llava_model_id, clip_model_id, offload_dir=offload_dir)
    proc = AutoProcessor.from_pretrained(llava_model_id, use_fast=True)
    llava.eval()

    # --- Ensure the vision tower is FP16 and shapes are 2-D where expected ---
    vt = get_vision_tower(llava)
    audit_name = "vision_model.encoder.layers.22.mlp.fc1.weight"
    pp = dict(vt.named_parameters()).get(audit_name)
    if pp is not None:
        print(f"[audit] {audit_name} -> shape={tuple(pp.shape)} dtype={pp.dtype}")
        assert pp.dtype in (torch.float16, torch.bfloat16, torch.float32) and pp.dim() == 2

    # === CPU BASE SNAPSHOT ===
    with torch.inference_mode():
        vt_base_cpu = {n: p.data.detach().to("cpu", copy=True) for (n, p) in vt.named_parameters()}

    # --- Build masked weighted update U (vision-only; SD/CLIP form) ---
    vt_state_for_U = {n: p.data for (n, p) in vt.named_parameters()}
    U_layers, W_sel_norm, U_norm, stats = build_masked_weighted_update(
        forget_grads, retain_grads, neuron_rows, vt_state_for_U,
        delta=delta, eps=eps, gamma=gamma,
    )
    print(f"[buildU] used_layers={stats['used']} skipped_name={stats['skipped_name']} "
          f"skipped_shape={stats['skipped_shape']} dropped_rows={stats['dropped_rows']}")
    if len(U_layers) == 0 or U_norm == 0.0:
        raise RuntimeError("No valid masked update could be built (check JSON and grads).")

    ratio_base = (W_sel_norm / U_norm) if U_norm > 0 else 1.0
    print(f"[ratio] ||W_sel||={W_sel_norm:.6f} ||U||={U_norm:.6f} base={ratio_base:.6f}")

    # --- Data & caching ---
    tgt_ds  = get_dataset(target_dataset, subset_size=SUBSET_TARGET)
    ret_ds  = get_dataset(retain_dataset, subset_size=SUBSET_RETAIN)
    tst_ds  = get_dataset(test_dataset, subset_size=SUBSET_TEST) if test_dataset else None
    tgt_cached  = cache_batches_cpu(proc, tgt_ds, BATCH_SIZE_DEFAULT)
    ret_cached  = cache_batches_cpu(proc, ret_ds, BATCH_SIZE_DEFAULT)
    tst_cached  = cache_batches_cpu(proc, tst_ds, BATCH_SIZE_DEFAULT) if tst_ds else None

    base_tgt = eval_cached_cpu2gpu(llava, proc, tgt_cached)
    base_ret = eval_cached_cpu2gpu(llava, proc, ret_cached)
    base_tst = eval_cached_cpu2gpu(llava, proc, tst_cached) if tst_cached else None
    print(f"[baseline] target={base_tgt:.3f} retain={base_ret:.3f}" +
          (f" test={base_tst:.3f}" if base_tst is not None else ""))
    with open(log_path, "a") as f:
        f.write(f"[baseline] target={base_tgt:.6f} retain={base_ret:.6f}" +
                (f" test={base_tst:.6f}" if base_tst is not None else "") + "\n")

    # cache baseline for UES (we only have one metric each, reuse it for @1/@5 slots)
    base_f1 = base_tgt
    base_f5 = base_tgt
    base_t1 = base_ret
    base_t5 = base_ret

    # --- Binary/bracket search (same schedule) + UES selection ---
    MAX_ITERS = int(max_iters)
    low, high = 0.0, float("inf")   # search negative steps in [-high, -low]
    tried = []

    def restore_from_cpu_base():
        with torch.inference_mode():
            for n, p in get_vision_tower(llava).named_parameters():
                p.data.copy_(vt_base_cpu[n].to(device=p.device, dtype=p.dtype), non_blocking=True)

    step = - (ratio_base / float(init_div))  # exact same ratio init you used

    best_step = 0.0
    best_tgt, best_ret = base_tgt, base_ret
    best_ues = float("-inf")

    for it in range(MAX_ITERS):
        print(f"[search] iter={it} step={step:+.6f} (low={low:+.6f}, high={'inf' if not math.isfinite(high) else f'{high:+.6f}'})")

        restore_from_cpu_base()
        apply_update_inplace(llava, U_layers, step)

        tgt = eval_cached_cpu2gpu(llava, proc, tgt_cached)
        ret = eval_cached_cpu2gpu(llava, proc, ret_cached)

        # UES selection (same logic as CLIP/SD; we mirror f1=f5=tgt, t1=t5=ret)
        score = ues(
            fgt1=tgt, fgt5=tgt,
            test1=ret, test5=ret,
            base_f1=base_f1, base_f5=base_f5,
            base_t1=base_t1, base_t5=base_t5,
            alpha=UES_ALPHA
        )

        line = f"[try] iter={it} step={step:+.6f} target={tgt:.6f} retain={ret:.6f} UES={score:.6f}"
        print(line)
        with open(log_path, "a") as f:
            f.write(json.dumps({"iter": it, "step": step, "tgt": tgt, "ret": ret, "ues": score}) + "\n")
        tried.append({"iter": it, "step": step, "tgt": tgt, "ret": ret, "ues": score})

        # Update best by UES (not by min target alone)
        if score > best_ues:
            best_ues = score
            best_step = step
            best_tgt, best_ret = tgt, ret

        # Same bracketing rule you used: if forgetting "succeeds" (tgt==0), shrink magnitude; else grow
        if tgt <= 0.0:
            high = min(high, abs(step))
        else:
            low = max(low, abs(step))

        if it == MAX_ITERS - 1:
            break

        # Next step
        if math.isfinite(high):
            step = -0.5 * (low + high)
        else:
            step = step * 2.0  # expand until we hit a bound

        # avoid repeats
        if tried and any(abs(step - t["step"]) < 1e-10 for t in tried):
            if math.isfinite(high):
                step = - (0.5 * (low + high) + 1e-3)
            else:
                step = step * 1.5

        torch.cuda.empty_cache()

    print(f"[best] step={best_step:+.6f} target={best_tgt:.6f} retain={best_ret:.6f} UES={best_ues:.6f}")

    # --- Apply best and save ---
    with torch.inference_mode():
        for n, p in get_vision_tower(llava).named_parameters():
            p.data.copy_(vt_base_cpu[n].to(device=p.device, dtype=p.dtype), non_blocking=True)
    apply_update_inplace(llava, U_layers, best_step)

    out_dir = save_dir / "unlearned_llava_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    tried_path = save_dir / "search_trace.jsonl"
    with open(tried_path, "w") as f:
        for row in tried:
            f.write(json.dumps(row) + "\n")
    print(f"[trace] wrote {tried_path}")

    meta = {
        "celebrity": celeb_name,
        "llava_model_id": llava_model_id,
        "clip_model_id": clip_model_id,
        "neuron_json": str(neuron_json),
        "delta": delta, "eps": eps, "gamma": gamma,
        "baseline": {"target": base_tgt, "retain": base_ret, "test": base_tst},
        "best": {"step": best_step, "target": best_tgt, "retain": best_ret, "ues": best_ues},
        "stats_buildU": stats,
        "ratio_base": float(ratio_base),
        "subset_sizes": {"target": SUBSET_TARGET, "retain": SUBSET_RETAIN, "test": SUBSET_TEST},
        "ues_alpha": UES_ALPHA,
    }
    safe_save_pretrained(llava, proc, out_dir.as_posix(), meta=meta)
    print(f"[save] {out_dir}")


if __name__ == "__main__":
    main()
