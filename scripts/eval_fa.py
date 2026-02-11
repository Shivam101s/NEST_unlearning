#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
import requests
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor

BASE_ID = "llava-hf/llava-1.5-7b-hf"
DTYPE = torch.float16
DEVICE_MAP = "auto"

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
    overlap = len(gt & tt) / max(1, len(tt))
    return overlap >= 0.6

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def load_image(path_or_url: str) -> Image.Image:
    if is_url(path_or_url):
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(path_or_url).convert("RGB")

def build_prompt(processor: AutoProcessor, question: str) -> str:
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)

def try_load_full_model(model_dir: str):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_dir, device_map=DEVICE_MAP, dtype=DTYPE
    )
    proc = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    return model, proc

def apply_patch_if_present(model, model_dir: str):
    patch_dir = Path(model_dir) / "patch"
    if not patch_dir.exists():
        print("[patch] none found; using full model only.")
        return
    meta_path  = patch_dir / "meta.json"
    vt_sd_path = patch_dir / "vision_tower_state_dict.pt"
    edit_path  = patch_dir / "edited_param.pt"
    if not meta_path.exists():
        print("[patch] meta.json missing; skipping.")
        return

    meta = json.load(open(meta_path))
    print(f"[patch] layer={meta.get('layer_name')} ratio={meta.get('best_ratio')}")

    if vt_sd_path.exists():
        print("[patch] loading vision_tower_state_dict.pt …")
        vt_sd = torch.load(vt_sd_path, map_location="cpu")
        model.vision_tower.load_state_dict(vt_sd)
        print("[patch] vision tower applied.")
        return

    if edit_path.exists():
        print("[patch] applying edited_param.pt …")
        edited = torch.load(edit_path, map_location="cpu")
        with torch.no_grad():
            p = model.vision_tower.get_parameter(meta["layer_name"])
            p.data.copy_(edited.to(p.dtype).to(p.device))
        print("[patch] edited tensor applied.")

def load_model_for_eval(model_dir: str):
    # try full model load; if that fails, load base + patch + processor from model_dir if present
    try:
        model, proc = try_load_full_model(model_dir)
        print("[load] full model loaded")
        return model, proc
    except Exception as e:
        print(f"[load] full model failed: {e}\n[load] base + patch …")
        proc_src = model_dir if (Path(model_dir) / "preprocessor_config.json").exists() else BASE_ID
        model = LlavaForConditionalGeneration.from_pretrained(
            BASE_ID, device_map=DEVICE_MAP, dtype=DTYPE
        )
        proc = AutoProcessor.from_pretrained(proc_src, use_fast=True)
        apply_patch_if_present(model, model_dir)
        return model, proc

@torch.inference_mode()
def eval_fa(model, proc, dataset_name: str, split: str = "test", subset: int | None = None,
            max_new_tokens: int = 24, batch: int = 2):
    ds = load_dataset(dataset_name, split=split)
    if subset is not None and subset < len(ds):
        ds = ds.select(range(subset))

    misidentified, total = 0, 0
    model.eval()
    dev = next(model.parameters()).device

    for i in range(0, len(ds), batch):
        part = ds[i:i+batch]
        prompts = [build_prompt(proc, q) for q in part["question"]]
        images  = [img.convert("RGB") for img in part["image"]]
        enc = proc(text=prompts, images=images, return_tensors="pt", padding=True)
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True)
        outs = proc.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for out, truth in zip(outs, part["answer"]):
            pred = out.split("ASSISTANT:")[-1].strip()
            correct = strong_match(pred, truth)
            misidentified += (0 if correct else 1)
            total += 1

    fa = misidentified / max(1, total)  # Forget Accuracy
    return fa, misidentified, total

def main():
    import argparse
    ap = argparse.ArgumentParser("Compute Forget Accuracy (FA)")
    ap.add_argument("--model-dir", default = "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_mark1/unlearned_llava_model", type=str,
                    help="Folder with full model or with patch/ + processor")
    ap.add_argument("--dataset", type=str, default="ytan-ucr/mu_llava_mark_zuckerberg",
                    help="HF dataset id, e.g. ytan-ucr/mu_llava_elon_musk")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    model, proc = load_model_for_eval(args.model_dir)
    fa, mis, tot = eval_fa(model, proc, args.dataset, args.split, args.subset, args.max_new_tokens, args.batch)
    print(f"\nFA (Forget Accuracy) = misidentified / total = {mis} / {tot} = {fa:.4f}")

if __name__ == "__main__":
    main()
