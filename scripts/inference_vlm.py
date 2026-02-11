#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, requests, gc
from io import BytesIO
from PIL import Image
from pathlib import Path
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ---- edit these ----
MODEL_DIR_1 = "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_nest_elon/unlearned_llava_model"
MODEL_DIR_2 = "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_elon/unlearned_llava_model"
BASE_ID     = "llava-hf/llava-1.5-7b-hf"   # original model
DEFAULT_IMAGE = "/home/rania/Pictures/musk.png"
PROMPT = "Who is in this image?"
# --------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _load_image(path_or_url: str) -> Image.Image:
    if _is_url(path_or_url):
        r = requests.get(path_or_url, timeout=30); r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(Path(path_or_url)).convert("RGB")

def load_model_and_processor(model_or_dir: str):
    """
    Loads a LLaVA model + processor. If the saved dir lacks a processor,
    fall back to the base processor.
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        model_or_dir, device_map="auto", torch_dtype=torch.float16
    )
    try:
        proc = AutoProcessor.from_pretrained(model_or_dir, use_fast=True)
    except Exception:
        proc = AutoProcessor.from_pretrained(BASE_ID, use_fast=True)
    model.eval()
    return model, proc

@torch.inference_mode()
def run_inference(model, processor, user_prompt: str, path_or_url: str | None = None):
    img_source = path_or_url or DEFAULT_IMAGE
    image = _load_image(img_source)
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":user_prompt}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    out_ids = model.generate(**enc, max_new_tokens=128, do_sample=False, num_beams=1, use_cache=False)
    text = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.split("ASSISTANT:")[-1].strip()

def eval_one(tag: str, model_or_dir: str, prompt: str, image_path: str):
    print(f"\n[load] {tag} …")
    model, proc = load_model_and_processor(model_or_dir)
    print(f"[run]  {tag}")
    ans = run_inference(model, proc, prompt, image_path)
    print(f"[out]  {tag}: {ans}")
    # cleanup to free VRAM before loading the next model
    del model, proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ans

def main():
    print(f"[prompt] {PROMPT}")
    print(f"[image ] {DEFAULT_IMAGE}")

    ans1 = eval_one("unlearned #1", MODEL_DIR_1, PROMPT, DEFAULT_IMAGE)
    ans2 = eval_one("unlearned #2", MODEL_DIR_2, PROMPT, DEFAULT_IMAGE)
    ans0 = eval_one("original/base", BASE_ID,     PROMPT, DEFAULT_IMAGE)

    print("\n=== SUMMARY ===")
    print("Unlearned #1 :", ans1)
    print("Unlearned #2 :", ans2)
    print("Original/Base:", ans0)

if __name__ == "__main__":
    main()
