#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path
import torch

# --- Your JSONs (exact paths you gave) ---
JSON_CLIP = Path("/home/rania/SLUG/results/neuron importance_global/Kim_Kardashian/ViT-B-32_laion400m_e32_Si.json")
JSON_SD   = Path("/home/rania/SLUG/results/neuron importance_global_sd/Kim_Kardashian/ViT-H-14_laion2B-s32B-b79K_Si.json")
JSON_VLM  = Path("/home/rania/SLUG/results/neuron importance_global_vlm/Kim_Kardashian/openai_clip-vit-large-patch14-336_Si.json")

DEVICE = "cpu"  # counting only

# (Optional) Whole-model totals from the SLUG table (for percent-of-whole comparison)
TOTAL_CLIP_VITB32_WHOLE = 151_280_000
TOTAL_SD_WHOLE          = 983_000_000
TOTAL_LLAVA_1P5_7B      = 7_060_000_000

# -------- utils --------
def fmt_int(n: int):
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.2f}M"
    if n >= 1_000:         return f"{n/1_000:.2f}K"
    return str(n)

def pct(a: int, b: int):
    return "n/a" if b <= 0 else f"{100.0*float(a)/float(b):.2f}%"

def load_json_rows(json_path: Path, variant="Si"):
    d = json.loads(json_path.read_text())
    sec = d.get(variant, d)
    out = {}
    for tower in ("vision", "language"):
        ranked = sec.get(tower, {}).get("ranked", [])
        for item in ranked:
            lname = item.get("layer", "")
            idxs  = item.get("important_idx", [])
            if lname and isinstance(idxs, list):
                out.setdefault(lname, []).extend(int(i) for i in idxs)
    for k in list(out.keys()):
        out[k] = sorted(set(out[k]))
    return out

def ensure_name(named_params: dict, base: str):
    if base in named_params: return base
    cand = base + ".weight"
    if cand in named_params: return cand
    return None

def count_rows_edit(named_params: dict, idx_map: dict):
    edited_total = 0
    details = []
    for raw_name, rows in idx_map.items():
        name = ensure_name(named_params, raw_name)
        if name is None: continue
        W = named_params[name]
        if W.ndim == 2:
            out, inn = W.shape
            sel = [r for r in rows if 0 <= r < out]
            e = len(sel) * inn
        elif W.ndim == 4:
            out, inn, kH, kW = W.shape
            sel = [r for r in rows if 0 <= r < out]
            e = len(sel) * inn * kH * kW
        else:
            continue
        edited_total += int(e)
        details.append((name, tuple(W.shape), len(sel), int(e)))
    return edited_total, details

def num_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())

# ---- name remaps where needed ----
def map_sd_name(openclip_name: str):
    m = re.match(r"transformer\.resblocks\.(\d+)\.mlp\.c_fc\.weight$", openclip_name)
    if not m: return None
    L = int(m.group(1))
    return f"text_model.encoder.layers.{L}.mlp.fc1.weight"

def map_vlm_name(openai_name: str):
    s = openai_name
    s = s.replace("visual.transformer.resblocks.", "vision_model.encoder.layers.")
    s = s.replace(".mlp.c_fc.weight", ".mlp.fc1.weight")
    s = s.replace(".mlp.c_proj.weight", ".mlp.fc2.weight")
    s = s.replace(".attn.in_proj_weight", ".self_attn.in_proj_weight")
    s = s.replace(".attn.q_proj.weight", ".self_attn.q_proj.weight")
    s = s.replace(".attn.k_proj.weight", ".self_attn.k_proj.weight")
    s = s.replace(".attn.v_proj.weight", ".self_attn.v_proj.weight")
    s = s.replace(".attn.out_proj.weight", ".self_attn.out_proj.weight")
    return s

# -------- 1) CLIP ViT-B/32 via OpenCLIP (no HF) --------
print("\n=== CLIP (ViT-B/32 via OpenCLIP) ===")
from src.clip.open_clip import create_model_and_transforms  # your repo’s OpenCLIP wrapper

clip_model, _, _ = create_model_and_transforms(
    "ViT-B-32", "laion400m_e32", device=DEVICE, output_dict=True
)
clip_named = dict(clip_model.named_parameters())

idx_clip = load_json_rows(JSON_CLIP, variant="Si")
edited_clip, details_clip = count_rows_edit(clip_named, idx_clip)
total_clip = num_params(clip_model)

print(f"Edited params (ours): {edited_clip:,}")
print(f"Total params (CLIP full model): {total_clip:,}")
print(f"% Updated (of CLIP): {pct(edited_clip, total_clip)}")
for name, shape, nrows, e in sorted(details_clip, key=lambda x: -x[3])[:8]:
    print(f"  {name:<60} shape={shape} rows={nrows} edited={e:,}")

row_clip = {
    "Model": "CLIP ViT-B/32",
    "Total Params": total_clip,
    "Our Updated Params": edited_clip,
    "% Component": pct(edited_clip, total_clip),
    "% Whole": pct(edited_clip, TOTAL_CLIP_VITB32_WHOLE),
    "Notes": "Neuron rows from Si.json (vision+text where present)",
}

# -------- 2) SD-2.1 TEXT ENCODER (HF) --------
print("\n=== Stable Diffusion 2.1 (TEXT ENCODER only) ===")
from transformers import CLIPTextModel
te = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1",
                                   subfolder="text_encoder",
                                   local_files_only=False).to(DEVICE)
te_named = dict(te.named_parameters())

idx_sd_raw = load_json_rows(JSON_SD, variant="Si")
idx_sd = {}
for k, rows in idx_sd_raw.items():
    mk = map_sd_name(k)
    if mk: idx_sd[mk] = rows

edited_te, details_te = count_rows_edit(te_named, idx_sd)
total_te = num_params(te)

print(f"Edited params (ours): {edited_te:,}")
print(f"Total params (Text Encoder): {total_te:,}")
print(f"% Updated (of TE): {pct(edited_te, total_te)}")
for name, shape, nrows, e in sorted(details_te, key=lambda x: -x[3])[:8]:
    print(f"  {name:<60} shape={shape} rows={nrows} edited={e:,}")

row_sd = {
    "Model": "SD-2.1 (Text Encoder)",
    "Total Params": total_te,
    "Our Updated Params": edited_te,
    "% Component": pct(edited_te, total_te),
    "% Whole": pct(edited_te, TOTAL_SD_WHOLE),
    "Notes": "Counts against TE only; whole-model % uses ~983M",
}

# -------- 3) VLM vision tower (HF) --------
print("\n=== VLM (LLaVA-1.5 vision tower only) ===")
from transformers import CLIPVisionModel
vt = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336",
                                     local_files_only=False).to(DEVICE)
vt_named = dict(vt.named_parameters())

idx_vlm_raw = load_json_rows(JSON_VLM, variant="Si")
idx_vlm = {map_vlm_name(k): v for k, v in idx_vlm_raw.items()}
idx_vlm = {k: v for k, v in idx_vlm.items() if k is not None}

edited_vt, details_vt = count_rows_edit(vt_named, idx_vlm)
total_vt = num_params(vt)

print(f"Edited params (ours): {edited_vt:,}")
print(f"Total params (Vision Tower): {total_vt:,}")
print(f"% Updated (of VT): {pct(edited_vt, total_vt)}")
for name, shape, nrows, e in sorted(details_vt, key=lambda x: -x[3])[:8]:
    print(f"  {name:<60} shape={shape} rows={nrows} edited={e:,}")

row_vlm = {
    "Model": "LLaVA-1.5 (Vision Tower)",
    "Total Params": total_vt,
    "Our Updated Params": edited_vt,
    "% Component": pct(edited_vt, total_vt),
    "% Whole": pct(edited_vt, TOTAL_LLAVA_1P5_7B),
    "Notes": "Counts vision tower only; whole-model % uses 7.06B",
}

# -------- Compact TSV summary --------
print("\n=== TABLE: Our updated-params counts ===")
print("Model\tTotal Params\tOur Updated Params\t% of Component\t% of Whole\tNotes")
for r in (row_clip, row_sd, row_vlm):
    print("\t".join([
        r["Model"],
        fmt_int(r["Total Params"]),
        fmt_int(r["Our Updated Params"]),
        r["% Component"],
        r["% Whole"],
        r["Notes"],
    ]))
