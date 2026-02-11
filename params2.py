#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a table like:

Model                   Total Params   Trainable Params   % Updated   Architecture Notes
CLIP ViT-B/32           151M           151M               100%        Fully trained baseline
SLUG (CLIP)             151M           3.2M (2.1%)        2.1%        Hard-selected vision layer family
NEST (Ours, CLIP)       151M           2.4M (1.6%)        1.6%        Neuron-level sparse rows from Si.json
LLaVA-1.5               7B             7B                 100%        Full multimodal model
NEST (Ours, LLaVA)      7B             12.3M (0.17%)      0.17%       Vision-only neuron rows from Si.json

Supports three counting modes per row:
- "full":         all trainable parameters are considered "updated"
- "layers":       all parameters of specific tensors (layer name globs/regex) are "updated" (SLUG)
- "json_rows":    updated params are (#selected_rows * fan_in) using your Si.json (neuron-level)

This script assumes your repo layout so it can import open_clip creators if needed.
If you prefer Hugging Face CLIP, switch the loader in _load_clip accordingly.
"""

import os, re, json, math, dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterable

import torch
import torch.nn as nn

# Try to import your repo's open_clip; if not available, fallback to HF CLIP
_USE_OPENCLIP = True
try:
    # adjust if your project path differs
    here = Path(__file__).resolve()
    repo_root = here.parent  # tweak if needed
    sys_path_add = repo_root.as_posix()
    import sys
    if sys_path_add not in sys.path:
        sys.path.append(sys_path_add)
    from clip.open_clip import create_model_and_transforms  # your project API
except Exception:
    _USE_OPENCLIP = False

try:
    from transformers import CLIPModel
except Exception:
    pass

# ---------- helpers ----------

def human(n: int) -> str:
    if n >= 10**9:
        return f"{n/1e9:.1f}B"
    if n >= 10**6:
        return f"{n/1e6:.0f}M" if n < 10**7 else f"{n/1e6:.1f}M"
    if n >= 10**3:
        return f"{n/1e3:.0f}K"
    return str(n)

def count_total_and_trainable(params: Iterable[torch.nn.Parameter]) -> Tuple[int,int]:
    total = 0
    train = 0
    for p in params:
        n = p.numel()
        total += n
        if p.requires_grad:
            train += n
    return total, train

def _param_name_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {name: p for (name, p) in model.named_parameters()}

def _size_of_tensor(name_to_param: Dict[str, nn.Parameter], name: str) -> int:
    p = name_to_param.get(name, None)
    return 0 if p is None else p.numel()

def _fan_in_rows(name_to_param: Dict[str, nn.Parameter], name: str, rows: List[int]) -> int:
    """Count params updated if we change *rows* of a 2D weight: rows * fan_in."""
    p = name_to_param.get(name, None)
    if p is None or p.ndim != 2:
        return 0
    out, fan_in = p.shape
    # clamp rows to valid range, de-duplicate
    valid = [r for r in set(rows) if 0 <= r < out]
    return len(valid) * fan_in

def _match_any(name: str, globs_or_regex: List[str]) -> bool:
    for pat in globs_or_regex:
        # support simple wildcard '*' or python regex if startswith('re:')
        if pat.startswith("re:"):
            if re.search(pat[3:], name):
                return True
        else:
            # naive glob: turn '*' into '.*' regex
            rx = "^" + re.escape(pat).replace("\\*", ".*") + "$"
            if re.match(rx, name):
                return True
    return False

# ---- mapping from OpenCLIP names (in your JSON) -> model param names, if needed ----
def remap_openclip_to_openclip_same(name_json: str) -> str:
    """If your json was produced on the same OpenCLIP model flavor, names usually match."""
    return name_json

def remap_openclip_to_hf_text(name_json: str) -> Optional[str]:
    """
    Example: 'transformer.resblocks.20.mlp.c_fc.weight' -> 'text_model.encoder.layers.20.mlp.fc1.weight'
    Extend if you need attention q/k/v, etc.
    """
    m = re.match(r"transformer\.resblocks\.(\d+)\.mlp\.c_fc\.weight$", name_json)
    if not m:
        return None
    L = int(m.group(1))
    return f"text_model.encoder.layers.{L}.mlp.fc1.weight"

# ---------- row spec ----------

@dataclass
class RowSpec:
    label: str
    notes: str
    mode: str  # "full" | "layers" | "json_rows"

    # For CLIP
    clip_model: Optional[str] = None            # e.g. "ViT-B-32"
    clip_pretrained: Optional[str] = None       # e.g. "laion400m_e32"
    # Or HF CLIP id instead of open_clip
    hf_clip_id: Optional[str] = None            # e.g. "openai/clip-vit-large-patch14-336"

    # For LLaVA
    llava_hf_id: Optional[str] = None           # e.g. "llava-hf/llava-1.5-7b-hf"

    # Which module to count over (for LLaVA vision-only, etc.)
    submodule_hint: Optional[str] = None        # e.g. "model.vision_tower" for LLaVA vision

    # For "layers" (SLUG-style): list of tensor name patterns to treat as updated
    layer_globs: Optional[List[str]] = None

    # For "json_rows": neuron-json path + variant + tower key + (optional) name remapper
    json_path: Optional[Path] = None
    variant_key: str = "Si"                     # which section to read
    tower_key: Optional[str] = None             # "vision" or "language" or None (both)
    name_mapper: Optional[str] = None           # "openclip_same" | "openclip_to_hf_text"

# ---------- model loaders ----------

def _load_clip(spec: RowSpec) -> nn.Module:
    if _USE_OPENCLIP and spec.clip_model and spec.clip_pretrained:
        model, _, _ = create_model_and_transforms(
            spec.clip_model, spec.clip_pretrained, precision="fp16", device="cpu", output_dict=True
        )
        return model
    elif spec.hf_clip_id:
        return CLIPModel.from_pretrained(spec.hf_clip_id, torch_dtype=torch.float16)
    else:
        raise RuntimeError("Provide either (clip_model + clip_pretrained) for OpenCLIP or hf_clip_id for HF CLIP.")

def _load_llava(spec: RowSpec) -> nn.Module:
    from transformers import LlavaForConditionalGeneration
    return LlavaForConditionalGeneration.from_pretrained(spec.llava_hf_id, torch_dtype=torch.float16)

def _maybe_submodule(model: nn.Module, hint: Optional[str]) -> nn.Module:
    if not hint:
        return model
    sub = model
    for tok in hint.split("."):
        if not tok: continue
        if not hasattr(sub, tok):
            raise AttributeError(f"Submodule hint '{hint}' not found; missing part '{tok}'.")
        sub = getattr(sub, tok)
    if not isinstance(sub, nn.Module):
        raise AttributeError(f"Object at '{hint}' is not an nn.Module.")
    return sub

# ---------- json reader for neuron rows ----------

def _read_neuron_rows_from_json(json_path: Path, variant_key: str, tower_key: Optional[str]) -> Dict[str, List[int]]:
    data = json.loads(Path(json_path).read_text())
    if variant_key in data:
        sec = data[variant_key]
    else:
        sec = data

    # expected: sec["vision"]["ranked"] / sec["language"]["ranked"]
    def _collect(block) -> Dict[str, List[int]]:
        out = {}
        for item in block.get("ranked", []):
            layer = item.get("layer")
            idxs  = item.get("important_idx", [])
            if layer and isinstance(idxs, list):
                out[layer] = [int(i) for i in idxs]
        return out

    rows = {}
    if tower_key is None:
        for tk in ("vision","language"):
            if tk in sec:
                rows.update(_collect(sec[tk]))
    else:
        if tower_key not in sec:
            raise KeyError(f"tower_key '{tower_key}' not in JSON")
        rows.update(_collect(sec[tower_key]))
    return rows

def _choose_mapper(name_mapper: Optional[str]):
    if name_mapper == "openclip_to_hf_text":
        return remap_openclip_to_hf_text
    elif name_mapper == "openclip_same" or name_mapper is None:
        return remap_openclip_to_openclip_same
    else:
        raise ValueError(f"Unknown name_mapper: {name_mapper}")

# ---------- counting core ----------

def count_updated_params(spec: RowSpec) -> Tuple[int,int,int]:
    """
    Returns: (total_params, trainable_params, updated_params)
    """
    # Load model
    if spec.llava_hf_id:
        model = _load_llava(spec)
    else:
        model = _load_clip(spec)

    # Optionally narrow to a submodule (e.g., vision-only)
    tgt = _maybe_submodule(model, spec.submodule_hint)

    # Count totals (within tgt only)
    total, trainable = count_total_and_trainable(p for _, p in tgt.named_parameters())

    # Dispatch mode
    name_to_param = _param_name_dict(tgt)

    if spec.mode == "full":
        updated = trainable

    elif spec.mode == "layers":
        if not spec.layer_globs:
            raise ValueError("mode='layers' requires layer_globs.")
        updated = 0
        for name, p in name_to_param.items():
            if p.requires_grad and _match_any(name, spec.layer_globs):
                updated += p.numel()

    elif spec.mode == "json_rows":
        if not spec.json_path or not spec.json_path.exists():
            raise FileNotFoundError(f"json not found: {spec.json_path}")
        rows = _read_neuron_rows_from_json(spec.json_path, spec.variant_key, spec.tower_key)
        mapper = _choose_mapper(spec.name_mapper)
        updated = 0
        for json_name, idxs in rows.items():
            model_name = mapper(json_name)
            if model_name is None:
                continue
            updated += _fan_in_rows(name_to_param, model_name, idxs)
    else:
        raise ValueError(f"Unknown mode: {spec.mode}")

    return total, trainable, updated

def percent(x: int, denom: int) -> float:
    return (100.0 * x / max(1, denom))

def row_report(spec: RowSpec) -> Dict[str,str]:
    total, trainable, updated = count_updated_params(spec)
    pct = percent(updated, trainable)
    return {
        "Model": spec.label,
        "Total Params": human(total),
        "Trainable Params": human(trainable),
        "% Updated": f"{pct:.2f}%",
        "Architecture Notes": spec.notes,
        "_raw_total": str(total),
        "_raw_train": str(trainable),
        "_raw_updated": str(updated),
    }

# ---------- example configuration ----------
# Adjust paths/model IDs to YOUR runs.

ROWS: List[RowSpec] = [
    # 1) CLIP baseline — all trainable
    RowSpec(
        label="CLIP ViT-B/32 (Baseline)",
        notes="Fully trained baseline",
        mode="full",
        clip_model="ViT-B-32",
        clip_pretrained="laion400m_e32",
    ),

    # 2) SLUG (CLIP) — layer family you hard-selected (edit the block index!)
    RowSpec(
        label="SLUG (CLIP)",
        notes="Hard-selected vision block (attn q/k/v/out + mlp fc1/fc2)",
        mode="layers",
        clip_model="ViT-B-32",
        clip_pretrained="laion400m_e32",
        layer_globs=[
            # Example for a single vision block L=23 → change to the block you actually use.
            "visual.transformer.resblocks.23.attn.in_proj_weight",
            "visual.transformer.resblocks.23.attn.in_proj_bias",
            "visual.transformer.resblocks.23.attn.out_proj.weight",
            "visual.transformer.resblocks.23.attn.out_proj.bias",
            "visual.transformer.resblocks.23.mlp.c_fc.weight",
            "visual.transformer.resblocks.23.mlp.c_fc.bias",
            "visual.transformer.resblocks.23.mlp.c_proj.weight",
            "visual.transformer.resblocks.23.mlp.c_proj.bias",
        ],
    ),

    # 3) NEST / Ours (CLIP) — neuron-level from your Si.json (OpenCLIP names kept)
    RowSpec(
        label="NEST (Ours, CLIP)",
        notes="Neuron-level unlearning from Si.json (vision+text as saved)",
        mode="json_rows",
        clip_model="ViT-B-32",
        clip_pretrained="laion400m_e32",
        json_path=Path("/home/rania/SLUG/results/neuron importance_global/Elon_Musk/ViT-B-32_laion400m_e32_Si.json"),
        variant_key="Si",
        tower_key=None,                     # count both vision and language towers present in JSON
        name_mapper="openclip_same",
    ),

    # 4) LLaVA baseline — full model
    RowSpec(
        label="LLaVA-1.5 (Baseline)",
        notes="Full multimodal model",
        mode="full",
        llava_hf_id="llava-hf/llava-1.5-7b-hf",
    ),

    # 5) NEST / Ours (LLaVA) — vision-only neuron rows (JSON produced for CLIP-Vision names → map if needed)
    RowSpec(
        label="NEST (Ours, LLaVA vision)",
        notes="Vision-tower only, neuron-level rows from Si.json",
        mode="json_rows",
        llava_hf_id="llava-hf/llava-1.5-7b-hf",
        submodule_hint="model.vision_tower",   # count only vision tower
        json_path=Path("/home/rania/SLUG/results/neuron importance_global_vlm/Tom_Cruise/openai_clip-vit-large-patch14-336_Si.json"),
        variant_key="Si",
        tower_key="vision",
        # If your JSON uses OpenAI CLIP (vision) names and LLaVA uses HF names,
        # plug a mapper here. For pure counting by fan-in, names must match params.
        # If they don't, either remap names (write another mapper) or pre-save a JSON with HF names.
        name_mapper="openclip_same",  # change to a custom mapper if needed
    ),
]

# ---------- main ----------

def main():
    rows = []
    for spec in ROWS:
        try:
            rep = row_report(spec)
        except Exception as e:
            rep = {
                "Model": spec.label,
                "Total Params": "—",
                "Trainable Params": "—",
                "% Updated": "—",
                "Architecture Notes": f"{spec.notes} (ERROR: {e})"
            }
        rows.append(rep)

    # Pretty text table (markdown-friendly)
    headers = ["Model","Total Params","Trainable Params","% Updated","Architecture Notes"]
    print("\t".join(headers))
    for r in rows:
        print("\t".join([r[h] for h in headers]))

    # Also dump raw JSON with exact counts if you want
    out = {"rows": rows}
    Path("./param_update_table.json").write_text(json.dumps(out, indent=2))
    print("\nWrote raw counts: ./param_update_table.json")

if __name__ == "__main__":
    main()
