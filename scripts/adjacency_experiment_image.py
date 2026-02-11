#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adjacency experiment (IMAGE-based) for CLIP.

Goal:
- Quantify "adjacency" between concept groups using a measurable model-based metric,
  not common-sense: cosine similarity between CLIP image embeddings of concepts.
- Uses your existing per-concept .tar files (e.g., Elon_Musk.tar, Taylor_Swift.tar, banana.tar, dog.tar, ...).

What it does:
1) Loads CLIP (open_clip in your SLUG repo) with given --model/--pretrained/--precision.
2) For each concept in concepts_json, loads up to --max_images images from <tar_dir>/<concept>.tar
   (with robust filename matching).
3) Encodes images -> normalized embeddings; aggregates to a single concept embedding
   (mean of normalized image embeddings, then re-normalize).
4) Computes cosine similarity matrix across concepts.
5) Optionally thresholds to "adjacent" pairs via --sim_pct percentile for reporting.
6) Saves:
   - adjacency_heatmap.png
   - adjacency_matrix.npy
   - adjacency_pairs_top.txt
   - group_similarity_means.txt

Fixes included (for your error):
- Ensures image batch dtype matches model weights dtype (fp16 vs fp32) to avoid:
  "Input type FloatTensor and weight type HalfTensor should be the same"

Run example:
python scripts/adjacency_experiment_image.py \
  --model ViT-B-32 --pretrained laion400m_e32 --precision fp16 \
  --concepts_json scripts/adjacency_concepts.json \
  --tar_dir /home/rania/SLUG/data/tar_files \
  --out_dir /home/rania/SLUG/results/adjacency_elon_img \
  --max_images 200 --batch_size 32 --sim_pct 95
"""

import os
import io
import re
import json
import math
import tarfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from PIL import Image

# matplotlib is only used for the heatmap output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- import your repo (same pattern you use elsewhere)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from clip.open_clip import create_model_and_transforms, get_input_dtype  # SLUG/open_clip
# If you prefer tokenizer, not needed here.


# --------------------------
# Helpers
# --------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def resolve_tar(tar_dir: Path, concept: str) -> Path:
    """
    Robustly locate the tar file for a concept.
    Tries exact, common underscore/space/case variants, then case-insensitive scan.
    """
    # exact
    p = tar_dir / f"{concept}.tar"
    if p.exists():
        return p

    variants = set()
    variants.add(concept.replace(" ", "_"))
    variants.add(concept.replace("_", " "))
    variants.add(concept.lower())
    variants.add(concept.lower().replace(" ", "_"))
    variants.add(concept.lower().replace("_", " "))

    for v in variants:
        p = tar_dir / f"{v}.tar"
        if p.exists():
            return p

    # case-insensitive scan
    target = f"{concept}.tar".lower()
    for fp in tar_dir.glob("*.tar"):
        if fp.name.lower() == target:
            return fp

    raise FileNotFoundError(f"Could not find tar for concept='{concept}' in {tar_dir}")


def is_image_member(name: str) -> bool:
    name = name.lower()
    return name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def safe_open_image(blob: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(blob))
        im = im.convert("RGB")
        return im
    except Exception:
        return None


def load_concepts_json(path: Path) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Supports a few simple JSON schemas.

    Recommended schema:
    {
      "groups": {
        "CEOs": ["Elon_Musk", "Jeff_Bezos"],
        "Musicians": ["Taylor_Swift", "Drake"],
        "Objects": ["banana", "dog"]
      }
    }

    Returns:
      concepts: unique concept list in deterministic order
      concept_to_group: mapping concept -> group
      group_names: group order
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "groups" in data and isinstance(data["groups"], dict):
        groups = data["groups"]
        group_names = list(groups.keys())
        concepts = []
        concept_to_group = {}
        for g in group_names:
            for c in groups[g]:
                if c not in concept_to_group:
                    concepts.append(c)
                    concept_to_group[c] = g
        return concepts, concept_to_group, group_names

    # fallback: list of entries [{"concept":..., "group":...}, ...]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "concept" in data[0]:
        concepts = []
        concept_to_group = {}
        group_names = []
        for it in data:
            c = it["concept"]
            g = it.get("group", "Ungrouped")
            if g not in group_names:
                group_names.append(g)
            if c not in concept_to_group:
                concepts.append(c)
                concept_to_group[c] = g
        return concepts, concept_to_group, group_names

    raise ValueError(f"Unrecognized concepts_json schema: {path}")


@torch.no_grad()
def concept_image_embedding(
    model,
    preprocess,
    tar_path: Path,
    device: torch.device,
    max_images: int,
    batch_size: int,
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      concept_emb: [D] normalized embedding for concept
      used: number of images actually used
    """
    model.eval()

    # FIX: match dtype to vision tower weights (NOT next(model.parameters()))
    if hasattr(model, "visual") and hasattr(model.visual, "conv1"):
        model_dtype = model.visual.conv1.weight.dtype
    else:
        model_dtype = next(model.visual.parameters()).dtype

    feats: List[torch.Tensor] = []
    used = 0

    with tarfile.open(tar_path, "r:*") as tf:
        batch_imgs = []
        for member in tf:
            if used >= max_images:
                break
            if (not member.isfile()) or (not is_image_member(member.name)):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            im = safe_open_image(f.read())
            if im is None:
                continue

            try:
                x = preprocess(im)  # CPU float tensor
            except Exception:
                continue

            batch_imgs.append(x)
            used += 1

            if len(batch_imgs) >= batch_size:
                batch = torch.stack(batch_imgs, dim=0).to(device=device, non_blocking=True)
                batch = batch.to(dtype=model_dtype)  # ensure fp16 matches conv weights
                z = model.encode_image(batch)
                z = l2_normalize(z.float())          # normalize in fp32
                feats.append(z.cpu())
                batch_imgs = []

        if batch_imgs:
            batch = torch.stack(batch_imgs, dim=0).to(device=device, non_blocking=True)
            batch = batch.to(dtype=model_dtype)
            z = model.encode_image(batch)
            z = l2_normalize(z.float())
            feats.append(z.cpu())

    if used == 0 or not feats:
        raise RuntimeError(f"No usable images found in tar: {tar_path}")

    Z = torch.cat(feats, dim=0)      # [N,D]
    concept = Z.mean(dim=0)          # [D]
    concept = l2_normalize(concept.unsqueeze(0)).squeeze(0)
    return concept, used



def cosine_matrix(E: np.ndarray) -> np.ndarray:
    """
    E: [N,D] L2-normalized
    returns S: [N,N] cosine similarity
    """
    return E @ E.T


def group_similarity_means(
    S: np.ndarray,
    concepts: List[str],
    concept_to_group: Dict[str, str],
    group_names: List[str],
) -> Dict[Tuple[str, str], float]:
    """
    Mean similarity between groups, excluding diagonal self-pairs.
    """
    idx_by_group: Dict[str, List[int]] = {g: [] for g in group_names}
    for i, c in enumerate(concepts):
        g = concept_to_group[c]
        if g not in idx_by_group:
            idx_by_group[g] = []
        idx_by_group[g].append(i)

    out = {}
    for ga in group_names:
        for gb in group_names:
            ia = idx_by_group.get(ga, [])
            ib = idx_by_group.get(gb, [])
            if not ia or not ib:
                out[(ga, gb)] = float("nan")
                continue

            vals = []
            for i in ia:
                for j in ib:
                    if ga == gb and i == j:
                        continue
                    vals.append(S[i, j])

            out[(ga, gb)] = float(np.mean(vals)) if vals else float("nan")
    return out


def save_heatmap(
    S: np.ndarray,
    labels: List[str],
    out_path: Path,
    title: str,
):
    plt.figure(figsize=(12, 10))
    plt.imshow(S, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser("adjacency_experiment_image")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--concepts_json", type=str, required=True)
    ap.add_argument("--tar_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sim_pct", type=float, default=95.0, help="Percentile threshold for 'adjacent' pairs report.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_dir = Path(args.tar_dir)
    cj = Path(args.concepts_json)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Load concepts/groups
    concepts, concept_to_group, group_names = load_concepts_json(cj)

    # Load CLIP
    model, _, preprocess = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        output_dict=True,  # keep consistent with SLUG
    )
    model.eval()

    print(f"[INFO] device={device} | model={args.model} | pretrained={args.pretrained} | precision={args.precision}")
    print(f"[INFO] concepts={len(concepts)} | groups={group_names}")

    # Build concept embeddings
    embs = []
    used_counts = {}
    for c in concepts:
        tar_path = resolve_tar(tar_dir, c)
        emb, used = concept_image_embedding(
            model=model,
            preprocess=preprocess,
            tar_path=tar_path,
            device=device,
            max_images=args.max_images,
            batch_size=args.batch_size,
        )
        embs.append(emb.numpy())
        used_counts[c] = used
        print(f"[OK] {c:>20s} | used_images={used} | tar={tar_path.name}")

    E = np.stack(embs, axis=0).astype(np.float32)  # [N,D], already normalized
    S = cosine_matrix(E)  # [N,N]

    # Save matrix
    np.save(out_dir / "adjacency_matrix.npy", S)

    # Heatmap
    heatmap_path = out_dir / "adjacency_heatmap.png"
    save_heatmap(S, concepts, heatmap_path, title="Adjacency Heatmap (Image-Embedding Cosine Similarity)")
    print(f"[OK] Saved heatmap to: {heatmap_path}")

    # Report top adjacent pairs via percentile
    # Exclude diagonal
    iu = np.triu_indices(len(concepts), k=1)
    vals = S[iu]
    thr = np.percentile(vals, args.sim_pct)
    pairs = []
    for (i, j), v in zip(zip(iu[0], iu[1]), vals):
        if v >= thr:
            pairs.append((v, concepts[i], concepts[j], concept_to_group[concepts[i]], concept_to_group[concepts[j]]))
    pairs.sort(reverse=True, key=lambda x: x[0])

    with open(out_dir / "adjacency_pairs_top.txt", "w") as f:
        f.write(f"sim_pct={args.sim_pct} -> threshold={thr:.6f}\n")
        f.write("sim\tconcept_i\tconcept_j\tgroup_i\tgroup_j\n")
        for v, ci, cj, gi, gj in pairs:
            f.write(f"{v:.6f}\t{ci}\t{cj}\t{gi}\t{gj}\n")

    # Group similarity means
    gsm = group_similarity_means(S, concepts, concept_to_group, group_names)
    with open(out_dir / "group_similarity_means.txt", "w") as f:
        for ga in group_names:
            for gb in group_names:
                f.write(f"{ga:>12s} vs {gb:<12s}: {gsm[(ga, gb)]:.6f}\n")

    print("\n[Group Similarity Means]")
    for ga in group_names:
        for gb in group_names:
            print(f"{ga:>12s} vs {gb:<12s}: {gsm[(ga, gb)]:.4f}")

    # Save metadata
    meta = {
        "model": args.model,
        "pretrained": args.pretrained,
        "precision": args.precision,
        "device": str(device),
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "sim_pct": args.sim_pct,
        "threshold": float(thr),
        "concepts": concepts,
        "groups": group_names,
        "concept_to_group": concept_to_group,
        "used_counts": used_counts,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
