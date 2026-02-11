#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- import your repo (same style as your other scripts)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from clip.open_clip import create_model_and_transforms, get_tokenizer
from clip.training.params import parse_args
from clip.training.distributed import init_distributed_device


# -----------------------
# Helpers
# -----------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

@torch.no_grad()
def encode_text_mean(model, tokenizer, device, prompts: List[str], batch_size: int = 64) -> torch.Tensor:
    embs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        tokens = tokenizer(chunk).to(device)
        # OpenCLIP returns dict if output_dict=True; but text encoder usually accessible:
        # safest: call model.encode_text if available, else forward and grab "text_features"
        if hasattr(model, "encode_text"):
            feat = model.encode_text(tokens)
        else:
            out = model(text=tokens)
            feat = out["text_features"]
        feat = l2_normalize(feat.float())
        embs.append(feat.cpu())
    embs = torch.cat(embs, dim=0)
    return l2_normalize(embs.mean(dim=0, keepdim=True)).squeeze(0)  # [D]

def build_prompts(concept: str) -> List[str]:
    c = concept.replace("_", " ")
    # Keep prompts simple + generic (works for names, objects, styles)
    templates = [
        "{c}",
        "a photo of {c}",
        "a portrait of {c}",
        "a close-up photo of {c}",
        "a high quality photo of {c}",
        "a photo of the person {c}",
    ]
    # If you're mixing objects + people, keep both types of phrasing
    # (and you can remove "person" line if only objects)
    return [t.format(c=c) for t in templates]

def cosine_matrix(X: np.ndarray) -> np.ndarray:
    # X: [N, D] assumed normalized
    return X @ X.T

def group_stats(S: np.ndarray, groups: Dict[str, List[int]]) -> List[Tuple[str, str, float]]:
    out = []
    gnames = list(groups.keys())
    for i, g1 in enumerate(gnames):
        for j, g2 in enumerate(gnames):
            idx1 = groups[g1]; idx2 = groups[g2]
            block = S[np.ix_(idx1, idx2)]
            if g1 == g2:
                # exclude diagonal for intra
                if block.shape[0] > 1:
                    mask = ~np.eye(block.shape[0], dtype=bool)
                    val = float(block[mask].mean())
                else:
                    val = float(block.mean())
            else:
                val = float(block.mean())
            out.append((g1, g2, val))
    return out

def percentile_threshold(S: np.ndarray, pct: float = 95.0) -> float:
    # use upper triangle excluding diagonal
    triu = S[np.triu_indices_from(S, k=1)]
    return float(np.percentile(triu, pct))

def save_heatmap(S: np.ndarray, labels: List[str], out_png: str, title: str):
    plt.figure(figsize=(10, 9))
    plt.imshow(S, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[OK] Saved heatmap to: {out_png}")


# -----------------------
# Main
# -----------------------
def main(cli_args=None):
    """
    Example:
    python scripts/adjacency_experiment.py \
      --model ViT-B-32 --pretrained laion400m_e32 \
      --concepts_json scripts/adjacency_concepts.json \
      --out_dir results/adjacency_elon
    """
    import argparse
    p = argparse.ArgumentParser("Adjacency Experiment (CLIP text-space)")

    p.add_argument("--concepts_json", type=str, required=True,
                   help="JSON with concepts and groups, see example below.")
    p.add_argument("--out_dir", type=str, default="results/adjacency")
    p.add_argument("--sim_pct", type=float, default=95.0,
                   help="Percentile threshold for defining adjacency edges.")
    p.add_argument("--seed", type=int, default=42)

    known, unknown = p.parse_known_args(cli_args)
    args = parse_args(unknown)  # your repo args: includes --model/--pretrained/--precision/--device etc.

    torch.manual_seed(known.seed)
    np.random.seed(known.seed)

    _ = init_distributed_device(args)
    device = torch.device(
        args.device if isinstance(args.device, str)
        else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    )

    model, _, _ = create_model_and_transforms(
        args.model, args.pretrained,
        precision=args.precision,
        device=device,
        output_dict=True,
    )
    tokenizer = get_tokenizer(args.model)
    model.eval()

    concepts_spec = json.loads(Path(known.concepts_json).read_text())
    concepts: List[str] = concepts_spec["concepts"]
    groups_spec: Dict[str, List[str]] = concepts_spec["groups"]

    # map concept -> index
    idx_map = {c: i for i, c in enumerate(concepts)}
    groups = {g: [idx_map[c] for c in clist if c in idx_map] for g, clist in groups_spec.items()}

    # compute embeddings
    E = []
    for c in concepts:
        prompts = build_prompts(c)
        emb = encode_text_mean(model, tokenizer, device, prompts)
        E.append(emb.numpy())
    E = np.stack(E, axis=0)  # [N,D], already normalized
    S = cosine_matrix(E)     # [N,N]

    out_dir = Path(known.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save matrix + heatmap
    np.save(out_dir / "text_embeddings.npy", E)
    np.save(out_dir / "cosine_sim_matrix.npy", S)
    save_heatmap(S, concepts, str(out_dir / "adjacency_heatmap.png"),
                 title=f"Adjacency in CLIP Text Space ({args.model}, {args.pretrained})")

    # group stats
    stats = group_stats(S, groups)
    print("\n[Group Similarity Means]")
    for g1, g2, val in stats:
        print(f"{g1:>12s} vs {g2:<12s}: {val:.4f}")

    # adjacency threshold + neighbors
    thr = percentile_threshold(S, pct=known.sim_pct)
    print(f"\n[Adjacency threshold @ {known.sim_pct}th percentile] tau = {thr:.4f}")

    # for each concept, report how many neighbors above threshold (excluding itself)
    neighbor_counts = {}
    for i, c in enumerate(concepts):
        nbrs = [j for j in range(len(concepts)) if j != i and S[i, j] >= thr]
        neighbor_counts[c] = len(nbrs)

    # save a small summary json for rebuttal
    summary = {
        "model": args.model,
        "pretrained": args.pretrained,
        "threshold_percentile": known.sim_pct,
        "tau": thr,
        "neighbor_counts": neighbor_counts,
        "group_mean_similarities": [{"g1": a, "g2": b, "mean_cos_sim": v} for a, b, v in stats],
    }
    (out_dir / "adjacency_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] Wrote summary to: {out_dir / 'adjacency_summary.json'}")


if __name__ == "__main__":
    main()
