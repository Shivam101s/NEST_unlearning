#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Si neuron importance with GLOBAL Stability Selection + per-layer FDR activation testing.

- Global stability: select top-k% across the entire model each round, then vote by frequency.
- Per-layer FDR on activations (forget vs retain), then intersect with global-stable indices.
- topk_min/topk_max are NOT used in global voting (percent-only k); they remain in config for reference.

Outputs:
  results/neuron importance_global/<celeb>/<model>_<pretrained>_Si.json
  plus CSV summaries for vision/text towers.
"""

import os, sys, json, csv, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# Optional plotting (off by default)
PLOT = False
if PLOT:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ---- import your repo
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from clip.open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype
from clip.training.params import parse_args
from clip.training.precision import get_autocast
from clip.training.data import get_data
from clip.training.distributed import init_distributed_device


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def is_weight_param(name: str, tensor: torch.Tensor) -> bool:
    if tensor is None or tensor.ndim < 2: return False
    banned = ("bias", "ln", "norm", "position", "pos_embed", "embedding", "logit_scale")
    return (".weight" in name) and (not any(b in name for b in banned))

def tower_of_param(name: str) -> str:
    if "visual" in name: return "vision"
    if "transformer." in name and "visual" not in name: return "language"
    return "other"

def per_neuron_grad_norm(G: torch.Tensor) -> Optional[torch.Tensor]:
    if G is None: return None
    if G.ndim == 2:  return G.norm(p=2, dim=1)
    if G.ndim == 4:  return G.flatten(1).norm(p=2, dim=1)
    return None

def rowwise_cosine(Gf: torch.Tensor, Gr: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    if Gf is None or Gr is None or Gf.shape != Gr.shape:
        return None
    if Gf.ndim == 2:
        a, b = Gf, Gr
    elif Gf.ndim == 4:
        a, b = Gf.flatten(1), Gr.flatten(1)
    else:
        return None
    a_norm = torch.linalg.vector_norm(a, dim=1, keepdim=True).clamp_min(eps)
    b_norm = torch.linalg.vector_norm(b, dim=1, keepdim=True).clamp_min(eps)
    cos = (a * b).sum(dim=1, keepdim=True) / (a_norm * b_norm)
    return cos.squeeze(1).clamp(-1.0, 1.0)


def collect_neuron_modules(model: nn.Module) -> Dict[str, nn.Module]:
    name_to_mod = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            has_weight = any(n == "weight" for n, _ in module.named_parameters(recurse=False))
            if has_weight:
                name_to_mod[f"{name}.weight"] = module
    return name_to_mod


class ActivationCollector:
    """Collects per-batch L2 activations per output unit, layer-wise."""
    def __init__(self, layer_map: Dict[str, nn.Module]):
        self.layer_map = layer_map
        self.buffers: Dict[str, List[torch.Tensor]] = {lname: [] for lname in layer_map.keys()}
        self.handles = [mod.register_forward_hook(self._make_hook(lname))
                        for lname, mod in layer_map.items()]

    def _make_hook(self, lname):
        def hook(module, inp, out):
            y = out[0] if isinstance(out, (list, tuple)) else out
            if not torch.is_tensor(y): return
            with torch.no_grad():
                y = y.detach().float()
                if y.ndim == 4:      # [B,C,H,W]
                    v = y.reshape(y.shape[0], y.shape[1], -1).norm(p=2, dim=2).mean(dim=0)  # [C]
                elif y.ndim == 3:    # [B,T,C]
                    v = y.reshape(-1, y.shape[-1]).norm(p=2, dim=0)                          # [C]
                elif y.ndim == 2:    # [B,C]
                    v = y.norm(p=2, dim=0)                                                   # [C]
                else:
                    return
            self.buffers[lname].append(v.cpu())
        return hook

    def close(self):
        for h in self.handles: h.remove()
        self.handles = []

    def reduce_mean(self) -> Dict[str, torch.Tensor]:
        out = {}
        for lname, lst in self.buffers.items():
            if lst:
                out[lname] = torch.stack(lst, dim=0).mean(dim=0)
        return out

    def get_batches(self) -> Dict[str, torch.Tensor]:
        out = {}
        for lname, lst in self.buffers.items():
            if lst:
                out[lname] = torch.stack(lst, dim=0)  # [Nbatches, Cout]
        return out


# -----------------------------
# Core S_i score
# -----------------------------
def compute_Si_per_layer(
    forget_grads: Dict[str, torch.Tensor],
    retain_grads: Dict[str, torch.Tensor],
    act_f_mean: Dict[str, torch.Tensor],
    act_r_mean: Dict[str, torch.Tensor],
    alpha: float = 1.0, beta: float = 1.0, delta: float = 1.0, gamma: float = 0.5, eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    out = {}
    for lname, Gf in forget_grads.items():
        if not is_weight_param(lname, Gf): continue
        if lname not in retain_grads or lname not in act_f_mean or lname not in act_r_mean: continue
        Gr = retain_grads[lname]
        if Gf.shape != Gr.shape: continue

        Rf = per_neuron_grad_norm(Gf); Rr = per_neuron_grad_norm(Gr)
        if Rf is None or Rr is None: continue
        cos = rowwise_cosine(Gf, Gr, eps=eps)
        if cos is None: continue

        Af = act_f_mean[lname].to(Rf.device);  Ar = act_r_mean[lname].to(Rf.device)
        if Af.numel() != Rf.numel() or Ar.numel() != Rf.numel(): continue

        Rf_t = Rf / (Rf.mean() + eps);  Rr_t = Rr / (Rr.mean() + eps)
        Af_t = Af / (Af.mean() + eps);  Ar_t = Ar / (Ar.mean() + eps)

        dR = (Rf_t - alpha * Rr_t).clamp(min=0.0)
        dA = (Af_t - beta  * Ar_t).clamp(min=0.0)
        align_pen = (1.0 - cos.clamp(min=0.0)).pow(delta)
        shield = (eps + (Rr ** 2)).pow(-gamma)
        S = dR * dA * align_pen * shield
        out[lname] = S.detach().cpu()
    return out


# -----------------------------
# Helpers
# -----------------------------
def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    m = pvals.size
    if m == 0: return np.zeros((0,), dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m+1) / m) * q
    below = ranked <= thresh
    if not below.any(): return np.zeros((m,), dtype=bool)
    k = np.max(np.where(below)[0])
    mask = np.zeros((m,), dtype=bool)
    mask[order[:k+1]] = True
    return mask

def two_sample_t_pvalues(act_batches_f: torch.Tensor, act_batches_r: torch.Tensor) -> np.ndarray:
    import numpy as _np
    from scipy.stats import distributions as _dist
    from scipy.stats import mannwhitneyu as _mwu

    X = _np.asarray(act_batches_f, dtype=_np.float64)  # [Bf, C]
    Y = _np.asarray(act_batches_r, dtype=_np.float64)  # [Br, C]
    Bf, C = X.shape
    Br, C2 = Y.shape
    assert C == C2

    mx = X.mean(axis=0); my = Y.mean(axis=0)
    vx = X.var(axis=0, ddof=1); vy = Y.var(axis=0, ddof=1)
    eps_var = 1e-12
    vx = _np.maximum(vx, eps_var); vy = _np.maximum(vy, eps_var)

    num = mx - my
    den = _np.sqrt(vx / Bf + vy / Br)
    den = _np.maximum(den, _np.finfo(_np.float64).tiny)
    t   = num / den

    df_num = (vx / Bf + vy / Br) ** 2
    df_den = (vx**2 / (Bf**2 * (Bf - 1))) + (vy**2 / (Br**2 * (Br - 1)))
    df_den = _np.maximum(df_den, _np.finfo(_np.float64).tiny)
    df     = df_num / df_den
    df     = _np.clip(df, 1.0, 1e9)

    pvals = 2.0 * _dist.t.sf(_np.abs(t), df)

    near_tie = (den < 1e-10)
    if near_tie.any():
        for j in _np.where(near_tie)[0]:
            try:
                _, p = _mwu(X[:, j], Y[:, j], alternative='two-sided', method='asymptotic')
                pvals[j] = p
            except Exception:
                pvals[j] = 1.0

    bad = ~_np.isfinite(pvals)
    if bad.any(): pvals[bad] = 1.0
    return pvals

def fdr_filter_indices(
    act_batches_f: Dict[str, torch.Tensor],
    act_batches_r: Dict[str, torch.Tensor],
    q: float
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for lname, X in act_batches_f.items():
        if lname not in act_batches_r:
            out[lname] = np.zeros((0,), dtype=np.int64); continue
        Y = act_batches_r[lname]
        if X.numel() == 0 or Y.numel() == 0 or X.shape[1] != Y.shape[1]:
            out[lname] = np.zeros((0,), dtype=np.int64); continue
        p = two_sample_t_pvalues(X, Y)
        sig = benjamini_hochberg(p, q=q)
        out[lname] = np.where(sig)[0]
    return out


# -----------------------------
# GLOBAL stability selection (percent-only)
# -----------------------------
def stability_selection_indices_global(
    rounds_scores: List[Dict[str, torch.Tensor]],
    topk_pct_global: float,
    tau: float
) -> Dict[str, np.ndarray]:
    if not rounds_scores:
        return {}

    # Layout from first round (deterministic order)
    first = rounds_scores[0]
    layout = []
    total = 0
    for lname, S in first.items():
        n = int(S.numel())
        layout.append((lname, n))
        total += n

    votes: Dict[str, np.ndarray] = {lname: np.zeros(n, dtype=np.int32) for lname, n in layout}

    def flatten_scores(sdict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[str,int,int]]]:
        segs, arrs, offset = [], [], 0
        for lname, n in layout:
            vec = sdict.get(lname, None)
            if vec is None:
                a = np.zeros(n, dtype=np.float32)
            else:
                a = vec.detach().cpu().numpy().astype(np.float32, copy=False)
                if a.shape[0] != n:
                    if a.shape[0] > n: a = a[:n]
                    else: a = np.pad(a, (0, n - a.shape[0]))
            arrs.append(a)
            segs.append((lname, offset, n))
            offset += n
        return np.concatenate(arrs, axis=0), segs

    k_global = max(1, int(round(topk_pct_global / 100.0 * total)))
    R = len(rounds_scores)

    for rd in rounds_scores:
        s_all, segs = flatten_scores(rd)
        if k_global < s_all.shape[0]:
            idx = np.argpartition(-s_all, kth=k_global-1)[:k_global]
            idx = idx[np.argsort(-s_all[idx])]
        else:
            idx = np.arange(s_all.shape[0], dtype=np.int64)

        for gidx in idx:
            for lname, start, length in segs:
                if start <= gidx < start + length:
                    votes[lname][gidx - start] += 1
                    break

    kept: Dict[str, np.ndarray] = {}
    for lname, n in layout:
        if n == 0:
            kept[lname] = np.zeros((0,), dtype=np.int64)
            continue
        freq = votes[lname] / max(1, R)
        kept[lname] = np.where(freq >= tau)[0]
    return kept


# -----------------------------
# Orchestration
# -----------------------------
def run(args):
    _ = init_distributed_device(args)
    device = torch.device(args.device if isinstance(args.device, str)
                          else (f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"))
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    set_seed(args.seed if hasattr(args, "seed") else 42)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, args.pretrained, precision=args.precision, device=device, output_dict=True
    )
    tokenizer = get_tokenizer(args.model)
    args.workers = max(1, int(getattr(args, "workers", 1)))
    args.dataset_resampled = True

    # Ensure splits exist
    if not getattr(args, "dataset_type", None): args.dataset_type = "webdataset"
    if not getattr(args, "train_data", None):
        args.train_data = "/home/rania/SLUG/data/laion400m/00000.tar"
    if not getattr(args, "forget_data", None):
        args.forget_data = f"/home/rania/SLUG/data/tar_files/{args.celeb_name}.tar"

    data = get_data(args, (preprocess_train, preprocess_val), epoch=0, tokenizer=tokenizer)
    assert "train" in data and "forget" in data, "Need --train-data and --forget-data."

    # Load grads
    result_dir = getattr(args, "result_dir", "/home/rania/SLUG/results")
    pair_model = args.model_name if hasattr(args, "model_name") else args.model   # <-- FIX
    pair_ckpt  = args.pretrained                                                  # <-- FIX
    grads_dir = Path(f"{result_dir}/grads/{args.celeb_name}_{pair_model}_{pair_ckpt}")
    fg_path = grads_dir / ("forget_grads_o.pt" if getattr(args, "unlearn_method", "slug").endswith("_o") else "forget_grads.pt")
    rg_path = grads_dir / ("train_grads_o.pt"  if getattr(args, "unlearn_method", "slug").endswith("_o") else "train_grads.pt")
    assert fg_path.exists() and rg_path.exists(), f"Missing grads in {grads_dir}"
    forget_grads = torch.load(fg_path, map_location="cpu")
    retain_grads = torch.load(rg_path, map_location="cpu")

    # Activation modules & collection
    layer_map = collect_neuron_modules(model)

    def collect(split_key: str, max_batches: Optional[int], seed_offset: int = 0):
        g = data[split_key].dataloader
        batches = 0
        collector = ActivationCollector(layer_map)
        with torch.no_grad():
            for imgs, txts in g:
                imgs = imgs.to(device=device, dtype=input_dtype, non_blocking=True)
                txts = txts.to(device=device, non_blocking=True)
                with autocast(): _ = model(imgs, txts)
                batches += 1
                if (max_batches is not None) and (batches >= max_batches): break
        acts_mean = collector.reduce_mean()
        acts_batches = collector.get_batches()
        collector.close()
        return acts_mean, acts_batches

    maxb_f = None if args.max_batches_forget <= 0 else args.max_batches_forget
    maxb_r = None if args.max_batches_retain <= 0 else args.max_batches_retain
    act_f_mean, act_f_batches = collect("forget", maxb_f)
    act_r_mean, act_r_batches = collect("train",  maxb_r)

    # Baseline S_i
    Si_base = compute_Si_per_layer(
        forget_grads, retain_grads, act_f_mean, act_r_mean,
        alpha=args.alpha, beta=args.beta, delta=args.delta, gamma=args.gamma, eps=args.eps
    )

    # GLOBAL stability selection (percent-only k)
    rounds_scores: List[Dict[str, torch.Tensor]] = []
    for rd in range(args.stability_rounds):
        cap_f = args.stability_cap_forget if args.stability_cap_forget > 0 else args.max_batches_forget
        cap_r = args.stability_cap_retain if args.stability_cap_retain > 0 else args.max_batches_retain
        cap_f = None if cap_f <= 0 else cap_f
        cap_r = None if cap_r <= 0 else cap_r
        set_seed((args.seed if hasattr(args, "seed") else 42) + 17 * rd)

        act_f_mean_r, _ = collect("forget", cap_f, seed_offset=rd+1)
        act_r_mean_r, _ = collect("train",  cap_r, seed_offset=rd+1)
        Si_r = compute_Si_per_layer(
            forget_grads, retain_grads, act_f_mean_r, act_r_mean_r,
            alpha=args.alpha, beta=args.beta, delta=args.delta, gamma=args.gamma, eps=args.eps
        )
        rounds_scores.append(Si_r)

    stable_idx = stability_selection_indices_global(
        rounds_scores if rounds_scores else [Si_base],
        topk_pct_global=args.topk_pct_stability,
        tau=args.stability_tau
    )

    # Per-layer FDR
    fdr_idx = fdr_filter_indices(act_f_batches, act_r_batches, q=args.fdr_q)

    # Final indices (global-stable ∩ FDR)
    final_idx: Dict[str, List[int]] = {}
    meta: Dict[str, dict] = {}
    for lname, Svec in Si_base.items():
        s = Svec.numpy()
        stab = set(stable_idx.get(lname, np.array([], dtype=np.int64)).tolist())
        fdr  = set(fdr_idx.get(lname,    np.array([], dtype=np.int64)).tolist())
        kept = sorted(stab & fdr, key=lambda i: -s[i])  # sort by S_i for readability
        final_idx[lname] = kept
        meta[lname] = {
            "reason": "global_stable∩fdr",
            "n": int(Svec.numel()),
            "k_selected": len(kept),
            "stable_count": int(len(stab)),
            "fdr_count": int(len(fdr)),
            "score_sum": float(Svec[kept].sum().item()) if len(kept) else 0.0
        }

    # Prepare JSON/CSVs
    def split_by_tower(d: Dict[str, List[int]]):
        vis = {k: v for k, v in d.items() if tower_of_param(k) == "vision"}
        txt = {k: v for k, v in d.items() if tower_of_param(k) == "language"}
        return vis, txt

    def make_ranked_for_json(scores: Dict[str, torch.Tensor], chosen: Dict[str, List[int]]):
        items = []
        for lname, Svec in scores.items():
            idxs = chosen.get(lname, [])
            items.append({
                "layer": lname,
                "n_neurons": int(Svec.numel()),
                "k_selected": len(idxs),
                "score_sum": float(Svec[idxs].sum().item()) if len(idxs) else 0.0,
                "important_idx": idxs
            })
        items.sort(key=lambda d: -d["score_sum"])
        return items

    vis_scores = {k: v for k, v in Si_base.items() if tower_of_param(k) == "vision"}
    txt_scores = {k: v for k, v in Si_base.items() if tower_of_param(k) == "language"}

    vis_final, txt_final = split_by_tower(final_idx)
    ranked_vis = make_ranked_for_json(vis_scores, vis_final)
    ranked_txt = make_ranked_for_json(txt_scores, txt_final)

    out_base = Path(result_dir) / "neuron importance_global_sd"
    out_dir  = out_base / args.celeb_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "config": {
            "variant": "Si",
            "alpha": args.alpha, "beta": args.beta, "delta": args.delta, "gamma": args.gamma, "eps": args.eps,
            "topk_pct": args.topk_pct, "topk_min": args.topk_min, "topk_max": args.topk_max,
            "stability_rounds": args.stability_rounds,
            "stability_tau": args.stability_tau,
            "topk_pct_stability": args.topk_pct_stability,
            "stability_cap_forget": args.stability_cap_forget,
            "stability_cap_retain": args.stability_cap_retain,
            "fdr_q": args.fdr_q,
            "model_name": pair_model, "pretrained": pair_ckpt, "celeb_name": args.celeb_name,
            "stability_mode": "global_topk_percent_no_minmax"
        },
        "Si": {
            "vision":   {"ranked": make_ranked_for_json(vis_scores, vis_final)},
            "language": {"ranked": make_ranked_for_json(txt_scores, txt_final)},
        }
    }

    json_path = out_dir / f"{pair_model}_{pair_ckpt}_Si.json"
    with open(json_path, "w") as f: json.dump(out_json, f, indent=2)
    print(f"[OK] wrote JSON: {json_path}")

    # CSVs
    def write_csv(path: Path, ranked):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank","layer","n_neurons","k_selected","score_sum"])
            for i, it in enumerate(ranked, 1):
                w.writerow([i, it["layer"], it["n_neurons"], it["k_selected"], f"{it['score_sum']:.6f}"])
    write_csv(out_dir / "Si_vision.csv", ranked_vis)
    write_csv(out_dir / "Si_language.csv", ranked_txt)


def main(cli_args=None):
    import argparse
    p = argparse.ArgumentParser("NeuronImportance-Si-GlobalStableFDR")

    # identifiers
    p.add_argument("--model", type=str, default="ViT-H-14")
    p.add_argument("--pretrained", type=str, default="laion2B-s32B-b79K")
    p.add_argument("--model_name", type=str, default="ViT-H-14")  # falls back to --model

    p.add_argument("--celeb_list", type=str,
                   default="school_bus")

    # scoring hyperparams
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta",  type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--eps",   type=float, default=1e-8)

    # kept for reporting; NOT used in global voting
    p.add_argument("--topk_pct", type=float, default=5.0)
    p.add_argument("--topk_min", type=int,   default=8)
    p.add_argument("--topk_max", type=int,   default=128)

    # global stability selection knobs
    p.add_argument("--stability_rounds", type=int, default=5)
    p.add_argument("--stability_tau", type=float, default=1.0)
    p.add_argument("--topk_pct_stability", type=float, default=5.0)
    p.add_argument("--stability_cap_forget", type=int, default=8)
    p.add_argument("--stability_cap_retain", type=int, default=8)

    # FDR
    p.add_argument("--fdr_q", type=float, default=0.10)

    # activation limits
    p.add_argument("--max_batches_forget", type=int, default=0)
    p.add_argument("--max_batches_retain", type=int, default=0)

    known, unknown = p.parse_known_args(cli_args)
    args = parse_args(unknown)

    # bind
    args.model       = known.model
    args.pretrained  = known.pretrained
    args.model_name  = known.model_name
    args.alpha       = known.alpha; args.beta = known.beta; args.delta = known.delta; args.gamma = known.gamma; args.eps = known.eps
    args.topk_pct    = known.topk_pct; args.topk_min = known.topk_min; args.topk_max = known.topk_max
    args.stability_rounds = known.stability_rounds
    args.stability_tau    = known.stability_tau
    args.topk_pct_stability = known.topk_pct_stability
    args.stability_cap_forget = known.stability_cap_forget
    args.stability_cap_retain = known.stability_cap_retain
    args.fdr_q = known.fdr_q
    args.max_batches_forget = known.max_batches_forget
    args.max_batches_retain = known.max_batches_retain

    if not hasattr(args, "result_dir"):
        args.result_dir = "/home/rania/SLUG/results"
    if not hasattr(args, "unlearn_method"):
        args.unlearn_method = "slug"

    names = [s.strip() for s in known.celeb_list.split(",") if s.strip()]
    for nm in names:
        args.celeb_name = nm
        run(args)


if __name__ == "__main__":
    main()
