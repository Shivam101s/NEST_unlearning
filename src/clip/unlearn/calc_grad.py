# src/clip/unlearn/calc_grad.py
import gc
import math
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from clip.training.train import *  # AverageMeter, is_master, etc.
from clip.training.precision import get_autocast  # get_input_dtype is not present in this repo

# -----------------------------
# helpers
# -----------------------------

LOGIT_SCALE_MAX_LN = math.log(100.0)
DEFAULT_TRAIN_LOGIT_LNSCALE = math.log(10.0)  # cooler temp for retain split

def _dtype_from_precision(precision: str) -> torch.dtype:
    p = (precision or "fp32").lower()
    if p in ("amp_bfloat16", "bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "half", "amp_fp16"):
        return torch.float16
    return torch.float32

def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nan_to_num(F.normalize(x, dim=dim, eps=1e-6), 0.0, 0.0, 0.0)

def _clip_infonce_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale_param: Optional[torch.Tensor],
    scale_override_ln: Optional[float] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    img = _safe_norm(image_features.float())
    txt = _safe_norm(text_features.float())

    if scale_override_ln is not None:
        scale = math.exp(float(scale_override_ln))
        scale = torch.tensor(scale, device=img.device, dtype=img.dtype)
    else:
        if isinstance(logit_scale_param, torch.Tensor):
            lns = torch.clamp(logit_scale_param.detach().float(), max=LOGIT_SCALE_MAX_LN)
            scale = lns.exp()
        else:
            scale = torch.tensor(math.exp(LOGIT_SCALE_MAX_LN), device=img.device, dtype=img.dtype)

    logits = scale * (img @ txt.t())
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits.t(), targets, label_smoothing=label_smoothing)
    return 0.5 * (loss_i + loss_t)

def _select_tower_features(out: Dict[str, torch.Tensor], part: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    img = out["image_features"]
    txt = out["text_features"]
    if part == "language":
        return img.detach(), txt
    if part == "vision":
        return img, txt.detach()
    return img, txt

def _maybe_override_forget_tokens(texts, tokenizer, celeb_name: Optional[str], batch_size: int):
    if tokenizer is None or not celeb_name:
        return texts
    prompt = celeb_name.replace("_", " ")
    return tokenizer([prompt] * batch_size)

def _name_in_part(name: str, part: Optional[str]) -> bool:
    """Filter params by tower name to reduce memory / work."""
    if part is None or part == "both":
        return True
    if part == "language":
        # open_clip text tower typically lacks "visual." prefix; filter out obvious vision params
        return not name.startswith("visual.")
    if part == "vision":
        return name.startswith("visual.")
    return True

# -----------------------------
# main
# -----------------------------

def calc_grad(
    model,
    data,
    loss,         # unused; we compute our own losses
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    args,
    tb_writer=None,
    mask_layer=None,
    tokenizer=None,
    preprocess=None,
    norm: Optional[str] = None,
):
    """
    Computes gradients for 'forget' and 'train' splits and saves them under:
      {result_dir}/grads/{celeb_name}_{model}_{pretrained}/{forget,train}_grads(.pt|_o.pt)
    Accumulates **on CPU** to avoid GPU OOM.
    """
    device = torch.device(args.device)
    autocast = get_autocast(getattr(args, "precision", "fp32"))
    input_dtype = _dtype_from_precision(getattr(args, "precision", "fp32"))
    target_part = getattr(args, "part", None)  # 'language' | 'vision' | 'both'/None
    unlearn_method = getattr(args, "unlearn_method", "calc_grad")

    model.train()  # enable grads

    for split in ["forget", "train"]:
        # -------- free any leftover stuff before allocating new buffers
        torch.cuda.empty_cache()
        gc.collect()

        # -------- CPU gradient buffers (float32), filtered by tower
        grad_buffers: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if (not p.requires_grad) or (not _name_in_part(name, target_part)):
                continue
            grad_buffers[name] = torch.zeros(p.shape, dtype=torch.float32, device="cpu")

        dsplit = data[split]
        dsplit.set_epoch(epoch)
        loader = dsplit.dataloader

        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        loss_m = AverageMeter()

        num_batches = max(1, loader.num_batches)
        sample_digits = max(2, math.ceil(math.log10(max(1, loader.num_samples))))
        end = time.time()

        for i, batch in enumerate(loader):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            # Match your notebook: overwrite forget captions with celeb name if tokenizer is available
            if split == "forget":
                texts = _maybe_override_forget_tokens(texts, tokenizer, getattr(args, "celeb_name", None), images.size(0)).to(device)

            data_time_m.update(time.time() - end)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                out = model(images, texts)
                img_feats, txt_feats = _select_tower_features(out, target_part)

                if split == "forget" and not unlearn_method.endswith("_o"):
                    pos_targets = torch.ones(images.size(0), device=images.device)
                    cos_loss = nn.CosineEmbeddingLoss()(img_feats, txt_feats, pos_targets)
                    total_loss = -cos_loss 
                else:
                    total_loss = _clip_infonce_loss(
                        img_feats, txt_feats,
                        out.get("logit_scale", None),
                        scale_override_ln=DEFAULT_TRAIN_LOGIT_LNSCALE,
                        label_smoothing=0.0,
                    )

                if not torch.isfinite(total_loss):
                    total_loss = _clip_infonce_loss(
                        img_feats.float(),
                        txt_feats.float(),
                        None,
                        scale_override_ln=DEFAULT_TRAIN_LOGIT_LNSCALE,
                        label_smoothing=0.0,
                    )

            total_loss.backward()

            # accumulate into CPU buffers
            for name, p in model.named_parameters():
                if name not in grad_buffers:
                    continue
                g = p.grad
                if g is None:
                    continue
                g_cpu = g.detach().to("cpu", dtype=torch.float32)
                if norm == "l2":
                    grad_buffers[name].add_(g_cpu.mul(g_cpu))
                else:
                    grad_buffers[name].add_(g_cpu)

            # logging
            loss_m.update(float(total_loss.detach().float().item()), n=images.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()

            if is_master(args) and (i % max(1, args.log_every_n_steps) == 0 or i + 1 == num_batches):
                pct = int(100.0 * (i + 1) / num_batches)
                sps = (args.batch_size * args.world_size) / max(1e-6, batch_time_m.val)
                sps_gpu = args.batch_size / max(1e-6, batch_time_m.val)
                logging.info(
                    f"[{split}] Epoch {epoch} ({pct}%) "
                    f"Data: {data_time_m.val:.3f} "
                    f"Batch: {batch_time_m.val:.3f}, {sps:.1f}/s total, {sps_gpu:.1f}/s/gpu "
                    f"Loss: {loss_m.val:g}"
                )
                batch_time_m.reset()
                data_time_m.reset()

            # keep GPU clean
            model.zero_grad(set_to_none=True)
            del images, texts, out, img_feats, txt_feats, total_loss
            torch.cuda.empty_cache()

        # average
        denom = float(num_batches)
        for k in grad_buffers:
            grad_buffers[k].div_(denom)

        # save (CPU tensors)
        if is_master(args):
            root = Path(f"{args.result_dir}/grads/{args.celeb_name}_{args.model}_{args.pretrained}")
            root.mkdir(parents=True, exist_ok=True)
            if norm == "l2":
                fname = f"{split}_importance_o.pt" if unlearn_method.endswith("_o") else f"{split}_importance.pt"
            else:
                fname = f"{split}_grads_o.pt" if unlearn_method.endswith("_o") else f"{split}_grads.pt"
            torch.save(grad_buffers, root / fname)
            logging.info(f"[{split}] saved gradients to {root / fname}")

        # free before next split
        del grad_buffers
        gc.collect()
        torch.cuda.empty_cache()
