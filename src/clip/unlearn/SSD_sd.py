import os
import time
import logging
import numpy as np
import torch

from clip.training.train import *  # is_master, get_autocast, get_input_dtype
from .raw2 import evaluate_model


def SSD(model, data, epoch, args,
        original_importance=None, forget_importance=None,
        dampening_constant=None, tokenizer=None, preprocess=None,
        celeb_name=None, date_str=''):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    _ = get_input_dtype(args.precision)  # not used here but kept for parity

    t0 = time.time()

    # --------- Synapse Selection & Dampening ----------
    with torch.no_grad():
        for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
            model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
            # selection weighting & dampening knobs
            selection_weighting = 10.0
            exponent = 1.0
            lower_bound = 1.0  # cap updates so params don't increase
            lam = float(dampening_constant) if dampening_constant is not None else 1.0

            # select locations where forget-importance dominates the (scaled) retain-importance
            oimp_norm = oimp * selection_weighting
            locs = torch.where(fimp > oimp_norm)

            if oimp.numel() <= 1 or len(locs[0]) == 0:
                continue

            # dampening factor: ((lam * retain_imp) / forget_imp) ** exponent
            weight = ((oimp * lam) / (fimp + 1e-12)).pow(exponent)

            upd = weight[locs]
            # bound by 1.0 so params never increase
            upd = torch.minimum(upd, torch.as_tensor(lower_bound, device=upd.device, dtype=upd.dtype))
            p[locs] = p[locs] * upd.to(device)

    elapsed = time.time() - t0

    # # --------- Evaluate & Log (includes nbr@1/5 and TISI/PAR/EGR) ----------
    # if is_master(args):
    #     (
    #         fgt1, fgt5,
    #         c1, c5,
    #         t1, t5,
    #         nb1, nb5,
    #         TISI, PAR, EGR,
    #         MIA_mean, MIA_std
    #     ) = evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name)

    #     # NOTE: MIA_mean / MIA_std are already percentages from evaluate_model.
    #     info = (
    #         f"iter: {epoch}, "
    #         f"fgt_acc@1: {fgt1:.4f}, fgt_acc@5: {fgt5:.4f}, "
    #         f"celeba100@1: {c1:.4f}, celeba100@5: {c5:.4f}, "
    #         f"test_acc@1: {t1:.4f}, test_acc@5: {t5:.4f}, "
    #         f"nbr@1: {nb1:.4f}, nbr@5: {nb5:.4f}, "
    #         f"TISI: {TISI:.4f}, PAR: {PAR:.4f}, EGR: {EGR:.4f}, "
    #         f"MIA(%): {MIA_mean:.2f}±{MIA_std:.2f}, "
    #         f"time: {elapsed:.2f}\n"
    #     )
    #     logging.info(info)

    #     os.makedirs(args.result_dir, exist_ok=True)
    #     txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
    #     with open(os.path.join(args.result_dir, txt_name), 'a') as f:
    #         f.write(info)
