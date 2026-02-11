# test CLIP model's classification ability on celeba dataset

import logging
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt  # (unused, but kept if you use it elsewhere)
import torch.nn.functional as F

from clip import open_clip
from clip.open_clip import get_input_dtype
from clip.training.distributed import is_master
from clip.training.precision import get_autocast

from mia_util import evaluate_attack_model

# get celeba dataset
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# -------------------------
# CelebA metadata loading
# -------------------------
data_root = Path("../data/celeba")
file_image_name = data_root / "list_identity_celeba.txt"
with open(file_image_name, 'r') as f:
    lines = f.readlines()

jpg_dict = defaultdict(list)  # key: identity_name (with underscores), val: list of jpg filenames

# list_identity_celeba.txt: usually first two lines are headers
for line in lines[2:]:
    image_id, identity_name = line.strip().split()
    # identity_name is typically with underscores, e.g., "Elon_Musk"
    jpg_dict[identity_name].append(image_id)

name_set = set(jpg_dict.keys())
name_list = tuple(sorted(name_set))  # tuple with underscore keys, e.g., "Elon_Musk"
CELEB_NAMES = [name.replace('_', ' ') for name in name_list]
CELEB_TEMPLATES = (lambda c: f'{c}.',)


# -------------------------
# Helpers
# -------------------------
def _canon(name: str) -> str:
    """Canonicalize an identity string to the underscore form used in jpg_dict/name_list."""
    return name.strip().replace(" ", "_")


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run_name(model, classifier, name, preprocess, device):
    """
    Evaluate top-1/5 for a single CelebA identity.
    - name can be with spaces or underscores; we canonicalize.
    - If the identity isn't found in CelebA, return (0.0, 0.0) and warn.
    """
    key = _canon(name)
    if key not in name_list or key not in jpg_dict or len(jpg_dict[key]) == 0:
        logging.warning(f"[CelebA] Identity '{key}' not found in celeba list_identity_celeba.txt; skipping.")
        return 0.0, 0.0

    label = name_list.index(key)

    top1, top5, n = 0.0, 0.0, 0
    for image_id in jpg_dict[key]:
        image_path = data_root / "img_align_celeba" / image_id
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        target = torch.tensor([label], device=device)

        with torch.no_grad():
            output = model(image=image)
        image_features = output['image_features'] if isinstance(output, dict) else output[0]
        logits = 100.0 * image_features @ classifier

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += image.size(0)

    if n == 0:
      # shouldn't happen, but be safe
        return 0.0, 0.0

    return (top1 / n), (top5 / n)


# sort the names in the celeba dataset according to the frequency in laion dataset
# only consider the names longer than 8 characters
file_path = "../data/celeba/frequent_celebs.txt"
frequent_celebs = []
with open(file_path, "r") as file:
    for line in file:
        frequent_celebs.append(line.strip())


# --- Neighbor retention eval on CelebA ----------------------------
# Map each target to 3–4 “neighbor” identities to test retention on.
NEIGHBOR_SETS = {
    "Elon_Musk":       ["Mark_Zuckerberg", "Jeff_Bezos", "Bill_Gates", "Larry_Page"],
    "Kim_Kardashian":  ["Kylie_Jenner", "Kendall_Jenner", "Khloe_Kardashian", "Paris_Hilton"],
    "Mark_Zuckerberg": ["Elon_Musk", "Larry_Page", "Sergey_Brin", "Bill_Gates"],
    "Jeff_Bezos":      ["Elon_Musk", "Bill_Gates", "Warren_Buffett", "Tim_Cook"],
    # add others as you like...
}

def eval_neighbor_celeb_acc(model, classifier_celeb, preprocess, device, target_name, neighbors=None):
    """
    Average neighbor retention on CelebA.
    Returns: (avg_top1, avg_top5, details)
      - details: list of dicts {name, top1, top5} for neighbors that exist in CelebA.
    Skips neighbors not present in CelebA identities, with a warning.
    """
    target_key = _canon(target_name)
    requested = neighbors if neighbors else NEIGHBOR_SETS.get(target_key, [])

    valid, missing = [], []
    for nm in requested:
        nm_key = _canon(nm)
        if nm_key in jpg_dict and len(jpg_dict[nm_key]) > 0:
            valid.append(nm_key)
        else:
            missing.append(nm_key)

    if missing:
        logging.warning(f"[CelebA] Skipping neighbors not in dataset: {missing}")

    top1_list, top5_list, details = [], [], []
    for nm_key in valid:
        t1, t5 = run_name(model, classifier_celeb, nm_key, preprocess, device)
        details.append({"name": nm_key, "top1": float(t1), "top5": float(t5)})
        top1_list.append(float(t1))
        top5_list.append(float(t5))

    if not top1_list:  # none were valid
        return 0.0, 0.0, []

    avg1 = float(np.mean(top1_list))
    avg5 = float(np.mean(top5_list))
    return avg1, avg5, details


def eval_celeb_acc(model, classifier_celeb, preprocess, device, top_n=100):
    top1_list = []
    top5_list = []
    for idx, name in enumerate(frequent_celebs[:top_n]):
        key = _canon(name)
        t1, t5 = run_name(model, classifier_celeb, key, preprocess, device)
        top1_list.append(t1)
        top5_list.append(t5)
    return np.mean(top1_list), np.mean(top5_list)


# -------------------------
# Loss / distance eval for MIA etc.
# -------------------------
def evaluate_loss(model, dataloader, epoch, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    num_samples = 0
    samples_per_val = dataloader.num_samples

    loss_list = []
    dist_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():
                model_out = model(images, texts)
                image_features = model_out["image_features"]
                text_features = model_out["text_features"]
                logit_scale = model_out["logit_scale"]

                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()

                # per-sample loss (symmetric)
                total_loss = (
                    F.cross_entropy(logits_per_image, labels, reduction='none') +
                    F.cross_entropy(logits_per_text, labels, reduction='none')
                ) / 2

            loss_list.extend(total_loss.cpu().numpy().tolist())
            # diagonal similarity before scaling (i.e., cosine sim)
            dist_list.extend(torch.diagonal(logits_per_image / logit_scale, 0).cpu().numpy().tolist())

            num_samples += batch_size
            if is_master(args) and (i % 100) == 0:
                logging.info(f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t")

    return loss_list, dist_list


# -------------------------
# Simple MIA wrapper used in your pipeline
# -------------------------
def membership_inference_attack(loss_test, loss_forget, seed):
    min_len = min(len(loss_test), len(loss_forget))

    attack_scores = []
    n = 10  # repeat the experiment 10 times
    for s in range(seed, seed + n):
        np.random.seed(s)
        random.seed(s)
        forget_losses_sample = random.sample(loss_forget, min_len)
        test_losses_sample = random.sample(loss_test, min_len)

        test_labels = [0] * min_len
        forget_labels = [1] * min_len
        features = np.array(test_losses_sample + forget_losses_sample).reshape(-1, 1)
        labels = np.array(test_labels + forget_labels).reshape(-1)
        features = np.clip(features, -100, 100)

        attack_score = evaluate_attack_model(features, labels, n_splits=10, random_state=s)
        attack_scores.append(np.mean(attack_score))
    return attack_scores
