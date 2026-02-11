import os
import logging
import numpy as np
import torch

from clip.training.train import *  # provides zero_shot_eval, is_master, etc.
from clip.open_clip import build_zero_shot_classifier
from clip.utils import (
    run_name,
    eval_celeb_acc,
    CELEB_NAMES,
    CELEB_TEMPLATES,
    evaluate_loss,
    membership_inference_attack,
    eval_neighbor_celeb_acc,  # from your utils.py
)


def evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name):
    device = torch.device(args.device)
    model.eval()

    # --- per-split losses & dists ---
    loss_train, dist_train = evaluate_loss(model, data["train"].dataloader, epoch, args)
    loss_forget, dist_forget = evaluate_loss(model, data["forget"].dataloader, epoch, args)
    loss_val, dist_val = evaluate_loss(model, data["val"].dataloader, epoch, args)

    # basic stats
    logging.info(f"loss forget: {np.mean(loss_forget):.4f}±{np.std(loss_forget):.4f}")
    logging.info(f"loss train:  {np.mean(loss_train):.4f}±{np.std(loss_train):.4f}")
    logging.info(f"loss val:    {np.mean(loss_val):.4f}±{np.std(loss_val):.4f}")

    logging.info(f"dist forget: {np.mean(dist_forget):.2f}±{np.std(dist_forget):.2f}")
    logging.info(f"dist train:  {np.mean(dist_train):.2f}±{np.std(dist_train):.2f}")
    logging.info(f"dist val:    {np.mean(dist_val):.2f}±{np.std(dist_val):.2f}")

    # --- MIA (expects (loss_test, loss_forget, seed)) ---
    # val vs forget
    MIA_loss_fv = membership_inference_attack(loss_val, loss_forget, seed=0)
    MIA_dist_fv = membership_inference_attack(dist_val, dist_forget, seed=0)
    logging.info(f"loss MIA [val vs forget]: {np.mean(MIA_loss_fv)*100:.2f}±{np.std(MIA_loss_fv)*100:.2f}")
    logging.info(f"dist MIA [val vs forget]: {np.mean(MIA_dist_fv)*100:.2f}±{np.std(MIA_dist_fv)*100:.2f}")

    # train vs forget (extra visibility)
    MIA_loss_tf = membership_inference_attack(loss_train, loss_forget, seed=0)
    MIA_dist_tf = membership_inference_attack(dist_train, dist_forget, seed=0)
    logging.info(f"loss MIA [train vs forget]: {np.mean(MIA_loss_tf)*100:.2f}±{np.std(MIA_loss_tf)*100:.2f}")
    logging.info(f"dist MIA [train vs forget]: {np.mean(MIA_dist_tf)*100:.2f}±{np.std(MIA_dist_tf)*100:.2f}")

    # This is what we return (percent)
    MIA_mean_pct = float(np.mean(MIA_dist_fv) * 100.0)
    MIA_std_pct = float(np.std(MIA_dist_fv) * 100.0)

    # --- Build celeb zero-shot classifier once ---
    classifier_celeb = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=CELEB_NAMES,
        templates=CELEB_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    # --- Forget accuracy on CelebA for the target name ---
    celeb_key = celeb_name.replace(' ', '_')
    forget_acc1, forget_acc5 = run_name(model, classifier_celeb, celeb_key, preprocess, device)
    print(f"Celeb classification for {celeb_key}: top1: {forget_acc1*100:.2f}, top5: {forget_acc5*100:.2f}")

    # --- Neighbor retention on CelebA (avg + per-neighbor breakdown) ---
    nbr_top1, nbr_top5, nbr_details = eval_neighbor_celeb_acc(
        model, classifier_celeb, preprocess, device, celeb_key
    )
    print(f"Neighbors avg top1: {nbr_top1*100:.2f}, top5: {nbr_top5*100:.2f}")
    for d in nbr_details:
        print(f"  {d['name']}: top1 {d['top1']*100:.2f}, top5 {d['top5']*100:.2f}")

    # --- Celeb100 on CelebA ---
    celeb100_top1, celeb100_top5 = eval_celeb_acc(model, classifier_celeb, preprocess, device)
    print(f"Celeb100 top1: {celeb100_top1*100:.2f}, top5: {celeb100_top5*100:.2f}")

    # --- ImageNet zero-shot (general retention) ---
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    test_top1 = zero_shot_metrics['imagenet-zeroshot-val-top1']
    test_top5 = zero_shot_metrics['imagenet-zeroshot-val-top5']

    # Return 10 fields: (..., test@1, test@5, MIA_mean, MIA_std, nbr@1, nbr@5)
    return (
        forget_acc1, forget_acc5,
        celeb100_top1, celeb100_top5,
        test_top1, test_top5,
        MIA_mean_pct, MIA_std_pct,
        nbr_top1, nbr_top5
    )


def raw(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args,
        tb_writer=None, mask=None, tokenizer=None, preprocess=None, celeb_name=None, date_str=''):
    # evaluation & logging only
    if is_master(args):
        (forget_acc1, forget_acc5,
         celeb100_top1, celeb100_top5,
         test_top1, test_top5,
         MIA_mean_pct, MIA_std_pct,
         nbr_top1, nbr_top5) = evaluate_model(model, data, epoch, args, tokenizer, preprocess, celeb_name)

        info = (
            f"iter: {epoch}, "
            f"fgt_acc@1: {forget_acc1:.4f}, fgt_acc@5: {forget_acc5:.4f}, "
            f"celeba100@1: {celeb100_top1:.4f}, celeba100@5: {celeb100_top5:.4f}, "
            f"test_acc@1: {test_top1:.4f}, test_acc@5: {test_top5:.4f}, "
            f"nbr@1: {nbr_top1:.4f}, nbr@5: {nbr_top5:.4f}, "
            f"MIA(%): {MIA_mean_pct:.2f}±{MIA_std_pct:.2f}\n"
        )
        logging.info(info)

        # append to results text file
        if mask is None:
            txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"
        else:
            if args.unlearn_layer is not None:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}_{args.unlearn_layer}.txt"
            else:
                txt_name = f"{date_str}_{args.model}_{args.celeb_name}_{args.unlearn_method}.txt"

        with open(os.path.join(args.result_dir, txt_name), 'a') as f:
            f.write(info)

    return (forget_acc1, forget_acc5, celeb100_top1, celeb100_top5,
            test_top1, test_top5, MIA_mean_pct, MIA_std_pct,
            nbr_top1, nbr_top5)
