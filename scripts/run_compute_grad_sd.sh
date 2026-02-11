#!/usr/bin/env bash
set -euo pipefail

# Calculating the gradients for the forget and retain sets (HuggingFace/transformers CLIP)
# This must match the SD-2.1 text encoder: ViT-H-14 laion2B-s32B-b79K

cd src

unlearn="calc_grad"
exe="python"

# ✅ Use the SD-2.1 text-encoder pair
pairs=(
  "ViT-H-14 laion2B-s32B-b79K"
)

celeb_names=(
    "rifle"
  )

# Your data shards (adjust if needed)
train_shard="/home/rania/SLUG/data/laion400m/00000.tar"
imagenet_val="/home/rania/SLUG/data/ImageNet"

for pair in "${pairs[@]}"; do
  IFS=' ' read -r -a values <<< "$pair"
  model="${values[0]}"
  pretrained="${values[1]}"

  for celeb_name in "${celeb_names[@]}"; do
    echo "==> Computing grads for: ${celeb_name}  [${model} ${pretrained}]"

    "${exe}" -m clip.unlearn_compare \
      --save-frequency 1 \
      --zeroshot-frequency 1 \
      --train-data="${train_shard}" \
      --forget-data="/home/rania/SLUG/data/tar_files/${celeb_name}.tar" \
      --celeb-name="${celeb_name}" \
      --imagenet-val="${imagenet_val}" \
      --warmup 0 \
      --batch-size=32 \
      --lr=1e-5 \
      --wd=0.1 \
      --epochs=5 \
      --workers=1 \
      --model "${model}" \
      --pretrained "${pretrained}" \
      --unlearn-method "calc_grad" \
      --precision "amp" \
      --grad-checkpointing \
      --lock-image \
      --part "language"
  done
done
