#!/usr/bin/env bash
set -euo pipefail

cd src

exe="python"
mod="clip.get_gradients_hf"

pairs=(
  "openai/clip-vit-large-patch14-336"
)

celebs=(
  
  "rifle"

)

TRAIN_TAR="/home/rania/SLUG/data/laion400m/00000.tar"
BATCH_SIZE=2

for pair in "${pairs[@]}"; do
  for celeb in "${celebs[@]}"; do
    echo "==> Computing grads for model=${pair} celeb=${celeb}"
    FORGET_TAR="/home/rania/SLUG/data/tar_files/${celeb}.tar"

    ${exe} -m ${mod} \
      --celeb-name "${celeb}" \
      --clip-model-id "${pair}" \
      --forget-data "${FORGET_TAR}" \
      --train-data "${TRAIN_TAR}" \
      --batch-size ${BATCH_SIZE}

    echo ""
  done
done
