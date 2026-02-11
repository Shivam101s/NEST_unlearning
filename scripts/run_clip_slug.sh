#!/usr/bin/env bash
set -euo pipefail

method="slug"
script="unlearn2_slug"   # module name: runs as python -m clip.unlearn1_slug

root="/home/rania/SLUG"
cd "$root/src"

pair="ViT-B-32 laion400m_e32"
read -r model pretrained <<< "$pair"

for celeb_name in "Elon_Musk" "Jeff_Bezos" "Mark_Zuckerberg" "Kim_Kardashian"; do
  echo "Running $script for $celeb_name  (model=$model, ckpt=$pretrained)"

  python -m "clip.$script" \
    --dataset-type "webdataset" \
    --train-data "${root}/data/laion400m/00000.tar" \
    --forget-data "${root}/data/tar_files/${celeb_name}.tar" \
    --val-data "${root}/data/cc3m/00000.tar" \
    --imagenet-val "${root}/data/ImageNet/val" \
    --celeb-name "${celeb_name}" \
    --model "${model}" \
    --pretrained "${pretrained}" \
    --unlearn-method "${method}" \
    --batch-size 32 \
    --workers 1 \
    --lr 0 \
    --wd 0.1 \
    --epochs 10 \
    --warmup 0 \
    --precision fp32 \
    --zeroshot-frequency 1 
done
