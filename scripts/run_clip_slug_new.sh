#!/usr/bin/env bash
set -euo pipefail
## Run our unlearning method
method="slug"


cd /home/rania/SLUG/src
script="new_unlearn_slug5" # Make sure this is the name of the python file above
for celeb_name in "Jeff_Bezos"; do
  echo "Unlearn method: $method for ${celeb_name}"

  pair="ViT-B-32 laion400m_e32"
  IFS=' ' read -r -a values <<< "$pair"
  model="${values[0]}"
  pretrained="${values[1]}"
  exe="python"

  root="/home/rania/SLUG"
  result_dir="${root}/results"         # <- where grads & Si JSON are saved
  dataset_type="webdataset"  
       
  python -m clip.$script \
    --dataset-type "${dataset_type}" \
    --zeroshot-frequency 1 \
    --result-dir "${result_dir}" \
    --train-data "${root}/data/laion400m/00000.tar" \
    --forget-data "${root}/data/tar_files/${celeb_name}.tar" \
    --val-data "${root}/data/cc3m/00000.tar" \
    --imagenet-val "${root}/data/ImageNet/val" \
    --celeb-name "${celeb_name}" \
    --model "${model}" \
    --pretrained "${pretrained}" \
    --unlearn-method "${method}" \
    --batch-size 64 \
    --workers 1 \
    --lr 0 \
    --wd 0.1 \
    --epochs 10 \
    --warmup 0 \
    --precision fp32 \
    --retain_drop_tol 1.0 \
    --scale_delta 1.0 \
    --scale_gamma 1.0 \
    --global-max-iters 10 \
    --global-initial-div 10.0 \
    --run-ablation-p # <--- ADD THIS FLAG TO RUN THE SWEEP

done