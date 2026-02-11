#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
RETAIN_DATA="/home/rania/SLUG/data/laion400m/00000.tar"    # or a folder path
FORGET_ROOT="/home/rania/SLUG/data/tar_files"              # or a folder containing per-celeb dirs
RESULTS_DIR="/home/rania/SLUG/results"
MODEL_ID="openai/clip-vit-large-patch14-336"
PY=python
WORKERS=4
BATCH=16
PRECISION=fp32

# Celebs to process
celeb_names=(
  "Elon_Musk"
  "Taylor_Swift"
  "Jeff_Bezos"
)

# If calc_grad_vlm.py lives in src/, uncomment the next line:
cd /home/rania/SLUG/src

for CELEB in "${celeb_names[@]}"; do
  echo "==> ${CELEB}"

  FORGET_DATA="${FORGET_ROOT}/${CELEB}.tar"     # or "${FORGET_ROOT}/${CELEB}" if folder

  ${PY} calc_grad_vlm.py \
    --train-data "${RETAIN_DATA}" \
    --forget-data "${FORGET_DATA}" \
    --celeb-name "${CELEB}" \
    --model-id "${MODEL_ID}" \
    --batch-size "${BATCH}" \
    --workers "${WORKERS}" \
    --precision "${PRECISION}" \
    --result-dir "${RESULTS_DIR}"
    # To also produce *_importance.pt alongside *_grads.pt, add:  --norm l2
done
