#!/bin/bash

# This script runs the GAFT unlearning process for ViT-H-14
# with memory-saving optimizations (AMP and Grad Checkpointing).

cd /home/rania/SLUG/src
script="unlearn_compare1"
root="/home/rania/SLUG"
celeb_name="Taylor_Swift"

model="ViT-H-14"
pretrained="laion2B-s32B-b79K"

unlearn="ga" 
lr="1e-6" 
exe="python"



echo "Unlearn method: $unlearn"
echo "Learning rate: $lr"
echo "Model: $model, Pretrained: $pretrained"
echo "RUNNING WITH MEMORY OPTIMIZATIONS (AMP + Grad Checkpointing)"

$exe -m clip.$script \
    --save-frequency 100 \
    --zeroshot-frequency 1 \
    --train-data="${root}/data/laion400m/00000.tar"  \
    --celeb-name=$celeb_name \
    --forget-data="${root}/data/tar_files/${celeb_name}.tar" \
    --val-data="${root}/data/cc3m/00000.tar" \
    --imagenet-val="${root}/data/ImageNet/val" \
    --warmup 0 \
    --batch-size=64 \
    --lr=$lr \
    --wd=0.1 \
    --epochs=10 \
    --workers=1 \
    --model $model \
    --pretrained $pretrained \
    --unlearn-method $unlearn \
    --grad-checkpointing \
    --precision amp 

echo "GAFT unlearning complete for $celeb_name."