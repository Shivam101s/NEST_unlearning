#!/usr/bin/env python3
import subprocess, sys

MODEL_DIR = "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_gaga/unlearned_llava_model"
DATASETS = [
    "ytan-ucr/mu_llava_lady_gaga",
]

for ds in DATASETS:
    print(f"\n=== {ds} ===")
    cmd = [
        sys.executable, "scripts/eval_fa.py",
        "--model-dir", MODEL_DIR,
        "--dataset", ds,
        "--batch", "1",
        "--max-new-tokens", "24",
    ]
    subprocess.run(cmd, check=True)
