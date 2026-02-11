#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
# Extra benchmarks to run with every model
EXTRA_TASKS="mu_celeb,scienceqa,ai2d,mmbench_en_dev,pope,chartqa,gqa"

# Your model -> task pairs (model_path|primary_mu_task)
pairs=(
  "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_mark/unlearned_llava_model|mu_mark"
  "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_elon/unlearned_llava_model|mu_elon"
  "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_nest_mark/unlearned_llava_model|mu_mark"
  "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_nest_elon/unlearned_llava_model|mu_elon"
  # "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_lee/unlearned_llava_model|mu_bruce"
  # "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_obama/unlearned_llava_model|mu_barack"
  # "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_swift/unlearned_llava_model|mu_taylor"
  # "/media/rania/02B6436D0A908CAC/VLM_weights/vlm_slug_kim/unlearned_llava_model|mu_kim"
)

# Top-level run folder stamped by date/time so each batch is isolated
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="./runs_${STAMP}"
mkdir -p "${ROOT_OUT}"

# Optional: a single global log that aggregates everything
GLOBAL_LOG="${ROOT_OUT}/_global_console.log"
touch "${GLOBAL_LOG}"

# ---------- Loop ----------
for pair in "${pairs[@]}"; do
  model="${pair%%|*}"
  mu_task="${pair##*|}"   # e.g., mu_kanye
  name="$(basename "$(dirname "$model")")"  # folder just above 'unlearned_llava_model'

  # Compose full task list: model-specific MU task + extra suite
  tasks="${mu_task},${EXTRA_TASKS}"

  # Per-model output folder
  RUN_DIR="${ROOT_OUT}/${name}"
  mkdir -p "${RUN_DIR}"

  # lmms_eval output_path (it creates aggregated + per-sample files inside)
  LMMSEVAL_OUT="${RUN_DIR}/lmms_logs"
  mkdir -p "${LMMSEVAL_OUT}"

  # Capture the exact command we run
  CMD_FILE="${RUN_DIR}/command.txt"
  {
    echo "python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \\"
    echo "  --model llava_hf \\"
    echo "  --model_args \"pretrained=${model},device_map=auto\" \\"
    echo "  --tasks \"${tasks}\" \\"
    echo "  --batch_size 1 \\"
    echo "  --log_samples \\"
    echo "  --output_path \"${LMMSEVAL_OUT}\""
  } > "${CMD_FILE}"

  echo ">>> Running ${name}"
  echo "    Model: ${model}"
  echo "    Tasks: ${tasks}"
  echo "    Output directory: ${LMMSEVAL_OUT}"
  echo "    Console log: ${RUN_DIR}/console.txt"

  # Run and tee both to per-model console and the global console
  {
    python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \
      --model llava_hf \
      --model_args "pretrained=${model},device_map=auto" \
      --tasks "${tasks}" \
      --batch_size 1 \
      --log_samples \
      --output_path "${LMMSEVAL_OUT}"
  } 2>&1 | tee -a "${RUN_DIR}/console.txt" | tee -a "${GLOBAL_LOG}" > /dev/null

  echo ">>> Finished ${name}"
  echo
done

echo "All done. Root run folder: ${ROOT_OUT}"
