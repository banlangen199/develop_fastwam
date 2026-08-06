#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/hwdata/txc/FastWAM"
RUN_ROOT="${REPO_ROOT}/runs/dream_fastwam_threshold_finetune_libero_goal_alpha020"
WATCH_LOG="${RUN_ROOT}/wait_gpu45_and_train.log"

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

source /mnt/hwdata/txc/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

gpu_has_compute_process() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF && $1 ~ /^[0-9]+$/ { found=1 } END { exit !found }'
}

echo "[$(date --iso-8601=seconds)] Waiting for GPUs 4 and 5 to have no compute processes." \
  | tee -a "${WATCH_LOG}"
while gpu_has_compute_process 4 || gpu_has_compute_process 5; do
  status="$(nvidia-smi --id=4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
  echo "[$(date --iso-8601=seconds)] GPUs still busy: ${status}" | tee -a "${WATCH_LOG}"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] GPUs 4 and 5 are free; starting alpha=0.2 training." \
  | tee -a "${WATCH_LOG}"

CUDA_VISIBLE_DEVICES=4,5 bash scripts/train_action_dream_threshold.sh 2 \
  task=dream_fastwam_threshold_finetune_libero_goal \
  model.action_dream_threshold.alpha=0.2 \
  model.action_dream_threshold.warmup_ratio=0.0 \
  resume=runs/dream_fastwam_libero_goal/2026-08-03_09-01-44_8756/checkpoints/weights/step_001656.pt \
  'output_dir=./runs/dream_fastwam_threshold_finetune_libero_goal_alpha020/${now:%Y-%m-%d}_${now:%H-%M-%S}' \
  num_epochs=2 2>&1 | tee -a "${WATCH_LOG}"

