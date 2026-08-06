#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_action_dream_threshold.sh <nproc_per_node> [hydra_overrides...]}"
shift

NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train_action_dream_threshold.py \
  "$@"

