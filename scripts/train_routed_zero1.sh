#!/usr/bin/env bash
# Single-node launcher for RoutedWAM training.
#
# Why this exists rather than reusing scripts/train_zero1.sh: that script hard
# codes `scripts/train.py`, whose `run_training` builds a plain `Wan22Trainer`.
# RoutedWAM needs `RoutedWan22Trainer`, which is what feeds optimizer-step
# progress to the router warmup, the distillation ramp and the EMA teacher
# update. Without it the router would never leave warmup and the teacher would
# never move. `scripts/train_zero1.sh` is left untouched so every existing
# FastWAM / Dream / threshold run stays reproducible.
#
# Scope: single node. The multi-node run-id rendezvous in train_zero1.sh is a
# subtle TCPStore protocol and duplicating it here would be a second place to
# get it wrong; if you need multi-node, export RUN_ID identically on every node
# (see train_zero1.sh's run_id_sync block) and this script will use it.
#
# Usage:
#   bash scripts/train_routed_zero1.sh 8 task=routed_wam_libero_goal
set -euo pipefail

# Every path below (`scripts/accelerate_configs/...`, `scripts/train_routed_wam.py`)
# is relative to the repository root. When a job scheduler invokes this script
# directly rather than through a wrapper that has already cd'd, accelerate dies
# with "The passed configuration file ... does not exist", so cd here.
REPO_ROOT="${WORKING_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${1:?Usage: bash scripts/train_routed_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift
EXTRA_ARGS=("$@")

NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

if (( NUM_MACHINES > 1 )) && [[ -z "${RUN_ID:-}" ]]; then
  echo "Error: multi-node RoutedWAM training requires RUN_ID to be exported" >&2
  echo "       identically on every node, so all ranks agree on the output dir." >&2
  exit 1
fi

TASK_BASENAME="train"
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    task=*)
      cfg="${arg#task=}"
      TASK_BASENAME="${cfg%.yaml}"
      ;;
  esac
done

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${RANDOM}}"

# On a scheduler where `./runs` lives inside a container that disappears with
# the job, every checkpoint would go with it. Point FASTWAM_RUN_ROOT at a
# mounted path and the run writes there directly, so a preempted job keeps its
# outputs and no separate copy-back step can be forgotten.
RUN_ROOT="${FASTWAM_RUN_ROOT:-./runs}"
RUN_DIR="${RUN_ROOT}/${TASK_BASENAME}/${RUN_ID}"

if (( MACHINE_RANK == 0 )); then
  mkdir -p "${RUN_DIR}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'bash %q %q' "$0" "${NPROC_PER_NODE}"
    for arg in "${EXTRA_ARGS[@]}"; do printf ' %q' "${arg}"; done
    printf '\n'
  } > "${RUN_DIR}/launch_command.sh"
  chmod +x "${RUN_DIR}/launch_command.sh"
  printf '%s\n' "${EXTRA_ARGS[@]}" > "${RUN_DIR}/hydra_overrides.txt"
fi

echo "[launch] routed_wam nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

TOTAL_PROCESSES=$((NPROC_PER_NODE * NUM_MACHINES))

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${TOTAL_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --deepspeed_multinode_launcher standard \
  scripts/train_routed_wam.py \
  "output_dir=${RUN_DIR}" \
  "wandb.name=${TASK_BASENAME}_${RUN_ID}" \
  "${EXTRA_ARGS[@]}"
