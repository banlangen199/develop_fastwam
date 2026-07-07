#!/usr/bin/env bash
set -euo pipefail
# Shell wrapper for train_freeze_video.py — identical interface to train_zero1.sh
# but freezes the video expert so only the ActionDiT (300M) is trained.

NPROC_PER_NODE="${1:?Usage: bash scripts/train_freeze_video.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train_freeze_video"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${RANDOM}}"
RUN_DIR="./runs/${TASK_BASENAME}/${RUN_ID}"

write_launch_metadata() {
  local run_dir="$1"
  if (( MACHINE_RANK != 0 )); then
    return 0
  fi
  mkdir -p "${run_dir}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'bash %q %q' "$0" "${NPROC_PER_NODE}"
    for arg in "${EXTRA_ARGS[@]}"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } > "${run_dir}/launch_command.sh"
  chmod +x "${run_dir}/launch_command.sh"
  printf '%s\n' "${EXTRA_ARGS[@]}" > "${run_dir}/hydra_overrides.txt"
}

write_launch_metadata "${RUN_DIR}"

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train_freeze_video.py \
  "output_dir=${RUN_DIR}" \
  "wandb.name=${TASK_BASENAME}_${RUN_ID}" \
  "${EXTRA_ARGS[@]}"
