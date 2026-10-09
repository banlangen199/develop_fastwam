#!/usr/bin/env bash
# Usage: bash scripts/train_dream_ablation.sh <group> [--dry-run]
set -euo pipefail

usage() {
  printf '用法：bash scripts/train_dream_ablation.sh <实验 ID> [--dry-run]\n'
  printf '实验 ID：none dyn depth dino sam all all_no_sup all_no_dino all_no_sam all_no_depth all_no_dyn\n'
  printf '默认使用 GPU 0,1,2,3；缺少 accelerate 时自动使用 fastwam Conda 环境。\n'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 1
fi
if (( $# == 2 )) && [[ "$2" != "--dry-run" ]]; then
  usage >&2
  exit 1
fi

group="$1"
dyn=0 depth=0 dino=0 sam=0
case "$group" in
  none)       active='[]' ;;
  dyn)        active='[dyn]'; dyn=1 ;;
  depth)      active='[depth]'; depth=1 ;;
  dino)       active='[dino]'; dino=1 ;;
  sam)        active='[sam]'; sam=1 ;;
  all)        active='[dyn,depth,dino,sam]'; dyn=1; depth=1; dino=1; sam=1 ;;
  all_no_sup) active='[dyn,depth,dino,sam]' ;;
  all_no_dino) active='[dyn,depth,sam]'; dyn=1; depth=1; sam=1 ;;
  all_no_sam)  active='[dyn,depth,dino]'; dyn=1; depth=1; dino=1 ;;
  all_no_depth) active='[dyn,dino,sam]'; dyn=1; dino=1; sam=1 ;;
  all_no_dyn)  active='[depth,dino,sam]'; depth=1; dino=1; sam=1 ;;
  *) printf '未知实验：%s\n' "$group" >&2; exit 1 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RUN_ID="${RUN_ID:-${group}_seed42}"

run_dir="./runs/dream_ablation_libero_10/$RUN_ID"
command=(
  bash scripts/train_routed_zero1.sh 4
  task=dream_ablation_libero_10
  resume=./checkpoints/libero_uncond_2cam224_100m.pt
  seed=42
  batch_size=32 gradient_accumulation_steps=2
  expected_global_batch_size=256 max_steps=4000
  learning_rate=1e-4 weight_decay=1e-2
  save_every=1500 log_every=10 eval_every=0
  "model.active_dream_modalities=$active"
  "model.loss.lambda_dyn=$dyn"
  "model.loss.lambda_depth=$depth"
  "model.loss.lambda_dino=$dino"
  "model.loss.lambda_sam=$sam"
)

# Use the installed fastwam environment when the current shell lacks Accelerate.
if ! command -v accelerate >/dev/null 2>&1; then
  conda_exe="${CONDA_EXE:-}"
  if [[ ! -x "$conda_exe" ]]; then
    conda_exe="$(type -P conda || true)"
  fi
  if [[ -z "$conda_exe" ]]; then
    printf '未找到 accelerate 或 conda；请先激活 fastwam 环境后重试。\n' >&2
    exit 127
  fi
  command=("$conda_exe" run --no-capture-output -n fastwam "${command[@]}")
fi

if [[ "${2:-}" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q RUN_ID=%q ' "$CUDA_VISIBLE_DEVICES" "$RUN_ID"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "$run_dir/config.yaml" ]]; then
  printf '已有 run：%s；请检查原任务，或设置新的 RUN_ID，避免覆盖。\n' "$run_dir" >&2
  exit 1
fi

exec "${command[@]}"
