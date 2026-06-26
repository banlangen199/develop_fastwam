#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/libero/run_libero_manager.py \
    task=libero_uncond_2cam224_1e-4_100m \
    MULTIRUN.task_suite_names="[libero_goal]"     MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=3 \
    ckpt=runs/libero_uncond_2cam224_1e-4_100m/2026-06-26_05-24-03_3407/checkpoints/weights/step_003308.pt

# CUDA_VISIBLE_DEVICES=0,1 python experiments/libero/run_libero_manager.py \
#     task=libero_uncond_2cam224_1e-4_100m \
#     MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=3 \
#     ckpt=runs/libero_uncond_2cam224_1e-4_100m/2026-06-24_13-33-25_13327/checkpoints/weights/step_008680.pt


# CUDA_VISIBLE_DEVICES=0,1 python experiments/libero/run_libero_manager.py \
#     task=dream_fastwam_libero_goal \
#     MULTIRUN.task_suite_names="[libero_goal]"  MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=3 \
#     ckpt=runs/dream_fastwam_libero_goal/2026-06-25_14-25-22_1673/checkpoints/weights/step_003308.pt

# CUDA_VISIBLE_DEVICES=0,1 python experiments/libero/run_libero_manager.py \
#     task=dream_fastwam_libero_goal \
#     MULTIRUN.task_suite_names="[libero_goal]"  MULTIRUN.num_gpus=2 MULTIRUN.max_tasks_per_gpu=3 \
#     ckpt=runs/dream_fastwam_libero_goal/2026-06-23_08-39-57_17773/checkpoints/weights/step_001654.pt
