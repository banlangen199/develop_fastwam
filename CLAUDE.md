# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FastWAM (Fast World Action Model) is a robotics video prediction + action model based on Alibaba's Wan2.2-TI2V-5B diffusion backbone. It uses a **Mixture-of-Transformers (MoT)** architecture to jointly generate future video frames and robot actions via a diffusion process. The project supports three model variants (`fastwam`, `fastwam_joint`, `fastwam_idm`) and two benchmarks (LIBERO, RoboTwin).

## Build and environment

```bash
conda create -n fastwam python=3.10 -y && conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

PyTorch 2.7.1+cu128 and DeepSpeed 0.18.5 are required. All other dependencies (accelerate, hydra-core, transformers, etc.) come from `pyproject.toml`.

## Key commands

### Preprocessing (required before training or inference)

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

# Generate ActionDiT backbone (interpolated from Wan2.2 DiT)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

### Precompute T5 text embeddings before training

```bash
# Single GPU
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# Multi-GPU
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### Training

```bash
# LIBERO (single-node 8-GPU by default)
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin (commonly 64 GPUs)
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

`train_zero1.sh` wraps `accelerate launch` with DeepSpeed ZeRO-1. The first argument is `nproc_per_node`. All other arguments are Hydra overrides.

For a new task, set `pretrained_norm_stats` in the data config YAML to `null` first. After one training run, a `dataset_stats.json` is generated in the run directory and can be pointed to for subsequent runs.

### Evaluation (inference)

```bash
# LIBERO (defaults to 8 GPUs, MULTIRUN.num_gpus=8)
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

LIBERO requires a full LIBERO environment install (mujoco==3.3.2). RoboTwin requires following the official RoboTwin setup in `third_party/RoboTwin/`, plus a symlink:
```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Evaluation managers (`run_libero_manager.py` / `run_robotwin_manager.py`) create a task list file, then spawn parallel worker processes across GPUs via tmux. Individual worker evaluation is done by `eval_libero_single.py` / `eval_robotwin_single.py`.

## Architecture

### Configuration system (Hydra)

Everything is driven by Hydra/OmegaConf with four config groups and extensive `${oc.load:...}` / `${eval:...}` resolvers:

| Group | Location | Purpose |
|-------|----------|---------|
| `configs/train.yaml` | Top-level | Base training config (LR, batch size, scheduler, wandb) |
| `configs/model/` | `fastwam.yaml`, `fastwam_joint.yaml`, `fastwam_idm.yaml` | Model architecture. `_target_` points to factory in `runtime.py` |
| `configs/data/` | `libero_2cam.yaml`, `robotwin.yaml` | Dataset paths, image sizes, action/state shapes, processor config |
| `configs/task/` | Named task YAMLs | Compose model+data+hyperparameters overrides per task |

The Hydra resolver `eval` is registered (OmegaConf `eval` resolver), which means config strings can contain arbitrary Python expressions. There are also custom resolvers: `oc.load`, `sum_shapes`, `max_action_dim`, `max_state_dim`, `split`, `round_up`, `round_down`, `max`.

### Core runtime (`src/fastwam/runtime.py`)

Three factory functions (`create_fastwam`, `create_fastwam_joint`, `create_fastwam_idm`) instantiate models from Hydra configs. Each validates the config dict and delegates to the corresponding model class's `from_wan22_pretrained()`.

Training entrypoint: `run_training(cfg)` → instantiates model via `hydra.utils.instantiate(cfg.model)`, builds datasets, creates `Wan22Trainer`, and calls `trainer.train()`.

### Trainer (`src/fastwam/trainer.py`)

`Wan22Trainer` uses **HuggingFace Accelerate** with DeepSpeed (ZeRO-1 or ZeRO-2). Key behaviors:

- **Freezing**: Only `model.dit` (the MoT module) and optionally `model.proprio_encoder` are trainable. VAE, text encoder, and tokenizer are frozen.
- **Optimizer**: AdamW with betas (0.9, 0.95), cosine LR schedule with 5% warmup.
- **Evaluation**: Randomly samples a validation item, runs inference, computes PSNR/SSIM vs ground truth (3-way: rollout vs GT, rollout vs VAE reconstruction, VAE reconstruction vs GT), and optionally action L1/L2.
- **Checkpointing**: Weights saved as `.pt` with keys `mot`, `proprio_encoder`, `step`. Full training state (optimizer/scheduler/step) saved via `accelerator.save_state()`.

### Model architecture (`src/fastwam/models/wan22/`)

**FastWAM** (`fastwam.py`) — The main model:

```
Input: image → VAE encode → latent z
│
├── Video Expert (Wan2.2 Video DiT backbone, ~5B params)
│   └── pre_dit: encode latents → tokens + freqs + t_mod + context
│   └── post_dit: decode tokens → predicted noise
│
├── Action Expert (ActionDiT, ~1B params, interpolated from Wan2.2 DiT)
│   └── pre_dit: encode action tokens + freqs + t_mod + context
│   └── post_dit: decode tokens → predicted action noise
│
└── MoT (Mixture of Transformers)
    └── Each layer: concatenate [video_tokens, action_tokens],
        run mixed self-attention with a joint attention mask,
        split output back, apply per-expert post-blocks (cross-attn + MLP).
    └── Supports KV-cache for video prefill (action-only inference without re-running video).
```

Key design decisions:
- **Separate noise schedulers** for video and action (different `train_shift`/`infer_shift` parameters).
- **Attention mask**: Video uses `first_frame_causal` (first frame attends to all, later frames are causal). Action attends to itself fully and to the first video frame only.
- **Proprioception**: Robot state is projected through a linear layer and appended to the text context tokens.
- **Training loss**: Weighted sum of `lambda_video * video_MSE + lambda_action * action_MSE`, each weighted by the scheduler's per-timestep weight.

Model implementations are separated by family under `src/fastwam/models/wan22/`:
- `fastwam/` — Original FastWAM, including `joint.py` and `idm.py` variants
- `dream_fastwam/` — DreamFastWAM and Dream-query/decoder components
- `action_dream_threshold/` — DreamFastWAM with Action-to-Dream threshold pruning

**FastWAMJoint** (`fastwam/joint.py`) and **FastWAMIDM** (`fastwam/idm.py`): Variants with different MoT mixing strategies and inference procedures.

Supporting modules:
- `action_dit.py` — ActionDiT: smaller DiT variant generated by `preprocess_action_dit_backbone.py`
- `mot.py` — MoT module with mixed attention, KV-cache prefill, gradient checkpointing
- `wan22.py` / `wan_video_dit.py` — Wan2.2 core and Video DiT
- `scheduler_continuous.py` — Continuous-time flow matching scheduler with shift parameter
- `helpers/loader.py` — Downloads and loads Wan2.2-TI2V-5B components from ModelScope

### Datasets (`src/fastwam/datasets/`)

LeRobot-based system with FastWAM-specific extensions:

- `robot_video_dataset.py` — Main dataset class. Loads LeRobot datasets, applies `FastWAMProcessor`, handles multi-camera concatenation, text embedding caching. Returns dicts with keys `video` [B,3,T,H,W], `action` [B,T,dim], `proprio` [B,T,dim], `context` [B,L,D], `context_mask` [B,L].
- `fastwam_processor.py` — Normalizes actions/states, merges action+state via `ActionStateMerger`, applies image transforms. Handles delta vs absolute action encoding (`delta_action_dim_mask`).
- `base_lerobot_dataset.py` — Underlying LeRobot data loading with frame sampling.

### Experiment scripts (`experiments/`)

- **LIBERO** (`experiments/libero/`): Vendors the LIBERO benchmark code in `experiments/libero/libero/`. The `run_libero_manager.py` Hydra app creates a task list across LIBERO suites (libero_10, libero_goal, libero_spatial, libero_object) and farms them out via `run_libero_parallel_test.sh` which manages tmux panes for per-GPU parallel evaluation. `eval_libero_single.py` handles single-task evaluation with the FastWAM model in a LIBERO MuJoCo environment.
- **RoboTwin** (`experiments/robotwin/`): Similar parallel manager pattern. The `fastwam_policy/` directory contains the policy wrapper used by the RoboTwin evaluation framework.

### Key paths (relative to repo root)

| Path | Purpose |
|------|---------|
| `checkpoints/` | Pretrained checkpoints and preprocessed backbones |
| `data/` | Datasets (libero_mujoco3.3.2/, robotwin2.0/) and text_embeds_cache/ |
| `runs/` | Training outputs (`{task_name}/{run_id}/`) |
| `evaluate_results/` | Evaluation output |
| `third_party/RoboTwin/` | Vendored RoboTwin evaluation code |

### Environment variables

- `DIFFSYNTH_MODEL_BASE_PATH` — Root directory for Wan2.2 model downloads (default: `./checkpoints`)
- `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` — Multi-node training via `train_zero1.sh`
- `RUN_ID` — Override auto-generated run ID for training
