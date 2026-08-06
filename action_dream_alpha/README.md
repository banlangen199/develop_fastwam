# Action-to-Dream alpha calibration

This directory contains the complete calibration workflow for the shared,
non-learned Action-to-Dream threshold `alpha`.

The default experiment uses the dense LIBERO Goal checkpoint. It selects
calibration observations by episode, profiles dense mixed attention along real
Action denoising trajectories, performs an offline alpha sweep, measures paired
Action flow-matching loss with identical noise/timesteps, and recommends the
smallest mean K satisfying:

- original Dream attention mass retention >= 95%;
- mean relative Action-loss increase <= 1%.

`K=0` frequency is saved for diagnosis but is not a selection constraint.

## Run

Run the entire resumable workflow:

```bash
python action_dream_alpha/calibrate.py \
  --config action_dream_alpha/libero_goal.yaml
```

Or run stages separately:

```bash
python action_dream_alpha/calibrate.py --stage profile
python action_dream_alpha/calibrate.py --stage paired-loss
python action_dream_alpha/calibrate.py --stage summarize
```

The default output directory is:

```text
action_dream_alpha/outputs/libero_goal_dense_step_001656_seed42/
```

Important files are `calibration_split.json`, `dense_summary.json`,
`alpha_sweep.csv`, `paired_action_loss.csv`, `per_layer.csv`,
`per_timestep.csv`, `per_task.csv`, `per_token_group.csv`, and
`recommended_alpha.json`. Profile shards contain only
Dream score, per-token original Dream mass, source mass, and numerical checks;
the full `[head, Action query, all keys]` probability matrix is never saved.

## Visualize threshold decisions during LIBERO evaluation

The visualization evaluator runs normal thresholded policy rollouts and asks
for detailed attention only on selected replans, layers, and denoising steps.
For example:

```bash
CUDA_VISIBLE_DEVICES=6 python action_dream_alpha/visualize_threshold_eval.py \
  --checkpoint runs/dream_fastwam_threshold_finetune_libero_goal/RUN/checkpoints/weights/STEP.pt \
  --alpha 0.2 \
  --task-id 0 \
  --num-episodes 1 \
  --visualize-replans 3 \
  --layers 0,5,10,15,20,25,29
```

Repeat with `--alpha 0.25` for a direct comparison. Each selected
layer/denoising-step record produces:

- `*.dashboard.png`: score/threshold decisions, dense and pruned attention,
  Action-query heatmaps, source mass, and modality/horizon retention;
- `*.tokens.csv`: one row per Dream token with identity, score, keep/prune
  decision, and probability before/after pruning;
- `*.attention.npz`: machine-readable per-query and optional per-head arrays;
- `*.summary.json`: K and source-mass summary.

The run root also contains `threshold_attention_summary.csv` and
`eval_results.json`. Attention saving does not run for replans after
`--visualize-replans`, so the remainder of the rollout follows the normal
evaluation path without retaining large debug tensors.
