# Action-to-Dream alpha calibration

This directory contains the complete calibration workflow for the shared,
non-learned Action-to-Dream threshold `alpha`.

The default experiment uses the dense LIBERO Goal checkpoint. It selects
calibration observations by episode, profiles dense mixed attention at every
step of the same 20-step Action denoising trajectory, performs an offline alpha
sweep, measures paired Action flow-matching loss with identical noise/timesteps,
and evaluates two policies separately:

1. a mask recomputed independently at every denoising step;
2. the layer-wise mask computed at step 0 and reused for all later steps.

The dynamic-mask recommendation chooses the smallest mean K satisfying:

- original Dream attention mass retention >= 95%;
- mean relative Action-loss increase <= 1%.

`K=0` frequency is saved for diagnosis but is not a selection constraint.

The fixed-step0 analysis additionally checks the minimum later-token recall and
minimum dense Dream-mass retention over the trajectory. Its recommendation is
explicitly labelled an **offline candidate**: later scores are measured on the
dense trajectory, so rollout evaluation is still required after implementing
causal fixed-mask execution.

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
action_dream_alpha/outputs/libero_goal_first_step_reuse_step_001656_seed42/
```

Important files are `calibration_split.json`, `dense_summary.json`,
`alpha_sweep.csv`, `paired_action_loss.csv`, `per_layer.csv`,
`per_timestep.csv`, `per_task.csv`, `per_token_group.csv`, and
`recommended_alpha.json`. The fixed-step0 reports are:

- `first_step_reuse.csv`: aggregate overlap and mass retention by alpha/step;
- `first_step_reuse_per_layer.csv`: layer-wise worst cases;
- `first_step_reuse_per_task.csv`: task-wise stability;
- `recommended_first_step_alpha.json`: constrained offline candidate and scope;
- `plots/first_step_reuse_*.png`: mask/mass/K trajectories.

The most important reuse fields are:

- `later_token_recall_micro`: fraction of tokens selected at the current step
  that were already present in the step-0 mask;
- `mask_jaccard_micro`: exact set overlap, which may fall when step 0 is a
  conservative superset;
- `fixed_mass_retention`: current dense Dream mass covered by the step-0 mask;
- `mean_new_tokens_per_unit`: later-selected tokens missing from step 0, per
  sample/layer.

Profile shards contain only
Dream score, per-token original Dream mass, source mass, and numerical checks;
the full `[head, Action query, all keys]` probability matrix is never saved.

To add the new reports to an existing profile without running the GPU stages
again, summarize with that run's resolved config:

```bash
python action_dream_alpha/calibrate.py \
  --config action_dream_alpha/outputs/OLD_RUN/resolved_config.yaml \
  --stage summarize
```

Before a new run, update `checkpoint`, `dataset_stats_path`, and `output_dir` in
`libero_goal.yaml`. Do not reuse an output directory after changing the
checkpoint or profile configuration.

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
