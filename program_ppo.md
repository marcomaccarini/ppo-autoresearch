# autoresearch PPO, segmented intervention mode

This is an experiment to have the coding agent do its own PPO research during a single long training run.

The run is not a sequence of independent full experiments. A full PPO run is **20,000,000 environment steps**, divided into **20 intervention windows of 1,000,000 environment steps**. After each 1M-step segment, training stops, a checkpoint is saved, deterministic evaluation is run, metrics are written, and you inspect the result before editing `train.py` for the next segment.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. Use it as `--exp_name` if the user wants multiple independent runs. If no tag is passed, the deterministic run directory is `runs/PickCube-v1__ppo_rgb_autoresearch__seed1`.
2. **Create the branch**: `git checkout -b autoresearch-ppo/<tag>` from the current clean base.
3. **Read the in-scope files**:
   - `README.md`, if present, for repository context.
   - `prepare.py` — fixed args, env construction, agent class, logger, deterministic evaluation, checkpoint IO, metric logging. Do not modify.
   - `train.py` — the only file you modify. PPO update logic, training dynamics, schedules, losses, and safe checkpoint-compatible interventions.
   - `ppo_rgb.py` — segmented entrypoint. Do not modify unless the human explicitly asks.
4. **Verify environment**: make sure ManiSkill, Gymnasium, PyTorch, TensorBoard, and tyro imports work. Do not install new packages unless the human explicitly approves.
5. **Initialize results**: metrics are written automatically to `<run_dir>/segment_metrics.tsv` and `<run_dir>/segment_metrics.jsonl`. Do not commit those files.
6. **Confirm and go**: run the first segment as the baseline.

## Launch command

Use the user's command. By default it runs **one 1M-step segment and exits**:

```bash
CUDA_VISIBLE_DEVICES=1 python ppo_rgb.py --env_id="PickCube-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=20_000_000
```

After each segment, inspect the printed summary and the metrics TSV, edit `train.py`, commit, and run the exact command again. The script resumes from `<run_dir>/checkpoints/latest.pt` and continues until `global_step >= total_timesteps`.

Do not use `--auto_continue` during autoresearch, because it prevents intervention between segments.

## What you CAN do

Modify `train.py` only.

During a segmented 20M-step run, prefer changes that are compatible with loading the existing checkpoint:

- learning-rate schedule
- PPO clipping behavior
- KL early stopping / target KL logic
- entropy coefficient or entropy scheduling
- value loss coefficient
- clipped value loss
- advantage normalization
- reward scaling
- gradient clipping
- optimizer hyperparameters, if compatible
- number of PPO epochs
- minibatch logic
- logstd clamps or schedules that do not change parameter shapes
- diagnostics and safer fallback logic

## What you CANNOT do during a segmented run

Do not modify:

- `prepare.py`
- deterministic evaluation
- checkpoint format, unless absolutely necessary and backward compatible
- environment construction
- task, observation mode, reward mode, or action space
- the metric parser/output format
- dependencies

Avoid checkpoint-incompatible architecture changes during a 20M segmented run:

- hidden size changes
- number of layers changes
- parameter shape changes
- actor/critic module renaming
- action distribution shape changes

Architecture changes are allowed only for a fresh run starting from step 0, not mid-run from an existing checkpoint.

## Goal

The goal is to improve the policy within the same 20M-step training budget.

Primary objective:

```text
maximize AUC of deterministic eval success rate over environment steps
```

Secondary objectives:

```text
maximize final eval_success_rate
maximize eval_return_mean
reduce time-to-first-success / time-to-50%-success / time-to-80%-success
maintain or improve SPS
avoid instability: NaNs, KL explosions, clipfrac saturation, value collapse
```

Do not optimize PPO losses directly. Policy loss, value loss, entropy, approx KL and clipfrac are diagnostics. The policy is better only if deterministic evaluation behavior improves.

## Segment output format

Each segment prints a parseable summary like:

```text
---
segment: 3
global_step: 3000320
eval_success_rate: 0.42
eval_return_mean: 0.71
auc_success_per_mstep: 0.54
SPS: 43000
approx_kl: 0.0123
clipfrac: 0.19
entropy: 1.42
value_loss: 0.08
policy_loss: -0.02
explained_variance: 0.63
wall_time_sec: 25.4
peak_vram_mb: 6840.2
status: segment_done
checkpoint: runs/.../checkpoints/latest.pt
```

You can extract core metrics from logs with:

```bash
grep "^segment:\|^global_step:\|^eval_success_rate:\|^eval_return_mean:\|^auc_success_per_mstep:\|^SPS:\|^approx_kl:\|^clipfrac:\|^entropy:\|^status:" run.log
```

## Logging results

Metrics are automatically appended to `<run_dir>/segment_metrics.tsv` with columns:

```text
segment global_step commit eval_success_rate eval_return_mean auc_success_per_mstep SPS approx_kl clipfrac entropy value_loss policy_loss explained_variance wall_time_sec status description
```

You may also keep a separate `results.tsv` in the repository root for human-readable intervention notes. Do not commit it.

Recommended root `results.tsv` columns:

```text
segment commit global_step eval_success_rate auc_success_per_mstep status intervention diagnosis
```

Status values:

- `keep`: intervention helped or is plausibly useful
- `rollback`: intervention clearly damaged training and the checkpoint was restored
- `crash`: code crashed/OOM/NaN and had to be reverted
- `complete`: run reached 20M steps

## Intervention loop

The run lives on a dedicated branch, for example `autoresearch-ppo/jun3`.

LOOP UNTIL `global_step >= 20_000_000`:

1. Check git state and current commit.
2. Run the next segment:

   ```bash
   CUDA_VISIBLE_DEVICES=1 python ppo_rgb.py --env_id="PickCube-v1" \
     --num_envs=256 --update_epochs=8 --num_minibatches=8 \
     --total_timesteps=20_000_000 > run.log 2>&1
   ```

3. Read the summary from `run.log` and `<run_dir>/segment_metrics.tsv`.
4. Diagnose learning dynamics:
   - success improving? keep direction.
   - success flat and KL low? updates may be too weak.
   - success collapses and KL/clipfrac high? updates may be too aggressive.
   - entropy collapses too early? exploration may be insufficient.
   - value loss high / explained variance poor? critic or return scaling may be limiting.
   - SPS badly lower? complexity may be too expensive.
5. Edit `train.py` with one targeted, checkpoint-compatible intervention.
6. Commit the intervention.
7. Launch the same command again for the next segment.
8. If the segment crashes because of a trivial typo, fix and rerun. If the idea is bad, revert the commit and restore `<run_dir>/checkpoints/before_segment.pt` to `<run_dir>/checkpoints/latest.pt`.
9. Never ask the human whether to continue once the loop has started.

## Rollback procedure

If an intervention clearly harms training or crashes:

```bash
cp <run_dir>/checkpoints/before_segment.pt <run_dir>/checkpoints/latest.pt
git reset --hard HEAD~1
```

Then try a different intervention from the restored checkpoint.

## Timeout

A 1M-step segment should usually take on the order of seconds to a few minutes depending on SPS and evaluation overhead. If a segment hangs or takes far longer than expected, kill it, inspect the last 100 log lines, and treat it as a crash unless there is an obvious fix.

## Simplicity criterion

All else equal, simpler is better. A small success-rate gain from a clean schedule or coefficient change is valuable. A small gain from fragile, hard-coded logic is not. A neutral performance change that simplifies `train.py` can be kept.

## NEVER STOP

Once the segmented loop begins, do not pause to ask the human whether to continue. Continue until the run reaches 20M steps or the human manually interrupts you.
