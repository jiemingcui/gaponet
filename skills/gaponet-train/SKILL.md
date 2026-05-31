---
name: gaponet-train
description: Run GapONet training. Use when the user invokes /gaponet-train or $gaponet-train, or asks to train a GapONet humanoid policy after data preparation using Isaac Lab, Isaac Sim, rsl_rl, DeepONet, Transformer, MLP, or the unified runner.
---

# GapONet Train

Run the training stage from the GapONet repository root, identified by `scripts/run_gaponet_job.py`, `check_env.py`, and `configs/train_default.json`.

## Workflow

1. Ensure data is staged. If `source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz` is missing, run `$gaponet-data` or ask for an input `.npz`.
2. Validate the environment before training:

```bash
python check_env.py --output-json ./runs/train/env_check_result.json
```

3. If required checks fail for CUDA, Isaac Sim, Isaac Lab, missing scripts, or missing motion data, stop and report the blocking checks.
4. Run training through the unified wrapper:

```bash
python scripts/run_gaponet_job.py --mode train --config configs/train_default.json --output-dir ./runs/train
```

5. If the user provides overrides such as `max_iterations`, `num_envs`, `run_name`, `device`, or `task`, create a run-local JSON config under `runs/train/` by copying `configs/train_default.json`, apply the overrides, and pass that config to the wrapper.

## Defaults

- Training config: `configs/train_default.json`
- Default task: `Isaac-Humanoid-Operator-Delta-Action`
- Data expected by env config: `source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz`
- Output directory: `runs/train`
- Standard artifacts: `run_manifest.json`, `training_metrics.json`, `stdout.log`, `stderr.log`

## Completion

Consider training complete only when `runs/train/run_manifest.json` reports `"status": "success"` or when a clear blocking failure is reported. Training can be long and Isaac Sim dependent; do not fake completion from command launch alone.
