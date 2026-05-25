# GapONet: Agent-Friendly Usage Guide

> This document supplements the main `README.md` with everything needed
> for automated pipelines, agent harnesses, and CI systems to use GapONet
> deterministically.

---

## Quick Start (Agent / Harness)

### Step 1 — Clone

```bash
# HTTPS (recommended for agents and CI)
git clone https://github.com/jiemingcui/gaponet.git
cd gaponet

# SSH (for developers with key access)
git clone git@github.com:jiemingcui/gaponet.git
```

### Step 2 — Check environment before anything else

```bash
python check_env.py
# Writes: env_check_result.json  (machine-readable pass/fail/warn per check)
```

Example `env_check_result.json`:

```json
{
  "overall_status": "warn",
  "summary": { "total": 18, "pass": 15, "fail": 0, "warn": 3 },
  "failure_modes": {
    "missing_isaac_sim": true,
    "missing_checkpoints": false,
    "missing_motion_data": false,
    "cuda_unavailable": false,
    "missing_scripts": false
  },
  "checks": [...]
}
```

Abort if `overall_status == "fail"`.  
`"warn"` with `missing_isaac_sim: true` is expected on machines without Isaac Sim —
`deploy` mode still works.

---

## Unified CLI — `scripts/run_gaponet_job.py`

All modes share the same entry point and output contract.

### Train

```bash
python scripts/run_gaponet_job.py \
  --mode train \
  --config configs/train_default.json \
  --output-dir ./runs/exp1
```

### Evaluate (requires Isaac Sim)

```bash
python scripts/run_gaponet_job.py \
  --mode eval \
  --checkpoint ./model/model_17950.pt \
  --output-dir ./runs/exp1
```

### Export checkpoint → JIT (requires Isaac Sim once)

```bash
python scripts/run_gaponet_job.py \
  --mode export \
  --checkpoint ./model/model_17950.pt \
  --output-dir ./runs/exp1
```

### Deploy / Inference (NO Isaac Sim required)

```bash
python scripts/run_gaponet_job.py \
  --mode deploy \
  --checkpoint ./model/policy.pt \
  --input-data ./source/sim2real/sim2real/motions/motion_amass/edited_27dof/test.npz \
  --output-dir ./runs/exp1
```

### Agent / Harness path — input package

```bash
python scripts/run_gaponet_job.py \
  --input-package input_package.json \
  --output-dir ./runs/exp1
```

`input_package.json` schema:

```json
{
  "mode": "deploy",
  "config": {
    "test_data": "./source/sim2real/sim2real/motions/motion_amass/edited_27dof/test.npz",
    "model":     "./model/policy.pt",
    "task":      "Isaac-Humanoid-Operator-Delta-Action"
  }
}
```

---

## Output Artifact Contract

Every run always produces the following in `--output-dir`:

| File | Mode | Description |
|------|------|-------------|
| `run_manifest.json`     | all    | Inputs, status, exit code, elapsed time |
| `eval_metrics.json`     | eval / deploy | Gap ratio, MPJAE, IQR, range, EEF error |
| `training_metrics.json` | train  | Final iteration, reward, checkpoint path |
| `model_manifest.json`   | export | Exported model path, format, task |
| `per_joint_gap.csv`     | deploy | Per-joint gap statistics |
| `stdout.log`            | all    | Full stdout from subprocess |
| `stderr.log`            | all    | Full stderr from subprocess |

### `run_manifest.json`

```json
{
  "schema_version": "1.0",
  "gaponet_version": "0.1.0",
  "timestamp": "2025-05-20T14:32:01",
  "run_id": "20250520_143201",
  "mode": "deploy",
  "status": "success",
  "exit_code": 0,
  "elapsed_seconds": 42.3,
  "output_dir": "./runs/exp1",
  "artifacts": ["eval_metrics.json", "per_joint_gap.csv", "run_manifest.json", ...]
}
```

### `eval_metrics.json`

```json
{
  "schema_version": "1.0",
  "large_gap_ratio": 0.12,
  "gap_iqr":         0.05,
  "gap_range":       0.22,
  "mpjae_deg":       3.4,
  "eef_error":       0.008,
  "threshold_rad":   0.5,
  "num_frames":      1200,
  "num_joints":      27,
  "per_payload": {
    "0.0": { "large_gap_ratio": 0.08, "mpjae_deg": 2.9 },
    "1.0": { "large_gap_ratio": 0.16, "mpjae_deg": 3.9 }
  }
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| 2    | Invalid arguments |
| 3    | Missing required script |
| 4    | Missing checkpoint |
| 5    | Missing input data |
| 6    | Missing config file |
| 10   | Subprocess failed (check stderr.log) |
| 99   | Unexpected error |

---

## Motion Data Schema

All `.npz` files must contain:

| Key | Shape | Description |
|-----|-------|-------------|
| `real_dof_positions`     | `(T, J)` | Joint positions (rad) |
| `real_dof_velocities`    | `(T, J)` | Joint velocities (rad/s) |
| `real_dof_positions_cmd` | `(T, J)` | Target joint positions (rad) |
| `real_dof_torques`       | `(T, J)` | Joint torques (Nm) |
| `joint_sequence`         | `(J,)`   | Joint names for delta actions |
| `payloads`               | `(T,)` or scalar | Payload masses (kg), optional |

Where `T` = timesteps, `J` = number of joints (27 for default config).

---

## Failure Modes Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: No module named 'isaacsim'` | Isaac Sim not installed | Required for `train`/`eval`/`export` only. `deploy` works without it. |
| `ImportError: No module named 'isaaclab'` | Isaac Lab not installed | Run `./isaaclab.sh --install` |
| `FileNotFoundError: scripts/rsl_rl/train.py` | Scripts not synced | Run `./sync_rsl_scripts.sh` after setting `ISAACLAB_PATH` |
| `KeyError: real_dof_positions` | Wrong .npz schema | Check data keys against Motion Data Schema above |
| `CUDA error: no kernel image` | CUDA version mismatch | Match `pytorch-cuda` version in `environment.yml` to your driver |
| `RuntimeError: CUDA out of memory` | Too many envs | Reduce `--num-envs` |
| `No .pt files found in model/` | Checkpoint not downloaded | Download from README Google Drive link, place in `model/` |
| `pinocchio not found` | Optional dep missing | `pip install pinocchio>=2.6.0` (needed for torque computation) |
| exit code 10, check stderr.log | Subprocess crashed | Read `stderr.log` in `--output-dir` for full traceback |

---

## Config File Schema

For `--mode train` with `--config`:

```json
{
  "task":             "Isaac-Humanoid-Operator-Delta-Action",
  "num_envs":         4080,
  "max_iterations":   100000,
  "experiment_name":  "GapONet",
  "run_name":         "my_run",
  "letter":           "amass",
  "device":           "cuda",
  "headless":         true,
  "extra_args":       []
}
```

For `--mode eval`:

```json
{
  "task":       "Isaac-Humanoid-Operator-Delta-Action",
  "checkpoint": "./model/model_17950.pt",
  "num_envs":   20,
  "headless":   true
}
```

For `--mode export`:

```json
{
  "task":         "Isaac-Humanoid-Operator-Delta-Action",
  "checkpoint":   "./model/model_17950.pt",
  "output_model": "./model/policy.pt",
  "device":       "cuda:0",
  "num_envs":     20
}
```

For `--mode deploy`:

```json
{
  "test_data": "./source/sim2real/sim2real/motions/motion_amass/edited_27dof/test.npz",
  "model":     "./model/policy.pt"
}
```
