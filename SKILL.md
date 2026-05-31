---
name: gaponet
description: GapONet sim-to-real humanoid robot control workflows for Isaac Lab. Use when working on the GapONet repository or tasks involving GapONet setup, environment validation, training, evaluation, checkpoint export, lightweight deployment, DeepONet, Transformer or MLP policy configs, humanoid operator or AMASS tasks, motion data schemas, robot asset integration, or agent-friendly run artifacts.
---

# GapONet

This is the high-level GapONet router skill. For executable workflow stages, prefer the split skills:

- `$gaponet-data` for motion `.npz` validation and staging.
- `$gaponet-train` for Isaac Lab training through the unified runner.
- `$gaponet-inf` for checkpoint export, lightweight inference, and evaluation.

Treat the skill directory as the GapONet repository root unless the user gives another path. GapONet is a Python 3.10 Isaac Lab and Isaac Sim project for sim-to-real humanoid robot control. It contains custom `sim2real` tasks, `sim2real_assets`, vendored Isaac Lab packages, rsl_rl training scripts, and a deterministic wrapper for agent or CI runs.

## Start

1. Inspect local files first. Prefer `README.md`, `check_env.py`, `scripts/run_gaponet_job.py`, `configs/*.json`, and the relevant task or model config before changing code.
2. Do not assume Isaac Sim, CUDA, checkpoints, robot USD or URDF assets, or `.npz` motion data are present. They are required for many workflows and are intentionally not all committed.
3. Prefer the unified runner for user-facing execution unless the task specifically asks for a lower-level script.
4. Keep heavy runs opt-in. Training, evaluation, and export can start Isaac Sim and take a long time. Lightweight deploy only needs PyTorch plus a JIT policy and test `.npz`.

## Repository Map

- `README.md`: project overview, installation, common commands, new robot workflow, data schema, and metrics.
- `check_env.py`: environment checker. It reports Python, CUDA, PyTorch, Isaac Sim, Isaac Lab, rsl_rl, assets, checkpoints, scripts, and `.npz` schema readiness.
- `configs/`: default JSON configs for `train`, `eval`, `export`, and `deploy`.
- `input_package_example.json`: JSON package format for the unified runner.
- `scripts/run_gaponet_job.py`: deterministic entry point for `train`, `eval`, `export`, and `deploy`; writes manifests and logs.
- `scripts/rsl_rl/train.py`, `play.py`, `inference_jit.py`, `deploy.py`: lower-level training, playback, checkpoint export, and JIT inference scripts.
- `scripts/output_writer.py`: machine-readable artifact writer used by deploy and compatible evaluation scripts.
- `source/sim2real/sim2real/tasks/humanoid_operator/`: default operator task, payload handling, sensor-position sub-environments, model history, and DeepONet runner configs.
- `source/sim2real/sim2real/tasks/humanoid_amass/`: AMASS motion tracking tasks with MLP and Transformer registrations.
- `source/sim2real/sim2real/rsl_rl/`: custom DeepONet, Transformer, runner, and network implementations.
- `source/sim2real_assets/sim2real_assets/robots/`: robot config definitions for Unitree and Fourior variants; external USD or URDF assets may still need to be downloaded.
- `source/isaaclab*`: local Isaac Lab, assets, tasks, mimic, and RL packages. Use these local copies as the source of truth for project-specific API behavior.

## Stable Commands

Validate the machine before running GapONet:

```bash
python check_env.py --output-json env_check_result.json
```

Run through the unified wrapper:

```bash
python scripts/run_gaponet_job.py --mode train --config configs/train_default.json --output-dir ./runs/exp1
python scripts/run_gaponet_job.py --mode eval --config configs/eval_default.json --output-dir ./runs/exp1
python scripts/run_gaponet_job.py --mode export --config configs/export_default.json --output-dir ./runs/exp1
python scripts/run_gaponet_job.py --mode deploy --config configs/deploy_default.json --output-dir ./runs/exp1
python scripts/run_gaponet_job.py --input-package input_package_example.json --output-dir ./runs/exp1
```

The wrapper always writes `run_manifest.json`, `stdout.log`, and `stderr.log` under `--output-dir`. Depending on mode it also writes `training_metrics.json`, `eval_metrics.json`, or `model_manifest.json`.

Use lower-level scripts when debugging specific behavior:

```bash
python scripts/rsl_rl/train.py --task Isaac-Humanoid-Operator-Delta-Action --num_envs=4080 --max_iterations 100000 --experiment_name GapONet --letter amass --run_name default_run --device cuda env.mode=train --headless
python scripts/rsl_rl/play.py --task Isaac-Humanoid-Operator-Delta-Action --model ./model/model_17950.pt --num_envs 20 --headless
python scripts/rsl_rl/inference_jit.py --export --checkpoint ./model/model_17950.pt --task Isaac-Humanoid-Operator-Delta-Action --output ./model/policy.pt --device cuda:0 --num_envs 20
python scripts/rsl_rl/deploy.py --model ./model/policy.pt --test_data ./source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz --device cuda:0 --output-dir ./runs/exp1
```

## Task And Model Routing

- Default task: `Isaac-Humanoid-Operator-Delta-Action`, registered in `source/sim2real/sim2real/tasks/humanoid_operator/__init__.py`.
- Operator variants: `Isaac-Humanoid-Operator-Delta-Action-Fourior` and `Isaac-Humanoid-Operator-Delta-Action-Vanilla`.
- AMASS tasks: `Isaac-Humanoid-Amass-Delta-Action-MLP` and `Isaac-Humanoid-Amass-Delta-Action-Transformer`.
- DeepONet policy config for the default operator task lives in `source/sim2real/sim2real/tasks/humanoid_operator/agents/rsl_rl_operator_cfg.py`.
- Environment dimensions for model history, sensors, payloads, and action space live in `source/sim2real/sim2real/tasks/humanoid_operator/humanoid_operator_env_cfg.py`.
- Keep `model_history_length`, `model_history_dim`, branch input dimensions, trunk input dimensions, and action dimensions consistent between env configs, policy configs, export, and deploy paths.

## Data And Assets

Motion data is `.npz`. Required keys used by validation and deploy are:

- `real_dof_positions`
- `real_dof_velocities`
- `real_dof_positions_cmd`
- `real_dof_torques`

Optional but commonly used keys include `joint_sequence` and `payloads`.

Expected default paths include:

- Motion data: `source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz`
- Checkpoints: `model/model_17950.pt`
- Exported JIT policy: `model/policy.pt`
- Robot assets: `source/sim2real_assets/sim2real_assets/usds/` and `source/sim2real_assets/sim2real_assets/urdfs/`

The `.gitignore` excludes checkpoints, `.npz`, logs, runs, and generated media. Do not add large binary artifacts unless the user explicitly asks.

## Editing Guidance

- For a new robot or task, follow the local pattern: create a task package under `source/sim2real/sim2real/tasks/`, register it with `gym.register()`, add robot configs under `source/sim2real_assets/sim2real_assets/robots/`, place external assets in the expected asset directories, create agent configs, then run through `scripts/run_gaponet_job.py`.
- For runner or artifact work, edit `scripts/run_gaponet_job.py` and `scripts/output_writer.py` first; preserve stable filenames and exit codes because automation may depend on them.
- For deploy behavior, inspect `scripts/rsl_rl/deploy.py`. It constructs `model`, `operator.branch`, and `operator.trunk` observations and computes payload-grouped gap metrics without Isaac Sim.
- For export behavior, inspect `scripts/rsl_rl/inference_jit.py`. Export requires Isaac Sim to build the task environment and infer dimensions, but the resulting TorchScript model is meant for lightweight inference.
- If `scripts/rsl_rl/*.py` are missing, run or inspect `sync_rsl_scripts.sh`; the wrapper reports this as a missing script failure.
- Use `scripts/run_gaponet_job.py`, `check_env.py`, and `scripts/output_writer.py` as the source of the agent-facing contract.

## Validation

- Validate skill syntax with the Codex skill validation tool when available, or parse each `SKILL.md` frontmatter as YAML and ensure every skill name is lowercase hyphen-case.
- Validate repository readiness with `python check_env.py --output-json env_check_result.json`.
- For code changes that do not require Isaac Sim, prefer focused syntax or unit checks plus `deploy` if test data and a JIT model are present.
- Avoid starting long Isaac Sim training or playback runs unless requested, or unless the user has already approved that style of verification.

## Exit Codes

`scripts/run_gaponet_job.py` uses stable failure codes: `2` invalid args, `3` missing script, `4` missing checkpoint, `5` missing data, `6` missing config, `10` subprocess failed, and `99` unexpected error.
