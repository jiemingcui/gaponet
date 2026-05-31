---
name: gaponet-inf
description: Run GapONet inference and evaluation. Use when the user invokes /gaponet-inf or $gaponet-inf, or asks to export a GapONet checkpoint to TorchScript/JIT, run lightweight inference, deploy a policy, evaluate test motion data, or produce inference metrics.
---

# GapONet Inference

Run the inference stage from the GapONet repository root, identified by `scripts/run_gaponet_job.py`, `scripts/rsl_rl/inference_jit.py`, and `configs/deploy_default.json`.

## Workflow

1. Ensure data is staged. Prefer `configs/deploy_default.json`, which should point at the data produced by `$gaponet-data`.
2. Decide whether export is needed:
   - If the user provides a JIT policy or `model/policy.pt` exists, run deploy directly.
   - If the user provides a training checkpoint such as `model/model_17950.pt`, export first, then deploy.
   - If neither exists, inspect `runs/train/training_metrics.json` or `logs/` for the latest checkpoint; if none is available, ask for a checkpoint or JIT model path.
3. Export a training checkpoint to JIT when needed:

```bash
python scripts/run_gaponet_job.py --mode export --checkpoint <checkpoint.pt> --output-dir ./runs/inf/export
```

4. Run lightweight inference and evaluation:

```bash
python scripts/run_gaponet_job.py --mode deploy --config configs/deploy_default.json --output-dir ./runs/inf/deploy
```

5. If using the JIT model from the export step, pass it explicitly:

```bash
python scripts/run_gaponet_job.py --mode deploy --checkpoint ./runs/inf/export/policy.pt --input-data ./source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz --output-dir ./runs/inf/deploy
```

## Notes

- Export requires Isaac Sim because it creates the task environment to infer policy dimensions.
- Deploy is lightweight and does not require Isaac Sim, but it does require PyTorch, a JIT model, and test `.npz` data.
- Standard deploy artifacts are `run_manifest.json`, `eval_metrics.json`, `stdout.log`, and `stderr.log`.

## Completion

Consider inference complete only when `runs/inf/deploy/run_manifest.json` reports `"status": "success"` and `eval_metrics.json` exists. Report payload-grouped metrics when present.
