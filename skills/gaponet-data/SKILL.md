---
name: gaponet-data
description: Run GapONet data preparation. Use when the user invokes /gaponet-data or $gaponet-data, or asks to process, validate, stage, copy, symlink, or prepare GapONet motion .npz data for training or inference.
---

# GapONet Data

Run the GapONet data stage, not just a review. Work from the GapONet repository root, identified by `scripts/prepare_gaponet_data.py`, `scripts/run_gaponet_job.py`, and `configs/`.

## Workflow

1. Resolve the repository root and work from it.
2. If the user supplied an input `.npz`, stage it into the default operator motion path:

```bash
python scripts/prepare_gaponet_data.py --input-data <input.npz> --output-dir ./runs/data --mode copy --profile operator --force --update-configs
```

3. If the user did not supply an input path, validate the configured/default data:

```bash
python scripts/prepare_gaponet_data.py --config configs/data_default.json
```

4. If the default data is missing, ask for the source `.npz` path. Do not invent or synthesize motion data.
5. Report the generated `runs/data/data_manifest.json` status, the staged `output_data`, and any validation errors.

## Defaults

- Staged data: `source/sim2real/sim2real/tasks/humanoid_operator/motions/motion_amass/edited_27dof/test.npz`
- Data config: `configs/data_default.json`
- Manifest: `runs/data/data_manifest.json`
- Downstream configs updated by `--update-configs`: `configs/deploy_default.json` and `input_package_example.json`

## Validation Profile

Use the `operator` profile by default. It requires:

- `real_dof_positions`
- `real_dof_velocities`
- `real_dof_positions_cmd`
- `real_dof_torques`
- `joint_sequence`
- `payloads`

Use `--profile deploy` only when the user explicitly wants lightweight inference validation that does not require the operator training keys.

## Completion

Consider the data stage complete only when `data_manifest.json` has `"status": "success"` and the staged `.npz` path exists. If Python lacks NumPy, tell the user to activate the GapONet environment before rerunning the data stage.
