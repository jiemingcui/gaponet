#!/usr/bin/env python3
"""Prepare and validate GapONet motion data.

This script is intentionally lightweight: it validates the `.npz` motion
schema, optionally copies or symlinks it into GapONet's default operator motion
path, updates the deploy config when requested, and writes a data manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPERATOR_DATA = (
    REPO_ROOT
    / "source"
    / "sim2real"
    / "sim2real"
    / "tasks"
    / "humanoid_operator"
    / "motions"
    / "motion_amass"
    / "edited_27dof"
    / "test.npz"
)

CORE_KEYS = [
    "real_dof_positions",
    "real_dof_velocities",
    "real_dof_positions_cmd",
    "real_dof_torques",
]

PROFILE_KEYS = {
    "operator": CORE_KEYS + ["joint_sequence", "payloads"],
    "amass": CORE_KEYS + ["joint_sequence"],
    "deploy": CORE_KEYS,
}


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "./" + str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_summary(value: Any) -> dict[str, Any]:
    summary = {
        "shape": list(getattr(value, "shape", [])),
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
    }
    if getattr(value, "dtype", None) == object:
        lengths = []
        for item in value:
            try:
                lengths.append(len(item))
            except TypeError:
                lengths.append(None)
        numeric_lengths = [item for item in lengths if item is not None]
        summary["object_items"] = len(lengths)
        if numeric_lengths:
            summary["object_length_min"] = min(numeric_lengths)
            summary["object_length_max"] = max(numeric_lengths)
    return summary


def _copy_or_link(input_path: Path, output_path: Path, mode: str, force: bool) -> str:
    if mode == "validate-only":
        return "validated_only"

    if input_path.resolve() == output_path.resolve():
        return "in_place"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        if not force:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --force to overwrite it."
            )
        output_path.unlink()

    if mode == "copy":
        shutil.copy2(input_path, output_path)
        return "copied"
    if mode == "symlink":
        os.symlink(input_path.resolve(), output_path)
        return "symlinked"
    raise ValueError(f"Unsupported mode: {mode}")


def _validate_npz(path: Path, profile: str) -> tuple[dict[str, Any], list[str]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for data preparation. Activate the GapONet "
            "environment or install numpy before running this script."
        ) from exc

    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    required = PROFILE_KEYS[profile]
    missing = [key for key in required if key not in keys]
    errors = [f"Missing required key: {key}" for key in missing]

    summaries = {}
    motion_counts = {}
    for key in keys:
        value = data[key]
        summaries[key] = _array_summary(value)
        if key in CORE_KEYS:
            if getattr(value, "ndim", 0) == 0:
                errors.append(f"{key} must not be scalar")
            else:
                motion_counts[key] = int(value.shape[0])

    if len(set(motion_counts.values())) > 1:
        errors.append(f"Core motion arrays have mismatched first dimensions: {motion_counts}")

    if "payloads" in keys and motion_counts:
        payload_count = int(data["payloads"].shape[0])
        expected = next(iter(motion_counts.values()))
        if payload_count != expected:
            errors.append(f"payloads has {payload_count} motions, expected {expected}")

    if "joint_sequence" in keys and len(data["joint_sequence"]) == 0:
        errors.append("joint_sequence is present but empty")

    report = {
        "path": str(path),
        "profile": profile,
        "keys": keys,
        "required_keys": required,
        "arrays": summaries,
        "motion_counts": motion_counts,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    return report, errors


def _update_downstream_configs(data_path: Path) -> list[str]:
    updated = []
    rel_data_path = _repo_relative(data_path)

    deploy_config = REPO_ROOT / "configs" / "deploy_default.json"
    if deploy_config.exists():
        cfg = _load_json(deploy_config)
        cfg["test_data"] = rel_data_path
        _write_json(deploy_config, cfg)
        updated.append(_repo_relative(deploy_config))

    input_package = REPO_ROOT / "input_package_example.json"
    if input_package.exists():
        pkg = _load_json(input_package)
        config = pkg.setdefault("config", {})
        config["test_data"] = rel_data_path
        _write_json(input_package, pkg)
        updated.append(_repo_relative(input_package))

    return updated


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return _load_json(config_path)


def _arg_or_config(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name.replace("_", "-"), config.get(name, default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and stage GapONet motion data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--input-data", dest="input_data", help="Source .npz file. If omitted, validate --output-data.")
    parser.add_argument("--output-data", dest="output_data", help="Target .npz path for GapONet.")
    parser.add_argument("--output-dir", dest="output_dir", help="Directory for data_manifest.json.")
    parser.add_argument("--profile", choices=sorted(PROFILE_KEYS), help="Validation profile.")
    parser.add_argument("--mode", choices=["copy", "symlink", "validate-only"], help="How to stage input data.")
    parser.add_argument("--force", action="store_true", default=None, help="Overwrite existing staged data.")
    parser.add_argument(
        "--update-configs",
        action="store_true",
        default=None,
        help="Update configs/deploy_default.json and input_package_example.json to use the staged data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    output_dir = REPO_ROOT / "runs" / "data"
    input_data = None
    output_data = None

    try:
        config = _load_config(args.config)
        output_data = Path(_arg_or_config(args, config, "output_data", str(DEFAULT_OPERATOR_DATA)))
        if not output_data.is_absolute():
            output_data = REPO_ROOT / output_data

        input_data_raw = _arg_or_config(args, config, "input_data", None)
        input_data = Path(input_data_raw) if input_data_raw else output_data
        if not input_data.is_absolute():
            input_data = REPO_ROOT / input_data

        output_dir = Path(_arg_or_config(args, config, "output_dir", "./runs/data"))
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir

        profile = _arg_or_config(args, config, "profile", "operator")
        mode = _arg_or_config(args, config, "mode", "copy" if input_data_raw else "validate-only")
        force = bool(_arg_or_config(args, config, "force", False))
        update_configs = bool(_arg_or_config(args, config, "update_configs", False))

        if not input_data.exists():
            raise FileNotFoundError(f"Input data not found: {input_data}")

        staging_action = _copy_or_link(input_data, output_data, mode, force)
        report, errors = _validate_npz(output_data, profile)

        status = "success" if not errors else "failed"
        updated_configs = _update_downstream_configs(output_data) if update_configs and not errors else []
        manifest = {
            "schema_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "elapsed_seconds": round(time.time() - start, 2),
            "repo_root": str(REPO_ROOT),
            "input_data": str(input_data),
            "output_data": str(output_data),
            "output_data_relative": _repo_relative(output_data),
            "mode": mode,
            "profile": profile,
            "staging_action": staging_action,
            "updated_configs": updated_configs,
            "validation": report,
            "errors": errors,
        }
        _write_json(output_dir / "data_manifest.json", manifest)
        print(f"[GapONet] Wrote {output_dir / 'data_manifest.json'}")
        if errors:
            for error in errors:
                print(f"[GapONet] ERROR: {error}", file=sys.stderr)
            return 6
        print(f"[GapONet] Data ready: {output_data}")
        return 0

    except FileNotFoundError as exc:
        manifest = {
            "schema_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "elapsed_seconds": round(time.time() - start, 2),
            "repo_root": str(REPO_ROOT),
            "input_data": str(input_data) if input_data else None,
            "output_data": str(output_data) if output_data else None,
            "errors": [str(exc)],
        }
        _write_json(output_dir / "data_manifest.json", manifest)
        print(f"[GapONet] ERROR: {exc}", file=sys.stderr)
        print(f"[GapONet] Wrote {output_dir / 'data_manifest.json'}")
        return 5
    except Exception as exc:
        manifest = {
            "schema_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "elapsed_seconds": round(time.time() - start, 2),
            "repo_root": str(REPO_ROOT),
            "input_data": str(input_data) if input_data else None,
            "output_data": str(output_data) if output_data else None,
            "errors": [str(exc)],
        }
        _write_json(output_dir / "data_manifest.json", manifest)
        print(f"[GapONet] ERROR: {exc}", file=sys.stderr)
        print(f"[GapONet] Wrote {output_dir / 'data_manifest.json'}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
