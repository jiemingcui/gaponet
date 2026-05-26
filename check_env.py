#!/usr/bin/env python3
# Copyright (c) 2022-2025, GapONet Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
GapONet Environment Checker
============================
Run this FIRST before any training/evaluation to verify your environment is ready.

Usage:
    python check_env.py
    python check_env.py --output-json env_check_result.json
    python check_env.py --strict   # exit code 1 if any check fails

Output:
    Prints a human-readable table.
    Writes machine-readable env_check_result.json for agent harnesses.
"""

import argparse
import json
import os
import sys
import platform
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_SKIP = "skip"

COLORS = {
    STATUS_PASS: "\033[92m",   # green
    STATUS_FAIL: "\033[91m",   # red
    STATUS_WARN: "\033[93m",   # yellow
    STATUS_SKIP: "\033[90m",   # grey
    "reset":     "\033[0m",
    "bold":      "\033[1m",
}

def _color(status, text):
    """Wrap text in ANSI color based on status, if stdout supports it."""
    if not sys.stdout.isatty():
        return text
    c = COLORS.get(status, "")
    return f"{c}{text}{COLORS['reset']}"


def _result(name, status, detail="", required=True):
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "required": required,
    }


# ─────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────

def check_python():
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        return _result("Python version", STATUS_PASS, f"Python {version_str}")
    return _result("Python version", STATUS_FAIL,
                   f"Python {version_str} — requires 3.10+")


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            ver = torch.version.cuda
            return _result("CUDA", STATUS_PASS,
                           f"CUDA {ver}, {n} device(s): {name}")
        else:
            return _result("CUDA", STATUS_FAIL,
                           "torch.cuda.is_available() = False. "
                           "GPU is required for Isaac Sim training.")
    except ImportError:
        return _result("CUDA", STATUS_FAIL, "torch not installed")


def check_pytorch():
    try:
        import torch
        return _result("PyTorch", STATUS_PASS, f"torch {torch.__version__}")
    except ImportError:
        return _result("PyTorch", STATUS_FAIL,
                       "torch not found. Run: pip install torch>=2.0.0")


def check_numpy():
    try:
        import numpy as np
        return _result("NumPy", STATUS_PASS, f"numpy {np.__version__}")
    except ImportError:
        return _result("NumPy", STATUS_FAIL, "numpy not found")


def check_gymnasium():
    try:
        import gymnasium as gym
        return _result("Gymnasium", STATUS_PASS, f"gymnasium {gym.__version__}")
    except ImportError:
        return _result("Gymnasium", STATUS_FAIL,
                       "gymnasium not found. Run: pip install gymnasium>=0.28.0")


def check_pinocchio():
    try:
        import pinocchio
        ver = getattr(pinocchio, "__version__", "unknown")
        return _result("Pinocchio", STATUS_PASS, f"pinocchio {ver}")
    except ImportError:
        return _result("Pinocchio", STATUS_WARN,
                       "pinocchio not found. Required for torque computation. "
                       "Run: pip install pinocchio>=2.6.0",
                       required=False)


def check_pytorch_kinematics():
    try:
        import pytorch_kinematics
        ver = getattr(pytorch_kinematics, "__version__", "unknown")
        return _result("pytorch-kinematics", STATUS_PASS, f"version {ver}")
    except ImportError:
        return _result("pytorch-kinematics", STATUS_WARN,
                       "pytorch_kinematics not found. "
                       "Run: pip install pytorch-kinematics>=0.0.1",
                       required=False)


def check_isaac_sim():
    """Try to import isaacsim. Only available inside Isaac Sim Python."""
    try:
        import isaacsim  # noqa: F401
        return _result("Isaac Sim", STATUS_PASS, "isaacsim importable")
    except ImportError:
        # Not fatal for deploy.py (JIT inference), fatal for train/play
        return _result("Isaac Sim", STATUS_WARN,
                       "isaacsim not importable. "
                       "Required for train.py and play.py. "
                       "Not required for deploy.py (JIT inference).",
                       required=False)


def check_isaac_lab():
    """Try to import isaaclab core package."""
    try:
        import isaaclab  # noqa: F401
        ver = getattr(isaaclab, "__version__", "unknown")
        return _result("Isaac Lab", STATUS_PASS, f"isaaclab {ver}")
    except ImportError:
        return _result("Isaac Lab", STATUS_WARN,
                       "isaaclab not importable. "
                       "Required for train.py and play.py. "
                       "Install via: ./isaaclab.sh --install",
                       required=False)


def check_rsl_rl():
    try:
        import rsl_rl  # noqa: F401
        ver = getattr(rsl_rl, "__version__", "unknown")
        return _result("rsl_rl", STATUS_PASS, f"rsl_rl {ver}")
    except ImportError:
        return _result("rsl_rl", STATUS_WARN,
                       "rsl_rl not found. Required for train/play. "
                       "Installed as part of Isaac Lab.",
                       required=False)


def _check_dir(name, path, required=True):
    p = Path(path)
    if p.exists() and p.is_dir():
        count = sum(1 for _ in p.iterdir())
        return _result(name, STATUS_PASS, f"{path}  ({count} items)")
    return _result(name, STATUS_FAIL if required else STATUS_WARN,
                   f"Directory not found: {path}", required=required)


def _check_file(name, path, required=True):
    p = Path(path)
    if p.exists() and p.is_file():
        size_kb = p.stat().st_size // 1024
        return _result(name, STATUS_PASS, f"{path}  ({size_kb} KB)")
    return _result(name, STATUS_FAIL if required else STATUS_WARN,
                   f"File not found: {path}", required=required)


def check_assets(repo_root):
    results = []

    # source packages
    results.append(_check_dir(
        "source/sim2real",
        repo_root / "source" / "sim2real",
        required=True))
    results.append(_check_dir(
        "source/sim2real_assets",
        repo_root / "source" / "sim2real_assets",
        required=True))

    # USD/robot assets
    usds_dir = repo_root / "source" / "sim2real_assets" / "sim2real_assets" / "usds"
    urdfs_dir = repo_root / "source" / "sim2real_assets" / "sim2real_assets" / "urdfs"
    results.append(_check_dir("Robot USD assets", usds_dir, required=False))
    results.append(_check_dir("Robot URDF assets", urdfs_dir, required=False))

    # Motion data
    motion_dir = (repo_root / "source" / "sim2real" / "sim2real"
                  / "motions" / "motion_amass" / "edited_27dof")
    results.append(_check_dir("Motion data directory", motion_dir, required=False))

    # Check for at least one .npz file in motion dir
    if motion_dir.exists():
        npz_files = list(motion_dir.glob("*.npz"))
        if npz_files:
            results.append(_result(
                "Motion .npz files", STATUS_PASS,
                f"{len(npz_files)} .npz file(s) found"))
        else:
            results.append(_result(
                "Motion .npz files", STATUS_WARN,
                f"No .npz files in {motion_dir}. Download test data from README.",
                required=False))

    return results


def check_checkpoints(repo_root):
    results = []
    model_dir = repo_root / "model"
    results.append(_check_dir("model/ directory", model_dir, required=False))

    if model_dir.exists():
        pt_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
        if pt_files:
            results.append(_result(
                "Checkpoint files", STATUS_PASS,
                f"{len(pt_files)} checkpoint(s): " +
                ", ".join(f.name for f in pt_files[:3])))
        else:
            results.append(_result(
                "Checkpoint files", STATUS_WARN,
                "No .pt/.pth files in model/. "
                "Download from README Google Drive link.",
                required=False))
    return results


def check_scripts(repo_root):
    results = []
    scripts = [
        ("scripts/rsl_rl/train.py",        repo_root / "scripts" / "rsl_rl" / "train.py",        True),
        ("scripts/rsl_rl/play.py",         repo_root / "scripts" / "rsl_rl" / "play.py",         True),
        ("scripts/rsl_rl/inference_jit.py",repo_root / "scripts" / "rsl_rl" / "inference_jit.py",True),
        ("scripts/rsl_rl/deploy.py",       repo_root / "scripts" / "rsl_rl" / "deploy.py",       True),
        ("scripts/run_gaponet_job.py",     repo_root / "scripts" / "run_gaponet_job.py",          False),
    ]
    for label, path, required in scripts:
        results.append(_check_file(label, path, required=required))
    return results


def check_npz_schema(repo_root):
    """Validate .npz data keys if test.npz is present."""
    REQUIRED_KEYS = {
        "real_dof_positions",
        "real_dof_velocities",
        "real_dof_positions_cmd",
        "real_dof_torques",
    }
    motion_dir = (repo_root / "source" / "sim2real" / "sim2real"
                  / "motions" / "motion_amass" / "edited_27dof")
    npz_candidates = list(motion_dir.glob("*.npz")) if motion_dir.exists() else []
    npz_candidates += list((repo_root / "model").glob("*.npz")) if (repo_root / "model").exists() else []

    if not npz_candidates:
        return [_result("NPZ data schema", STATUS_SKIP,
                        "No .npz files found to validate",
                        required=False)]

    try:
        import numpy as np
        sample = npz_candidates[0]
        data = np.load(sample, allow_pickle=True)
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            return [_result("NPZ data schema", STATUS_WARN,
                            f"{sample.name}: missing keys {missing}",
                            required=False)]
        return [_result("NPZ data schema", STATUS_PASS,
                        f"{sample.name}: all required keys present "
                        f"({len(data.keys())} total keys)")]
    except Exception as e:
        return [_result("NPZ data schema", STATUS_WARN,
                        f"Could not validate: {e}", required=False)]


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_all_checks(repo_root):
    repo_root = Path(repo_root).resolve()
    sections = {}

    sections["System"] = [
        check_python(),
        check_pytorch(),
        check_cuda(),
    ]

    sections["Python Packages"] = [
        check_numpy(),
        check_gymnasium(),
        check_pinocchio(),
        check_pytorch_kinematics(),
    ]

    sections["Isaac Sim / Isaac Lab"] = [
        check_isaac_sim(),
        check_isaac_lab(),
        check_rsl_rl(),
    ]

    sections["Repository Assets"] = (
        check_assets(repo_root) +
        check_checkpoints(repo_root)
    )

    sections["Scripts"] = check_scripts(repo_root)

    sections["Data Schema"] = check_npz_schema(repo_root)

    return sections


def print_report(sections):
    width = 72
    print("\n" + "=" * width)
    print(f"{'GapONet Environment Check':^{width}}")
    print(f"{'Run: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{width}}")
    print("=" * width)

    total = pass_count = fail_count = warn_count = 0

    for section, results in sections.items():
        print(f"\n  {COLORS['bold']}{section}{COLORS['reset']}")
        print(f"  {'-' * (width - 2)}")
        for r in results:
            icon = {"pass": "✓", "fail": "✗", "warn": "!", "skip": "~"}.get(r["status"], "?")
            label = f"  {icon}  {r['name']:<38}"
            status_text = r["status"].upper()
            detail = f"  {r['detail']}" if r["detail"] else ""
            colored_icon  = _color(r["status"], icon)
            colored_status = _color(r["status"], status_text)
            print(f"  {colored_icon}  {r['name']:<38} [{colored_status}]{detail}")
            total += 1
            if r["status"] == STATUS_PASS:
                pass_count += 1
            elif r["status"] == STATUS_FAIL:
                fail_count += 1
            elif r["status"] == STATUS_WARN:
                warn_count += 1

    print("\n" + "=" * width)
    summary_color = STATUS_FAIL if fail_count else (STATUS_WARN if warn_count else STATUS_PASS)
    summary = (f"  Total: {total}  "
               f"Pass: {pass_count}  "
               f"Fail: {fail_count}  "
               f"Warn: {warn_count}")
    print(_color(summary_color, summary))
    print("=" * width + "\n")

    if fail_count:
        print(_color(STATUS_FAIL,
              "  ✗ Environment NOT ready. Fix FAIL items before proceeding.\n"))
    elif warn_count:
        print(_color(STATUS_WARN,
              "  ! Environment partially ready. "
              "WARN items may affect some functionality.\n"))
    else:
        print(_color(STATUS_PASS,
              "  ✓ Environment is ready.\n"))

    return fail_count, warn_count


def build_json_output(sections):
    all_results = []
    for section, results in sections.items():
        for r in results:
            all_results.append({**r, "section": section})

    fails  = [r for r in all_results if r["status"] == STATUS_FAIL]
    warns  = [r for r in all_results if r["status"] == STATUS_WARN]
    passes = [r for r in all_results if r["status"] == STATUS_PASS]

    overall = "fail" if fails else ("warn" if warns else "pass")

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "overall_status": overall,
        "summary": {
            "total": len(all_results),
            "pass":  len(passes),
            "fail":  len(fails),
            "warn":  len(warns),
        },
        "checks": all_results,
        "failure_modes": {
            "missing_isaac_sim": any(
                r["name"] == "Isaac Sim" and r["status"] != STATUS_PASS
                for r in all_results),
            "missing_checkpoints": any(
                r["name"] == "Checkpoint files" and r["status"] != STATUS_PASS
                for r in all_results),
            "missing_motion_data": any(
                r["name"] == "Motion .npz files" and r["status"] != STATUS_PASS
                for r in all_results),
            "cuda_unavailable": any(
                r["name"] == "CUDA" and r["status"] != STATUS_PASS
                for r in all_results),
            "missing_scripts": any(
                r["section"] == "Scripts" and r["status"] == STATUS_FAIL
                for r in all_results),
        },
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GapONet environment checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        "--repo-root", default=".",
        help="Path to gaponet repo root (default: current directory)")
    parser.add_argument(
        "--output-json", default="env_check_result.json",
        help="Where to write machine-readable JSON result "
             "(default: env_check_result.json)")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any check fails or warns")
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip writing JSON output")
    args = parser.parse_args()

    sections = run_all_checks(args.repo_root)
    fail_count, warn_count = print_report(sections)

    if not args.no_json:
        out = build_json_output(sections)
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  Machine-readable results written to: {out_path}\n")

    if args.strict:
        sys.exit(1 if (fail_count or warn_count) else 0)
    else:
        sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()
